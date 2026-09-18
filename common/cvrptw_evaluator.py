"""Shared batch-aware evaluation driver; method modules provide runtime hooks."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from common.cvrptw_artifacts import (
    append_batch, finalize, initialize, require_our2_gate, require_our5_gate,
    require_our5_prefix_matches_our2, require_small_gate,
    require_validation_gate, require_validation_prefix_matches_small,
    scope_instance_count,
)
from common.cvrptw_formal import (TIMING_SEMANTICS, dataset_config,
                                  instance_drop_percent, kit_validate,
                                  validate_solution)
from common.cvrptw_runtime import require_cuda_4090, stack_native
from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from common.paper_results import write_json
from common.provenance import (environment_provenance, git_provenance,
                               normalize_git_repository_identity, source_provenance)


def _package_versions():
    values = {}
    for name in ("rl4co", "tensordict", "torchrl", "lightning",
                 "pytorch-lightning", "einops", "entmax", "ml4co-kit"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def _parser(description, formal_batch_sizes):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-metadata", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, choices=(50, 100), required=True)
    parser.add_argument("--batch-size", type=int, choices=formal_batch_sizes, default=1)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=("preflight", "small_gate", "validation_gate", "production",
                 "our_2", "our_5"), required=True)
    parser.add_argument("--validation-gate-evidence", type=Path)
    parser.add_argument("--small-gate-evidence", type=Path)
    parser.add_argument("--our5-evidence", type=Path)
    parser.add_argument("--our2-evidence", type=Path)
    parser.add_argument("--warmup-batches", type=int, choices=range(0, 6))
    parser.add_argument("--warmup-instances", type=int, choices=range(0, 6),
                        help="legacy BS=1 alias for --warmup-batches")
    parser.add_argument("--device", default="cuda:0")
    return parser


def _warmup_batches(args):
    if args.warmup_batches is not None and args.warmup_instances is not None:
        raise ValueError("choose only one warm-up option")
    if args.warmup_instances is not None:
        if args.batch_size != 1:
            raise ValueError("--warmup-instances is a BS=1 compatibility option")
        return args.warmup_instances
    return 2 if args.warmup_batches is None else args.warmup_batches


def _load_prepared(args):
    cfg = dataset_config(args.problem_size)
    metadata_path = args.input_metadata or args.input.with_suffix(args.input.suffix + ".json")
    prepared = json.loads(metadata_path.read_text())
    if prepared.get("format") != "remaining-cvrptw-input-v1":
        raise ValueError("unsupported shared CVRPTW prepared-input schema")
    input_hash = sha256_file(args.input)
    required = {
        "problem_size": args.problem_size, "dataset_filename": cfg["filename"],
        "dataset_sha256": cfg["sha256"], "dataset_count": cfg["count"],
        "input_npz_sha256": input_hash,
    }
    if any(prepared.get(key) != value for key, value in required.items()):
        raise ValueError("prepared input identity differs from the pinned dataset")
    if args.dataset.name != cfg["filename"] or sha256_file(args.dataset) != cfg["sha256"]:
        raise ValueError("Kit dataset filename/SHA256 differs from pinned identity")
    with np.load(args.input, allow_pickle=False) as data:
        arrays = {name: data[name] for name in (
            "depots", "points", "raw_demands", "raw_capacities", "time_windows",
            "service_times", "time_tolerances", "dataset_indices", "reference_objectives")}
    indices = [int(value) for value in arrays["dataset_indices"]]
    expected_count = scope_instance_count(args.scope, args.batch_size, cfg["count"])
    if indices != list(range(expected_count)):
        raise ValueError(
            f"{args.scope}/BS{args.batch_size} requires exact dataset indices "
            f"0..{expected_count - 1}")
    if (prepared.get("dataset_indices") != indices or
            len(prepared.get("instance_names", [])) != len(indices)):
        raise ValueError("prepared metadata and NPZ identities differ")
    return cfg, prepared, arrays, indices, input_hash, metadata_path


def _gate_evidence(args, method_spec, cfg, checkpoint_hash):
    common = {
        "method": method_spec["name"], "problem_size": args.problem_size,
        "dataset_sha256": cfg["sha256"], "checkpoint_sha256": checkpoint_hash,
    }
    small_gate = None
    if args.scope == "production":
        if args.validation_gate_evidence is not None:
            require_validation_gate(
                args.validation_gate_evidence, batch_size=args.batch_size, **common)
        elif args.batch_size == 1 and args.our5_evidence is not None:
            require_our5_gate(args.our5_evidence, **common)
        else:
            raise ValueError("production requires matching --validation-gate-evidence")
    elif args.validation_gate_evidence is not None or args.our5_evidence is not None:
        raise ValueError("production gate evidence is only valid for production")
    if args.scope == "validation_gate":
        if args.small_gate_evidence is None:
            raise ValueError("validation_gate requires --small-gate-evidence")
        small_gate = require_small_gate(
            args.small_gate_evidence, batch_size=args.batch_size, **common)
    elif args.scope == "our_5":
        if args.batch_size != 1 or args.our2_evidence is None:
            raise ValueError("legacy our_5/BS1 requires --our2-evidence")
        small_gate = require_our2_gate(args.our2_evidence, **common)
    elif args.small_gate_evidence is not None or args.our2_evidence is not None:
        raise ValueError("small gate evidence is only valid for its validation scope")
    return small_gate


def run(method_spec, adapter, decoder, runtime_class, *, source_files):
    formal_batch_sizes = tuple(method_spec.get("formal_batch_sizes", (1,)))
    parser = _parser(
        f"Formal {method_spec['name']} CVRPTW50/100 evaluation", formal_batch_sizes)
    args = parser.parse_args()
    warmup_batches = _warmup_batches(args)
    cfg, prepared, arrays, indices, input_hash, metadata_path = _load_prepared(args)
    project = git_provenance(method_spec["root"])
    upstream = git_provenance(args.upstream)
    if project["dirty"]:
        raise ValueError("formal evaluation requires a clean project checkout")
    if (upstream["dirty"] or upstream["commit"] != method_spec["upstream_commit"] or
            normalize_git_repository_identity(upstream["url"]) !=
            normalize_git_repository_identity(method_spec["upstream_url"])):
        raise ValueError("official upstream identity/cleanliness mismatch")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != args.expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA256 differs from the audited expected value")
    pinned_hash = method_spec.get("checkpoint_hashes", {}).get(args.problem_size)
    if pinned_hash is not None and checkpoint_hash != pinned_hash:
        raise ValueError("checkpoint SHA256 differs from repository-pinned official asset")
    expected_relative = method_spec["checkpoints"][args.problem_size]
    try:
        actual_relative = str(args.checkpoint.resolve().relative_to(args.upstream.resolve()))
    except ValueError as exc:
        raise ValueError("checkpoint must be inside the pinned official checkout") from exc
    if actual_relative != expected_relative:
        raise ValueError("checkpoint path is not the exact size-specific official asset")
    small_gate = _gate_evidence(args, method_spec, cfg, checkpoint_hash)

    import ml4co_kit as kit
    import torch
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal remaining-CVRPTW evaluation requires CUDA")
    require_cuda_4090(torch.cuda.get_device_name(device))
    environment = environment_provenance(device)
    environment["method_packages"] = _package_versions()
    if method_spec.get("batch_aware_protocol"):
        protocol = method_spec["protocol"](args.problem_size, args.batch_size)
    else:
        protocol = method_spec["protocol"](args.problem_size)
    if protocol.get("original_instance_batch_size") != args.batch_size:
        raise RuntimeError("method protocol did not bind the requested original batch size")
    sources = source_provenance([
        *source_files, method_spec["root"] / "common/cvrptw_formal.py",
        method_spec["root"] / "common/cvrptw_artifacts.py",
        method_spec["root"] / "common/cvrptw_runtime.py",
        method_spec["root"] / "common/cvrptw_evaluator.py",
        method_spec["root"] / "problems/cvrptw/validate.py",
    ], root=method_spec["root"])
    identity = {
        "method": method_spec["name"], "variant": method_spec["variant"],
        "problem": "CVRPTW", "problem_size": args.problem_size, "scope": args.scope,
        "project": project, "upstream": upstream,
        "checkpoint": {"path": str(args.checkpoint.resolve()),
                       "relative_path": expected_relative, "sha256": checkpoint_hash,
                       "size_bytes": args.checkpoint.stat().st_size},
        "dataset": {"path": str(args.dataset.resolve()), "filename": cfg["filename"],
                    "sha256": cfg["sha256"], "count": cfg["count"]},
        "prepared_input": {"path": str(args.input.resolve()), "sha256": input_hash,
                           "metadata_path": str(metadata_path.resolve())},
        "chunk": {"offset": 0, "count": len(indices), "expected_indices": indices,
                  "native_batch_size": args.batch_size,
                  "expected_batches": len(indices) // args.batch_size},
        "protocol": protocol, "environment": environment,
        "validation": {
            "independent": "problems/cvrptw/validate.py original-unit simulation",
            "secondary": "ML4CO-Kit exact CVRPTWTask.check_constraints/evaluate",
            "ml4co_kit_version": environment["method_packages"].get("ml4co-kit"),
        },
        "warmup": {"batches": warmup_batches, "batch_size": args.batch_size,
                   "policy": "first prepared native batches, rerun formally, excluded from timing"},
        "source_provenance": sources, "timing_semantics": TIMING_SEMANTICS,
    }
    _, completed = initialize(args.output_dir, identity)
    runtime = runtime_class(args.upstream, args.checkpoint, args.problem_size, device, torch)
    write_json(args.output_dir / "checkpoint_state.json", runtime.checkpoint_state)
    if args.scope == "preflight":
        smoke = runtime.official_format_smoke()
        smoke_canonical = decoder(smoke["raw_action"], args.problem_size)
        write_json(args.output_dir / "official_format_smoke.json", {
            "status": "PASS", "actual_action": smoke["raw_action"],
            "canonical_solution": smoke_canonical,
            "selected_candidate": smoke["selected_candidate"],
            "official_reward": smoke["official_reward"],
            "official_objective": -float(smoke["official_reward"]),
            "official_environment_feasible": True,
            "semantics": "official generator format smoke; formal batch runs separately",
        })
    wrapper = kit.CVRPTWWrapper()
    wrapper.from_pickle(args.dataset)

    def native_at(local_index):
        return adapter(
            arrays["depots"][local_index], arrays["points"][local_index],
            arrays["raw_demands"][local_index], arrays["raw_capacities"][local_index],
            arrays["time_windows"][local_index], arrays["service_times"][local_index],
            problem_size=args.problem_size)

    total_batches = len(indices) // args.batch_size
    for batch_index in range(min(warmup_batches, total_batches)):
        first = batch_index * args.batch_size
        natives = [native_at(first + offset)[0] for offset in range(args.batch_size)]
        native_batch = stack_native(natives)
        if args.batch_size == 1 and not hasattr(runtime, "solve_batch"):
            runtime.solve(native_batch, timed=False)
        else:
            runtime.solve_batch(native_batch, timed=False)

    for batch_index in range(total_batches):
        first = batch_index * args.batch_size
        local_indices = list(range(first, first + args.batch_size))
        dataset_indices = [indices[index] for index in local_indices]
        completed_here = [index in completed for index in dataset_indices]
        if all(completed_here):
            continue
        if any(completed_here):
            raise RuntimeError("resume encountered a partial native inference batch")
        adapted = [native_at(index) for index in local_indices]
        native_batch = stack_native([value[0] for value in adapted])
        if args.batch_size == 1 and not hasattr(runtime, "solve_batch"):
            selection, elapsed = runtime.solve(native_batch, timed=True)
            selections = [selection]
        else:
            selections, elapsed = runtime.solve_batch(native_batch, timed=True)
        if len(selections) != args.batch_size:
            raise RuntimeError("official runtime returned the wrong original batch size")
        records = []
        for position, (local_index, dataset_index, selection) in enumerate(
                zip(local_indices, dataset_indices, selections)):
            mapping = adapted[position][1]
            canonical = decoder(selection["raw_action"], args.problem_size)
            reported = -float(selection["official_reward"])
            validation = validate_solution(
                arrays, local_index, canonical, reported_objective=reported)
            independent = float(validation["independent_objective"])
            kit_result = kit_validate(
                wrapper.task_list[dataset_index], canonical, independent)
            reference = float(arrays["reference_objectives"][local_index])
            details = validation["constraint_details"]
            records.append({
                "dataset_instance_index": dataset_index,
                "instance_id": prepared["instance_names"][local_index],
                "batch_index": batch_index, "position_in_batch": position,
                "raw_official_action": selection["raw_action"],
                "canonical_solution": canonical,
                "selected_candidate": selection["selected_candidate"],
                "official_reward": selection["official_reward"],
                "reported_objective": reported, "independent_objective": independent,
                **kit_result, "reference_objective": reference,
                "instance_drop_percent": instance_drop_percent(independent, reference),
                "runtime_seconds": elapsed,
                "runtime_seconds_semantics": "shared native inference-batch latency",
                "independent_feasible": bool(validation["feasible"]),
                "reported_objective_agrees": objective_agrees(reported, independent),
                "route_count": len(validation["routes"]),
                "route_loads": details["route_loads"],
                "route_timelines": details["route_timelines"],
                "adapter_mapping": mapping, "status": "KIT_VALIDATED",
            })
        append_batch(args.output_dir, records, {
            "batch_index": batch_index, "dataset_indices": dataset_indices,
            "batch_size": args.batch_size, "runtime_seconds": elapsed,
        })
    if small_gate is not None:
        if args.scope == "validation_gate":
            require_validation_prefix_matches_small(args.output_dir, *small_gate)
        else:
            require_our5_prefix_matches_our2(args.output_dir, *small_gate)
    summary = finalize(args.output_dir, paper_ready=args.scope == "production")
    print(json.dumps(summary, indent=2))
