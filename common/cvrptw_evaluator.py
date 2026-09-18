"""Shared evaluation driver; method modules provide only official runtime hooks."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

from common.cvrptw_artifacts import (append, finalize, initialize,
                                     require_our2_gate,
                                     require_our5_gate,
                                     require_our5_prefix_matches_our2)
from common.cvrptw_formal import (TIMING_SEMANTICS, dataset_config,
                                  instance_drop_percent, kit_validate,
                                  validate_solution)
from common.cvrptw_runtime import require_cuda_4090
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


def _parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-metadata", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, choices=(50, 100), required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("preflight", "our_2", "our_5", "production"), required=True)
    parser.add_argument("--our5-evidence", type=Path)
    parser.add_argument("--our2-evidence", type=Path)
    parser.add_argument("--warmup-instances", type=int, choices=range(0, 6), default=2)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _load_prepared(args, method_spec):
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
    expected_counts = {"preflight": 1, "our_2": 2, "our_5": 5,
                       "production": cfg["count"]}
    if indices != list(range(expected_counts[args.scope])):
        raise ValueError(f"{args.scope} requires exact dataset indices 0..{expected_counts[args.scope]-1}")
    if prepared.get("dataset_indices") != indices or len(prepared.get("instance_names", [])) != len(indices):
        raise ValueError("prepared metadata and NPZ identities differ")
    return cfg, prepared, arrays, indices, input_hash, metadata_path


def run(method_spec, adapter, decoder, runtime_class, *, source_files):
    parser = _parser(f"Formal {method_spec['name']} CVRPTW50/100 evaluation")
    args = parser.parse_args()
    cfg, prepared, arrays, indices, input_hash, metadata_path = _load_prepared(args, method_spec)
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
        raise ValueError("checkpoint SHA256 differs from the repository-pinned official asset")
    expected_relative = method_spec["checkpoints"][args.problem_size]
    try:
        actual_relative = str(args.checkpoint.resolve().relative_to(args.upstream.resolve()))
    except ValueError as exc:
        raise ValueError("checkpoint must be inside the pinned official checkout") from exc
    if actual_relative != expected_relative:
        raise ValueError("checkpoint path is not the exact size-specific official asset")
    if args.scope == "production":
        if args.our5_evidence is None:
            raise ValueError("production requires --our5-evidence")
        require_our5_gate(
            args.our5_evidence, method=method_spec["name"], problem_size=args.problem_size,
            dataset_sha256=cfg["sha256"], checkpoint_sha256=checkpoint_hash)
    elif args.our5_evidence is not None:
        raise ValueError("--our5-evidence is only valid for production")
    our2_gate = None
    if args.scope == "our_5":
        if args.our2_evidence is None:
            raise ValueError("our_5 requires --our2-evidence")
        our2_gate = require_our2_gate(
            args.our2_evidence, method=method_spec["name"], problem_size=args.problem_size,
            dataset_sha256=cfg["sha256"], checkpoint_sha256=checkpoint_hash)
    elif args.our2_evidence is not None:
        raise ValueError("--our2-evidence is only valid for our_5")

    import ml4co_kit as kit
    import torch
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal remaining-CVRPTW evaluation requires CUDA")
    require_cuda_4090(torch.cuda.get_device_name(device))
    environment = environment_provenance(device)
    environment["method_packages"] = _package_versions()
    protocol = method_spec["protocol"](args.problem_size)
    sources = source_provenance([*source_files, method_spec["root"] / "common/cvrptw_formal.py",
                                 method_spec["root"] / "common/cvrptw_artifacts.py",
                                 method_spec["root"] / "common/cvrptw_runtime.py",
                                 method_spec["root"] / "common/cvrptw_evaluator.py",
                                 method_spec["root"] / "problems/cvrptw/validate.py"],
                                root=method_spec["root"])
    identity = {
        "method": method_spec["name"], "variant": method_spec["variant"],
        "problem": "CVRPTW", "problem_size": args.problem_size, "scope": args.scope,
        "project": project, "upstream": upstream,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "relative_path": expected_relative,
                       "sha256": checkpoint_hash, "size_bytes": args.checkpoint.stat().st_size},
        "dataset": {"path": str(args.dataset.resolve()), "filename": cfg["filename"],
                    "sha256": cfg["sha256"], "count": cfg["count"]},
        "prepared_input": {"path": str(args.input.resolve()), "sha256": input_hash,
                           "metadata_path": str(metadata_path.resolve())},
        "chunk": {"offset": 0, "count": len(indices), "expected_indices": indices},
        "protocol": protocol, "environment": environment,
        "validation": {
            "independent": "problems/cvrptw/validate.py original-unit simulation",
            "secondary": "ML4CO-Kit exact CVRPTWTask.check_constraints/evaluate",
            "ml4co_kit_version": environment["method_packages"].get("ml4co-kit"),
        },
        "warmup": {"instances": args.warmup_instances,
                   "policy": "first prepared instances, rerun formally, excluded from timing"},
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
            "semantics": (
                "official generator CVRPTW; actual selected action; official env "
                "check_solution enabled; not a paper/ML4CO result"),
        })
    wrapper = kit.CVRPTWWrapper()
    wrapper.from_pickle(args.dataset)

    def native_at(local_index):
        return adapter(
            arrays["depots"][local_index], arrays["points"][local_index],
            arrays["raw_demands"][local_index], arrays["raw_capacities"][local_index],
            arrays["time_windows"][local_index], arrays["service_times"][local_index],
            problem_size=args.problem_size)

    for local_index in range(min(args.warmup_instances, len(indices))):
        native, _ = native_at(local_index)
        runtime.solve(native, timed=False)
    for local_index, dataset_index in enumerate(indices):
        if dataset_index in completed:
            continue
        native, mapping = native_at(local_index)
        selection, elapsed = runtime.solve(native, timed=True)
        canonical = decoder(selection["raw_action"], args.problem_size)
        reported = -float(selection["official_reward"])
        validation = validate_solution(arrays, local_index, canonical,
                                       reported_objective=reported)
        independent = float(validation["independent_objective"])
        task = wrapper.task_list[dataset_index]
        kit_result = kit_validate(task, canonical, independent)
        reference = float(arrays["reference_objectives"][local_index])
        details = validation["constraint_details"]
        record = {
            "dataset_instance_index": dataset_index,
            "instance_id": prepared["instance_names"][local_index],
            "raw_official_action": selection["raw_action"],
            "canonical_solution": canonical,
            "selected_candidate": selection["selected_candidate"],
            "official_reward": selection["official_reward"],
            "reported_objective": reported, "independent_objective": independent,
            **kit_result, "reference_objective": reference,
            "instance_drop_percent": instance_drop_percent(independent, reference),
            "runtime_seconds": elapsed,
            "independent_feasible": bool(validation["feasible"]),
            "reported_objective_agrees": objective_agrees(reported, independent),
            "route_count": len(validation["routes"]),
            "route_loads": details["route_loads"],
            "route_timelines": details["route_timelines"],
            "adapter_mapping": mapping, "status": "KIT_VALIDATED",
        }
        append(args.output_dir, record)
    if our2_gate is not None:
        require_our5_prefix_matches_our2(args.output_dir, *our2_gate)
    summary = finalize(args.output_dir, paper_ready=args.scope == "production")
    print(json.dumps(summary, indent=2))
