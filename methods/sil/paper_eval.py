#!/usr/bin/env python3
"""SIL ML4CO smoke, preflight, and resumable formal BS1 production."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from common.provenance import (environment_provenance, git_provenance,
                               normalize_git_repository_identity, source_provenance)
from methods.sil.config import (FORMAL_BATCH_SIZE, UPSTREAM_COMMIT, UPSTREAM_URL,
                                resolve_config, validate_checkpoint_path)
from methods.sil.paper_results import (TIMING_SEMANTICS, append, capture_rng_state,
                                       finalize, initialize, restore_rng_state)


def comparison(left, right):
    left, right = float(left), float(right)
    absolute = abs(left - right)
    return {
        "left": left, "right": right, "abs_diff": absolute,
        "rel_diff": absolute / max(abs(left), abs(right), 1e-300),
        "pass": objective_agrees(left, right),
    }


def _require_file(value, name):
    if value is None or not Path(value).is_file():
        raise ValueError(f"{name} must be an existing file")
    return Path(value).resolve()


def _require_directory(value, name):
    if value is None or not Path(value).is_dir():
        raise ValueError(f"{name} must be an existing directory")
    return Path(value).resolve()


def _cuda_device(torch, value):
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SIL server evaluation requires CUDA")
    if device.index is None:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpu = torch.cuda.get_device_name(device)
    if "RTX 4090" not in gpu:
        raise RuntimeError(f"formal SIL evaluation requires RTX 4090; observed {gpu!r}")
    return device


def _load_tasks(kit, problem, dataset):
    wrapper = kit.TSPWrapper() if problem == "tsp" else kit.CVRPWrapper()
    wrapper.from_pickle(dataset)
    expected_type = kit.TSPTask if problem == "tsp" else kit.CVRPTask
    if not wrapper.task_list or any(type(task) is not expected_type for task in wrapper.task_list):
        raise ValueError("dataset contains an unexpected ML4CO task type")
    return wrapper.task_list


def _task_name(task, index):
    value = getattr(task, "name", None)
    return str(value) if value is not None else f"instance-{index}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["tsp", "cvrp"], required=True)
    parser.add_argument("--problem-size", type=int, required=True)
    parser.add_argument("--budget", choices=["fewer", "more", "greedy_diagnostic"], required=True)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--dump-config", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scope", choices=["our-smoke", "preflight", "fullset"])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--device", default="cuda:0")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true")
    resume_group.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    protocol = resolve_config(args.problem, args.problem_size, args.budget,
                              batch_size=args.batch_size)
    if args.dump_config:
        print(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.scope is None or args.count is None or args.output_dir is None:
        parser.error("execution requires --scope, --count, and --output-dir")
    if args.offset < 0 or args.count <= 0:
        parser.error("--offset must be nonnegative and --count positive")
    if args.scope == "our-smoke" and (args.offset != 0 or args.count != 2):
        parser.error("our-smoke is exactly dataset indices 0 and 1")
    if args.scope == "fullset" and args.budget == "greedy_diagnostic":
        parser.error("greedy_diagnostic cannot produce formal fullset artifacts")
    if not args.expected_dataset_sha256 or not args.expected_checkpoint_sha256:
        parser.error("execution requires both expected SHA256 arguments")

    dataset = _require_file(args.dataset, "dataset")
    checkpoint = _require_file(args.checkpoint, "checkpoint")
    upstream_path = _require_directory(args.upstream, "upstream")
    validate_checkpoint_path(protocol, checkpoint)
    dataset_sha = sha256_file(dataset)
    checkpoint_sha = sha256_file(checkpoint)
    if dataset_sha != args.expected_dataset_sha256:
        raise ValueError("SIL dataset SHA256 mismatch")
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("SIL checkpoint SHA256 mismatch")

    project = git_provenance(ROOT)
    upstream = git_provenance(upstream_path)
    if project["dirty"]:
        raise ValueError("SIL server evaluation requires a clean project checkout")
    if (upstream["dirty"] or upstream["commit"] != UPSTREAM_COMMIT
            or normalize_git_repository_identity(upstream["url"])
            != normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("official SIL checkout identity/cleanliness mismatch")

    import ml4co_kit as kit
    tasks = _load_tasks(kit, args.problem, dataset)
    dataset_count = len(tasks)
    if args.offset + args.count > dataset_count:
        raise ValueError("requested SIL dataset slice exceeds task count")
    if args.scope == "fullset" and (args.offset != 0 or args.count != dataset_count):
        raise ValueError("SIL fullset must cover the complete ordered dataset")
    indices = list(range(args.offset, args.offset + args.count))

    import torch
    device = _cuda_device(torch, args.device)
    environment = environment_provenance(device)
    sources = [
        Path(__file__), ROOT / "methods/sil/config.py", ROOT / "methods/sil/runtime.py",
        ROOT / "methods/sil/paper_results.py",
        ROOT / f"methods/sil/{args.problem}/adapter.py",
        ROOT / f"problems/{args.problem}/validate.py",
    ]
    identity = {
        "method": "SIL", "scope": args.scope, "protocol": protocol,
        "offset": args.offset, "count": args.count, "indices": indices,
        "dataset": {"path": str(dataset), "filename": dataset.name,
                    "sha256": dataset_sha, "count": dataset_count},
        "checkpoint": {"path": str(checkpoint), "filename": checkpoint.name,
                       "sha256": checkpoint_sha,
                       "drive_file_id": protocol["checkpoint"]["drive_file_id"]},
        "project": project, "upstream": upstream, "environment": environment,
        "source_files": source_provenance(sources, root=ROOT),
        "timing_semantics": TIMING_SEMANTICS,
        "official_source_modified": False,
        "checkpoint_load_semantics": (
            "pinned official Tester torch.load followed by model.load_state_dict "
            "with PyTorch strict=True default"),
        "compatibility_shims": [],
    }
    output_dir = args.output_dir.resolve()
    metadata, records, timings, rng_checkpoint = initialize(
        output_dir, identity, resume=(args.resume or args.skip_existing))
    if metadata["state"] in {"KIT_VALIDATED", "PAPER_READY"}:
        print(output_dir / "summary.json")
        return 0

    from methods.sil.runtime import build_tester, solve_one
    tester, runtime_config = build_tester(
        problem=args.problem, problem_size=args.problem_size, budget_label=args.budget,
        upstream=upstream_path, checkpoint=checkpoint, device=device, torch=torch)
    if runtime_config != protocol:
        raise RuntimeError("runtime SIL config differs from artifact protocol")
    if rng_checkpoint is not None:
        restore_rng_state(rng_checkpoint["rng_state"], torch)

    if args.problem == "tsp":
        from methods.sil.tsp.adapter import (adapt_task, decode_official_solution,
                                             inject_official_data)
        from problems.tsp.validate import validate as independent_validate
    else:
        from methods.sil.cvrp.adapter import (adapt_task, decode_official_solution,
                                              inject_official_data)
        from problems.cvrp.validate import validate as independent_validate

    for index in indices[len(records):]:
        task = tasks[index]
        reference = float(task.evaluate(task.ref_sol))
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError(f"invalid reference objective at dataset index {index}")
        if args.problem == "tsp":
            source_points, model_points, reference_order, adaptation = adapt_task(
                task, problem_size=args.problem_size)
            inject = lambda env: inject_official_data(
                env, model_points[None], reference_order[None], device=device, torch=torch)
        else:
            (depot, points, raw_demands, capacity, coordinates, native_demands,
             reference_native, adaptation) = adapt_task(task, problem_size=args.problem_size)
            inject = lambda env: inject_official_data(
                env, coordinates[None], native_demands[None], np.asarray([capacity]),
                reference_native[None], device=device, torch=torch)

        solved = solve_one(
            tester, problem=args.problem, problem_size=args.problem_size,
            inject=inject, device=device, torch=torch)
        decoded = decode_official_solution(
            solved["solution"], problem_size=args.problem_size)
        if args.problem == "tsp":
            checked = independent_validate(source_points, decoded["canonical_solution"])
        else:
            checked = independent_validate(
                depot, points, raw_demands, capacity, decoded["canonical_solution"],
                capacity_tolerance=1e-5 * capacity)
        if not checked["feasible"]:
            raise RuntimeError(f"independent SIL validation failed at dataset index {index}")
        independent = float(checked["independent_objective"])
        kit_solution = np.asarray(decoded["canonical_solution"], dtype=np.int64)
        kit_feasible = bool(task.check_constraints(kit_solution))
        kit_objective = float(task.evaluate(kit_solution))
        official_check = comparison(solved["official_objective"], independent)
        kit_check = comparison(independent, kit_objective)
        if not kit_feasible or not official_check["pass"] or not kit_check["pass"]:
            raise RuntimeError(f"SIL objective/Kit validation failed at dataset index {index}")
        record = {
            "schema": "sil-validated-record.v1", "method": "SIL",
            "problem": args.problem.upper(), "problem_size": args.problem_size,
            "dataset_instance_index": index, "instance_id": _task_name(task, index),
            "protocol": args.budget, "budget": protocol["budget"],
            "config_origin": protocol["config_origin"],
            "adapted_from_size": protocol["adapted_from_size"],
            "checkpoint_sha256": checkpoint_sha, "dataset_sha256": dataset_sha,
            **decoded, "adaptation": adaptation,
            "official_objective": solved["official_objective"],
            "independent_objective": independent, "reference_objective": reference,
            "gap_percent": (independent - reference) / reference * 100.0,
            "independent_feasible": True,
            "independent_constraint_details": checked["constraint_details"],
            "kit_feasible": kit_feasible, "kit_objective": kit_objective,
            "official_vs_independent": official_check,
            "independent_vs_kit": kit_check,
            "solution_capture": solved["capture"], "evidence_status": "KIT_VALIDATED",
        }
        timing = {
            "schema": "sil-batch-timing.v1", "dataset_instance_index": index,
            "original_instance_batch_size": 1,
            "runtime_seconds": solved["runtime_seconds"],
            "timing_semantics": TIMING_SEMANTICS,
        }
        append(output_dir, metadata, records, timings, record, timing,
               capture_rng_state(torch))

    summary = finalize(output_dir, metadata, records, timings)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
