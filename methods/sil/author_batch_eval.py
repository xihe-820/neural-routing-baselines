#!/usr/bin/env python3
"""Official-style batched SIL quality evaluation; never a BS1 timing result."""
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
from methods.sil.config import (FORMAL_GPU, FORMAL_PROTOCOLS, UPSTREAM_COMMIT,
                                UPSTREAM_URL, resolve_author_batch_config,
                                validate_checkpoint_path, validate_dataset_path)


# Keep the frozen provenance label separate from NVIDIA's runtime product naming.
FORMAL_GPU_MATCH_TOKEN = "RTX 4090"


_AUTHOR_BATCH_REBOUND_FIELDS = {
    "artifact_class", "evaluation_path", "expected_dataset_count",
    "original_instance_batch_size", "batch_size_requested", "author_batch_size",
    "batch_protocol_origin", "batch_override_reason", "rng_semantics",
    "paper_result_eligible_for_quality", "timing_column_eligible",
}


def _verify_runtime_algorithm_config(runtime_protocol: dict, author_protocol: dict) -> None:
    """Bind every solver-relevant BS1 config field before author batching begins."""
    expected = {key: value for key, value in runtime_protocol.items()
                if key not in _AUTHOR_BATCH_REBOUND_FIELDS}
    if expected != {key: author_protocol[key] for key in expected}:
        raise RuntimeError("SIL author-batch protocol diverges from frozen BS1 algorithm config")


def batch_slices(count: int, batch_size: int) -> list[tuple[int, int]]:
    """Return ordered real original-instance batch ranges, including a final tail."""
    if int(count) <= 0 or int(batch_size) <= 1:
        raise ValueError("author-batch count must be positive and batch_size must exceed one")
    return [(start, min(start + batch_size, count))
            for start in range(0, count, batch_size)]


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_jsonl(path: Path, values: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n" for value in values))
    temporary.replace(path)


def _comparison(left: float, right: float) -> dict:
    left, right = float(left), float(right)
    return {"left": left, "right": right, "abs_diff": abs(left - right),
            "pass": objective_agrees(left, right)}


def _require_file(value, name: str) -> Path:
    if value is None or not Path(value).is_file():
        raise ValueError(f"{name} must be an existing file")
    return Path(value).resolve()


def _require_directory(value, name: str) -> Path:
    if value is None or not Path(value).is_dir():
        raise ValueError(f"{name} must be an existing directory")
    return Path(value).resolve()


def _cuda_device(torch, value):
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SIL author-batch evaluation requires CUDA")
    if device.index is None:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpu = torch.cuda.get_device_name(device)
    if FORMAL_GPU_MATCH_TOKEN not in gpu:
        raise RuntimeError(
            f"SIL formal GPU must match {FORMAL_GPU_MATCH_TOKEN!r} "
            f"(canonical label {FORMAL_GPU!r}); observed {gpu!r}")
    return device


def _load_tasks(kit, problem: str, dataset: Path):
    wrapper = kit.TSPWrapper() if problem == "tsp" else kit.CVRPWrapper()
    wrapper.from_pickle(dataset)
    expected_type = kit.TSPTask if problem == "tsp" else kit.CVRPTask
    if not wrapper.task_list or any(type(task) is not expected_type for task in wrapper.task_list):
        raise ValueError("SIL dataset contains an unexpected ML4CO task type")
    return wrapper.task_list


def prepare_batch(problem: str, tasks: list, problem_size: int, *, device, torch):
    """Stack ML4CO tasks once for a single official [B,...] Tester invocation."""
    if not tasks:
        raise ValueError("author-batch preparation requires at least one task")
    if problem == "tsp":
        from methods.sil.tsp.adapter import adapt_task, inject_official_data

        adapted = [adapt_task(task, problem_size=problem_size) for task in tasks]
        source_points = [item[0] for item in adapted]
        model_points = np.stack([item[1] for item in adapted], axis=0)
        references = np.stack([item[2] for item in adapted], axis=0)
        return {
            "source_points": source_points, "adaptations": [item[3] for item in adapted],
            "inject": lambda env: inject_official_data(
                env, model_points, references, device=device, torch=torch),
        }
    from methods.sil.cvrp.adapter import adapt_task, inject_official_data

    adapted = [adapt_task(task, problem_size=problem_size) for task in tasks]
    capacities = np.asarray([item[3] for item in adapted], dtype=np.float32)
    if not np.array_equal(capacities, np.full_like(capacities, capacities[0])):
        raise ValueError(
            "SIL CVRP author batch requires identical true capacities; official "
            "environment indexes raw_data_capacity[0]")
    return {
        "depots": [item[0] for item in adapted], "points": [item[1] for item in adapted],
        "raw_demands": [item[2] for item in adapted],
        "adaptations": [item[7] for item in adapted],
        "inject": lambda env: inject_official_data(
            env, np.stack([item[4] for item in adapted], axis=0),
            np.stack([item[5] for item in adapted], axis=0), capacities,
            np.stack([item[6] for item in adapted], axis=0),
            device=device, torch=torch),
    }


def validate_batch(problem: str, tasks: list, prepared: dict, solutions: np.ndarray,
                   official_objectives: np.ndarray, problem_size: int,
                   *, dataset_offset: int) -> list[dict]:
    if len(tasks) != len(solutions) or len(tasks) != len(official_objectives):
        raise RuntimeError("SIL batch output does not match the injected task count")
    if problem == "tsp":
        from methods.sil.tsp.adapter import decode_official_solution
        from problems.tsp.validate import validate as independent_validate
    else:
        from methods.sil.cvrp.adapter import decode_official_solution
        from problems.cvrp.validate import validate as independent_validate
    records = []
    for local_index, (task, solution, official) in enumerate(
            zip(tasks, solutions, official_objectives)):
        decoded = decode_official_solution(solution, problem_size=problem_size)
        if problem == "tsp":
            checked = independent_validate(
                prepared["source_points"][local_index], decoded["canonical_solution"])
        else:
            capacity = float(prepared["adaptations"][local_index]["capacity"])
            checked = independent_validate(
                prepared["depots"][local_index], prepared["points"][local_index],
                prepared["raw_demands"][local_index], capacity,
                decoded["canonical_solution"], capacity_tolerance=1e-5 * capacity)
        if not checked["feasible"]:
            raise RuntimeError(f"SIL independent validation failed at {dataset_offset + local_index}")
        independent = float(checked["independent_objective"])
        canonical = np.asarray(decoded["canonical_solution"], dtype=np.int64)
        kit_feasible = bool(task.check_constraints(canonical))
        kit_objective = float(task.evaluate(canonical))
        official_check, kit_check = (_comparison(float(official), independent),
                                     _comparison(independent, kit_objective))
        if not kit_feasible or not official_check["pass"] or not kit_check["pass"]:
            raise RuntimeError(f"SIL objective/Kit validation failed at {dataset_offset + local_index}")
        reference = float(task.evaluate(task.ref_sol))
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError(f"invalid SIL reference at {dataset_offset + local_index}")
        records.append({
            "schema": "sil-author-batch-record.v1", "method": "SIL",
            "dataset_instance_index": dataset_offset + local_index,
            "problem": problem.upper(), "problem_size": problem_size,
            "official_objective": float(official), "independent_objective": independent,
            "kit_objective": kit_objective, "reference_objective": reference,
            "gap_percent": (independent - reference) / reference * 100.0,
            "independent_feasible": True, "kit_feasible": kit_feasible,
            "evidence_status": "KIT_VALIDATED",
            "official_vs_independent": official_check,
            "independent_vs_kit": kit_check,
            "adaptation": prepared["adaptations"][local_index], **decoded,
        })
    return records


def _source_files(problem: str) -> list[dict]:
    return source_provenance([
        Path(__file__), ROOT / "methods/sil/config.py", ROOT / "methods/sil/runtime.py",
        ROOT / f"methods/sil/{problem}/adapter.py", ROOT / f"problems/{problem}/validate.py",
    ], root=ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["tsp", "cvrp"], required=True)
    parser.add_argument("--problem-size", type=int, required=True)
    parser.add_argument("--budget", choices=FORMAL_PROTOCOLS, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-override-reason")
    parser.add_argument("--dump-config", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    protocol = resolve_author_batch_config(
        args.problem, args.problem_size, args.budget, batch_size=args.batch_size,
        batch_override_reason=args.batch_override_reason)
    if args.dump_config:
        print(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if not args.expected_dataset_sha256 or not args.expected_checkpoint_sha256:
        parser.error("execution requires expected dataset and checkpoint SHA256")
    if args.output_dir is None:
        parser.error("execution requires --output-dir")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError("author-batch output directory already exists")
    dataset, checkpoint = _require_file(args.dataset, "dataset"), _require_file(args.checkpoint, "checkpoint")
    upstream_path = _require_directory(args.upstream, "upstream")
    validate_checkpoint_path(protocol, checkpoint)
    validate_dataset_path(protocol, dataset)
    dataset_sha, checkpoint_sha = sha256_file(dataset), sha256_file(checkpoint)
    if dataset_sha != args.expected_dataset_sha256 or checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("SIL author-batch asset SHA256 mismatch")
    project, upstream = git_provenance(ROOT), git_provenance(upstream_path)
    if project["dirty"]:
        raise ValueError("SIL author-batch evaluation requires a clean project checkout")
    if (upstream["dirty"] or upstream["commit"] != UPSTREAM_COMMIT or
            normalize_git_repository_identity(upstream["url"]) != normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("SIL upstream identity/cleanliness mismatch")
    import ml4co_kit as kit
    import torch
    tasks = _load_tasks(kit, args.problem, dataset)
    if len(tasks) != protocol["expected_dataset_count"]:
        raise ValueError("SIL dataset count differs from frozen author-batch protocol")
    device = _cuda_device(torch, args.device)
    output_dir.mkdir(parents=True)
    metadata = {
        "schema": "sil-author-batch-eval.v1", "state": "RUNNING",
        "artifact_class": "baseline_result_reproduction", "protocol": protocol,
        "dataset": {"path": str(dataset), "sha256": dataset_sha, "count": len(tasks)},
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "project": project, "upstream": upstream, "environment": environment_provenance(device),
        "source_files": _source_files(args.problem), "official_source_modified": False,
        "author_batch_runtime_diagnostic_only": True,
        "not_comparable_to_bs1_timing_protocol": True,
    }
    _atomic_json(output_dir / "metadata.json", metadata)
    from methods.sil.runtime import build_tester, solve_batch
    tester, runtime_protocol = build_tester(
        problem=args.problem, problem_size=args.problem_size, budget_label=args.budget,
        upstream=upstream_path, checkpoint=checkpoint, device=device, torch=torch)
    _verify_runtime_algorithm_config(runtime_protocol, protocol)
    records, effective_sizes, total_wall = [], [], 0.0
    for start, end in batch_slices(len(tasks), protocol["batch_size_requested"]):
        batch = tasks[start:end]
        prepared = prepare_batch(args.problem, batch, args.problem_size, device=device, torch=torch)
        solved = solve_batch(
            tester, problem=args.problem, problem_size=args.problem_size,
            batch_size=len(batch), inject=prepared["inject"], device=device, torch=torch,
            timed=True)
        total_wall += float(solved["runtime_seconds"])
        effective_sizes.append(len(batch))
        records.extend(validate_batch(
            args.problem, batch, prepared, solved["solutions"], solved["official_objectives"],
            args.problem_size, dataset_offset=start))
    if len(records) != len(tasks):
        raise RuntimeError("SIL author-batch evaluation did not validate every dataset instance")
    _atomic_jsonl(output_dir / "validated_records.jsonl", records)
    objectives = np.asarray([row["independent_objective"] for row in records], dtype=float)
    references = np.asarray([row["reference_objective"] for row in records], dtype=float)
    gaps = np.asarray([row["gap_percent"] for row in records], dtype=float)
    summary = {
        "schema": "sil-author-batch-eval.v1", "status": "KIT_VALIDATED",
        "artifact_class": "baseline_result_reproduction", "method": "SIL",
        "problem": protocol["problem"], "problem_size": protocol["actual_problem_size"],
        "protocol": protocol["budget_label"], "budget": protocol["budget"],
        "dataset_sha256": dataset_sha, "checkpoint_sha256": checkpoint_sha,
        "validated_count": len(records), "failed_count": 0,
        "mean_objective": float(objectives.mean()),
        "mean_reference_objective": float(references.mean()),
        "mean_instance_gap_percent": float(gaps.mean()),
        "batch_size_requested": protocol["batch_size_requested"],
        "original_instance_batch_size": protocol["original_instance_batch_size"],
        "effective_batch_sizes": effective_sizes, "number_of_batches": len(effective_sizes),
        "total_wall_time_seconds": total_wall,
        "author_batch_runtime_diagnostic_only": True,
        "not_comparable_to_bs1_timing_protocol": True,
        "paper_result_eligible_for_quality": True, "timing_column_eligible": False,
        "config_origin": protocol["config_origin"],
        "batch_protocol_origin": protocol["batch_protocol_origin"],
        "environment": metadata["environment"], "project": project, "upstream": upstream,
        "source_files": metadata["source_files"],
    }
    _atomic_json(output_dir / "summary.json", summary)
    metadata.update(state="KIT_VALIDATED", summary_file="summary.json",
                    records_file="validated_records.jsonl", validated_count=len(records))
    _atomic_json(output_dir / "metadata.json", metadata)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
