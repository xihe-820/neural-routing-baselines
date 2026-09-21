#!/usr/bin/env python3
"""Auditable LEHD appendix parallel evaluator and explicit derived BS1 row builder."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from common.protocol_rebind import verify_quality_artifact, verify_timing_artifact
from common.provenance import (environment_provenance, git_provenance,
                               normalize_git_repository_identity, source_provenance)
from methods.lehd.author_batch_eval import (_atomic_json, _atomic_jsonl, _cuda_device,
                                            _load_tasks, _require_directory,
                                            _require_file, prepare_batch,
                                            validate_batch)
from methods.lehd.config import (AUTHOR_BATCH_REGISTRY, CHECKPOINTS,
                                 DATASET_FILENAMES, FORMAL_GPU,
                                 TIMING_PROBE_COUNTS, UPSTREAM_COMMIT,
                                 UPSTREAM_URL, resolve_author_batch_config,
                                 resolve_config, validate_checkpoint_location,
                                 validate_dataset_path)


PARALLEL_SIZES = {"tsp": (100, 500, 1000), "cvrp": (50, 100, 200)}
PARALLEL_BATCH_SIZE = {"tsp": 128, "cvrp": 100}
PARALLEL_PROTOCOLS = ("fewer", "more")
DERIVED_MODE = "derived_bs1_parallel_row_from_verified_quality_and_timing"
MEASURED_MODE = "measured_full_parallel"
TABLE_RENDERING = {
    "objective_decimals": 3, "drop_percent_decimals": 3,
    "total_runtime_decimals": 3, "mean_batch_time_decimals": 3,
    "total_runtime_unit": "human-readable seconds/minutes/hours in LaTeX table",
}


def parallel_scope(problem: str, problem_size: int, protocol: str,
                   batch_size: int) -> dict:
    problem, size, batch = problem.lower(), int(problem_size), int(batch_size)
    if problem not in PARALLEL_SIZES or size not in PARALLEL_SIZES[problem]:
        raise ValueError("LEHD parallel scope is TSP100/500/1000 or CVRP50/100/200")
    if protocol not in PARALLEL_PROTOCOLS:
        raise ValueError("LEHD parallel protocol must be fewer or more")
    if batch not in (1, PARALLEL_BATCH_SIZE[problem]):
        raise ValueError("LEHD parallel batch size is outside the frozen matrix")
    config = resolve_config(problem, size, protocol)
    return {
        "method": "LEHD", "problem": problem.upper(), "problem_size": size,
        "protocol": protocol, "RRC_budget": config["RRC_budget"],
        "requested_batch_size": batch,
        "expected_dataset_count": AUTHOR_BATCH_REGISTRY[(problem, size)]["dataset_count"],
        "formal_parallel_batch_size": PARALLEL_BATCH_SIZE[problem],
        "paper_protocol": config,
    }


def batch_ranges(count: int, batch_size: int) -> list[tuple[int, int]]:
    if int(count) <= 0 or int(batch_size) <= 0:
        raise ValueError("parallel count and batch size must be positive")
    return [(start, min(start + batch_size, count))
            for start in range(0, count, batch_size)]


def _runtime_statistics(values: list[float]) -> dict:
    if not values or any(not isinstance(value, (int, float)) or
                         not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("parallel runtimes must be finite and positive")
    numbers = [float(value) for value in values]
    return {
        "total_solver_runtime_seconds": float(sum(numbers)),
        "mean_batch_runtime_seconds": statistics.fmean(numbers),
        "median_batch_runtime_seconds": statistics.median(numbers),
        "min_batch_runtime_seconds": min(numbers),
        "max_batch_runtime_seconds": max(numbers),
    }


def summarize_measured(records: list[dict], batch_timings: list[dict],
                       *, scope: dict) -> dict:
    count, batch_size = scope["expected_dataset_count"], scope["requested_batch_size"]
    ranges = batch_ranges(count, batch_size)
    if len(records) != count or [row.get("dataset_instance_index") for row in records] != list(
            range(count)):
        raise ValueError("parallel records do not exactly cover the ordered dataset")
    for row in records:
        if (row.get("evidence_status") != "KIT_VALIDATED" or
                row.get("independent_feasible") is not True or
                row.get("kit_feasible") is not True or
                row.get("official_vs_independent", {}).get("pass") is not True or
                row.get("independent_vs_kit", {}).get("pass") is not True):
            raise ValueError("parallel record failed independent or Kit validation")
    if len(batch_timings) != len(ranges):
        raise ValueError("parallel timing count differs from batch count")
    runtimes, effective = [], []
    for index, ((start, stop), timing) in enumerate(zip(ranges, batch_timings)):
        if (timing.get("batch_index") != index or
                timing.get("dataset_index_start") != start or
                timing.get("dataset_index_stop_exclusive") != stop or
                timing.get("original_instance_count") != stop - start):
            raise ValueError("parallel timing row batch identity mismatch")
        runtimes.append(timing.get("solver_runtime_seconds"))
        effective.append(stop - start)
    runtime = _runtime_statistics(runtimes)
    objectives = [float(row["independent_objective"]) for row in records]
    references = [float(row["reference_objective"]) for row in records]
    gaps = [float(row["gap_percent"]) for row in records]
    return {
        "schema": "lehd-parallel-evaluation.v1", "status": "KIT_VALIDATED",
        "artifact_class": "baseline_parallel_evaluation", "method": "LEHD",
        "problem": scope["problem"], "problem_size": scope["problem_size"],
        "protocol": scope["protocol"], "RRC_budget": scope["RRC_budget"],
        "artifact_generation_mode": MEASURED_MODE,
        "requested_batch_size": batch_size, "effective_batch_sizes": effective,
        "number_of_batches": len(effective), "validated_count": count,
        "failed_count": 0, "mean_objective": statistics.fmean(objectives),
        "mean_reference_objective": statistics.fmean(references),
        "mean_instance_gap_percent": statistics.fmean(gaps),
        "individual_batch_runtimes": [float(value) for value in runtimes],
        **runtime, "parallel_table_eligible": True,
        "table_rendering": TABLE_RENDERING,
        "timing_semantics": (
            "CUDA-synchronized official solve_batch only; preparation, validation, "
            "artifact I/O, and ML4CO-Kit evaluation excluded; Time is mean native "
            "batch latency and Total is the sum of native batch latencies"
        ),
    }


def derive_bs1_summary(*, scope: dict, quality: dict, timing: dict,
                       current_project: dict) -> tuple[dict, dict]:
    if scope["requested_batch_size"] != 1:
        raise ValueError("derived parallel rows are supported only for BS1")
    quality_summary, timing_summary = quality["summary"], timing["summary"]
    count = scope["expected_dataset_count"]
    if (quality_summary.get("status") != "KIT_VALIDATED" or
            quality_summary.get("protocol") != scope["protocol"] or
            quality_summary.get("RRC_budget") != scope["RRC_budget"] or
            quality_summary.get("validated_count") != count or
            quality_summary.get("failed_count") != 0 or
            timing_summary.get("status") != "KIT_VALIDATED" or
            timing_summary.get("protocol") != scope["protocol"] or
            timing_summary.get("RRC_budget") != scope["RRC_budget"] or
            timing_summary.get("count") != TIMING_PROBE_COUNTS[scope["protocol"]] or
            timing_summary.get("validated_count") != timing_summary.get("count") or
            timing_summary.get("failed_count") != 0 or
            quality_summary.get("dataset_sha256") != timing_summary.get("dataset_sha256") or
            quality_summary.get("checkpoint_sha256") !=
            timing_summary.get("checkpoint_sha256")):
        raise ValueError("derived BS1 quality/timing identity mismatch")
    mean_runtime = float(timing_summary["mean_runtime_seconds"])
    if not math.isfinite(mean_runtime) or mean_runtime <= 0:
        raise ValueError("derived BS1 timing mean must be finite and positive")
    summary = {
        "schema": "lehd-parallel-evaluation.v1", "status": "KIT_VALIDATED",
        "artifact_class": "baseline_parallel_evaluation", "method": "LEHD",
        "problem": scope["problem"], "problem_size": scope["problem_size"],
        "protocol": scope["protocol"], "RRC_budget": scope["RRC_budget"],
        "artifact_generation_mode": DERIVED_MODE,
        "requested_batch_size": 1, "dataset_count": count,
        "number_of_batches": count, "validated_count": count, "failed_count": 0,
        "mean_objective": quality_summary["mean_objective"],
        "mean_reference_objective": quality_summary["mean_reference_objective"],
        "mean_instance_gap_percent": quality_summary["mean_instance_gap_percent"],
        "mean_batch_runtime_seconds": mean_runtime,
        "total_solver_runtime_seconds": mean_runtime * count,
        "timing_sample_count": timing_summary["count"],
        "timing_sample_runtimes": timing_summary["individual_runtimes"],
        "total_runtime_is_derived": True, "parallel_table_eligible": True,
        "table_rendering": TABLE_RENDERING,
        "project": current_project,
        "quality_source": {"path": str(quality["directory"]),
                           "summary_sha256": quality["hashes"]["summary"],
                           "metadata_sha256": quality["hashes"]["metadata"],
                           "records_sha256": quality["hashes"]["records"]},
        "timing_source": {"path": str(timing["path"]), "sha256": timing["sha256"]},
        "dataset_sha256": quality["dataset"]["sha256"],
        "checkpoint_sha256": quality["checkpoint"]["sha256"],
        "upstream": quality_summary["upstream"],
        "timing_semantics": (
            "Obj/Drop come from verified full quality evidence; Time is the verified "
            "three-sample BS1 mean; Total is explicitly derived as Time*dataset_count"
        ),
    }
    metadata = {
        "schema": "lehd-parallel-evaluation.v1", "state": "KIT_VALIDATED",
        "artifact_class": "baseline_parallel_evaluation",
        "artifact_generation_mode": DERIVED_MODE, "parallel_scope": scope,
        "project": current_project, "official_source_modified": False,
        "quality_source": summary["quality_source"],
        "timing_source": summary["timing_source"],
        "parallel_table_eligible": True,
    }
    return metadata, summary


def _verify_derived_sources(args, scope, current_project):
    current_quality = resolve_author_batch_config(
        args.problem, args.problem_size, args.protocol)
    quality = verify_quality_artifact(
        args.quality_source, method="LEHD", problem=args.problem,
        problem_size=args.problem_size, protocol_label=args.protocol,
        budget_key="RRC_budget", budget=scope["RRC_budget"],
        expected_protocol=current_quality,
        expected_count=scope["expected_dataset_count"],
        expected_batch_size=AUTHOR_BATCH_REGISTRY[(args.problem, args.problem_size)]["batch_size"],
        expected_project_commit=None, upstream_commit=UPSTREAM_COMMIT,
        dataset_filename=DATASET_FILENAMES[(args.problem, args.problem_size)],
        checkpoint_filename=CHECKPOINTS[args.problem]["filename"])
    timing = verify_timing_artifact(
        args.timing_source, method="LEHD", problem=args.problem,
        problem_size=args.problem_size, protocol_label=args.protocol,
        budget_key="RRC_budget", budget=scope["RRC_budget"],
        expected_count=TIMING_PROBE_COUNTS[args.protocol],
        expected_project_commit=None, upstream_commit=UPSTREAM_COMMIT,
        dataset_sha256=quality["dataset"]["sha256"],
        checkpoint_sha256=quality["checkpoint"]["sha256"])
    return derive_bs1_summary(
        scope=scope, quality=quality, timing=timing, current_project=current_project)


def _source_files(problem: str) -> list[dict]:
    return source_provenance([
        Path(__file__), ROOT / "methods/lehd/config.py", ROOT / "methods/lehd/runtime.py",
        ROOT / "methods/lehd/author_batch_eval.py",
        ROOT / f"methods/lehd/{problem}/adapter.py",
        ROOT / f"problems/{problem}/validate.py",
    ], root=ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["tsp", "cvrp"], required=True)
    parser.add_argument("--problem-size", type=int, required=True)
    parser.add_argument("--protocol", choices=PARALLEL_PROTOCOLS, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--quality-source", type=Path)
    parser.add_argument("--timing-source", type=Path)
    parser.add_argument("--derive-bs1-from-verified-artifacts", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    scope = parallel_scope(args.problem, args.problem_size, args.protocol, args.batch_size)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("LEHD parallel output already exists")
    project = git_provenance(ROOT)
    if project["dirty"]:
        raise ValueError("LEHD parallel evaluation requires a clean project checkout")
    if args.derive_bs1_from_verified_artifacts:
        if args.batch_size != 1 or args.quality_source is None or args.timing_source is None:
            parser.error("derived BS1 requires batch-size 1, --quality-source, and --timing-source")
        metadata, summary = _verify_derived_sources(args, scope, project)
        output.mkdir(parents=True)
        _atomic_json(output / "metadata.json", metadata)
        _atomic_json(output / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0
    required = (args.dataset, args.expected_dataset_sha256, args.upstream,
                args.checkpoint, args.expected_checkpoint_sha256)
    if any(value is None for value in required):
        parser.error("measured parallel execution requires dataset/upstream/checkpoint paths and SHA256")
    dataset, checkpoint = _require_file(args.dataset, "dataset"), _require_file(
        args.checkpoint, "checkpoint")
    upstream_path = _require_directory(args.upstream, "upstream")
    protocol = scope["paper_protocol"]
    validate_checkpoint_location(protocol, checkpoint, upstream_path)
    validate_dataset_path(protocol, dataset)
    dataset_sha, checkpoint_sha = sha256_file(dataset), sha256_file(checkpoint)
    if (dataset_sha != args.expected_dataset_sha256 or
            checkpoint_sha != args.expected_checkpoint_sha256):
        raise ValueError("LEHD parallel asset SHA256 mismatch")
    upstream = git_provenance(upstream_path)
    if (upstream["dirty"] or upstream["commit"] != UPSTREAM_COMMIT or
            normalize_git_repository_identity(upstream["url"]) !=
            normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("LEHD parallel upstream identity/cleanliness mismatch")
    import ml4co_kit as kit
    import torch
    tasks = _load_tasks(kit, args.problem, dataset)
    if len(tasks) != scope["expected_dataset_count"]:
        raise ValueError("LEHD parallel dataset count differs from frozen scope")
    device = _cuda_device(torch, args.device)
    environment = environment_provenance(device)
    if environment["gpu"] is None or "RTX 4090" not in environment["gpu"]:
        raise ValueError(f"LEHD formal parallel GPU must be {FORMAL_GPU}")
    from methods.lehd.runtime import build_tester, seed_project_rng, solve_batch
    tester, runtime_protocol = build_tester(
        problem=args.problem, problem_size=args.problem_size,
        protocol_label=args.protocol, upstream=upstream_path,
        checkpoint=checkpoint, device=device, torch=torch)
    if runtime_protocol != protocol:
        raise RuntimeError("LEHD parallel runtime config differs from frozen protocol")
    seed_project_rng(torch)
    metadata = {
        "schema": "lehd-parallel-evaluation.v1", "state": "RUNNING",
        "artifact_class": "baseline_parallel_evaluation",
        "artifact_generation_mode": MEASURED_MODE, "parallel_scope": scope,
        "dataset": {"path": str(dataset), "sha256": dataset_sha, "count": len(tasks)},
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "project": project, "upstream": upstream, "environment": environment,
        "source_files": _source_files(args.problem), "official_source_modified": False,
        "parallel_table_eligible": False,
    }
    output.mkdir(parents=True)
    _atomic_json(output / "metadata.json", metadata)
    records, timings = [], []
    try:
        for batch_index, (start, stop) in enumerate(batch_ranges(
                len(tasks), args.batch_size)):
            batch = tasks[start:stop]
            prepared = prepare_batch(
                args.problem, batch, args.problem_size, device=device, torch=torch)
            solved = solve_batch(
                tester, problem=args.problem, problem_size=args.problem_size,
                batch_size=len(batch), inject=prepared["inject"], device=device,
                torch=torch, timed=True)
            batch_records = validate_batch(
                args.problem, batch, prepared, solved["solutions"],
                solved["official_objectives"], args.problem_size,
                dataset_offset=start)
            for row in batch_records:
                row.update(batch_index=batch_index,
                           effective_batch_size=stop - start)
            records.extend(batch_records)
            timings.append({
                "batch_index": batch_index, "dataset_index_start": start,
                "dataset_index_stop_exclusive": stop,
                "original_instance_count": stop - start,
                "solver_runtime_seconds": float(solved["runtime_seconds"]),
            })
        summary = summarize_measured(records, timings, scope=scope)
        summary.update(dataset_sha256=dataset_sha, checkpoint_sha256=checkpoint_sha,
                       project=project, upstream=upstream, environment=environment,
                       source_files=metadata["source_files"])
        _atomic_jsonl(output / "validated_records.jsonl", records)
        _atomic_jsonl(output / "batch_timings.jsonl", timings)
        _atomic_json(output / "summary.json", summary)
        metadata.update(state="KIT_VALIDATED", parallel_table_eligible=True,
                        validated_count=len(records), number_of_batches=len(timings),
                        summary_file="summary.json",
                        records_file="validated_records.jsonl",
                        batch_timings_file="batch_timings.jsonl")
        _atomic_json(output / "metadata.json", metadata)
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
    except Exception as exc:
        metadata.update(state="FAILED", parallel_table_eligible=False,
                        failure_type=type(exc).__name__, failure_message=str(exc),
                        validated_count=len(records), number_of_batches=len(timings))
        _atomic_json(output / "metadata.json", metadata)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
