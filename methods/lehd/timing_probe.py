#!/usr/bin/env python3
"""Small-sample BS1 LEHD timing probe with full post-timing correctness gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.hashing import sha256_file
from common.provenance import (environment_provenance, git_provenance,
                               normalize_git_repository_identity, source_provenance)
from methods.lehd.author_batch_eval import (_cuda_device, _load_tasks, _require_directory,
                                            _require_file, prepare_batch, validate_batch)
from methods.lehd.config import (AUTHOR_BATCH_REGISTRY, FORMAL_PROTOCOLS,
                                 TIMING_PROBE_COUNTS, UPSTREAM_COMMIT, UPSTREAM_URL,
                                 WARMUP_POLICY, resolve_config,
                                 validate_checkpoint_location, validate_dataset_path)
from methods.lehd.paper_results import TIMING_SEMANTICS


def timing_count(protocol: str, count: int | None) -> int:
    selected = TIMING_PROBE_COUNTS[protocol] if count is None else int(count)
    if selected <= 0:
        raise ValueError("LEHD timing probe count must be positive")
    return selected


def _source_files(problem: str) -> list[dict]:
    return source_provenance([
        Path(__file__), ROOT / "methods/lehd/config.py", ROOT / "methods/lehd/runtime.py",
        ROOT / "methods/lehd/author_batch_eval.py",
        ROOT / f"methods/lehd/{problem}/adapter.py", ROOT / f"problems/{problem}/validate.py",
    ], root=ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["tsp", "cvrp"], required=True)
    parser.add_argument("--problem-size", type=int, required=True)
    parser.add_argument("--protocol", choices=FORMAL_PROTOCOLS, required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--dump-config", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    protocol = resolve_config(args.problem, args.problem_size, args.protocol)
    count = timing_count(args.protocol, args.count)
    if args.dump_config:
        print(json.dumps({"protocol": protocol, "timing_count": count,
                          "original_instance_batch_size": 1}, indent=2,
                         sort_keys=True, allow_nan=False))
        return 0
    if not args.expected_dataset_sha256 or not args.expected_checkpoint_sha256:
        parser.error("execution requires expected dataset and checkpoint SHA256")
    if args.output is None:
        parser.error("execution requires --output")
    output = args.output.resolve()
    if output.exists():
        raise ValueError("LEHD timing probe output already exists")
    dataset, checkpoint = _require_file(args.dataset, "dataset"), _require_file(args.checkpoint, "checkpoint")
    upstream_path = _require_directory(args.upstream, "upstream")
    validate_checkpoint_location(protocol, checkpoint, upstream_path)
    validate_dataset_path(protocol, dataset)
    dataset_sha, checkpoint_sha = sha256_file(dataset), sha256_file(checkpoint)
    if dataset_sha != args.expected_dataset_sha256 or checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("LEHD timing probe asset SHA256 mismatch")
    project, upstream = git_provenance(ROOT), git_provenance(upstream_path)
    if project["dirty"]:
        raise ValueError("LEHD timing probe requires a clean project checkout")
    if (upstream["dirty"] or upstream["commit"] != UPSTREAM_COMMIT or
            normalize_git_repository_identity(upstream["url"]) != normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("LEHD upstream identity/cleanliness mismatch")
    import ml4co_kit as kit
    import torch
    tasks = _load_tasks(kit, args.problem, dataset)
    expected_count = AUTHOR_BATCH_REGISTRY[(args.problem, args.problem_size)]["dataset_count"]
    if len(tasks) != expected_count or count > len(tasks):
        raise ValueError("LEHD timing probe dataset/count differs from frozen protocol")
    device = _cuda_device(torch, args.device)
    from methods.lehd.runtime import build_tester, run_isolated_warmup, seed_project_rng, solve_one

    def build():
        tester, runtime_protocol = build_tester(
            problem=args.problem, problem_size=args.problem_size,
            protocol_label=args.protocol, upstream=upstream_path,
            checkpoint=checkpoint, device=device, torch=torch)
        if runtime_protocol != protocol:
            raise RuntimeError("LEHD timing runtime config differs from frozen protocol")
        return tester

    seed_project_rng(torch)
    warmup_prepared = prepare_batch(
        args.problem, [tasks[0]], args.problem_size, device=device, torch=torch)
    run_isolated_warmup(
        build, lambda tester: solve_one(
            tester, problem=args.problem, problem_size=args.problem_size,
            inject=warmup_prepared["inject"], device=device, torch=torch, timed=False),
        torch=torch)
    tester = build()
    records, runtimes = [], []
    for index, task in enumerate(tasks[:count]):
        prepared = prepare_batch(args.problem, [task], args.problem_size, device=device, torch=torch)
        solved = solve_one(
            tester, problem=args.problem, problem_size=args.problem_size,
            inject=prepared["inject"], device=device, torch=torch, timed=True)
        validated = validate_batch(
            args.problem, [task], prepared,
            np.asarray([solved["solution"]]), np.asarray([solved["official_objective"]]),
            args.problem_size, dataset_offset=index)[0]
        validated.update(runtime_seconds=float(solved["runtime_seconds"]),
                         solution_capture=solved["capture"])
        records.append(validated)
        runtimes.append(float(solved["runtime_seconds"]))
    values = np.asarray(runtimes, dtype=float)
    summary = {
        "schema": "lehd-bs1-timing-probe.v1", "status": "KIT_VALIDATED",
        "artifact_class": "baseline_bs1_timing_probe", "method": "LEHD",
        "problem": protocol["problem"], "problem_size": protocol["actual_problem_size"],
        "protocol": args.protocol, "RRC_budget": protocol["RRC_budget"], "count": count,
        "original_instance_batch_size": 1, "individual_runtimes": runtimes,
        "mean_runtime_seconds": float(values.mean()), "median_runtime_seconds": float(np.median(values)),
        "min_runtime_seconds": float(values.min()), "max_runtime_seconds": float(values.max()),
        "validation_records": records, "validated_count": len(records), "failed_count": 0,
        "warmup": WARMUP_POLICY, "dataset_sha256": dataset_sha,
        "checkpoint_sha256": checkpoint_sha, "project": project, "upstream": upstream,
        "environment": environment_provenance(device), "source_files": _source_files(args.problem),
        "timing_semantics": TIMING_SEMANTICS,
        "paper_result_eligible_for_quality": False, "timing_column_eligible": True,
        "official_source_modified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
