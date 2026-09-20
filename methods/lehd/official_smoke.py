#!/usr/bin/env python3
"""Pinned LEHD native-data strict-load/import/CUDA smoke without source writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.hashing import sha256_file
from common.provenance import (environment_provenance, git_provenance,
                               normalize_git_repository_identity)
from methods.lehd.config import (PROJECT_SEED, UPSTREAM_COMMIT, UPSTREAM_URL,
                                 resolve_config, validate_checkpoint_location)


NATIVE_DATASETS = {
    "tsp": "single_objective/LEHD/TSP/data/test_TSP1000_n128.txt",
    "cvrp": "single_objective/LEHD/CVRP/data/vrp1000_test_lkh.txt",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["tsp", "cvrp"], required=True)
    parser.add_argument("--problem-size", type=int, choices=[1000], default=1000)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--official-dataset", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--count", type=int, choices=[2], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("LEHD official smoke output already exists")
    config = resolve_config(args.problem, args.problem_size, "greedy")
    upstream_path = args.upstream.resolve()
    validate_checkpoint_location(config, args.checkpoint, upstream_path)
    expected_native = (upstream_path / NATIVE_DATASETS[args.problem]).resolve()
    if args.official_dataset.resolve() != expected_native:
        raise ValueError(f"LEHD native dataset must be pinned repository file: {expected_native}")
    if not args.checkpoint.is_file() or not args.official_dataset.is_file():
        raise ValueError("LEHD checkpoint and native dataset must exist")
    checkpoint_sha = sha256_file(args.checkpoint)
    dataset_sha = sha256_file(args.official_dataset)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("LEHD official smoke checkpoint SHA256 mismatch")
    if dataset_sha != args.expected_dataset_sha256:
        raise ValueError("LEHD official smoke dataset SHA256 mismatch")
    project, upstream = git_provenance(ROOT), git_provenance(upstream_path)
    if project["dirty"] or upstream["dirty"]:
        raise ValueError("LEHD official smoke requires clean project and upstream trees")
    if (upstream["commit"] != UPSTREAM_COMMIT
            or normalize_git_repository_identity(upstream["url"])
            != normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("LEHD official smoke upstream identity mismatch")

    import torch
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LEHD official smoke requires CUDA")
    if device.index is None:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpu = torch.cuda.get_device_name(device)
    if "RTX 4090" not in gpu:
        raise RuntimeError(f"current LEHD correctness workflow requires RTX 4090; observed {gpu!r}")
    from methods.lehd.runtime import build_tester, seed_project_rng
    seed_project_rng(torch, PROJECT_SEED)
    tester, runtime_config = build_tester(
        problem=args.problem, problem_size=1000, protocol_label="greedy",
        upstream=upstream_path, checkpoint=args.checkpoint.resolve(),
        device=device, torch=torch)
    if runtime_config != config:
        raise RuntimeError("LEHD native smoke runtime config mismatch")
    tester.tester_params["test_episodes"] = args.count
    tester.tester_params["test_batch_size"] = 1
    tester.env.data_path = str(args.official_dataset.resolve())
    tester.env.env_params["data_path"] = str(args.official_dataset.resolve())
    scores = tester.run()
    payload = {
        "schema": "lehd-official-native-smoke.v1", "status": "PASS",
        "problem": args.problem.upper(), "problem_size": 1000, "count": args.count,
        "artifact_class": "diagnostic_native_compatibility",
        "paper_result_eligible": False, "protocol": config,
        "project": project, "upstream": upstream,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": checkpoint_sha},
        "official_dataset": {
            "path": str(args.official_dataset.resolve()), "sha256": dataset_sha},
        "environment": environment_provenance(device),
        "script_sha256": sha256_file(Path(__file__)),
        "official_tester_return": [float(value) for value in scores],
        "purpose": "native-format compatibility smoke; not a paper result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
