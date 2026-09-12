#!/usr/bin/env python3
"""Export exact official ML4CO CVRP50/100 tasks to a neutral NPZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from methods.neuopt.cvrp.config import supported_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, choices=[50, 100], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    if args.offset < 0 or args.count <= 0:
        parser.error("offset must be nonnegative and count positive")
    config = supported_config(args.problem_size)
    dataset_hash = sha256_file(args.dataset)
    if dataset_hash != config["dataset_sha256"]:
        raise ValueError(f"unexpected CVRP{args.problem_size} dataset SHA256: {dataset_hash}")

    import ml4co_kit as kit
    wrapper = kit.CVRPWrapper()
    wrapper.from_pickle(args.dataset)
    if len(wrapper.task_list) != config["dataset_count"]:
        raise ValueError("official dataset count does not match pinned identity")
    tasks = wrapper.task_list[args.offset:args.offset + args.count]
    if len(tasks) != args.count:
        raise ValueError("requested instance range exceeds dataset")
    for task in tasks:
        arrays = (task.depots, task.points, task.demands)
        if type(task) is not kit.CVRPTask or task.points.shape != (args.problem_size, 2):
            raise ValueError(f"source task is not an exact ML4CO CVRP{args.problem_size} task")
        if float(task.capacity) != config["capacity"]:
            raise ValueError(f"unexpected CVRP{args.problem_size} capacity")
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("source task contains NaN or Inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        depots=np.stack([task.depots for task in tasks]),
        points=np.stack([task.points for task in tasks]),
        raw_demands=np.stack([task.demands for task in tasks]),
        raw_capacities=np.asarray([task.capacity for task in tasks], dtype=np.float32),
        dataset_indices=np.arange(args.offset, args.offset + args.count, dtype=np.int64),
        reference_objectives=np.asarray([task.evaluate(task.ref_sol) for task in tasks], dtype=np.float64),
    )
    metadata = {
        "format": "neuopt-cvrp-neutral-input-v1",
        "problem_size": args.problem_size,
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "dataset_indices": list(range(args.offset, args.offset + args.count)),
        "instance_names": [task.name for task in tasks],
        "input_npz_path": str(args.output.resolve()),
        "input_npz_sha256": sha256_file(args.output),
        "normalization": "not applied; NPZ contains raw demands and raw capacity; no dummy nodes",
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(metadata_path)


if __name__ == "__main__":
    main()
