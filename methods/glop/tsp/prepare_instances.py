#!/usr/bin/env python3
"""Export exact ML4CO TSP50/100 tasks to a neutral NPZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from methods.glop.tsp.adapter import adapt_points
from methods.glop.tsp.config import supported_config


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
        raise ValueError("TSP dataset SHA256 does not match the pinned benchmark")

    import ml4co_kit as kit
    wrapper = kit.TSPWrapper()
    wrapper.from_pickle(args.dataset)
    if len(wrapper.task_list) != config["dataset_count"]:
        raise ValueError("official TSP dataset count does not match its pinned identity")
    tasks = wrapper.task_list[args.offset:args.offset + args.count]
    if len(tasks) != args.count:
        raise ValueError("requested instance range exceeds dataset")
    for task in tasks:
        if type(task) is not kit.TSPTask or task.points.shape != (args.problem_size, 2):
            raise ValueError("source task is not the exact expected ML4CO TSPTask")
        if not np.isfinite(task.points).all():
            raise ValueError("source task contains NaN or Inf")
    points = np.stack([task.points for task in tasks]).astype(np.float32, copy=False)
    adapt_points(points, problem_size=args.problem_size, device="cpu")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points=points,
        dataset_indices=np.arange(args.offset, args.offset + args.count, dtype=np.int64),
        reference_objectives=np.asarray([task.evaluate(task.ref_sol) for task in tasks], dtype=np.float64),
    )
    metadata = {
        "format": "glop-tsp-neutral-input-v1",
        "problem_size": args.problem_size,
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "dataset_indices": list(range(args.offset, args.offset + args.count)),
        "instance_names": [task.name for task in tasks],
        "input_npz_path": str(args.output.resolve()),
        "input_npz_sha256": sha256_file(args.output),
        "transformation": "none; original float32 points",
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(metadata_path)


if __name__ == "__main__":
    main()
