#!/usr/bin/env python3
"""Export selected official ML4CO CVRPTW50/100 tasks to a neutral NPZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from methods.mvmoe.cvrptw.config import SUPPORTED_SIZES, get_size_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, choices=SUPPORTED_SIZES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    problem_size = args.problem_size
    size_config = get_size_config(problem_size)
    if args.offset < 0 or args.count <= 0:
        parser.error("offset must be nonnegative and count positive")
    actual_hash = sha256_file(args.dataset)
    if actual_hash != size_config["dataset_sha256"]:
        raise ValueError(f"unexpected CVRPTW{problem_size} dataset SHA256: {actual_hash}")

    import ml4co_kit as kit
    wrapper = kit.CVRPTWWrapper()
    wrapper.from_pickle(args.dataset)
    if len(wrapper.task_list) != size_config["dataset_count"]:
        raise ValueError(
            f"expected {size_config['dataset_count']} CVRPTW{problem_size} tasks")
    tasks = wrapper.task_list[args.offset:args.offset + args.count]
    if len(tasks) != args.count:
        raise ValueError("requested instance range exceeds dataset")
    for task in tasks:
        if type(task) is not kit.CVRPTWTask:
            raise ValueError("source task is not an exact ML4CO CVRPTWTask")
        if (task.points.shape != (problem_size, 2) or
                task.tw.shape != (problem_size + 1, 2)):
            raise ValueError(
                f"source task does not have exact CVRPTW{problem_size} point/TW shapes")
        if task.service.shape != (problem_size + 1,):
            raise ValueError(
                f"source task does not have exact CVRPTW{problem_size} service shape")
        if float(task.capacity) != size_config["capacity"]:
            raise ValueError(f"unexpected CVRPTW{problem_size} capacity")
        if float(task.service[0]) != 0.0:
            raise ValueError("benchmark depot service must be zero")
        arrays = (task.depots, task.points, task.demands, task.tw, task.service)
        if not all(np.isfinite(np.asarray(array)).all() for array in arrays):
            raise ValueError(f"CVRPTW{problem_size} task contains nonfinite values")

    depots = np.stack([task.depots for task in tasks])
    points = np.stack([task.points for task in tasks])
    demands = np.stack([task.demands for task in tasks])
    capacities = np.asarray([task.capacity for task in tasks], dtype=np.float32)
    time_windows = np.stack([task.tw for task in tasks])
    service_times = np.stack([task.service for task in tasks])
    tolerances = np.asarray([task.threshold for task in tasks], dtype=np.float64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, depots=depots, points=points, raw_demands=demands,
        raw_capacities=capacities, time_windows=time_windows,
        service_times=service_times, time_tolerances=tolerances,
        dataset_indices=np.arange(args.offset, args.offset + args.count, dtype=np.int64),
        reference_objectives=np.asarray(
            [task.evaluate(task.ref_sol) for task in tasks], dtype=np.float64),
    )
    metadata = {
        "format": "mvmoe-cvrptw-input-v1",
        "problem_size": problem_size,
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": actual_hash,
        "dataset_count": len(wrapper.task_list),
        "dataset_indices": list(range(args.offset, args.offset + args.count)),
        "instance_names": [task.name for task in tasks],
        "input_npz_path": str(args.output.resolve()),
        "input_npz_sha256": sha256_file(args.output),
        "depot_time_windows": [task.tw[0].astype(float).tolist() for task in tasks],
        "time_tolerances": tolerances.tolist(),
        "time_tolerance_source": "ML4CO-Kit CVRPTWTask.threshold",
        "normalization": "not applied; NPZ contains raw demand/capacity, coordinates, TW and service",
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(metadata_path)


if __name__ == "__main__":
    main()
