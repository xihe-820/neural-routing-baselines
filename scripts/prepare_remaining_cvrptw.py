#!/usr/bin/env python3
"""Export pinned ML4CO CVRPTW tasks to the shared neutral formal NPZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.cvrptw_formal import dataset_config, validate_task
from common.hashing import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, choices=(50, 100), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = dataset_config(args.problem_size)
    if args.offset < 0 or args.count <= 0:
        parser.error("offset must be nonnegative and count positive")
    if args.dataset.name != cfg["filename"] or sha256_file(args.dataset) != cfg["sha256"]:
        raise ValueError("dataset filename/SHA256 differs from pinned formal identity")
    import ml4co_kit as kit
    wrapper = kit.CVRPTWWrapper()
    wrapper.from_pickle(args.dataset)
    if len(wrapper.task_list) != cfg["count"]:
        raise ValueError("pinned CVRPTW dataset count mismatch")
    for task in wrapper.task_list:
        if type(task) is not kit.CVRPTWTask:
            raise ValueError("dataset contains a non-exact ML4CO CVRPTWTask")
        validate_task(task, args.problem_size)
    tasks = wrapper.task_list[args.offset:args.offset + args.count]
    if len(tasks) != args.count:
        raise ValueError("requested range exceeds the pinned dataset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        depots=np.stack([task.depots for task in tasks]),
        points=np.stack([task.points for task in tasks]),
        raw_demands=np.stack([task.demands for task in tasks]),
        raw_capacities=np.asarray([task.capacity for task in tasks], dtype=np.float32),
        time_windows=np.stack([task.tw for task in tasks]),
        service_times=np.stack([task.service for task in tasks]),
        time_tolerances=np.asarray([task.threshold for task in tasks], dtype=np.float64),
        dataset_indices=np.arange(args.offset, args.offset + args.count, dtype=np.int64),
        reference_objectives=np.asarray(
            [task.evaluate(task.ref_sol) for task in tasks], dtype=np.float64),
    )
    metadata = {
        "format": "remaining-cvrptw-input-v1", "problem_size": args.problem_size,
        "dataset_path": str(args.dataset.resolve()), "dataset_filename": args.dataset.name,
        "dataset_sha256": cfg["sha256"], "dataset_count": cfg["count"],
        "dataset_indices": list(range(args.offset, args.offset + args.count)),
        "instance_names": [task.name for task in tasks],
        "task_class": f"{type(tasks[0]).__module__}.{type(tasks[0]).__name__}",
        "first_instance_schema": {
            "depot_shape": list(np.asarray(tasks[0].depots).shape),
            "points_shape": list(np.asarray(tasks[0].points).shape),
            "raw_demands_shape": list(np.asarray(tasks[0].demands).shape),
            "capacity": float(tasks[0].capacity),
            "time_windows_shape": list(np.asarray(tasks[0].tw).shape),
            "service_time_shape": list(np.asarray(tasks[0].service).shape),
            "depot_time_window": np.asarray(tasks[0].tw[0]).astype(float).tolist(),
            "depot_service_time": float(tasks[0].service[0]),
            "coordinate_dtype": str(np.asarray(tasks[0].points).dtype),
            "demand_dtype": str(np.asarray(tasks[0].demands).dtype),
            "time_window_dtype": str(np.asarray(tasks[0].tw).dtype),
            "service_dtype": str(np.asarray(tasks[0].service).dtype),
            "reference_solution_present": tasks[0].ref_sol is not None,
            "reference_objective": float(tasks[0].evaluate(tasks[0].ref_sol)),
            "closed_routes": not bool(tasks[0].cvrp_open),
            "threshold": float(tasks[0].threshold),
        },
        "input_npz_path": str(args.output.resolve()),
        "input_npz_sha256": sha256_file(args.output),
        "normalization": (
            "neutral original units: coordinates, raw demand/capacity, time windows and "
            "service; method adapters normalize demand exactly once"),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(metadata_path)


if __name__ == "__main__":
    main()
