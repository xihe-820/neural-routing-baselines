#!/usr/bin/env python3
"""Prepare an explicit formal GLOP manuscript CVRP dataset slice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from methods.glop.cvrp.adapter import validate_instance
from methods.glop.paper_protocol import expected_dataset_filename, formal_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    if args.offset != 0 or args.count <= 0:
        parser.error(
            "formal CVRP preparation requires offset=0 and a positive count")
    protocol = formal_protocol("CVRP", args.problem_size, args.protocol)
    expected_name = expected_dataset_filename("CVRP", args.problem_size)
    if args.dataset.name != expected_name:
        raise ValueError(
            f"formal CVRP dataset filename must be exactly {expected_name}")

    import ml4co_kit as kit
    wrapper = kit.CVRPWrapper()
    wrapper.from_pickle(args.dataset)
    dataset_count = len(wrapper.task_list)
    tasks = wrapper.task_list[args.offset:args.offset + args.count]
    if len(tasks) != args.count:
        raise ValueError("requested instance range exceeds dataset")
    for task in tasks:
        if type(task) is not kit.CVRPTask:
            raise ValueError("dataset does not contain exact ML4CO CVRPTask objects")
        validate_instance(task.depots, task.points, task.demands, task.capacity,
                          problem_size=args.problem_size)
    depots = np.stack([np.asarray(task.depots).reshape(-1, 2)[0] for task in tasks])
    points = np.stack([task.points for task in tasks])
    demands = np.stack([task.demands for task in tasks])
    capacities = np.asarray([task.capacity for task in tasks], dtype=np.float64)
    references = np.asarray(
        [task.evaluate(task.ref_sol) for task in tasks], dtype=np.float64)
    if not np.isfinite(references).all() or (references <= 0).any():
        raise ValueError("dataset contains invalid reference objectives")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, depots=depots.astype(np.float32, copy=False),
        points=points.astype(np.float32, copy=False), raw_demands=demands,
        raw_capacities=capacities,
        dataset_indices=np.arange(args.offset, args.offset + args.count, dtype=np.int64),
        reference_objectives=references)
    dataset_hash = sha256_file(args.dataset)
    metadata = {
        "format": "glop-paper-cvrp-input-v1", "problem": "CVRP",
        "problem_size": args.problem_size,
        "official_protocol_name": protocol["official_protocol_name"],
        "expected_dataset_filename": expected_name,
        "dataset_path": str(args.dataset.resolve()), "dataset_sha256": dataset_hash,
        "dataset_size_bytes": args.dataset.stat().st_size,
        "dataset_count": dataset_count,
        "dataset_task_class": f"{type(tasks[0]).__module__}.{type(tasks[0]).__name__}",
        "coordinate_shape": [args.problem_size, 2],
        "coordinate_min": float(min(depots.min(), points.min())),
        "coordinate_max": float(max(depots.max(), points.max())),
        "demand_min": float(demands.min()), "demand_max": float(demands.max()),
        "capacity_values": sorted(set(map(float, capacities))),
        "dataset_indices": list(range(args.offset, args.offset + args.count)),
        "instance_names": [task.name for task in tasks],
        "reference_source": "ML4CO task.ref_sol evaluated by task.evaluate",
        "input_npz_path": str(args.output.resolve()),
        "input_npz_sha256": sha256_file(args.output),
        "transformation": "none; raw demands/capacity and original float32 coordinates",
    }
    path = args.output.with_suffix(args.output.suffix + ".json")
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
