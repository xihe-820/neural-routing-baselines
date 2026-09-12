#!/usr/bin/env python3
"""Secondary ML4CO-Kit validation of MVMoE CVRPTW50/100 result JSON."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from common.objective_agreement import OBJECTIVE_ATOL, OBJECTIVE_RTOL, objective_agrees
from common.result_schema import complete_validation, read_result_bundle, write_result_bundle
from methods.mvmoe.cvrptw.config import SUPPORTED_SIZES, get_size_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-size", type=int, choices=SUPPORTED_SIZES, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    size_config = get_size_config(args.problem_size)
    payload = read_result_bundle(args.input)
    dataset_hash = sha256_file(args.dataset)
    if dataset_hash != size_config["dataset_sha256"]:
        raise ValueError(
            f"validation dataset is not the pinned CVRPTW{args.problem_size} dataset")
    import ml4co_kit as kit
    wrapper = kit.CVRPTWWrapper()
    wrapper.from_pickle(args.dataset)
    failures = []
    for row in payload["results"]:
        if (row["problem"] != "CVRPTW" or
                row["problem_size"] != args.problem_size or
                row["dataset_sha256"] != dataset_hash):
            raise ValueError(
                f"result identity does not match CVRPTW{args.problem_size} validation dataset")
        task = wrapper.task_list[row["dataset_instance_index"]]
        solution = np.asarray(row["canonical_solution"], dtype=np.int64)
        feasible = bool(task.check_constraints(solution))
        objective = float(task.evaluate(solution))
        error = abs(objective - row["independent_objective"])
        row["kit_feasible"] = feasible
        row["kit_objective"] = objective
        row["kit_objective_abs_error"] = error
        row["kit_objective_agrees"] = objective_agrees(
            objective, row["independent_objective"])
        complete_validation(row)
        if row["evidence_status"] == "FAILED":
            failures.append(row["dataset_instance_index"])
    payload["run_metadata"]["kit_validation"] = {
        "kit_module": kit.__file__, "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash, "rtol": OBJECTIVE_RTOL,
        "atol": OBJECTIVE_ATOL, "task": "CVRPTWTask",
    }
    write_result_bundle(args.output, payload["results"], run_metadata=payload["run_metadata"])
    print(args.output)
    if failures:
        raise SystemExit(f"Kit validation failed for dataset indices {failures}")


if __name__ == "__main__":
    main()
