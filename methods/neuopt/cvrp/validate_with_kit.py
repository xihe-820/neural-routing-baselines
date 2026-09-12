#!/usr/bin/env python3
"""Secondary exact-task ML4CO-Kit validation for NeuOpt CVRP results."""
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
from methods.neuopt.cvrp.config import supported_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, choices=[50, 100], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = supported_config(args.problem_size)
    payload = read_result_bundle(args.input)
    dataset_hash = sha256_file(args.dataset)
    if dataset_hash != config["dataset_sha256"]:
        raise ValueError("validation dataset is not the pinned official dataset")
    import ml4co_kit as kit
    wrapper = kit.CVRPWrapper()
    wrapper.from_pickle(args.dataset)
    if len(wrapper.task_list) != config["dataset_count"]:
        raise ValueError("official dataset count does not match pinned identity")
    failures = []
    for row in payload["results"]:
        if (row["method"], row["problem"], row["problem_size"], row["dataset_sha256"]) != (
                "NeuOpt", "CVRP", args.problem_size, dataset_hash):
            raise ValueError("result identity does not match this validation run")
        task = wrapper.task_list[row["dataset_instance_index"]]
        if type(task) is not kit.CVRPTask:
            raise ValueError("benchmark wrapper returned a non-exact CVRPTask")
        solution = np.asarray(row["canonical_solution"], dtype=np.int64)
        row["kit_feasible"] = bool(task.check_constraints(solution))
        row["kit_objective"] = float(task.evaluate(solution))
        row["kit_objective_abs_error"] = abs(row["kit_objective"] - row["independent_objective"])
        row["kit_objective_agrees"] = objective_agrees(
            row["kit_objective"], row["independent_objective"]
        )
        complete_validation(row)
        if row["evidence_status"] == "FAILED":
            failures.append(row["dataset_instance_index"])
    payload["run_metadata"]["kit_validation"] = {
        "kit_module": kit.__file__, "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash, "exact_task_type": "ml4co_kit.CVRPTask",
        "rtol": OBJECTIVE_RTOL, "atol": OBJECTIVE_ATOL,
    }
    write_result_bundle(args.output, payload["results"], run_metadata=payload["run_metadata"])
    print(args.output)
    if failures:
        raise SystemExit(f"Kit validation failed for dataset indices {failures}")


if __name__ == "__main__":
    main()
