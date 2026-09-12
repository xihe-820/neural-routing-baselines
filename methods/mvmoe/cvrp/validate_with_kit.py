#!/usr/bin/env python3
"""Secondary ML4CO-Kit validation of MVMoE canonical result JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from common.result_schema import read_result_bundle, write_result_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = read_result_bundle(args.input)
    dataset_hash = sha256_file(args.dataset)
    import ml4co_kit as kit
    wrapper = kit.CVRPWrapper()
    wrapper.from_pickle(args.dataset)
    failures = []
    for row in payload["results"]:
        if row["dataset_sha256"] != dataset_hash:
            raise ValueError("result dataset hash does not match validation dataset")
        task = wrapper.task_list[row["dataset_instance_index"]]
        solution = np.asarray(row["canonical_solution"], dtype=np.int64)
        feasible = bool(task.check_constraints(solution))
        objective = float(task.evaluate(solution))
        error = abs(objective - row["independent_objective"])
        agrees = bool(np.isclose(objective, row["independent_objective"], rtol=1e-6, atol=1e-6))
        row["kit_feasible"] = feasible
        row["kit_objective"] = objective
        row["kit_objective_abs_error"] = error
        if row["independent_feasible"] and feasible and agrees:
            row["evidence_status"] = "LOCAL_VERIFIED"
            row["error"] = None
        else:
            row["evidence_status"] = "FAILED"
            row["error"] = "Kit feasibility/objective does not agree with independent validation"
            failures.append(row["dataset_instance_index"])
    payload["run_metadata"]["kit_validation"] = {
        "kit_module": kit.__file__, "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash, "rtol": 1e-6, "atol": 1e-6,
    }
    write_result_bundle(args.output, payload["results"], run_metadata=payload["run_metadata"])
    print(args.output)
    if failures:
        raise SystemExit(f"Kit validation failed for dataset indices {failures}")


if __name__ == "__main__":
    main()
