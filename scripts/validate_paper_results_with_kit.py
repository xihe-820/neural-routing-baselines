#!/usr/bin/env python3
"""Validate one or more MVMoE paper chunks with one ML4CO-Kit dataset load."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from common.objective_agreement import (OBJECTIVE_ATOL, OBJECTIVE_RTOL,
                                        objective_agrees)
from common.paper_results import (METADATA_FILE, RECORDS_FILE, SCHEMA_VERSION,
                                  VALIDATED_RECORDS_FILE, read_jsonl, utc_now,
                                  json_fingerprint, validate_record, write_json)
from methods.mvmoe.cvrp.config import supported_config
from methods.mvmoe.cvrptw.config import get_size_config


def _write_jsonl(path, records):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--chunk-dirs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    metadatas = []
    identity = None
    for directory in args.chunk_dirs:
        metadata = json.loads((directory / METADATA_FILE).read_text())
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported paper schema in {directory}")
        if metadata.get("resume_fingerprint") != json_fingerprint(
                metadata.get("resume_identity")):
            raise ValueError(f"paper metadata fingerprint mismatch in {directory}")
        if metadata.get("state") not in ("INFERENCE_COMPLETE", "KIT_VALIDATED"):
            raise ValueError(f"paper inference is incomplete in {directory}")
        current = metadata["resume_identity"]
        key = (current["method"], current["problem"], current["problem_size"],
               current["dataset"]["sha256"], current["dataset"]["count"])
        if identity is None:
            identity = key
        elif key != identity:
            raise ValueError("Kit validation inputs mix method/problem/size/dataset identity")
        records_path = directory / RECORDS_FILE
        if metadata.get("inference_records_sha256") != sha256_file(records_path):
            raise ValueError(f"inference record hash mismatch in {directory}")
        records = read_jsonl(records_path)
        expected_indices = current["chunk"]["expected_indices"]
        indices = [validate_record(record) for record in records]
        if len(indices) != len(set(indices)) or set(indices) != set(expected_indices):
            raise ValueError(f"chunk coverage mismatch in {directory}")
        metadatas.append((directory, metadata, records))

    _, problem, problem_size, expected_hash, expected_count = identity
    dataset_hash = sha256_file(args.dataset)
    if dataset_hash != expected_hash:
        raise ValueError("validation dataset SHA256 does not match paper chunks")
    config = (supported_config(problem_size) if problem == "CVRP"
              else get_size_config(problem_size))
    if (config["dataset_sha256"] != dataset_hash or
            config["dataset_count"] != expected_count):
        raise ValueError("dataset identity/count differs from pinned configuration")

    import ml4co_kit as kit
    wrapper = kit.CVRPWrapper() if problem == "CVRP" else kit.CVRPTWWrapper()
    wrapper.from_pickle(args.dataset)
    if len(wrapper.task_list) != expected_count:
        raise ValueError(f"expected {expected_count} tasks in validation dataset")

    failures = []
    for directory, metadata, records in metadatas:
        validated = []
        for record in records:
            index = record["dataset_instance_index"]
            task = wrapper.task_list[index]
            solution = np.asarray(record["canonical_solution"], dtype=np.int64)
            feasible = bool(task.check_constraints(solution))
            objective = float(task.evaluate(solution))
            agrees = objective_agrees(objective, record["independent_objective"])
            output = dict(record)
            output.update(
                kit_feasible=feasible,
                kit_objective=objective,
                kit_objective_abs_error=abs(objective - record["independent_objective"]),
                kit_objective_agrees=agrees,
            )
            if not feasible or not agrees:
                failures.append(index)
            validated.append(output)
        output_path = directory / VALIDATED_RECORDS_FILE
        _write_jsonl(output_path, validated)
        metadata.update(
            state="KIT_VALIDATED" if not any(
                record["dataset_instance_index"] in failures for record in records)
            else "KIT_VALIDATION_FAILED",
            validated_records_sha256=sha256_file(output_path),
            kit_validation={
                "dataset_path": str(args.dataset.resolve()),
                "dataset_sha256": dataset_hash,
                "kit_module": kit.__file__,
                "objective_rtol": OBJECTIVE_RTOL,
                "objective_atol": OBJECTIVE_ATOL,
                "completed_at": utc_now(),
            },
        )
        write_json(directory / METADATA_FILE, metadata)
        print(directory / METADATA_FILE)
    if failures:
        raise SystemExit(f"Kit validation failed for dataset indices {sorted(failures)}")


if __name__ == "__main__":
    main()
