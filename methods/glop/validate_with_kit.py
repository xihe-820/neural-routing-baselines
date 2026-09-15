#!/usr/bin/env python3
"""Apply the ML4CO-Kit gate to completed formal GLOP chunks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from common.objective_agreement import (OBJECTIVE_ATOL, OBJECTIVE_RTOL,
                                        objective_agrees)
from methods.glop.paper_results import (METADATA_FILE, RECORDS_FILE,
                                        SCHEMA_VERSION, VALIDATED_RECORDS_FILE,
                                        fingerprint, read_jsonl, utc_now,
                                        validate_record, write_json)


def _write_jsonl(path, records):
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
    dataset_hash = sha256_file(args.dataset)
    loaded = []
    common = None
    for directory in args.chunk_dirs:
        metadata = json.loads((directory / METADATA_FILE).read_text())
        if (metadata.get("schema_version") != SCHEMA_VERSION or
                metadata.get("resume_fingerprint") != fingerprint(
                    metadata.get("resume_identity"))):
            raise ValueError(f"invalid GLOP chunk metadata in {directory}")
        if metadata.get("state") not in ("INFERENCE_COMPLETE", "KIT_VALIDATED"):
            raise ValueError(f"GLOP inference incomplete in {directory}")
        identity = metadata["resume_identity"]
        key = (identity["problem"], identity["problem_size"],
               identity["dataset"]["sha256"], identity["dataset"]["count"],
               identity["official_protocol_name"])
        if common is None:
            common = key
        elif key != common:
            raise ValueError("Kit inputs mix GLOP dataset or protocol identity")
        if dataset_hash != identity["dataset"]["sha256"]:
            raise ValueError("Kit dataset hash does not match GLOP chunk")
        records_path = directory / RECORDS_FILE
        if metadata.get("inference_records_sha256") != sha256_file(records_path):
            raise ValueError("GLOP inference record hash mismatch")
        records = read_jsonl(records_path)
        indices = [validate_record(record) for record in records]
        if set(indices) != set(identity["chunk"]["expected_indices"]):
            raise ValueError("GLOP chunk coverage mismatch before Kit")
        loaded.append((directory, metadata, records))

    problem, size, _, dataset_count, _ = common
    import ml4co_kit as kit
    wrapper = kit.TSPWrapper() if problem == "TSP" else kit.CVRPWrapper()
    wrapper.from_pickle(args.dataset)
    if len(wrapper.task_list) != dataset_count:
        raise ValueError("Kit dataset count differs from GLOP identity")
    failures = []
    for directory, metadata, records in loaded:
        output = []
        for record in records:
            index = record["dataset_instance_index"]
            task = wrapper.task_list[index]
            solution = np.asarray(record["canonical_solution"], dtype=np.int64)
            feasible = bool(task.check_constraints(solution))
            objective = float(task.evaluate(solution))
            agrees = objective_agrees(objective, record["independent_objective"])
            row = dict(record, kit_feasible=feasible, kit_objective=objective,
                       kit_objective_abs_error=abs(
                           objective - record["independent_objective"]),
                       kit_objective_agrees=agrees)
            if not feasible or not agrees:
                failures.append(index)
            output.append(row)
        path = directory / VALIDATED_RECORDS_FILE
        _write_jsonl(path, output)
        metadata.update(
            state="KIT_VALIDATED" if not any(
                row["dataset_instance_index"] in failures for row in records)
            else "KIT_VALIDATION_FAILED",
            validated_records_sha256=sha256_file(path),
            kit_validation={"dataset_path": str(args.dataset.resolve()),
                            "dataset_sha256": dataset_hash,
                            "kit_module": kit.__file__,
                            "kit_version": getattr(kit, "__version__", None),
                            "validator_sha256": sha256_file(Path(__file__)),
                            "objective_rtol": OBJECTIVE_RTOL,
                            "objective_atol": OBJECTIVE_ATOL,
                            "completed_at": utc_now()})
        write_json(directory / METADATA_FILE, metadata)
    if failures:
        raise SystemExit(f"Kit validation failed for indices {sorted(failures)}")


if __name__ == "__main__":
    main()
