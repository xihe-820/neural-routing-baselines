#!/usr/bin/env python3
"""Fail-closed summary printer for the six remaining CVRPTW production cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.cvrptw_artifacts import RECORDS, SCHEMA, validate_record
from common.cvrptw_formal import DATASETS
from common.hashing import sha256_file
from common.paper_results import read_jsonl, write_json


def read_cell(path):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    identity = metadata.get("resume_identity", {})
    size = identity.get("problem_size")
    if (metadata.get("schema") != SCHEMA or metadata.get("state") != "PAPER_READY" or
            summary.get("state") != "PAPER_READY" or size not in DATASETS or
            identity.get("scope") != "production"):
        raise ValueError(f"{path} is not a formal production artifact")
    if identity.get("project", {}).get("dirty") or identity.get("upstream", {}).get("dirty"):
        raise ValueError(f"{path} provenance is dirty")
    if (identity.get("dataset", {}).get("sha256") != DATASETS[size]["sha256"] or
            identity.get("dataset", {}).get("count") != DATASETS[size]["count"]):
        raise ValueError(f"{path} dataset identity mismatch")
    records_path = path / RECORDS
    if metadata.get("validated_records_sha256") != sha256_file(records_path):
        raise ValueError(f"{path} records hash mismatch")
    records = read_jsonl(records_path)
    indices = [validate_record(record, problem_size=size) for record in records]
    if len(indices) != DATASETS[size]["count"] or set(indices) != set(range(DATASETS[size]["count"])):
        raise ValueError(f"{path} full-set coverage mismatch")
    if "RTX 4090" not in str(identity.get("environment", {}).get("gpu")):
        raise ValueError(f"{path} was not produced on RTX 4090")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for method in ("rfte", "cada", "moses-cada"):
        for size in (50, 100):
            parser.add_argument(f"--{method}-{size}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for method in ("rfte", "cada", "moses_cada"):
        for size in (50, 100):
            path = getattr(args, f"{method}_{size}")
            summary = read_cell(path)
            rows.append({
                "method": summary["method"], "problem": "CVRPTW",
                "size": size, "Obj": summary["mean_independent_objective"],
                "Drop_percent": summary["mean_instance_drop_percent"],
                "Time_seconds": summary["mean_runtime_seconds"],
                "artifact": str(path.resolve()),
            })
    payload = {"schema": "remaining-cvrptw-six-cell-summary-v1",
               "status": "PAPER_READY", "rows": rows}
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
