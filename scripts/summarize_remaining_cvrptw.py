#!/usr/bin/env python3
"""Fail-closed summary printer for remaining CVRPTW production cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.cvrptw_artifacts import (RECORDS, SCHEMA, TIMINGS,
                                     validate_batch_timing, validate_record)
from common.cvrptw_formal import DATASETS
from common.hashing import sha256_file
from common.paper_results import read_jsonl, write_json


def read_cell(path, *, expected_batch_size=None):
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
    batch_size = identity.get("protocol", {}).get("original_instance_batch_size")
    if (batch_size not in (1, 10) or summary.get("batch_size") != batch_size or
            (expected_batch_size is not None and batch_size != expected_batch_size)):
        raise ValueError(f"{path} original-instance batch-size identity mismatch")
    timings_path = path / TIMINGS
    if metadata.get("batch_timings_sha256") != sha256_file(timings_path):
        raise ValueError(f"{path} batch-timing hash mismatch")
    timings = read_jsonl(timings_path)
    timing_indices = [validate_batch_timing(row, batch_size=batch_size)
                      for row in timings]
    if timing_indices != list(range(DATASETS[size]["count"] // batch_size)):
        raise ValueError(f"{path} native-batch timing coverage mismatch")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for method in ("rfte", "cada", "moses-cada"):
        for size in (50, 100):
            parser.add_argument(f"--{method}-{size}", type=Path)
    parser.add_argument(
        "--cell", action="append", nargs=4, metavar=("METHOD", "SIZE", "BS", "PATH"),
        help="batch-aware cell; repeat for RF-TE/MoSES BS1 and BS10")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = []
    if args.cell:
        for method, size_text, batch_text, path_text in args.cell:
            size, batch_size = int(size_text), int(batch_text)
            if method not in ("rfte", "moses_cada") or size not in (50, 100) or batch_size not in (1, 10):
                raise ValueError("--cell is restricted to RF-TE/MoSES CVRPTW50/100 BS1/10")
            cells.append((method, size, batch_size, Path(path_text)))
    else:
        for method in ("rfte", "cada", "moses_cada"):
            for size in (50, 100):
                path = getattr(args, f"{method}_{size}")
                if path is None:
                    raise ValueError("provide all legacy cells or repeat --cell")
                cells.append((method, size, 1, path))
    rows = []
    for method, size, batch_size, path in cells:
        summary = read_cell(path, expected_batch_size=batch_size)
        rows.append({
            "method": summary["method"], "problem": "CVRPTW",
            "size": size, "batch_size": batch_size,
            "Obj": summary["mean_independent_objective"],
            "Drop_percent": summary["mean_instance_drop_percent"],
            "Time_seconds": summary["mean_batch_runtime_seconds"],
            "Total_seconds": summary["total_runtime_seconds"],
            "artifact": str(path.resolve()),
        })
    payload = {"schema": "remaining-cvrptw-batch-summary-v2",
               "status": "PAPER_READY", "rows": rows}
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
