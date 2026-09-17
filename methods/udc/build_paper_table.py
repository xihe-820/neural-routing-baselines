#!/usr/bin/env python3
"""Build the unique 20-cell UDC paper table from complete production evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from methods.udc.paper_protocol import (OFFICIAL_COMMIT, REGISTRY_SHA256,
                                        SCALE_DATASET_FILENAMES,
                                        formal_cells, load_budget_registry)
from methods.udc.paper_results import (METADATA_FILE, RECORDS_FILE, SUMMARY_FILE,
                                       atomic_json, fingerprint, read_jsonl,
                                       summarize_records)


def _same_number(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def load_complete_run(root: Path, problem: str, size: int, label: str) -> dict:
    directory = Path(root) / f"{problem}{size}" / label
    metadata_path = directory / METADATA_FILE
    records_path = directory / RECORDS_FILE
    summary_path = directory / SUMMARY_FILE
    metadata = json.loads(metadata_path.read_text())
    summary = json.loads(summary_path.read_text())
    identity = metadata.get("resume_identity")
    if (metadata.get("state") != "KIT_VALIDATED" or not isinstance(identity, dict)
            or metadata.get("resume_fingerprint") != fingerprint(identity)):
        raise ValueError(f"UDC production run is incomplete: {directory}")
    if (identity.get("problem") != problem or identity.get("size") != size
            or identity.get("budget", {}).get("label") != label):
        raise ValueError(f"UDC production identity mismatches paper cell: {directory}")
    frozen_budget = load_budget_registry()["entries"][(problem, label)]
    if identity.get("budget") != frozen_budget:
        raise ValueError(f"UDC production budget differs from frozen registry: {directory}")
    if (identity.get("registry", {}).get("sha256") != REGISTRY_SHA256
            or Path(identity.get("dataset", {}).get("filename", "")).name
            != SCALE_DATASET_FILENAMES[(problem, size)]):
        raise ValueError(f"UDC registry or dataset filename mismatch: {directory}")
    if (sha256_file(identity["dataset"]["path"]) != identity["dataset"]["sha256"]
            or metadata.get("validated_records_sha256") != sha256_file(records_path)
            or summary.get("validated_records_sha256") != sha256_file(records_path)
            or metadata.get("summary_sha256") != sha256_file(summary_path)):
        raise ValueError(f"UDC production artifact hash integrity failed: {directory}")
    records = read_jsonl(records_path)
    names = identity["dataset"].get("instance_names")
    if (not isinstance(names, list) or len(names) != identity["dataset"]["count"]
            or fingerprint(names) != identity["dataset"].get("instance_names_sha256")
            or [record.get("instance_id") for record in records] != names):
        raise ValueError(f"UDC dataset instance identity/order mismatch: {directory}")
    for checkpoint in identity.get("checkpoints", {}).values():
        if sha256_file(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError(f"UDC checkpoint bytes changed: {directory}")
    recomputed = summarize_records(records, identity)
    required = ("mean_objective", "mean_reference_objective",
                "mean_instance_drop_percent", "mean_runtime_seconds",
                "objective_std", "drop_std", "runtime_std")
    if any(not _same_number(summary.get(key), recomputed[key]) for key in required):
        raise ValueError(f"UDC summary differs from validated records: {directory}")
    if (summary.get("count") != identity["dataset"]["count"]
            or summary.get("validated_count") != identity["dataset"]["count"]
            or summary.get("failed_count") != 0
            or metadata.get("project_post", {}).get("pass") is not True
            or metadata.get("project_post", {}).get("head") != identity["project"]["head"]
            or metadata.get("official_post", {}).get("pass") is not True
            or metadata.get("official_post", {}).get("head") != OFFICIAL_COMMIT
            or identity.get("environment", {}).get("gpu_name", "").find("RTX 4090") < 0
            or identity.get("batch_size") != 1 or identity.get("seed_once") != 1234):
        raise ValueError(f"UDC formal production conditions failed: {directory}")
    return {"directory": str(directory.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "records_sha256": sha256_file(records_path),
            "summary_sha256": sha256_file(summary_path), "summary": summary}


def build_table(production_root: Path) -> dict:
    registry = load_budget_registry()
    cells = []
    project_heads = set()
    for problem, size, label in formal_cells():
        artifact = load_complete_run(production_root, problem, size, label)
        summary = artifact["summary"]
        identity = summary["resume_identity"]
        project_heads.add(identity["project"]["head"])
        cells.append({
            "problem": problem.upper(), "size": size, "method": "UDC",
            "budget": label, "objective": summary["mean_objective"],
            "drop_percent": summary["mean_instance_drop_percent"],
            "time_seconds": summary["mean_runtime_seconds"],
            "count": summary["count"], "dataset_sha256": summary["dataset_sha256"],
            "artifact_path": artifact["directory"],
            "metadata_sha256": artifact["metadata_sha256"],
            "records_sha256": artifact["records_sha256"],
            "summary_sha256": artifact["summary_sha256"],
            "main_table_key": f"{problem}{size}/{label}",
            "appendix_table_key": f"{problem}{size}/{label}",
        })
    if len(cells) != 20 or len(project_heads) != 1:
        raise ValueError("UDC paper table requires exactly 20 cells from one project commit")
    if any(cell["problem"] == "CVRP" and cell["size"] in (50, 100) for cell in cells):
        raise ValueError("CVRP50/100 UDC cells are outside paper scope")
    return {"schema": "udc-paper-table.v1", "method": "UDC",
            "registry_sha256": registry["sha256"],
            "project_head": project_heads.pop(),
            "scope": {"tsp_sizes": [100, 500, 1000, 2000, 5000, 10000],
                      "cvrp_sizes": [200, 500, 1000, 2000],
                      "budgets": ["fewer", "more"], "augmentation": "none"},
            "metrics": {"objective": "mean(obj_i)",
                        "drop_percent": "mean((obj_i-ref_i)/ref_i*100)",
                        "time_seconds": "mean BS=1 production solver wall-clock"},
            "cells": cells}


def write_csv(path: Path, cells):
    fields = ("problem", "size", "method", "budget", "objective",
              "drop_percent", "time_seconds")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            writer.writerow({key: cell[key] for key in fields})


def write_markdown(path: Path, cells):
    lines = ["| Problem | Size | Method | Budget | Obj | Drop (%) | Time (s) |",
             "|---|---:|---|---|---:|---:|---:|"]
    for cell in cells:
        lines.append("| {problem} | {size} | {method} | {budget} | {objective:.6f} | "
                     "{drop_percent:.3f} | {time_seconds:.3f} |".format(**cell))
    path.write_text("\n".join(lines) + "\n")


def build_and_write(production_root: Path, output_dir: Path):
    production_root, output_dir = Path(production_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    value = build_table(production_root)
    atomic_json(output_dir / "paper_table_udc.json", value)
    write_csv(output_dir / "paper_table_udc.csv", value["cells"])
    write_markdown(output_dir / "paper_table_udc.md", value["cells"])
    print(output_dir / "paper_table_udc.json")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_and_write(args.production_root, args.output_dir)


if __name__ == "__main__":
    main()
