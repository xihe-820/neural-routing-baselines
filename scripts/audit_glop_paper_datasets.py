#!/usr/bin/env python3
"""Audit the nine GLOP manuscript datasets without running GLOP inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_datasets import audit
from methods.glop.paper_protocol import PAPER_DATASET_FILENAMES

SCOPE = (("TSP", 100), ("TSP", 500), ("TSP", 1000), ("TSP", 2000),
         ("TSP", 5000), ("TSP", 10000), ("CVRP", 500),
         ("CVRP", 1000), ("CVRP", 2000))


def discover(dataset_root):
    found = {}
    if not dataset_root.is_dir():
        return found
    for key, filename in PAPER_DATASET_FILENAMES.items():
        found[key] = sorted(dataset_root.rglob(filename))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=os.environ.get("ML4CO_DATA_ROOT"), required=False)
    parser.add_argument("--dataset", nargs=3, action="append", default=[],
                        metavar=("PROBLEM", "SIZE", "PATH"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="user-executed-server-audit")
    args = parser.parse_args()
    if args.dataset_root is None and not args.dataset:
        parser.error("provide --dataset-root/ML4CO_DATA_ROOT or explicit --dataset entries")

    candidates = discover(args.dataset_root) if args.dataset_root else {}
    explicit = {}
    for problem, size, path in args.dataset:
        key = (problem.upper(), int(size))
        if key not in SCOPE:
            parser.error(f"outside GLOP paper scope: {key}")
        if Path(path).name != PAPER_DATASET_FILENAMES[key]:
            parser.error(
                f"formal dataset for {key} must be named "
                f"{PAPER_DATASET_FILENAMES[key]}")
        explicit.setdefault(key, []).append(Path(path))
    # Explicit paths override filename discovery for that problem/size. This
    # prevents similarly named archives or copies from entering a formal audit.
    candidates.update(explicit)

    rows = []
    coverage = []
    for problem, size in SCOPE:
        paths = list(dict.fromkeys(candidates.get((problem, size), [])))
        audited = [audit(path, problem, size) for path in paths]
        for row in audited:
            row["expected_filename"] = PAPER_DATASET_FILENAMES[(problem, size)]
            row["filename_matches"] = (
                Path(row["realpath"]).name == row["expected_filename"])
        rows.extend(audited)
        valid = [row["realpath"] for row in audited
                 if row.get("all_expected_size") and row.get("all_expected_task_class")
                 and row.get("filename_matches") and not row.get("error")]
        coverage.append({"problem": problem, "size": size,
                         "expected_filename":
                         PAPER_DATASET_FILENAMES[(problem, size)],
                         "candidate_count": len(paths), "verified_files": valid,
                         "status": "VERIFIED" if len(valid) == 1 else
                                   "AMBIGUOUS" if len(valid) > 1 else "MISSING"})

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "python": sys.version,
        "executable": sys.executable, "label": args.label,
        "dataset_root": str(args.dataset_root.resolve()) if args.dataset_root else None,
        "root_exists": args.dataset_root.is_dir() if args.dataset_root else None,
        "scope": [{"problem": p, "size": n} for p, n in SCOPE],
        "datasets": rows, "coverage": coverage,
        "complete": all(row["status"] == "VERIFIED" for row in coverage),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
