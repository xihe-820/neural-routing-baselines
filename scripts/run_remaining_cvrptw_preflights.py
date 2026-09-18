#!/usr/bin/env python3
"""Run method preflights independently: one failure does not hide the others."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

METHODS = {
    "rfte": ("methods/rfte/cvrptw/paper_eval.py", "routefinder"),
    "cada": ("methods/cada/cvrptw/paper_eval.py", "CaDA"),
    "moses_cada": ("methods/moses_cada/cvrptw/paper_eval.py", "moses_vrp"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--audit-evidence", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rfte-python", type=Path, required=True)
    parser.add_argument("--cada-python", type=Path, required=True)
    parser.add_argument("--moses-cada-python", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS),
                        default=list(METHODS))
    parser.add_argument("--batch-size", type=int, choices=(1, 10), default=1)
    args = parser.parse_args()
    if args.batch_size == 10 and "cada" in args.methods:
        raise ValueError("CaDA is outside this BS10 phase")
    audit = json.loads(args.audit_evidence.read_text())
    status = {}
    for method in args.methods:
        runner, upstream_name = METHODS[method]
        method_audit = audit.get("methods", {}).get(method, {})
        for size in (50, 100):
            key = f"{method}_{size}"
            asset = method_audit.get("checkpoints", {}).get(str(size), {})
            dataset_audit = audit.get("datasets", {}).get(str(size), {})
            if (not method_audit.get("upstream_pass") or
                    not asset.get("identity_pass") or
                    not dataset_audit.get("identity_pass") or
                    audit.get("project", {}).get("dirty") is not False):
                status[key] = {"status": "BLOCKED_AUDIT", "returncode": None}
                continue
            python = getattr(args, f"{method}_python")
            dataset_cfg = dataset_audit["expected"]
            command = [
                str(python), "-B", str(args.project_root / runner),
                "--scope", "preflight", "--problem-size", str(size),
                "--batch-size", str(args.batch_size),
                "--input", str(args.prepared_root / f"cvrptw{size}" /
                               f"preflight_bs{args.batch_size}.npz"),
                "--dataset", str(args.dataset_root / dataset_cfg["filename"]),
                "--upstream", str(args.project_root / "external" / upstream_name),
                "--checkpoint", asset["path"],
                "--expected-checkpoint-sha256", asset["sha256"],
                "--output-dir", str(args.output_root / method / f"cvrptw{size}" /
                                    f"bs{args.batch_size}" / "preflight"),
                "--warmup-batches", "0",
            ]
            proc = subprocess.run(command, cwd=args.project_root, check=False)
            status[key] = {"status": "PASS" if proc.returncode == 0 else "FAIL",
                           "returncode": proc.returncode, "command": command}
    print(json.dumps(status, indent=2))
    if any(row["status"] != "PASS" for row in status.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
