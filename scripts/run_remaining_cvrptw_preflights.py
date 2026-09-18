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
    args = parser.parse_args()
    audit = json.loads(args.audit_evidence.read_text())
    status = {}
    for method, (runner, upstream_name) in METHODS.items():
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
                "--input", str(args.prepared_root / f"cvrptw{size}" / "preflight.npz"),
                "--dataset", str(args.dataset_root / dataset_cfg["filename"]),
                "--upstream", str(args.project_root / "external" / upstream_name),
                "--checkpoint", asset["path"],
                "--expected-checkpoint-sha256", asset["sha256"],
                "--output-dir", str(args.output_root / method / f"cvrptw{size}" / "preflight"),
                "--warmup-instances", "0",
            ]
            proc = subprocess.run(command, cwd=args.project_root, check=False)
            status[key] = {"status": "PASS" if proc.returncode == 0 else "FAIL",
                           "returncode": proc.returncode, "command": command}
    print(json.dumps(status, indent=2))
    if any(row["status"] != "PASS" for row in status.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
