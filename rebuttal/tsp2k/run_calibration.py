"""Serial subprocess driver; fullset requires an actual reviewed calibration report."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from common.hashing import sha256_file
from rebuttal.tsp2k.aggregate import aggregate
from rebuttal.tsp2k.config import SWEEPS


def commands(lock_path, output_root, stage, *, udc_budgets=None):
    lock = json.loads(Path(lock_path).read_text())
    jobs = []
    for method, sweep in SWEEPS.items():
        root = Path(lock["projects"][method]["path"])
        script = {"GLOP": "methods/glop/tsp/rebuttal_eval.py",
                  "UDC": "methods/udc/rebuttal_eval.py",
                  "SIL": "methods/sil/rebuttal_eval.py"}[method]
        budgets = udc_budgets if method == "UDC" and udc_budgets is not None else sweep
        if len(set(budgets)) != len(budgets) or not budgets or any(b not in sweep for b in budgets):
            raise ValueError("invalid or duplicate UDC selected budgets")
        for budget in budgets:
            command = [sys.executable, "-B", str(root / script),
                       "--budget", str(budget),
                       "--phase", "fullset" if stage == "fullset" else "calibration",
                       "--asset-lock", str(Path(lock_path).resolve()),
                       "--output-dir", str(Path(output_root).resolve() / stage / method / str(budget)),
                       "--timing-count", "3"]
            if stage != "fullset":
                command += ["--count", "2"]
            jobs.append(command)
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=["preflight", "calibration", "fullset"], required=True)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--udc-budgets", type=int, nargs="+")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.stage != "fullset" and args.udc_budgets is not None:
        raise ValueError("initial calibration must retain every specified candidate")
    if args.execute and args.stage in ("calibration", "fullset"):
        preflight_root = args.output_root / "preflight"
        preflight = aggregate(sorted(p.parent for p in preflight_root.glob("*/*/metadata.json")))
        if preflight["phase"] != "calibration":
            raise ValueError("missing measured preflight")
        lock_sha = sha256_file(args.asset_lock)
        for point in preflight["points"]:
            meta = json.loads((Path(point["artifact_path"]) / "metadata.json").read_text())
            if meta["identity"]["asset_lock"]["sha256"] != lock_sha:
                raise ValueError("preflight lock differs from current execution lock")
    if args.execute and args.stage == "fullset":
        if args.calibration_report is None or args.udc_budgets is None:
            raise ValueError("fullset needs a reviewed actual calibration report and explicit UDC budgets")
        report = json.loads(args.calibration_report.read_text())
        actual = aggregate([p["artifact_path"] for p in report["points"]])
        if report != actual or report["phase"] != "calibration":
            raise ValueError("calibration report failed identity checks")
        # Calibration assets and checkout must match the exact fullset lock.
        lock_sha = sha256_file(args.asset_lock)
        for point in report["points"]:
            meta = json.loads((Path(point["artifact_path"]) / "metadata.json").read_text())
            if meta["identity"]["asset_lock"]["sha256"] != lock_sha:
                raise ValueError("calibration lock differs from fullset lock")
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for command in commands(args.asset_lock, args.output_root, args.stage,
                            udc_budgets=args.udc_budgets):
        print(json.dumps(command), flush=True)
        if args.execute:
            # Sequential blocking execution intentionally prevents concurrent solvers.
            subprocess.run(command, check=True, env=env, cwd=args.output_root.parent)


if __name__ == "__main__":
    main()
