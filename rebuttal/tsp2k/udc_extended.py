"""UDC-only serial stages with current-commit, exact-lock and transitive evidence gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from common.hashing import sha256_file
from methods.glop.paper_results import fingerprint, write_json
from rebuttal.tsp2k.artifacts import read_artifact
from rebuttal.tsp2k.audit import project, verify_lock
from rebuttal.tsp2k.config import UDC_SWEEP
from rebuttal.tsp2k.manifest import verify_evidence

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("preflight", "calibration", "fullset")


def checked_lock(path):
    path = Path(path).resolve()
    lock = json.loads(path.read_text())
    if "reference_lock" not in lock or lock.get("gpu_status") != "VERIFIED_RTX4090_INVENTORY":
        raise ValueError("UDC extended stages require a newly audited RTX4090 reference-bound lock")
    verify_lock(lock, "UDC", Path(lock["projects"]["UDC"]["path"]))
    aggregation = lock["projects"]["SIL"]
    if project(Path(aggregation["path"]), "SIL") != aggregation:
        raise ValueError("reviewed aggregation worktree changed after audit")
    return lock


def stage_report(output_root, stage, lock_path):
    if stage not in ("preflight", "calibration"):
        raise ValueError("only preflight/calibration provide stage gate evidence")
    lock = checked_lock(lock_path)
    points, cache = [], {}
    for budget in UDC_SWEEP:
        directory = Path(output_root).resolve() / stage / "UDC" / str(budget)
        identity, records, summary = verify_evidence(
            directory, method="UDC", budget=budget,
            commit=lock["projects"]["UDC"]["commit"],
            repository=lock["projects"]["UDC"]["path"], tasks_cache=cache)
        if (identity.get("experiment_stage") != stage or identity["phase"] != "calibration" or
                identity["dataset_indices"] != [0, 1] or
                identity.get("udc_evidence_root") != str(Path(output_root).resolve()) or
                identity["asset_lock"]["path"] != str(Path(lock_path).resolve()) or
                identity["asset_lock"]["sha256"] != sha256_file(lock_path)):
            raise ValueError("UDC stage/indices/output/lock identity mismatch")
        points.append({**summary, "artifact_path": str(directory),
                       "metadata_sha256": sha256_file(directory / "metadata.json")})
    return {
        "schema": "tsp2k-udc-stage.v1", "stage": stage,
        "status": "MEASURED_TIMING_SUBSET",
        "project_commit": lock["projects"]["UDC"]["commit"],
        "asset_lock": {"path": str(Path(lock_path).resolve()), "sha256": sha256_file(lock_path)},
        "dataset_sha256": lock["dataset"]["sha256"], "dataset_indices": [0, 1],
        "points": points,
        "minimum_seconds": min(p["mean_time_seconds"] for p in points),
        "maximum_seconds": max(p["mean_time_seconds"] for p in points),
        "over_30_second_budgets": [p["budget_value"] for p in points if p["mean_time_seconds"] > 30],
        "selection_policy": "manual, measured latency only; never select by quality",
    }


def stage_gate(stage, output_root, lock_path, calibration_report=None):
    if stage not in STAGES:
        raise ValueError("unsupported UDC stage")
    lock = checked_lock(lock_path)
    result = {"stage": stage, "asset_lock_sha256": sha256_file(lock_path),
              "project_commit": lock["projects"]["UDC"]["commit"]}
    if stage in ("calibration", "fullset"):
        preflight = stage_report(output_root, "preflight", lock_path)
        result["preflight"] = fingerprint(preflight)
    if stage == "fullset":
        if calibration_report is None:
            raise ValueError("fullset requires a complete current calibration report")
        path = Path(calibration_report).resolve()
        supplied = json.loads(path.read_text())
        actual = stage_report(output_root, "calibration", lock_path)
        if supplied != actual:
            raise ValueError("calibration report differs from current 11-budget evidence")
        # Calibration artifacts must themselves bind the exact current preflight.
        for point in actual["points"]:
            meta = json.loads((Path(point["artifact_path"]) / "metadata.json").read_text())
            previous = meta["identity"]["stage_evidence"]
            if previous != {**result, "stage": "calibration"}:
                raise ValueError("calibration was not gated by this exact preflight")
        result["calibration"] = {"path": str(path), "sha256": sha256_file(path),
                                 "verified_report_fingerprint": fingerprint(actual)}
    return result


def commands(lock_path, output_root, stage, budgets=None):
    if stage not in STAGES:
        raise ValueError("unsupported UDC stage")
    if stage != "fullset" and budgets is not None:
        raise ValueError("preflight/calibration must remeasure all 11 budgets")
    selected = list(UDC_SWEEP) if budgets is None else budgets
    if (not selected or len(set(selected)) != len(selected) or
            any(type(b) is not int or b not in UDC_SWEEP for b in selected)):
        raise ValueError("invalid/duplicate UDC selection")
    lock = json.loads(Path(lock_path).read_text())
    root = Path(lock["projects"]["UDC"]["path"])
    result = []
    for budget in selected:
        result.append([sys.executable, "-B", str(root / "methods/udc/rebuttal_eval.py"),
            "--budget", str(budget), "--phase", "fullset" if stage == "fullset" else "calibration",
            "--experiment-stage", stage, "--evidence-root", str(Path(output_root).resolve()),
            "--asset-lock", str(Path(lock_path).resolve()), "--output-dir",
            str(Path(output_root).resolve() / stage / "UDC" / str(budget)),
            "--count", "64" if stage == "fullset" else "2"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report-stage", choices=["preflight", "calibration"])
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args(argv)
    if args.report_stage is not None:
        if args.stage or args.execute or args.budgets or args.report_output is None:
            parser.error("report mode requires only --report-stage and --report-output")
        if args.report_output.exists():
            raise ValueError("stage report already exists")
        write_json(args.report_output, stage_report(args.output_root, args.report_stage, args.asset_lock))
        return
    if args.stage is None:
        parser.error("--stage is required")
    if args.stage == "fullset" and (args.budgets is None or args.calibration_report is None):
        raise ValueError("fullset needs explicit latency-reviewed budgets and calibration report")
    jobs = commands(args.asset_lock, args.output_root, args.stage, args.budgets)
    for command in jobs:
        if args.calibration_report is not None:
            command += ["--calibration-report", str(args.calibration_report.resolve())]
        output = Path(command[command.index("--output-dir") + 1])
        if output.exists():
            raise ValueError(f"existing UDC output must be retained: {output}")
    if args.execute:
        stage_gate(args.stage, args.output_root, args.asset_lock, args.calibration_report)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for command in jobs:
        print(json.dumps(command), flush=True)
        if args.execute:
            # Each child repeats its gates, preserving failures and stopping the sweep.
            subprocess.run(command, check=True, env=env, cwd=args.asset_lock.resolve().parent)


if __name__ == "__main__":
    main()
