"""Verify each measurement and export measured rows; never extrapolate."""
import argparse
import csv
import json
from pathlib import Path

from common.hashing import sha256_file
from methods.glop.paper_results import fingerprint, write_json
from rebuttal.tsp2k.artifacts import read_artifact
from rebuttal.tsp2k.config import SWEEPS


def aggregate(directories, *, require_fullset=False):
    points, seen, datasets, checkpoints, phases = [], set(), set(), {}, set()
    timing_samples, implementations, references, quality_indices = set(), {}, set(), set()
    for directory in directories:
        identity, records, summary = read_artifact(directory)
        if identity.get("evidence_kind") != "SERVER_MEASUREMENT":
            raise ValueError("synthetic/test data cannot be used as experimental curves")
        key = (identity["method"], identity["budget_value"])
        if key in seen:
            raise ValueError("duplicate method/budget")
        seen.add(key)
        method = key[0]
        datasets.add(identity["dataset"]["sha256"])
        references.add(identity["dataset"].get("reference_objectives_sha256"))
        quality_indices.add(tuple(identity["dataset_indices"]))
        phases.add(identity["phase"])
        current = identity["checkpoint_sha256"]
        if method in checkpoints and checkpoints[method] != current:
            raise ValueError("checkpoint identity differs across budgets")
        checkpoints[method] = current
        implementation = (identity["project_commit"], identity["upstream_commit"],
                          fingerprint(identity["source_provenance"]))
        if method in implementations and implementations[method] != implementation:
            raise ValueError("solver source/provenance differs across budgets")
        implementations[method] = implementation
        if method == "SIL":
            timing_samples.add(tuple(identity["timing_indices"]))
        if require_fullset and summary["status"] != "MEASURED_FULLSET":
            raise ValueError("calibration cannot stand in for fullset")
        meta_path = Path(directory).resolve() / "metadata.json"
        points.append({
            **summary, "artifact_path": str(Path(directory).resolve()),
            "quality_source": str(meta_path.parent / "inference_records.jsonl") + ":quality",
            "time_source": str(meta_path.parent / "inference_records.jsonl") +
                           (":timing" if method == "SIL" else ":quality"),
            "metadata_sha256": sha256_file(meta_path),
            "project_commit": identity["project_commit"],
            "upstream_commit": identity["upstream_commit"],
            "checkpoint_sha256": current,
        })
    if (not points or len(datasets) != 1 or len(references) != 1 or len(quality_indices) != 1 or
            len(phases) != 1 or len(timing_samples) > 1):
        raise ValueError("mixed/missing dataset, phase or SIL timing indices")
    for method in ("GLOP", "SIL"):
        expected = {(method, value) for value in SWEEPS[method]}
        if not expected.issubset(seen):
            raise ValueError(f"missing prescribed {method} budget")
    if not any(method == "UDC" for method, budget in seen):
        raise ValueError("missing UDC curve")
    if phases == {"calibration"} and not {("UDC", b) for b in SWEEPS["UDC"]}.issubset(seen):
        raise ValueError("calibration requires every initial UDC candidate")
    return {
        "schema": "tsp2k-curves.v1", "evidence_kind": "SERVER_MEASUREMENT",
        "status": "MEASURED_FULLSET" if require_fullset else "MEASURED_TIMING_SUBSET",
        "dataset_sha256": next(iter(datasets)), "phase": next(iter(phases)),
        "points": sorted(points, key=lambda p: (p["method"], p["budget_value"])),
        "runtime_coverage": {method: {
            "minimum_seconds": min(p["mean_time_seconds"] for p in points if p["method"] == method),
            "maximum_seconds": max(p["mean_time_seconds"] for p in points if p["method"] == method),
            "over_30_second_budgets": [p["budget_value"] for p in points
                if p["method"] == method and p["mean_time_seconds"] > 30],
        } for method in SWEEPS},
        "selection_policy": "UDC budgets chosen by user from actual latency only; no quality selection",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--artifact-root", type=Path)
    inputs.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-fullset", action="store_true")
    args = parser.parse_args(argv)
    verification_path = args.output.with_name(args.output.stem + "_verification.json")
    if (args.output.exists() or args.output.with_suffix(".csv").exists() or
            (args.manifest is not None and verification_path.exists())):
        raise ValueError("aggregation output already exists")
    if args.manifest is not None:
        from rebuttal.tsp2k.manifest import verify_manifest
        report, verification = verify_manifest(args.manifest)
        write_json(verification_path, verification)
    else:
        directories = sorted(p.parent for p in args.artifact_root.glob("*/*/metadata.json"))
        report = aggregate(directories, require_fullset=args.require_fullset)
    write_json(args.output, report)
    fields = ["method", "budget_parameter", "budget_value", "mean_objective",
              "mean_drop_percent", "mean_time_seconds", "quality_instance_count",
              "timing_sample_count", "quality_batch_size", "timing_batch_size",
              "status", "reuse_status", "quality_source", "time_source",
              "artifact_path", "metadata_sha256"]
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["points"])
    print(args.output)


if __name__ == "__main__":
    main()
