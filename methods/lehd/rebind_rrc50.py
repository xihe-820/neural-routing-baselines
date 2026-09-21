#!/usr/bin/env python3
"""Rebind verified legacy LEHD fewer/RRC50 artifacts to current more/RRC50."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.protocol_rebind import (LEGACY_SOLVER_COMMIT,
                                    verify_quality_artifact,
                                    verify_timing_artifact,
                                    write_rebound_artifacts)
from common.provenance import git_provenance, source_provenance
from methods.lehd.config import (AUTHOR_BATCH_REGISTRY, CHECKPOINTS,
                                 DATASET_FILENAMES, FORMAL_SIZES,
                                 UPSTREAM_COMMIT, resolve_author_batch_config,
                                 resolve_config)


LEGACY_RRC50_BATCH_OVERRIDES = {
    ("cvrp", 2000): {"batch_size": 50},
}


def _require_legacy_override_reason(value) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("legacy CVRP2000 batch override reason must be a nonempty exact string")
    normalized = value.casefold()
    if (re.search(r"\bbatch\b|batch[_ -]?size|\bbs\s*=?\s*\d", normalized) is None or
            re.search(r"\boom\b|out[ -]of[ -]memory", normalized) is None):
        raise ValueError(
            "legacy CVRP2000 batch override reason must explicitly identify batch OOM")
    return value


def _legacy_override_reason(source_quality: Path, problem: str, size: int) -> str | None:
    if (problem, size) not in LEGACY_RRC50_BATCH_OVERRIDES:
        return None
    metadata_path = source_quality / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid legacy quality metadata: {metadata_path}") from exc
    protocol = metadata.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("legacy CVRP2000 metadata lacks protocol identity")
    return _require_legacy_override_reason(protocol.get("batch_override_reason"))


def _legacy_quality_protocol(problem: str, size: int,
                             batch_override_reason: str | None = None) -> dict:
    override = LEGACY_RRC50_BATCH_OVERRIDES.get((problem, size))
    if override is None:
        if batch_override_reason is not None:
            raise ValueError("legacy batch override is not allowed for this LEHD cell")
        protocol = resolve_author_batch_config(problem, size, "more")
    else:
        reason = _require_legacy_override_reason(batch_override_reason)
        protocol = resolve_author_batch_config(
            problem, size, "more", batch_size=override["batch_size"],
            batch_override_reason=reason)
    protocol["protocol_label"] = "fewer"
    protocol["budget_mapping_origin"] = "project_protocol_mapping_of_author_reported_budgets"
    return protocol


def _legacy_batch_size(problem: str, size: int) -> int:
    override = LEGACY_RRC50_BATCH_OVERRIDES.get((problem, size))
    return (override["batch_size"] if override is not None else
            AUTHOR_BATCH_REGISTRY[(problem, size)]["batch_size"])


def _cells(problem: str | None, problem_size: int | None) -> list[tuple[str, int]]:
    if (problem is None) != (problem_size is None):
        raise ValueError("--problem and --problem-size must be supplied together")
    if problem is not None:
        key = (problem, int(problem_size))
        if key not in AUTHOR_BATCH_REGISTRY:
            raise ValueError("unsupported LEHD rebind problem/size")
        return [key]
    return [(name, size) for name, sizes in FORMAL_SIZES.items() for size in sizes]


def build_plan(*, legacy_root: Path, output_root: Path, problem: str | None = None,
               problem_size: int | None = None, current_project: dict,
               rebind_source_files: list[dict]) -> list[dict]:
    plan = []
    for name, size in _cells(problem, problem_size):
        source_quality = legacy_root / "author_batch" / f"{name}{size}" / "fewer"
        source_timing = legacy_root / "bs1_timing" / f"{name}{size}" / "fewer.json"
        quality_destination = output_root / "author_batch" / f"{name}{size}" / "more"
        timing_destination = output_root / "bs1_timing" / f"{name}{size}" / "more.json"
        if quality_destination.exists() or timing_destination.exists():
            raise ValueError("LEHD rebind destination already exists")
        legacy_override_reason = _legacy_override_reason(source_quality, name, size)
        current_quality = resolve_author_batch_config(name, size, "more")
        current_timing = resolve_config(name, size, "more")
        quality = verify_quality_artifact(
            source_quality, method="LEHD", problem=name, problem_size=size,
            protocol_label="fewer", budget_key="RRC_budget", budget=50,
            expected_protocol=_legacy_quality_protocol(
                name, size, legacy_override_reason),
            expected_count=AUTHOR_BATCH_REGISTRY[(name, size)]["dataset_count"],
            expected_batch_size=_legacy_batch_size(name, size),
            expected_project_commit=LEGACY_SOLVER_COMMIT,
            upstream_commit=UPSTREAM_COMMIT,
            dataset_filename=DATASET_FILENAMES[(name, size)],
            checkpoint_filename=CHECKPOINTS[name]["filename"])
        timing = verify_timing_artifact(
            source_timing, method="LEHD", problem=name, problem_size=size,
            protocol_label="fewer", budget_key="RRC_budget", budget=50,
            expected_count=3, expected_project_commit=LEGACY_SOLVER_COMMIT,
            upstream_commit=UPSTREAM_COMMIT,
            dataset_sha256=quality["dataset"]["sha256"],
            checkpoint_sha256=quality["checkpoint"]["sha256"])
        plan.append({
            "problem": name, "problem_size": size, "quality": quality,
            "timing": timing, "quality_destination": quality_destination,
            "timing_destination": timing_destination,
            "current_quality_protocol": current_quality,
            "current_timing_protocol": current_timing,
            "current_project": deepcopy(current_project),
            "rebind_source_files": deepcopy(rebind_source_files),
        })
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--problem", choices=["tsp", "cvrp"])
    parser.add_argument("--problem-size", type=int)
    args = parser.parse_args(argv)
    current_project = git_provenance(ROOT)
    if current_project["dirty"]:
        raise ValueError("LEHD rebind requires a clean project checkout")
    sources = source_provenance(
        [Path(__file__), ROOT / "common/protocol_rebind.py",
         ROOT / "methods/lehd/config.py"], root=ROOT)
    plan = build_plan(
        legacy_root=args.legacy_root.resolve(), output_root=args.output_root.resolve(),
        problem=args.problem, problem_size=args.problem_size,
        current_project=current_project, rebind_source_files=sources)
    for cell in plan:
        write_rebound_artifacts(
            quality=cell["quality"], timing=cell["timing"],
            quality_destination=cell["quality_destination"],
            timing_destination=cell["timing_destination"],
            current_quality_protocol=cell["current_quality_protocol"],
            current_timing_protocol=cell["current_timing_protocol"],
            current_project=cell["current_project"],
            label_key="protocol_label", budget_key="RRC_budget",
            rebind_source_files=cell["rebind_source_files"],
            preserve_solver_execution_batch_identity=True)
    print(json.dumps({"status": "PASS", "method": "LEHD", "rebound_cells": [
        f"{cell['problem']}{cell['problem_size']}" for cell in plan]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
