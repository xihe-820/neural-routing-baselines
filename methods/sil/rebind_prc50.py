#!/usr/bin/env python3
"""Historical PRC50 protocol reconstruction; PRC50 is outside the formal mapping."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.protocol_rebind import (LEGACY_SOLVER_COMMIT,
                                    verify_quality_artifact,
                                    verify_timing_artifact)
from methods.sil.config import (AUTHOR_BATCH_REGISTRY, CHECKPOINTS, FORMAL_SIZES,
                                UPSTREAM_COMMIT, resolve_author_batch_config,
                                resolve_config)


def _legacy_quality_protocol(problem: str, size: int) -> dict:
    protocol = resolve_author_batch_config(problem, size, "more")
    protocol["budget_label"] = "fewer"
    protocol["budget"] = 50
    protocol["evaluation_mapping"] = "project mapping to paper-reported PRC50"
    protocol.pop("budget_mapping_origin", None)
    return protocol


def _cells(problem: str | None, problem_size: int | None) -> list[tuple[str, int]]:
    if (problem is None) != (problem_size is None):
        raise ValueError("--problem and --problem-size must be supplied together")
    if problem is not None:
        key = (problem, int(problem_size))
        if key not in AUTHOR_BATCH_REGISTRY:
            raise ValueError("unsupported SIL rebind problem/size")
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
            raise ValueError("SIL rebind destination already exists")
        current_quality = resolve_author_batch_config(name, size, "more")
        current_timing = resolve_config(name, size, "more")
        checkpoint = current_quality["checkpoint"]
        quality = verify_quality_artifact(
            source_quality, method="SIL", problem=name, problem_size=size,
            protocol_label="fewer", budget_key="budget", budget=50,
            expected_protocol=_legacy_quality_protocol(name, size),
            expected_count=AUTHOR_BATCH_REGISTRY[(name, size)]["dataset_count"],
            expected_batch_size=AUTHOR_BATCH_REGISTRY[(name, size)]["batch_size"],
            expected_project_commit=LEGACY_SOLVER_COMMIT,
            upstream_commit=UPSTREAM_COMMIT,
            dataset_filename=current_quality["expected_dataset_filename"],
            checkpoint_filename=checkpoint["filename"])
        timing = verify_timing_artifact(
            source_timing, method="SIL", problem=name, problem_size=size,
            protocol_label="fewer", budget_key="budget", budget=50,
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
    parser.parse_args(argv)
    raise ValueError(
        "PRC50 is legacy higher-budget evidence only and cannot be rebound into "
        "the current formal fewer/more mapping")


if __name__ == "__main__":
    raise SystemExit(main())
