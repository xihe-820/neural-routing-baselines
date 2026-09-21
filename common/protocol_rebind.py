"""Fail-closed verification and provenance-preserving protocol-label rebinding."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import statistics

from common.hashing import sha256_file
from common.objective_agreement import objective_agrees


LEGACY_SOLVER_COMMIT = "6630301338640e2918b8a62885d8d21a2a6d338f"
GENERATION_MODE = "verified_protocol_rebind_no_solver_execution"


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid or unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> tuple[list[dict], bytes]:
    try:
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid or unreadable JSONL: {path}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"validated records must be nonempty JSON objects: {path}")
    return rows, raw


def _require_sha(value, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} is not a lowercase SHA256")
    return value


def _require_project(value, *, commit: str | None, name: str) -> dict:
    if not isinstance(value, dict) or value.get("dirty") is not False:
        raise ValueError(f"{name} must identify a clean checkout")
    if commit is not None and value.get("commit") != commit:
        raise ValueError(f"{name} commit mismatch")
    if not isinstance(value.get("commit"), str) or not value["commit"]:
        raise ValueError(f"{name} lacks a commit")
    return value


def _require_upstream(value, *, commit: str) -> dict:
    return _require_project(value, commit=commit, name="upstream")


def _require_asset(metadata: dict, summary: dict, name: str,
                   expected_filename: str | None) -> dict:
    asset = metadata.get(name)
    if not isinstance(asset, dict):
        raise ValueError(f"metadata lacks {name} identity")
    digest = _require_sha(asset.get("sha256"), f"{name} SHA256")
    if summary.get(f"{name}_sha256") != digest:
        raise ValueError(f"{name} SHA256 mismatch between metadata and summary")
    path = Path(asset.get("path", ""))
    if expected_filename is not None and path.name != expected_filename:
        raise ValueError(f"{name} filename mismatch")
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{name} bytes are missing or differ from recorded SHA256")
    return {"path": str(path.resolve()), "sha256": digest}


def _require_correct_record(row: dict, *, problem: str, problem_size: int,
                            expected_index: int | None = None) -> None:
    if (row.get("problem") != problem.upper() or
            row.get("problem_size") != int(problem_size)):
        raise ValueError("validation record problem identity mismatch")
    if expected_index is not None and row.get("dataset_instance_index") != expected_index:
        raise ValueError("validation records have missing, duplicate, or reordered indices")
    if (row.get("evidence_status") != "KIT_VALIDATED" or
            row.get("independent_feasible") is not True or
            row.get("kit_feasible") is not True or
            row.get("official_vs_independent", {}).get("pass") is not True or
            row.get("independent_vs_kit", {}).get("pass") is not True):
        raise ValueError("validation record correctness gate failed")
    objective = row.get("independent_objective")
    reference = row.get("reference_objective")
    gap = row.get("gap_percent")
    if not all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in (objective, reference, gap)) or reference <= 0:
        raise ValueError("validation record contains invalid objective values")
    expected_gap = (float(objective) - float(reference)) / float(reference) * 100.0
    if not objective_agrees(float(gap), expected_gap):
        raise ValueError("validation record per-instance gap mismatch")


def expected_batch_sizes(count: int, batch_size: int) -> list[int]:
    if int(count) <= 0 or int(batch_size) <= 0:
        raise ValueError("count and batch size must be positive")
    return [min(batch_size, count - start) for start in range(0, count, batch_size)]


def verify_quality_artifact(
        source_dir: Path, *, method: str, problem: str, problem_size: int,
        protocol_label: str, budget_key: str, budget: int,
        expected_protocol: dict, expected_count: int, expected_batch_size: int,
        expected_project_commit: str | None, upstream_commit: str,
        dataset_filename: str, checkpoint_filename: str) -> dict:
    source_dir = Path(source_dir).resolve()
    metadata_path = source_dir / "metadata.json"
    summary_path = source_dir / "summary.json"
    records_path = source_dir / "validated_records.jsonl"
    for path in (metadata_path, summary_path, records_path):
        if not path.is_file():
            raise ValueError(f"required quality artifact file is missing: {path}")
    metadata, summary = _load_json(metadata_path), _load_json(summary_path)
    records, records_bytes = _load_jsonl(records_path)
    if (metadata.get("state") != "KIT_VALIDATED" or
            metadata.get("artifact_class") != "baseline_result_reproduction" or
            metadata.get("protocol") != expected_protocol or
            metadata.get("official_source_modified") is not False):
        raise ValueError("quality metadata identity or protocol mismatch")
    if (summary.get("status") != "KIT_VALIDATED" or
            summary.get("artifact_class") != "baseline_result_reproduction" or
            summary.get("method") != method or summary.get("problem") != problem.upper() or
            summary.get("problem_size") != int(problem_size) or
            summary.get("protocol") != protocol_label or
            summary.get(budget_key) != int(budget) or
            summary.get("failed_count") != 0 or
            summary.get("paper_result_eligible_for_quality") is not True or
            summary.get("timing_column_eligible") is not False):
        raise ValueError("quality summary identity or eligibility mismatch")
    metadata_project = _require_project(
        metadata.get("project"), commit=expected_project_commit, name="project")
    summary_project = _require_project(
        summary.get("project"), commit=expected_project_commit, name="summary project")
    if metadata_project != summary_project:
        raise ValueError("quality project provenance differs across files")
    metadata_upstream = _require_upstream(metadata.get("upstream"), commit=upstream_commit)
    summary_upstream = _require_upstream(summary.get("upstream"), commit=upstream_commit)
    if metadata_upstream != summary_upstream:
        raise ValueError("quality upstream provenance differs across files")
    dataset = _require_asset(metadata, summary, "dataset", dataset_filename)
    checkpoint = _require_asset(metadata, summary, "checkpoint", checkpoint_filename)
    if metadata["dataset"].get("count") != int(expected_count):
        raise ValueError("quality metadata dataset count mismatch")
    if (len(records) != expected_count or summary.get("validated_count") != expected_count or
            metadata.get("validated_count") != expected_count):
        raise ValueError("quality validated record count mismatch")
    for index, row in enumerate(records):
        _require_correct_record(
            row, problem=problem, problem_size=problem_size, expected_index=index)
    objectives = [float(row["independent_objective"]) for row in records]
    references = [float(row["reference_objective"]) for row in records]
    gaps = [float(row["gap_percent"]) for row in records]
    recomputed = {
        "mean_objective": statistics.fmean(objectives),
        "mean_reference_objective": statistics.fmean(references),
        "mean_instance_gap_percent": statistics.fmean(gaps),
    }
    if any(not objective_agrees(float(summary.get(key, math.nan)), value)
           for key, value in recomputed.items()):
        raise ValueError("quality objective aggregate mismatch")
    sizes = expected_batch_sizes(expected_count, expected_batch_size)
    if (summary.get("batch_size_requested") != expected_batch_size or
            summary.get("original_instance_batch_size") != expected_batch_size or
            summary.get("effective_batch_sizes") != sizes or
            summary.get("number_of_batches") != len(sizes)):
        raise ValueError("quality batch mapping mismatch")
    if not isinstance(summary.get("total_wall_time_seconds"), (int, float)) or not math.isfinite(
            summary["total_wall_time_seconds"]) or summary["total_wall_time_seconds"] <= 0:
        raise ValueError("quality runtime must be finite and positive")
    if (metadata.get("source_files") != summary.get("source_files") or
            not isinstance(metadata.get("source_files"), list) or
            not metadata["source_files"]):
        raise ValueError("quality source provenance mismatch")
    if "RTX 4090" not in str(metadata.get("environment", {}).get("gpu", "")):
        raise ValueError("quality artifact was not produced on the formal RTX 4090")
    return {
        "directory": source_dir, "metadata": metadata, "summary": summary,
        "records": records, "records_bytes": records_bytes,
        "paths": {"metadata": metadata_path, "summary": summary_path,
                  "records": records_path},
        "hashes": {"metadata": sha256_file(metadata_path),
                   "summary": sha256_file(summary_path),
                   "records": sha256_file(records_path)},
        "dataset": dataset, "checkpoint": checkpoint,
    }


def verify_timing_artifact(
        source_path: Path, *, method: str, problem: str, problem_size: int,
        protocol_label: str, budget_key: str, budget: int, expected_count: int,
        expected_project_commit: str | None, upstream_commit: str,
        dataset_sha256: str, checkpoint_sha256: str) -> dict:
    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        raise ValueError(f"required timing artifact is missing: {source_path}")
    summary = _load_json(source_path)
    if (summary.get("status") != "KIT_VALIDATED" or
            summary.get("artifact_class") != "baseline_bs1_timing_probe" or
            summary.get("method") != method or summary.get("problem") != problem.upper() or
            summary.get("problem_size") != int(problem_size) or
            summary.get("protocol") != protocol_label or
            summary.get(budget_key) != int(budget) or
            summary.get("original_instance_batch_size") != 1 or
            summary.get("count") != expected_count or
            summary.get("validated_count") != expected_count or
            summary.get("failed_count") != 0 or
            summary.get("timing_column_eligible") is not True or
            summary.get("paper_result_eligible_for_quality") is not False or
            summary.get("official_source_modified") is not False):
        raise ValueError("timing artifact identity or eligibility mismatch")
    _require_project(summary.get("project"), commit=expected_project_commit, name="timing project")
    _require_upstream(summary.get("upstream"), commit=upstream_commit)
    if (summary.get("dataset_sha256") != dataset_sha256 or
            summary.get("checkpoint_sha256") != checkpoint_sha256):
        raise ValueError("timing dataset/checkpoint SHA256 mismatch")
    records, runtimes = summary.get("validation_records"), summary.get("individual_runtimes")
    if not isinstance(records, list) or not isinstance(runtimes, list) or (
            len(records) != expected_count or len(runtimes) != expected_count):
        raise ValueError("timing records/runtime count mismatch")
    for index, (row, runtime) in enumerate(zip(records, runtimes)):
        _require_correct_record(
            row, problem=problem, problem_size=problem_size, expected_index=index)
        if (not isinstance(runtime, (int, float)) or not math.isfinite(runtime) or
                runtime <= 0 or not objective_agrees(
                    float(row.get("runtime_seconds", math.nan)), float(runtime))):
            raise ValueError("timing runtime is invalid or differs from validation record")
    values = [float(value) for value in runtimes]
    aggregates = {
        "mean_runtime_seconds": statistics.fmean(values),
        "median_runtime_seconds": statistics.median(values),
        "min_runtime_seconds": min(values), "max_runtime_seconds": max(values),
    }
    if any(not objective_agrees(float(summary.get(key, math.nan)), value)
           for key, value in aggregates.items()):
        raise ValueError("timing aggregate mismatch")
    if (not isinstance(summary.get("source_files"), list) or not summary["source_files"] or
            "RTX 4090" not in str(summary.get("environment", {}).get("gpu", ""))):
        raise ValueError("timing source or hardware provenance mismatch")
    return {"path": source_path, "summary": summary,
            "sha256": sha256_file(source_path)}


def _derived_from(quality: dict, timing: dict, *, legacy_solver_commit: str,
                  legacy_protocol_label: str, legacy_budget: int) -> dict:
    return {
        "legacy_project_commit": legacy_solver_commit,
        "legacy_protocol_label": legacy_protocol_label,
        "legacy_budget": int(legacy_budget),
        "source_artifact_path": str(quality["directory"]),
        "source_metadata_sha256": quality["hashes"]["metadata"],
        "source_summary_sha256": quality["hashes"]["summary"],
        "source_records_sha256": quality["hashes"]["records"],
        "source_timing_path": str(timing["path"]),
        "source_timing_sha256": timing["sha256"],
    }


def _solver_execution_batch_identity(quality: dict) -> dict:
    """Retain the verified legacy batch execution without relabelling it."""
    protocol = quality["metadata"]["protocol"]
    summary = quality["summary"]
    return {
        "author_batch_size": protocol.get("author_batch_size"),
        "batch_size_requested": summary.get("batch_size_requested"),
        "original_instance_batch_size": summary.get("original_instance_batch_size"),
        "effective_batch_sizes": deepcopy(summary.get("effective_batch_sizes")),
        "number_of_batches": summary.get("number_of_batches"),
        "batch_override_reason": protocol.get("batch_override_reason"),
    }


def write_rebound_artifacts(
        *, quality: dict, timing: dict, quality_destination: Path,
        timing_destination: Path, current_quality_protocol: dict,
        current_timing_protocol: dict, current_project: dict,
        label_key: str, budget_key: str, rebind_source_files: list[dict],
        preserve_solver_execution_batch_identity: bool = False,
        legacy_solver_commit: str = LEGACY_SOLVER_COMMIT,
        legacy_protocol_label: str = "fewer", legacy_budget: int = 50,
        target_label: str = "more", target_budget: int = 50) -> None:
    quality_destination = Path(quality_destination)
    timing_destination = Path(timing_destination)
    if quality_destination.exists() or timing_destination.exists():
        raise ValueError("rebind destination already exists")
    _require_project(current_project, commit=None, name="protocol rebinding project")
    if current_project["commit"] == legacy_solver_commit:
        raise ValueError("protocol rebind must run from the new reviewed project commit")
    derived = _derived_from(
        quality, timing, legacy_solver_commit=legacy_solver_commit,
        legacy_protocol_label=legacy_protocol_label, legacy_budget=legacy_budget)
    execution_batch = None
    if preserve_solver_execution_batch_identity:
        execution_batch = _solver_execution_batch_identity(quality)
        derived["solver_execution_batch_identity"] = deepcopy(execution_batch)
    metadata = deepcopy(quality["metadata"])
    metadata.update({
        "schema": f"{quality['summary']['method'].lower()}-verified-protocol-rebind.v1",
        "state": "KIT_VALIDATED", "protocol": deepcopy(current_quality_protocol),
        "artifact_generation_mode": GENERATION_MODE,
        "new_protocol_label": target_label, "new_budget": int(target_budget),
        "solver_execution_project_commit": quality["metadata"]["project"]["commit"],
        "protocol_rebinding_project_commit": current_project["commit"],
        "solver_execution_project": deepcopy(quality["metadata"]["project"]),
        "protocol_rebinding_project": deepcopy(current_project),
        "project": deepcopy(current_project), "derived_from": deepcopy(derived),
        "solver_execution_source_files": deepcopy(quality["metadata"]["source_files"]),
        "source_files": deepcopy(rebind_source_files),
        "official_source_modified": False,
    })
    summary = deepcopy(quality["summary"])
    summary.update({
        "schema": f"{quality['summary']['method'].lower()}-verified-protocol-rebind.v1",
        "protocol": target_label, budget_key: int(target_budget),
        "artifact_generation_mode": GENERATION_MODE,
        "new_protocol_label": target_label, "new_budget": int(target_budget),
        "protocol_config": deepcopy(current_quality_protocol),
        "solver_execution_project_commit": quality["summary"]["project"]["commit"],
        "protocol_rebinding_project_commit": current_project["commit"],
        "solver_execution_project": deepcopy(quality["summary"]["project"]),
        "protocol_rebinding_project": deepcopy(current_project),
        "project": deepcopy(current_project), "derived_from": deepcopy(derived),
        "solver_execution_source_files": deepcopy(quality["summary"]["source_files"]),
        "source_files": deepcopy(rebind_source_files),
        "official_source_modified": False,
    })
    if execution_batch is not None:
        metadata["solver_execution_batch_identity"] = deepcopy(execution_batch)
        summary["solver_execution_batch_identity"] = deepcopy(execution_batch)
    timing_summary = deepcopy(timing["summary"])
    timing_summary.update({
        "schema": f"{quality['summary']['method'].lower()}-verified-timing-rebind.v1",
        "protocol": target_label, budget_key: int(target_budget),
        "artifact_generation_mode": GENERATION_MODE,
        "new_protocol_label": target_label, "new_budget": int(target_budget),
        "protocol_config": deepcopy(current_timing_protocol),
        "solver_execution_project_commit": timing["summary"]["project"]["commit"],
        "protocol_rebinding_project_commit": current_project["commit"],
        "solver_execution_project": deepcopy(timing["summary"]["project"]),
        "protocol_rebinding_project": deepcopy(current_project),
        "project": deepcopy(current_project), "derived_from": deepcopy(derived),
        "solver_execution_source_files": deepcopy(timing["summary"]["source_files"]),
        "source_files": deepcopy(rebind_source_files),
        "official_source_modified": False,
    })
    if (metadata["protocol"].get(label_key) != target_label or
            metadata["protocol"].get(budget_key) != int(target_budget)):
        raise ValueError("current rebound protocol does not match target label/budget")
    quality_destination.mkdir(parents=True)
    (quality_destination / "validated_records.jsonl").write_bytes(
        quality["records_bytes"])
    (quality_destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (quality_destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    timing_destination.parent.mkdir(parents=True, exist_ok=True)
    timing_destination.write_text(
        json.dumps(timing_summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
