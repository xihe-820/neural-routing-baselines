"""Small, strict JSON result contract for per-instance inference evidence."""
from __future__ import annotations

import json
import math
from pathlib import Path


RESULT_FIELDS = (
    "method", "variant", "problem", "problem_size", "instance_id",
    "project_repo_commit", "project_repo_dirty", "upstream_url",
    "upstream_commit", "upstream_dirty", "checkpoint_path",
    "checkpoint_sha256", "dataset_path", "dataset_sha256",
    "dataset_instance_index", "adapter_provenance", "inference_config",
    "selection_metadata", "canonical_solution", "reported_objective",
    "independent_objective", "objective_abs_error", "objective_rel_error",
    "reference_objective", "gap_percent", "kit_objective_abs_error",
    "independent_feasible", "constraint_details", "kit_feasible",
    "kit_objective", "runtime_seconds", "environment", "evidence_status",
    "error",
)

REQUIRED_IDENTITY_FIELDS = (
    "method", "variant", "problem", "problem_size", "instance_id",
    "project_repo_commit", "project_repo_dirty", "upstream_url",
    "upstream_commit", "upstream_dirty", "checkpoint_path",
    "checkpoint_sha256", "dataset_path", "dataset_sha256",
    "dataset_instance_index", "adapter_provenance",
)


def new_result(**values):
    """Return a complete result row; unexecuted fields remain explicit null/NOT_RUN."""
    unknown = set(values) - set(RESULT_FIELDS)
    if unknown:
        raise ValueError(f"unknown result fields: {sorted(unknown)}")
    row = {field: None for field in RESULT_FIELDS}
    row["evidence_status"] = "NOT_RUN"
    row.update(values)
    validate_result(row)
    return row


def _check_finite(value, path="result"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains NaN or Inf")
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_finite(item, f"{path}[{index}]")


def validate_result(row):
    missing = [field for field in RESULT_FIELDS if field not in row]
    if missing:
        raise ValueError(f"missing result fields: {missing}")
    null_identity = [field for field in REQUIRED_IDENTITY_FIELDS if row[field] is None]
    if null_identity:
        raise ValueError(f"missing identity fields: {null_identity}")
    if row["independent_feasible"] is True:
        if row["independent_objective"] is None or row["constraint_details"] is None:
            raise ValueError("independent_feasible=True requires objective and constraint details")
    if row["kit_feasible"] is True and row["kit_objective"] is None:
        raise ValueError("kit_feasible=True requires kit_objective")
    if row["evidence_status"] == "FAILED" and not row["error"]:
        raise ValueError("FAILED result requires a nonempty error")
    _check_finite(row)
    return row


def write_result_bundle(path, results, *, run_metadata=None):
    for row in results:
        validate_result(row)
    payload = {"schema_version": 1, "run_metadata": run_metadata or {}, "results": results}
    _check_finite(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return destination


def read_result_bundle(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("results"), list):
        raise ValueError("unsupported result bundle")
    for row in payload["results"]:
        validate_result(row)
    return payload
