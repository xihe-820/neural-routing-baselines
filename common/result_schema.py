"""Small, strict JSON result contract for per-instance inference evidence."""
from __future__ import annotations

import json
import math
from pathlib import Path

from common.objective_agreement import objective_agrees


RESULT_FIELDS = (
    "method", "variant", "problem", "problem_size", "instance_id",
    "project_repo_commit", "project_repo_dirty", "upstream_url",
    "upstream_commit", "upstream_dirty", "checkpoint_path",
    "checkpoint_sha256", "dataset_path", "dataset_sha256",
    "dataset_instance_index", "adapter_provenance", "inference_config",
    "selection_metadata", "canonical_solution", "reported_objective",
    "independent_objective", "objective_abs_error", "objective_rel_error",
    "reported_objective_agrees",
    "reference_objective", "gap_percent", "kit_objective_abs_error",
    "kit_objective_agrees",
    "independent_feasible", "constraint_details", "kit_feasible",
    "kit_objective", "runtime_seconds", "runtime_semantics", "environment", "evidence_status",
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
    agreement_specs = (
        ("reported_objective_agrees", "reported_objective"),
        ("kit_objective_agrees", "kit_objective"),
    )
    for flag, compared_objective in agreement_specs:
        if row[flag] is not None:
            if row[compared_objective] is None or row["independent_objective"] is None:
                raise ValueError(f"{flag} requires both objective values")
            actual = objective_agrees(row[compared_objective], row["independent_objective"])
            if row[flag] is not actual:
                raise ValueError(f"{flag} contradicts objective values")
    if row["runtime_seconds"] is not None and not row["runtime_semantics"]:
        raise ValueError("runtime_seconds requires explicit runtime_semantics")
    if row["evidence_status"] == "LOCAL_VERIFIED":
        gates = ("independent_feasible", "kit_feasible",
                 "reported_objective_agrees", "kit_objective_agrees")
        failed = [field for field in gates if row[field] is not True]
        if failed:
            raise ValueError(f"LOCAL_VERIFIED requires all completion gates: {failed}")
    if row["evidence_status"] == "FAILED" and not row["error"]:
        raise ValueError("FAILED result requires a nonempty error")
    _check_finite(row)
    return row


def complete_validation(row):
    """Apply the four-way full-evidence gate after independent and Kit checks."""
    gates = ("independent_feasible", "kit_feasible",
             "reported_objective_agrees", "kit_objective_agrees")
    failed = [field for field in gates if row.get(field) is not True]
    if failed:
        row["evidence_status"] = "FAILED"
        row["error"] = "full validation gate failed: " + ", ".join(failed)
    else:
        row["evidence_status"] = "LOCAL_VERIFIED"
        row["error"] = None
    validate_result(row)
    return row


def make_run_metadata(*, started_at, finished_at, total_runtime_seconds,
                      batch_size, **values):
    """Build explicit rollout timing metadata; timestamps must bracket the run."""
    from datetime import datetime
    start = datetime.fromisoformat(started_at)
    finish = datetime.fromisoformat(finished_at)
    if finish < start:
        raise ValueError("finished_at precedes started_at")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(total_runtime_seconds, (int, float)) or total_runtime_seconds < 0:
        raise ValueError("total_runtime_seconds must be nonnegative")
    metadata = {
        "started_at": started_at, "finished_at": finished_at,
        "total_runtime_seconds": float(total_runtime_seconds),
        "batch_size": batch_size,
        "per_row_runtime_semantics": (
            "amortized_batch_runtime_seconds = total rollout runtime / original batch size; "
            "not single-instance latency and not paper-comparable"
        ),
    }
    metadata.update(values)
    _check_finite(metadata, "run_metadata")
    return metadata


def write_result_bundle(path, results, *, run_metadata=None):
    for row in results:
        validate_result(row)
    payload = {"schema_version": 2, "run_metadata": run_metadata or {}, "results": results}
    _check_finite(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return destination


def read_result_bundle(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 2 or not isinstance(payload.get("results"), list):
        raise ValueError("unsupported result bundle")
    for row in payload["results"]:
        validate_result(row)
    return payload
