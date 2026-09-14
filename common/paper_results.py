"""Compact, resumable paper-result artifacts and strict full-set aggregation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from common.hashing import sha256_file


SCHEMA_VERSION = "mvmoe-paper-results-v1"
RECORDS_FILE = "inference_records.jsonl"
VALIDATED_RECORDS_FILE = "validated_records.jsonl"
METADATA_FILE = "metadata.json"
TIMING_SEMANTICS = (
    "single-original-instance wall-clock seconds; starts after input adaptation with a "
    "CUDA synchronization; includes official Aug8/POMO rollout, device-to-host result "
    "transfer, and best-candidate selection; ends after candidate selection; excludes "
    "dataset/checkpoint/model loading, warm-up, provenance, independent/Kit validation, "
    "artifact I/O, and aggregation"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    records = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def _finite(value, field, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{field} must be positive")
    if nonnegative and number < 0:
        raise ValueError(f"{field} must be nonnegative")
    return number


def validate_record(record, *, require_kit=False):
    required = (
        "dataset_instance_index", "instance_id", "canonical_solution",
        "reported_objective", "independent_objective", "reference_objective",
        "gap_percent", "runtime_seconds", "independent_feasible",
        "reported_objective_agrees", "evidence_status",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"paper record missing fields: {missing}")
    index = record["dataset_instance_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("dataset_instance_index must be a nonnegative integer")
    if not isinstance(record["canonical_solution"], list) or not record["canonical_solution"]:
        raise ValueError(f"record {index} has no canonical solution")
    independent = _finite(record["independent_objective"], "independent_objective")
    reference = _finite(record["reference_objective"], "reference_objective", positive=True)
    _finite(record["reported_objective"], "reported_objective")
    gap = _finite(record["gap_percent"], "gap_percent")
    _finite(record["runtime_seconds"], "runtime_seconds", nonnegative=True)
    expected_gap = (independent - reference) / reference * 100.0
    if not math.isclose(gap, expected_gap, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"record {index} gap is not its per-instance percentage gap")
    if record["independent_feasible"] is not True:
        raise ValueError(f"record {index} failed independent feasibility")
    if record["reported_objective_agrees"] is not True:
        raise ValueError(f"record {index} failed reported-objective agreement")
    if record["evidence_status"] != "INDEPENDENT_VERIFIED":
        raise ValueError(f"record {index} is not independently verified")
    if require_kit:
        for field in ("kit_feasible", "kit_objective_agrees", "kit_objective"):
            if field not in record:
                raise ValueError(f"record {index} missing {field}")
        if record["kit_feasible"] is not True:
            raise ValueError(f"record {index} failed Kit feasibility")
        if record["kit_objective_agrees"] is not True:
            raise ValueError(f"record {index} failed Kit objective agreement")
        _finite(record["kit_objective"], "kit_objective")
    return index


def initialize_chunk(output_dir, resume_identity):
    """Create a chunk or validate it before resuming; return completed indices."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / METADATA_FILE
    fingerprint = json_fingerprint(resume_identity)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("existing paper artifact has an unsupported schema")
        if metadata.get("resume_fingerprint") != fingerprint:
            raise ValueError("resume refused: provenance or configuration differs")
    else:
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "MVMoE paper evaluation chunk",
            "resume_identity": resume_identity,
            "resume_fingerprint": fingerprint,
            "records_file": RECORDS_FILE,
            "validated_records_file": VALIDATED_RECORDS_FILE,
            "state": "INFERENCE_IN_PROGRESS",
            "created_at": utc_now(),
            "completed_records": 0,
        }
        write_json(metadata_path, metadata)
    expected = set(resume_identity["chunk"]["expected_indices"])
    records_path = output_dir / RECORDS_FILE
    records = read_jsonl(records_path) if records_path.exists() else []
    completed = set()
    for record in records:
        index = validate_record(record)
        if index not in expected:
            raise ValueError(f"existing record index {index} is outside this chunk")
        if index in completed:
            raise ValueError(f"duplicate existing record index {index}")
        completed.add(index)
    return metadata, completed


def append_record(output_dir, record):
    validate_record(record)
    path = Path(output_dir) / RECORDS_FILE
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                allow_nan=False) + "\n")
        stream.flush()


def finalize_chunk(output_dir):
    output_dir = Path(output_dir)
    metadata_path = output_dir / METADATA_FILE
    metadata = json.loads(metadata_path.read_text())
    records = read_jsonl(output_dir / RECORDS_FILE)
    actual = [validate_record(record) for record in records]
    expected = metadata["resume_identity"]["chunk"]["expected_indices"]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("chunk cannot complete with duplicate, missing, or extra indices")
    prior_state = metadata.get("state")
    metadata.update(
        state="KIT_VALIDATED" if prior_state == "KIT_VALIDATED" else "INFERENCE_COMPLETE",
        completed_records=len(actual),
        inference_records_sha256=sha256_file(output_dir / RECORDS_FILE),
        inference_completed_at=utc_now(),
    )
    write_json(metadata_path, metadata)
    return metadata


def _consistency_identity(metadata):
    identity = metadata["resume_identity"]
    return {
        "method": identity["method"],
        "variant": identity["variant"],
        "problem": identity["problem"],
        "problem_size": identity["problem_size"],
        "paper_protocol": identity["paper_protocol"],
        "project": identity["project"],
        "upstream": identity["upstream"],
        "checkpoint_sha256": identity["checkpoint"]["sha256"],
        "dataset_sha256": identity["dataset"]["sha256"],
        "dataset_count": identity["dataset"]["count"],
        "warmup": identity["warmup"],
        "environment": identity["environment"],
        "source_provenance": identity["source_provenance"],
        "timing_semantics": identity["timing_semantics"],
    }


def summarize_chunks(chunk_dirs):
    """Validate exact full-set coverage and compute paper Obj/Drop/Time."""
    paths = [Path(path) for path in chunk_dirs]
    if not paths:
        raise ValueError("at least one paper chunk is required")
    all_records = []
    baseline = None
    chunk_descriptions = []
    for directory in paths:
        metadata_path = directory / METADATA_FILE
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported paper schema in {directory}")
        if metadata.get("resume_fingerprint") != json_fingerprint(
                metadata.get("resume_identity")):
            raise ValueError(f"paper metadata fingerprint mismatch in {directory}")
        if metadata.get("state") != "KIT_VALIDATED":
            raise ValueError(f"chunk {directory} is not KIT_VALIDATED")
        identity = _consistency_identity(metadata)
        if baseline is None:
            baseline = identity
        elif identity != baseline:
            raise ValueError(f"mixed paper provenance/configuration in {directory}")
        project = identity["project"]
        upstream = identity["upstream"]
        if project.get("dirty") or upstream.get("dirty"):
            raise ValueError("PAPER_READY requires clean project and upstream checkouts")
        records_path = directory / metadata["validated_records_file"]
        if metadata.get("validated_records_sha256") != sha256_file(records_path):
            raise ValueError(f"validated record hash mismatch in {directory}")
        records = read_jsonl(records_path)
        expected_chunk = metadata["resume_identity"]["chunk"]["expected_indices"]
        indices = [validate_record(record, require_kit=True) for record in records]
        if len(indices) != len(set(indices)) or set(indices) != set(expected_chunk):
            raise ValueError(f"chunk {directory} has duplicate, missing, or extra indices")
        all_records.extend(records)
        chunk_descriptions.append({
            "path": str(directory.resolve()),
            "offset": metadata["resume_identity"]["chunk"]["offset"],
            "count": metadata["resume_identity"]["chunk"]["count"],
            "validated_records_sha256": metadata["validated_records_sha256"],
        })
    indices = [record["dataset_instance_index"] for record in all_records]
    expected_count = baseline["dataset_count"]
    from methods.mvmoe.paper_config import paper_inference_config
    expected_protocol = paper_inference_config(
        baseline["problem_size"], problem=baseline["problem"])
    if (baseline["method"] != "MVMoE" or baseline["variant"] != "MVMoE/4E" or
            baseline["paper_protocol"] != expected_protocol):
        raise ValueError("paper chunks do not use the exact formal MVMoE/4E protocol")
    required_count = 10000 if baseline["problem"] == "CVRP" else 1000
    if expected_count != required_count:
        raise ValueError("paper chunks have an incorrect full-dataset count")
    environment = baseline["environment"]
    if ("RTX 4090" not in str(environment.get("gpu")) or
            not str(environment.get("device", "")).startswith("cuda")):
        raise ValueError("paper summary requires the approved RTX 4090 CUDA environment")
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate dataset index across paper chunks")
    expected = set(range(expected_count))
    actual = set(indices)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"full-set coverage failed: missing={missing[:10]} extra={extra[:10]}")
    mean_objective = sum(float(r["independent_objective"]) for r in all_records) / expected_count
    mean_drop = sum(float(r["gap_percent"]) for r in all_records) / expected_count
    mean_time = sum(float(r["runtime_seconds"]) for r in all_records) / expected_count
    for value, field in ((mean_objective, "mean_objective"),
                         (mean_drop, "mean_drop_percent"),
                         (mean_time, "mean_runtime_seconds")):
        _finite(value, field)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "MVMoE paper full-set summary",
        "status": "PAPER_READY",
        "method": baseline["method"],
        "variant": baseline["variant"],
        "problem": baseline["problem"],
        "problem_size": baseline["problem_size"],
        "instance_count": expected_count,
        "obj_mean_independent_objective": mean_objective,
        "drop_mean_per_instance_gap_percent": mean_drop,
        "time_mean_single_instance_seconds": mean_time,
        "drop_definition": "mean_i((independent_objective_i-reference_objective_i)/reference_objective_i*100)",
        "timing_semantics": baseline["timing_semantics"],
        "consistency_identity": baseline,
        "chunks": chunk_descriptions,
        "created_at": utc_now(),
    }
