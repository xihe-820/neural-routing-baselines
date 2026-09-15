"""Resumable GLOP chunks and strict formal full-set aggregation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from common.hashing import sha256_file
from methods.glop.paper_protocol import formal_protocol

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "glop-paper-results-v1"
METADATA_FILE = "metadata.json"
RECORDS_FILE = "inference_records.jsonl"
VALIDATED_RECORDS_FILE = "validated_records.jsonl"
TIMING_SEMANTICS = (
    "single-original-instance wall-clock seconds after checkpoint/model/prepared-input "
    "loading; includes official solver preparation, random insertion, partitioning when "
    "applicable, all revisions/augmentation/pruning/selection, final D2H, exact node-ID "
    "decode and CUDA synchronization; excludes warm-up, provenance, independent/Kit "
    "validation, artifact I/O and aggregation"
)
TSP_TIMING_SEMANTICS = (
    "single-original-instance wall-clock seconds after checkpoint/model/prepared-input "
    "loading; includes one shared RI-order generation cost amortized through dataset "
    "index 0, per-instance random insertion, all revisions/augmentation/pruning/selection, "
    "final D2H, exact node-ID decode and CUDA synchronization; excludes warm-up, "
    "provenance, independent/Kit validation, artifact I/O and aggregation"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    output = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return output


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def tsp_runtime_accounting(dataset_index, per_instance_solve_seconds,
                           shared_ri_order_generation_seconds):
    """Charge the one TSP shared-order setup cost to global index zero."""
    solve = _number(per_instance_solve_seconds,
                    "per_instance_solve_seconds", nonnegative=True)
    shared = _number(shared_ri_order_generation_seconds,
                     "shared_ri_order_generation_seconds", nonnegative=True)
    charged_shared = shared if dataset_index == 0 else 0.0
    components = {
        "per_instance_solve_seconds": solve,
        "shared_ri_order_generation_seconds": charged_shared,
    }
    return solve + charged_shared, components


def _validate_tsp_runtime_components(record, *, required):
    components = record.get("runtime_components")
    if components is None:
        if required:
            raise ValueError("TSP record lacks runtime_components")
        return None
    expected = {
        "per_instance_solve_seconds",
        "shared_ri_order_generation_seconds",
    }
    if not isinstance(components, dict) or set(components) != expected:
        raise ValueError("TSP runtime_components fields differ from protocol")
    solve = _number(components["per_instance_solve_seconds"],
                    "per_instance_solve_seconds", nonnegative=True)
    shared = _number(components["shared_ri_order_generation_seconds"],
                     "shared_ri_order_generation_seconds", nonnegative=True)
    runtime = _number(record["runtime_seconds"], "runtime_seconds",
                      nonnegative=True)
    if not math.isclose(runtime, solve + shared,
                        rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("TSP runtime_seconds differs from runtime_components sum")
    return shared


def validate_record(record, *, require_kit=False):
    required = (
        "dataset_instance_index", "instance_id", "canonical_solution",
        "reported_objective", "independent_objective", "reference_objective",
        "gap_percent", "runtime_seconds", "independent_feasible",
        "reported_objective_agrees", "evidence_status",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"GLOP paper record missing fields: {missing}")
    index = record["dataset_instance_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("dataset index must be a nonnegative integer")
    if not isinstance(record["canonical_solution"], list) or not record["canonical_solution"]:
        raise ValueError(f"record {index} has no canonical solution")
    independent = _number(record["independent_objective"], "independent_objective")
    reference = _number(record["reference_objective"], "reference_objective", positive=True)
    _number(record["reported_objective"], "reported_objective")
    runtime = _number(record["runtime_seconds"], "runtime_seconds", nonnegative=True)
    gap = _number(record["gap_percent"], "gap_percent")
    expected_gap = (independent - reference) / reference * 100.0
    if not math.isclose(gap, expected_gap, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"record {index} gap is not the per-instance percentage")
    if record["independent_feasible"] is not True:
        raise ValueError(f"record {index} failed independent feasibility")
    if record["reported_objective_agrees"] is not True:
        raise ValueError(f"record {index} failed objective agreement")
    if record["evidence_status"] != "INDEPENDENT_VERIFIED":
        raise ValueError(f"record {index} is not independently verified")
    if require_kit:
        if record.get("kit_feasible") is not True or record.get("kit_objective_agrees") is not True:
            raise ValueError(f"record {index} failed Kit validation")
        _number(record.get("kit_objective"), "kit_objective")
    return index


def initialize_chunk(output_dir, identity):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / METADATA_FILE
    records_path = output_dir / RECORDS_FILE
    expected_fingerprint = fingerprint(identity)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("schema_version") != SCHEMA_VERSION or
                metadata.get("resume_fingerprint") != expected_fingerprint):
            raise ValueError("resume refused: GLOP provenance or protocol differs")
        finalized = metadata.get("inference_records_sha256")
        if finalized and (not records_path.is_file() or sha256_file(records_path) != finalized):
            raise ValueError("resume refused: finalized inference records changed")
        validated = metadata.get("validated_records_sha256")
        validated_path = output_dir / VALIDATED_RECORDS_FILE
        if validated and (not validated_path.is_file() or
                          sha256_file(validated_path) != validated):
            raise ValueError("resume refused: validated GLOP records changed")
    else:
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "GLOP formal paper evaluation chunk",
            "resume_identity": identity,
            "resume_fingerprint": expected_fingerprint,
            "records_file": RECORDS_FILE,
            "validated_records_file": VALIDATED_RECORDS_FILE,
            "state": "INFERENCE_IN_PROGRESS", "created_at": utc_now(),
        }
        write_json(metadata_path, metadata)
    expected = set(identity["chunk"]["expected_indices"])
    completed = set()
    if records_path.exists():
        for record in read_jsonl(records_path):
            index = validate_record(record)
            if index not in expected or index in completed:
                raise ValueError("existing GLOP record has extra or duplicate index")
            completed.add(index)
    return metadata, completed


def append_record(output_dir, record):
    validate_record(record)
    with (Path(output_dir) / RECORDS_FILE).open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                allow_nan=False) + "\n")
        stream.flush()


def finalize_chunk(output_dir):
    output_dir = Path(output_dir)
    metadata_path = output_dir / METADATA_FILE
    metadata = json.loads(metadata_path.read_text())
    records = read_jsonl(output_dir / RECORDS_FILE)
    indices = [validate_record(record) for record in records]
    expected = metadata["resume_identity"]["chunk"]["expected_indices"]
    if len(indices) != len(set(indices)) or set(indices) != set(expected):
        raise ValueError("GLOP chunk has duplicate, missing, or extra indices")
    prior_state = metadata.get("state")
    metadata.update(
        state="KIT_VALIDATED" if prior_state == "KIT_VALIDATED"
        else "INFERENCE_COMPLETE", completed_records=len(indices),
        inference_records_sha256=sha256_file(output_dir / RECORDS_FILE),
        inference_completed_at=utc_now())
    write_json(metadata_path, metadata)
    return metadata


def _consistent_identity(metadata):
    identity = metadata["resume_identity"]
    return {key: identity[key] for key in (
        "method", "variant", "problem", "problem_size",
        "official_protocol_name", "paper_protocol", "project", "upstream",
        "assets", "dataset", "warmup", "rng", "environment",
        "source_provenance", "timing_semantics")}


def _verify_current_file_hashes(identity):
    dataset = identity["dataset"]
    if sha256_file(dataset["path"]) != dataset["sha256"]:
        raise ValueError("GLOP dataset bytes changed after inference")
    assets = identity["assets"]
    partitioner = assets.get("partitioner")
    if partitioner and sha256_file(partitioner["path"]) != partitioner["sha256"]:
        raise ValueError("GLOP partitioner bytes changed after inference")
    for reviser in assets["revisers"]:
        if sha256_file(reviser["checkpoint_path"]) != reviser["checkpoint_sha256"]:
            raise ValueError("GLOP reviser bytes changed after inference")
        if sha256_file(reviser["args_path"]) != reviser["args_sha256"]:
            raise ValueError("GLOP reviser args bytes changed after inference")
    for source in identity["source_provenance"]:
        path = ROOT / source["path"]
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"GLOP source bytes changed after inference: {path}")


def summarize_chunks(chunk_dirs):
    paths = [Path(path) for path in chunk_dirs]
    if not paths:
        raise ValueError("at least one GLOP chunk is required")
    baseline = None
    records = []
    chunks = []
    for directory in paths:
        metadata = json.loads((directory / METADATA_FILE).read_text())
        if (metadata.get("schema_version") != SCHEMA_VERSION or
                metadata.get("resume_fingerprint") != fingerprint(
                    metadata.get("resume_identity"))):
            raise ValueError(f"invalid GLOP metadata in {directory}")
        if metadata.get("state") != "KIT_VALIDATED":
            raise ValueError(f"GLOP chunk {directory} is not KIT_VALIDATED")
        raw_identity = metadata["resume_identity"]
        prepared = raw_identity["prepared_input"]
        if sha256_file(prepared["path"]) != prepared["sha256"]:
            raise ValueError(f"prepared GLOP input hash mismatch in {directory}")
        if (sha256_file(prepared["metadata_path"]) !=
                prepared["metadata_sha256"]):
            raise ValueError(
                f"prepared GLOP input metadata hash mismatch in {directory}")
        current = _consistent_identity(metadata)
        if baseline is None:
            baseline = current
        elif current != baseline:
            raise ValueError(f"mixed GLOP protocol/provenance in {directory}")
        path = directory / VALIDATED_RECORDS_FILE
        if metadata.get("validated_records_sha256") != sha256_file(path):
            raise ValueError(f"validated GLOP record hash mismatch in {directory}")
        current_records = read_jsonl(path)
        indices = [validate_record(record, require_kit=True)
                   for record in current_records]
        expected = metadata["resume_identity"]["chunk"]["expected_indices"]
        if len(indices) != len(set(indices)) or set(indices) != set(expected):
            raise ValueError(f"GLOP chunk coverage mismatch in {directory}")
        records.extend(current_records)
        chunks.append({"path": str(directory.resolve()),
                       "offset": metadata["resume_identity"]["chunk"]["offset"],
                       "count": metadata["resume_identity"]["chunk"]["count"],
                       "validated_records_sha256": metadata["validated_records_sha256"]})

    expected_protocol = formal_protocol(
        baseline["problem"], baseline["problem_size"],
        baseline["official_protocol_name"])
    if baseline["problem"] == "CVRP" and len(paths) != 1:
        raise ValueError(
            "formal CVRP aggregation requires one sequential offset-zero chunk")
    if (baseline["method"] != "GLOP" or baseline["variant"] !=
            baseline["official_protocol_name"] or
            baseline["paper_protocol"] != expected_protocol):
        raise ValueError("chunks do not use an exact enabled GLOP protocol")
    expected_rng = expected_protocol["rng_semantics"]
    if any(baseline["rng"].get(key) != value
           for key, value in expected_rng.items()):
        raise ValueError("chunks do not use the exact formal GLOP RNG semantics")
    if (baseline["problem"] == "TSP" and
            len(baseline["rng"].get("shared_ri_orders_fingerprint", "")) != 64):
        raise ValueError("TSP chunks lack shared RI-order identity")
    _verify_current_file_hashes(baseline)
    if baseline["project"].get("dirty") or baseline["upstream"].get("dirty"):
        raise ValueError("GLOP paper evidence requires clean repositories")
    if ("RTX 4090" not in str(baseline["environment"].get("gpu")) or
            not str(baseline["environment"].get("device", "")).startswith("cuda")):
        raise ValueError("GLOP paper results require approved RTX 4090 CUDA")
    count = baseline["dataset"]["count"]
    indices = [record["dataset_instance_index"] for record in records]
    if len(indices) != len(set(indices)) or set(indices) != set(range(count)):
        raise ValueError("GLOP full-set coverage is incomplete or duplicated")
    if baseline["problem"] == "TSP":
        if 0 not in indices:
            raise ValueError("TSP full-set timing requires dataset index 0")
        if baseline["timing_semantics"] != TSP_TIMING_SEMANTICS:
            raise ValueError("TSP timing semantics differ from formal protocol")
        index_zero_charges = 0
        for record in records:
            shared = _validate_tsp_runtime_components(record, required=True)
            if record["dataset_instance_index"] == 0:
                index_zero_charges += 1
            elif shared != 0.0:
                raise ValueError(
                    "TSP shared RI-order setup may only be charged to dataset index 0")
        if index_zero_charges != 1:
            raise ValueError(
                "TSP full-set timing must contain exactly one index-0 shared setup charge")
    objective = sum(float(row["independent_objective"]) for row in records) / count
    drop = sum(float(row["gap_percent"]) for row in records) / count
    runtime = sum(float(row["runtime_seconds"]) for row in records) / count
    for value, name in ((objective, "objective"), (drop, "drop"), (runtime, "runtime")):
        _number(value, name)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "GLOP formal full-set summary",
        "status": "PAPER_READY",
        "method": "GLOP", "variant": baseline["official_protocol_name"],
        "problem": baseline["problem"], "problem_size": baseline["problem_size"],
        "instance_count": count,
        "obj_mean_independent_objective": objective,
        "drop_mean_per_instance_gap_percent": drop,
        "time_mean_single_instance_seconds": runtime,
        "drop_definition": "mean_i((objective_i-reference_i)/reference_i*100)",
        "timing_semantics": baseline["timing_semantics"],
        "consistency_identity": baseline, "chunks": chunks,
        "created_at": utc_now(),
    }
