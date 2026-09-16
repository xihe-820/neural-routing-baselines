"""Fail-closed artifacts, aggregation, and hybrid summaries for final NeuOpt runs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev

from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from methods.neuopt.cvrp.config import supported_config
from methods.neuopt.cvrp.paper_protocol import (
    UPSTREAM_COMMIT, paper_protocol, protocol_fingerprint,
)


SCHEMA_VERSION = "neuopt-paper-production-v1"
SUMMARY_SCHEMA_VERSION = "neuopt-paper-summary-v1"
HYBRID_SCHEMA_VERSION = "neuopt-paper-hybrid-v1"
METADATA_FILE = "metadata.json"
RECORDS_FILE = "validated_records.jsonl"
BATCH_TIMINGS_FILE = "batch_timings.jsonl"
SUMMARY_FILE = "summary.json"
TIMING_SEMANTICS = (
    "solver-only wall-clock seconds for one original-instance batch; CUDA synchronized "
    "immediately before and after the official record=False rollout; excludes untimed "
    "record=True evidence replay, model/checkpoint loading, dataset parsing, adapter "
    "construction, warm-up, resume RNG replay, decoding, validation, and artifact I/O"
)


def json_fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _finite(value, field, *, nonnegative=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if (not math.isfinite(result) or (nonnegative and result < 0) or
            (positive and result <= 0)):
        raise ValueError(f"{field} has an invalid value")
    return result


def validate_record(record):
    required = (
        "dataset_instance_index", "instance_id", "batch_index",
        "position_in_batch", "canonical_solution", "successor",
        "reported_objective", "timed_official_objective",
        "replay_official_objective", "official_recomputed_objective",
        "independent_objective", "reference_objective", "kit_reference_objective",
        "gap_percent", "independent_feasible", "reported_objective_agrees",
        "kit_feasible", "kit_objective", "kit_objective_agrees",
        "constraint_details", "evidence_status",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"NeuOpt production record missing fields: {missing}")
    index = record["dataset_instance_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("dataset_instance_index must be a nonnegative integer")
    for field in ("batch_index", "position_in_batch"):
        if isinstance(record[field], bool) or not isinstance(record[field], int) or record[field] < 0:
            raise ValueError(f"record {index} has invalid {field}")
    if not isinstance(record["canonical_solution"], list) or not record["canonical_solution"]:
        raise ValueError(f"record {index} has no actual canonical solution")
    independent = _finite(record["independent_objective"], "independent_objective")
    reference = _finite(record["reference_objective"], "reference_objective", positive=True)
    for field in ("reported_objective", "timed_official_objective",
                  "replay_official_objective", "official_recomputed_objective",
                  "kit_objective", "kit_reference_objective"):
        _finite(record[field], field, nonnegative=True)
    if record["timed_official_objective"] != record["replay_official_objective"]:
        raise ValueError(f"record {index} replay objective differs from timed objective")
    if record["reported_objective"] != record["timed_official_objective"]:
        raise ValueError(f"record {index} reported objective is not the timed objective")
    for field in ("reported_objective", "official_recomputed_objective", "kit_objective"):
        if not objective_agrees(record[field], independent):
            raise ValueError(f"record {index} {field} disagrees with independent objective")
    if not objective_agrees(record["kit_reference_objective"], reference):
        raise ValueError(f"record {index} Kit reference disagrees with reference")
    expected_gap = (independent - reference) / reference * 100.0
    if not math.isclose(float(record["gap_percent"]), expected_gap,
                        rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"record {index} gap is not its per-instance gap")
    gates = ("independent_feasible", "reported_objective_agrees",
             "kit_feasible", "kit_objective_agrees")
    if any(record[field] is not True for field in gates):
        raise ValueError(f"record {index} failed a correctness gate")
    if record["evidence_status"] != "KIT_VALIDATED":
        raise ValueError(f"record {index} is not KIT_VALIDATED")
    return index


def validate_batch_timing(timing, *, batch_size=None):
    required = ("batch_index", "dataset_indices", "batch_size", "runtime_seconds",
                "timed_replay")
    missing = [field for field in required if field not in timing]
    if missing:
        raise ValueError(f"NeuOpt batch timing missing fields: {missing}")
    if (isinstance(timing["batch_index"], bool) or
            not isinstance(timing["batch_index"], int) or timing["batch_index"] < 0):
        raise ValueError("batch_index must be a nonnegative integer")
    size = timing["batch_size"]
    if size not in (1, 100) or (batch_size is not None and size != batch_size):
        raise ValueError("batch timing has an invalid or mixed batch size")
    indices = timing["dataset_indices"]
    if (not isinstance(indices, list) or len(indices) != size or
            any(isinstance(i, bool) or not isinstance(i, int) for i in indices) or
            len(indices) != len(set(indices))):
        raise ValueError("batch timing indices do not form one complete original batch")
    _finite(timing["runtime_seconds"], "runtime_seconds", nonnegative=True)
    expected = {
        "timed_rollout_record": False,
        "evidence_replay_record": True,
        "rng_state_restored": True,
        "timed_replay_official_objective_exact": True,
        "timed_replay_rng_after_exact": True,
        "cuda_synchronized_before_timing": True,
        "cuda_synchronized_after_timing": True,
        "evidence_replay_excluded_from_runtime": True,
    }
    if timing["timed_replay"] != expected:
        raise ValueError("batch timing replay/timing provenance mismatch")
    return timing["batch_index"]


def validate_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError("NeuOpt production identity is missing")
    protocol = identity.get("paper_protocol", {})
    expected_protocol = paper_protocol(
        identity.get("problem_size"), T_max=protocol.get("T_max"),
        batch_size=protocol.get("original_batch_size"), d2a=protocol.get("D2A"),
        stall_limit=protocol.get("stall_limit"), k=protocol.get("k"))
    if protocol != expected_protocol:
        raise ValueError("NeuOpt production protocol is not one of the eight frozen settings")
    if identity.get("protocol_fingerprint") != protocol_fingerprint(protocol):
        raise ValueError("NeuOpt production protocol fingerprint mismatch")
    size = identity["problem_size"]
    config = supported_config(size)
    if (identity.get("method") != "NeuOpt" or identity.get("variant") != "NeuOpt-GIRE" or
            identity.get("problem") != "CVRP" or
            identity.get("upstream", {}).get("commit") != UPSTREAM_COMMIT or
            identity.get("upstream", {}).get("dirty") is not False or
            identity.get("project", {}).get("dirty") is not False or
            identity.get("checkpoint", {}).get("sha256") != config["checkpoint_sha256"] or
            identity.get("dataset", {}).get("sha256") != config["dataset_sha256"] or
            identity.get("dataset", {}).get("count") != config["dataset_count"] or
            identity.get("timing_semantics") != TIMING_SEMANTICS):
        raise ValueError("NeuOpt production identity/assets/provenance mismatch")
    batch_size = protocol["original_batch_size"]
    compatibility = identity.get("decoder_compatibility", {})
    if (compatibility.get("original_batch_size") != batch_size or
            compatibility.get("D2A") != 1 or compatibility.get("val_m") != 1 or
            compatibility.get("internal_decoder_batch_size") != batch_size or
            compatibility.get("bs1_shape_shim") is not (batch_size == 1) or
            compatibility.get("official_source_modified") is not False or
            compatibility.get("action_reward_logits_rng_budget_changed") is not False):
        raise ValueError("NeuOpt decoder compatibility provenance mismatch")
    environment = identity.get("environment", {})
    if ("RTX 4090" not in str(environment.get("gpu")) or
            not str(environment.get("device", "")).startswith("cuda")):
        raise ValueError("NeuOpt production requires the approved RTX 4090 CUDA environment")
    chunk = identity.get("chunk", {})
    expected_indices = chunk.get("expected_indices")
    if (not isinstance(expected_indices, list) or not expected_indices or
            expected_indices != list(range(chunk.get("offset", -1),
                                           chunk.get("offset", -1) + chunk.get("count", 0))) or
            len(expected_indices) % batch_size):
        raise ValueError("NeuOpt chunk must contain contiguous complete batches")
    return identity


def _validate_progress(identity, records, timings, *, require_complete=False):
    validate_identity(identity)
    batch_size = identity["paper_protocol"]["original_batch_size"]
    expected = identity["chunk"]["expected_indices"]
    indices = [validate_record(record) for record in records]
    if len(indices) != len(set(indices)) or indices != expected[:len(indices)]:
        raise ValueError("NeuOpt records are not an exact completed prefix")
    timing_indices = []
    for expected_batch, timing in enumerate(timings):
        if validate_batch_timing(timing, batch_size=batch_size) != expected_batch:
            raise ValueError("NeuOpt batch timings are not an ordered batch prefix")
        batch_indices = expected[expected_batch * batch_size:(expected_batch + 1) * batch_size]
        if timing["dataset_indices"] != batch_indices:
            raise ValueError("NeuOpt batch timing index mapping mismatch")
        timing_indices.extend(batch_indices)
    if indices != timing_indices:
        raise ValueError("NeuOpt records and batch timings cover different instances")
    if require_complete and indices != expected:
        raise ValueError("NeuOpt chunk is incomplete")
    return len(timings)


def initialize_or_resume(output_dir, identity):
    output_dir = Path(output_dir)
    validate_identity(identity)
    fingerprint = json_fingerprint(identity)
    metadata_path = output_dir / METADATA_FILE
    records_path = output_dir / RECORDS_FILE
    timings_path = output_dir / BATCH_TIMINGS_FILE
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("schema_version") != SCHEMA_VERSION or
                metadata.get("resume_fingerprint") != fingerprint or
                metadata.get("resume_identity") != identity):
            raise ValueError("resume refused: NeuOpt provenance or protocol differs")
        records = _read_jsonl(records_path)
        timings = _read_jsonl(timings_path)
        if metadata.get("state") == "IN_PROGRESS":
            batch_size = identity["paper_protocol"]["original_batch_size"]
            committed_records = len(timings) * batch_size
            if len(records) == committed_records + batch_size:
                records = records[:committed_records]
                _write_jsonl(records_path, records)
        completed = _validate_progress(
            identity, records, timings,
            require_complete=metadata.get("state") == "KIT_VALIDATED")
        if metadata.get("state") == "KIT_VALIDATED" and (
                metadata.get("completed_batches") != completed or
                metadata.get("completed_records") != len(records)):
            raise ValueError("resume metadata completed-batch count mismatch")
        if metadata.get("state") == "IN_PROGRESS" and (
                metadata.get("completed_batches") != completed or
                metadata.get("completed_records") != len(records)):
            metadata["completed_batches"] = completed
            metadata["completed_records"] = len(records)
            _write_json(metadata_path, metadata)
        return metadata, records, timings
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("resume refused: output directory is nonempty without metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(records_path, [])
    _write_jsonl(timings_path, [])
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "NeuOpt final paper production chunk",
        "state": "IN_PROGRESS", "resume_identity": identity,
        "resume_fingerprint": fingerprint,
        "records_file": RECORDS_FILE, "batch_timings_file": BATCH_TIMINGS_FILE,
        "completed_records": 0, "completed_batches": 0,
    }
    _write_json(metadata_path, metadata)
    return metadata, [], []


def append_batch(output_dir, records, timing):
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / METADATA_FILE).read_text())
    if metadata.get("state") != "IN_PROGRESS":
        raise ValueError("cannot append to a finalized NeuOpt chunk")
    identity = metadata["resume_identity"]
    current_records = _read_jsonl(output_dir / RECORDS_FILE)
    current_timings = _read_jsonl(output_dir / BATCH_TIMINGS_FILE)
    completed = _validate_progress(identity, current_records, current_timings)
    batch_size = identity["paper_protocol"]["original_batch_size"]
    if len(records) != batch_size or validate_batch_timing(timing, batch_size=batch_size) != completed:
        raise ValueError("append does not contain exactly the next complete batch")
    expected = identity["chunk"]["expected_indices"]
    expected_indices = expected[completed * batch_size:(completed + 1) * batch_size]
    if [validate_record(record) for record in records] != expected_indices:
        raise ValueError("appended record indices do not match the next batch")
    if timing["dataset_indices"] != expected_indices:
        raise ValueError("appended timing indices do not match the next batch")
    _write_jsonl(output_dir / RECORDS_FILE, current_records + records)
    _write_jsonl(output_dir / BATCH_TIMINGS_FILE, current_timings + [timing])
    metadata["completed_records"] = len(current_records) + len(records)
    metadata["completed_batches"] = completed + 1
    _write_json(output_dir / METADATA_FILE, metadata)


def _statistics(identity, records, timings):
    _validate_progress(identity, records, timings, require_complete=True)
    objectives = [float(record["independent_objective"]) for record in records]
    gaps = [float(record["gap_percent"]) for record in records]
    runtimes = [float(timing["runtime_seconds"]) for timing in timings]
    total = sum(runtimes)
    batch_size = identity["paper_protocol"]["original_batch_size"]
    return {
        "instance_count": len(records), "num_batches": len(timings),
        "batch_size": batch_size,
        "obj_mean_independent_objective": mean(objectives),
        "drop_mean_per_instance_gap_percent": mean(gaps),
        "drop_definition": (
            "mean_i((independent_objective_i-reference_objective_i)/"
            "reference_objective_i*100)"
        ),
        "total_inference_seconds": total,
        "mean_batch_latency_seconds": mean(runtimes),
        "median_batch_latency_seconds": median(runtimes),
        "min_batch_latency_seconds": min(runtimes),
        "max_batch_latency_seconds": max(runtimes),
        "std_batch_latency_seconds": pstdev(runtimes),
        "throughput_instances_per_second": len(records) / total if total else None,
        "mean_instance_latency_seconds": total / len(records),
        "table_time_seconds": mean(runtimes),
        "table_time_semantics": (
            "mean solver latency per original-instance batch; not divided by batch size"
            if batch_size == 100 else "mean solver latency per instance because BS=1"
        ),
        "feasible_count": len(records), "validation_status": "KIT_VALIDATED",
    }


def finalize_chunk(output_dir):
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / METADATA_FILE).read_text())
    identity = metadata["resume_identity"]
    records = _read_jsonl(output_dir / RECORDS_FILE)
    timings = _read_jsonl(output_dir / BATCH_TIMINGS_FILE)
    statistics = _statistics(identity, records, timings)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "artifact_type": "NeuOpt production chunk summary",
        "status": "KIT_VALIDATED", "fullset": False,
        "resume_identity": identity, **statistics,
    }
    _write_json(output_dir / SUMMARY_FILE, summary)
    metadata.update({
        "state": "KIT_VALIDATED", "completed_records": len(records),
        "completed_batches": len(timings),
        "validated_records_sha256": sha256_file(output_dir / RECORDS_FILE),
        "batch_timings_sha256": sha256_file(output_dir / BATCH_TIMINGS_FILE),
        "summary_sha256": sha256_file(output_dir / SUMMARY_FILE),
    })
    _write_json(output_dir / METADATA_FILE, metadata)
    return summary


def load_chunk(directory):
    directory = Path(directory)
    metadata = json.loads((directory / METADATA_FILE).read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("state") != "KIT_VALIDATED":
        raise ValueError(f"NeuOpt chunk {directory} is not finalized production evidence")
    identity = validate_identity(metadata.get("resume_identity"))
    if metadata.get("resume_fingerprint") != json_fingerprint(identity):
        raise ValueError(f"NeuOpt chunk fingerprint mismatch in {directory}")
    records_path, timings_path = directory / RECORDS_FILE, directory / BATCH_TIMINGS_FILE
    if (metadata.get("validated_records_sha256") != sha256_file(records_path) or
            metadata.get("batch_timings_sha256") != sha256_file(timings_path) or
            metadata.get("summary_sha256") != sha256_file(directory / SUMMARY_FILE)):
        raise ValueError(f"NeuOpt chunk evidence hash mismatch in {directory}")
    records, timings = _read_jsonl(records_path), _read_jsonl(timings_path)
    expected = _statistics(identity, records, timings)
    summary = json.loads((directory / SUMMARY_FILE).read_text())
    if (metadata.get("completed_records") != len(records) or
            metadata.get("completed_batches") != len(timings)):
        raise ValueError(f"NeuOpt chunk completion metadata mismatch in {directory}")
    if (summary.get("schema_version") != SUMMARY_SCHEMA_VERSION or
            summary.get("status") != "KIT_VALIDATED" or
            summary.get("resume_identity") != identity or
            any(summary.get(key) != value for key, value in expected.items())):
        raise ValueError(f"NeuOpt chunk summary mismatch in {directory}")
    return metadata, records, timings


def _consistency_identity(identity):
    return {
        key: identity[key] for key in (
            "method", "variant", "problem", "problem_size", "paper_protocol",
            "protocol_fingerprint", "project", "upstream", "checkpoint", "dataset",
            "environment", "decoder_compatibility", "tensorboard_compatibility",
            "source_provenance", "timing_semantics", "rng_policy",
        )
    }


def aggregate_chunks(chunk_dirs, *, expected_offset, expected_count, scope):
    if scope not in ("fullset", "timing_subset"):
        raise ValueError("summary scope must be fullset or timing_subset")
    baseline = None
    records, timings, sources = [], [], []
    for directory in chunk_dirs:
        metadata, current_records, current_timings = load_chunk(directory)
        identity = metadata["resume_identity"]
        current = _consistency_identity(identity)
        if baseline is None:
            baseline = current
        elif current != baseline:
            raise ValueError("mixed T, D2A, batch size, assets, environment, or provenance")
        records.extend(current_records)
        timings.extend(current_timings)
        sources.append({
            "path": str(Path(directory).resolve()),
            "offset": identity["chunk"]["offset"], "count": identity["chunk"]["count"],
            "records_sha256": metadata["validated_records_sha256"],
            "batch_timings_sha256": metadata["batch_timings_sha256"],
        })
    if baseline is None:
        raise ValueError("at least one NeuOpt chunk is required")
    indices = [validate_record(record) for record in records]
    expected = list(range(expected_offset, expected_offset + expected_count))
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate dataset index across NeuOpt chunks")
    if sorted(indices) != expected:
        raise ValueError("NeuOpt summary has missing or extra dataset indices")
    ordered_records = sorted(records, key=lambda item: item["dataset_instance_index"])
    flattened_timing_indices = [i for timing in timings for i in timing["dataset_indices"]]
    if len(flattened_timing_indices) != len(set(flattened_timing_indices)) or sorted(flattened_timing_indices) != expected:
        raise ValueError("NeuOpt batch timings have missing or duplicate coverage")
    batch_size = baseline["paper_protocol"]["original_batch_size"]
    if expected_count % batch_size:
        raise ValueError("NeuOpt summary coverage contains a partial batch")
    dataset_count = baseline["dataset"]["count"]
    if scope == "fullset" and (expected_offset != 0 or expected_count != dataset_count):
        raise ValueError("NeuOpt fullset summary requires exact 0..9999 coverage")
    if scope == "timing_subset" and (batch_size != 1 or expected_count >= dataset_count):
        raise ValueError("NeuOpt timing subset must be a strict BS1 subset")
    synthetic_identity = dict(baseline)
    synthetic_identity["chunk"] = {
        "offset": expected_offset, "count": expected_count,
        "expected_indices": expected,
    }
    ordered_timings = []
    for batch_index, timing in enumerate(
            sorted(timings, key=lambda item: item["dataset_indices"][0])):
        normalized = dict(timing)
        normalized["batch_index"] = batch_index
        ordered_timings.append(normalized)
    statistics = _statistics(synthetic_identity, ordered_records, ordered_timings)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "artifact_type": "NeuOpt final paper aggregate",
        "status": "PAPER_READY" if scope == "fullset" else "TIMING_SUBSET_VALIDATED",
        "scope": scope, "fullset": scope == "fullset",
        "estimated": scope == "timing_subset",
        "sample_count": expected_count if scope == "timing_subset" else None,
        "sample_indices": expected if scope == "timing_subset" else None,
        "consistency_identity": baseline, "source_chunks": sources, **statistics,
    }


def write_summary(path, summary):
    _write_json(path, summary)
    return Path(path)


def _records_from_summary(summary):
    records = []
    for source in summary["source_chunks"]:
        _, current, _ = load_chunk(source["path"])
        records.extend(current)
    return {record["dataset_instance_index"]: record for record in records}


def _verify_aggregate_summary(summary):
    scope = summary.get("scope")
    count = summary.get("instance_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("hybrid source has an invalid instance count")
    if scope == "fullset":
        offset = 0
    elif scope == "timing_subset":
        indices = summary.get("sample_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError("hybrid timing source has no exact sample indices")
        offset = indices[0]
    else:
        raise ValueError("hybrid source has an invalid aggregation scope")
    expected = aggregate_chunks(
        [source["path"] for source in summary.get("source_chunks", [])],
        expected_offset=offset, expected_count=count, scope=scope)
    if summary != expected:
        raise ValueError("hybrid source summary does not reproduce from its chunks")


def build_hybrid_summary(quality_summary, time_summary, *, hybrid_quality_warning=False):
    if (quality_summary.get("schema_version") != SUMMARY_SCHEMA_VERSION or
            time_summary.get("schema_version") != SUMMARY_SCHEMA_VERSION):
        raise ValueError("hybrid inputs are not NeuOpt production summaries")
    _verify_aggregate_summary(quality_summary)
    _verify_aggregate_summary(time_summary)
    quality_identity = quality_summary["consistency_identity"]
    time_identity = time_summary["consistency_identity"]
    qp, tp = quality_identity["paper_protocol"], time_identity["paper_protocol"]
    if (not quality_summary.get("fullset") or qp["original_batch_size"] != 100 or
            quality_summary.get("instance_count") != quality_identity["dataset"]["count"]):
        raise ValueError("hybrid quality source must be a BS100 fullset")
    if (time_summary.get("scope") != "timing_subset" or
            time_summary.get("estimated") is not True or tp["original_batch_size"] != 1):
        raise ValueError("hybrid time source must be an explicitly estimated BS1 subset")
    comparable_keys = ("problem_size", "T_max", "D2A", "val_m", "stall_limit", "k", "seed")
    if any(qp[key] != tp[key] for key in comparable_keys):
        raise ValueError("hybrid sources use different problem or search protocols")
    for field in ("project", "upstream", "checkpoint", "dataset"):
        if quality_identity[field] != time_identity[field]:
            raise ValueError(f"hybrid sources mix {field} provenance")
    for field in ("environment", "tensorboard_compatibility", "source_provenance",
                  "timing_semantics", "rng_policy"):
        if quality_identity[field] != time_identity[field]:
            raise ValueError(f"hybrid sources mix {field}")
    quality_records = _records_from_summary(quality_summary)
    time_records = _records_from_summary(time_summary)
    indices = time_summary["sample_indices"]
    if sorted(time_records) != indices or any(index not in quality_records for index in indices):
        raise ValueError("hybrid same-subset records are unavailable or mismatched")
    bs1_obj = mean(time_records[i]["independent_objective"] for i in indices)
    bs100_obj = mean(quality_records[i]["independent_objective"] for i in indices)
    bs1_gap = mean(time_records[i]["gap_percent"] for i in indices)
    bs100_gap = mean(quality_records[i]["gap_percent"] for i in indices)
    return {
        "schema_version": HYBRID_SCHEMA_VERSION,
        "artifact_type": "NeuOpt explicit hybrid Complete Results candidate",
        "status": "PENDING_USER_DECISION", "hybrid": True,
        "bs1_fullset_completed": False,
        "quality_source": {
            "batch_size": 100, "num_instances": quality_summary["instance_count"],
            "fullset": True, "summary": quality_summary,
        },
        "time_source": {
            "batch_size": 1, "num_instances": time_summary["instance_count"],
            "estimated": True, "sample_indices": indices, "summary": time_summary,
        },
        "table10_candidate": {
            "Obj": quality_summary["obj_mean_independent_objective"],
            "Drop_percent": quality_summary["drop_mean_per_instance_gap_percent"],
            "Time_seconds": time_summary["mean_batch_latency_seconds"],
            "quality_and_time_from_different_experiments": True,
        },
        "same_subset_comparison": {
            "sample_indices": indices,
            "bs1_mean_objective": bs1_obj, "bs100_mean_objective": bs100_obj,
            "bs1_minus_bs100_mean_objective": bs1_obj - bs100_obj,
            "bs1_mean_gap_percent": bs1_gap, "bs100_mean_gap_percent": bs100_gap,
            "bs1_minus_bs100_mean_gap_percent": bs1_gap - bs100_gap,
            "hybrid_quality_warning": (
                "HYBRID_QUALITY_WARNING" if hybrid_quality_warning else None),
            "warning_status": (
                "USER_FLAGGED_HYBRID_QUALITY_WARNING" if hybrid_quality_warning
                else "PENDING_USER_REVIEW_NO_AUTOMATIC_THRESHOLD"
            ),
        },
    }
