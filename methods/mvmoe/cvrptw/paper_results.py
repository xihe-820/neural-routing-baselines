"""Strict aggregation for canonical scaled MVMoE CVRPTW paper chunks."""
from __future__ import annotations

import json
from pathlib import Path

from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from common.paper_results import (METADATA_FILE, SCHEMA_VERSION,
                                  _consistency_identity, _finite,
                                  json_fingerprint, read_jsonl, utc_now,
                                  validate_record)
from methods.mvmoe.cvrptw.paper_protocol import scaled_paper_inference_config
from methods.mvmoe.cvrptw.batch_artifacts import (BATCH_TIMINGS_FILE,
                                                  validate_batch_timing)


def validate_scaled_record(record, *, require_kit=False):
    """Apply the common contract plus scaled-to-original conversion gates."""
    index = validate_record(record, require_kit=require_kit)
    if record.get("input_scaling_protocol") != "continuous_official_style":
        raise ValueError(f"scaled paper record {index} has the wrong protocol identity")
    numeric_fields = (
        "scaler", "original_depot_tw_end", "original_coordinate_max",
        "scaled_depot_tw_end", "scaled_coordinate_max",
        "scaled_reported_objective", "scaled_route_objective",
        "scaled_objective_times_s",
    )
    missing = [field for field in numeric_fields if field not in record]
    missing.extend(field for field in (
        "scaled_reported_objective_agrees",
        "scaled_to_original_objective_agrees") if field not in record)
    if missing:
        raise ValueError(f"scaled paper record {index} missing fields: {missing}")
    scaler = _finite(record["scaler"], "scaler", positive=True)
    for field in numeric_fields[1:]:
        _finite(record[field], field)
    if record["scaled_reported_objective_agrees"] is not True:
        raise ValueError(f"record {index} failed scaled reported-objective agreement")
    if record["scaled_to_original_objective_agrees"] is not True:
        raise ValueError(f"record {index} failed scaled-to-original objective agreement")
    conversions = (
        (record["scaled_route_objective"] * scaler,
         record["scaled_objective_times_s"], "scaled objective conversion"),
        (record["scaled_objective_times_s"], record["independent_objective"],
         "scaled-to-original objective"),
        (record["scaled_reported_objective"], record["scaled_route_objective"],
         "scaled reported objective"),
        (record["scaled_reported_objective"] * scaler,
         record["reported_objective"], "reported objective conversion"),
    )
    for actual, expected, field in conversions:
        if not objective_agrees(actual, expected):
            raise ValueError(f"record {index} failed {field}")
    return index


def summarize_scaled_chunks(chunk_dirs):
    """Validate exact scaled CVRPTW full-set coverage and compute paper metrics."""
    paths = [Path(path) for path in chunk_dirs]
    if not paths:
        raise ValueError("at least one paper chunk is required")
    all_records = []
    all_batch_timings = []
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
        if identity["project"].get("dirty") or identity["upstream"].get("dirty"):
            raise ValueError("PAPER_READY requires clean project and upstream checkouts")
        records_path = directory / metadata["validated_records_file"]
        if metadata.get("validated_records_sha256") != sha256_file(records_path):
            raise ValueError(f"validated record hash mismatch in {directory}")
        records = read_jsonl(records_path)
        expected_chunk = metadata["resume_identity"]["chunk"]["expected_indices"]
        indices = [validate_scaled_record(record, require_kit=True)
                   for record in records]
        if len(indices) != len(set(indices)) or set(indices) != set(expected_chunk):
            raise ValueError(f"chunk {directory} has duplicate, missing, or extra indices")
        all_records.extend(records)
        batch_size = identity["paper_protocol"].get("original_batch_size", 1)
        timings_hash = metadata.get("batch_timings_sha256")
        if timings_hash is not None:
            timings_path = directory / metadata.get(
                "batch_timings_file", BATCH_TIMINGS_FILE)
            if timings_hash != sha256_file(timings_path):
                raise ValueError(f"native-batch timing hash mismatch in {directory}")
            timings = read_jsonl(timings_path)
            timing_indices = [validate_batch_timing(row, batch_size=batch_size)
                              for row in timings]
            if timing_indices != list(range(len(expected_chunk) // batch_size)):
                raise ValueError(f"native-batch timing coverage mismatch in {directory}")
            expected_groups = [
                expected_chunk[start:start + batch_size]
                for start in range(0, len(expected_chunk), batch_size)]
            if [row["dataset_indices"] for row in timings] != expected_groups:
                raise ValueError(f"native-batch timing indices mismatch in {directory}")
            all_batch_timings.extend(timings)
        elif batch_size == 1:
            all_batch_timings.extend({
                "batch_index": index,
                "dataset_indices": [record["dataset_instance_index"]],
                "batch_size": 1,
                "runtime_seconds": record["runtime_seconds"],
            } for index, record in enumerate(records))
        else:
            raise ValueError(f"BS{batch_size} chunk has no native-batch timings")
        chunk_descriptions.append({
            "path": str(directory.resolve()),
            "offset": metadata["resume_identity"]["chunk"]["offset"],
            "count": metadata["resume_identity"]["chunk"]["count"],
            "validated_records_sha256": metadata["validated_records_sha256"],
            "batch_timings_sha256": timings_hash,
        })

    if (baseline["method"] != "MVMoE" or baseline["variant"] != "MVMoE/4E" or
            baseline["problem"] != "CVRPTW" or
            baseline["problem_size"] not in (50, 100) or
            baseline["paper_protocol"] != scaled_paper_inference_config(
                baseline["problem_size"],
                baseline["paper_protocol"].get("original_batch_size", 1))):
        raise ValueError("paper chunks do not use the exact scaled MVMoE CVRPTW protocol")
    expected_count = baseline["dataset_count"]
    if expected_count != 1000:
        raise ValueError("scaled CVRPTW chunks have an incorrect full-dataset count")
    environment = baseline["environment"]
    if ("RTX 4090" not in str(environment.get("gpu")) or
            not str(environment.get("device", "")).startswith("cuda")):
        raise ValueError("paper summary requires the approved RTX 4090 CUDA environment")
    indices = [record["dataset_instance_index"] for record in all_records]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate dataset index across paper chunks")
    actual = set(indices)
    expected = set(range(expected_count))
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"full-set coverage failed: missing={missing[:10]} extra={extra[:10]}")

    mean_objective = sum(float(r["independent_objective"])
                         for r in all_records) / expected_count
    mean_drop = sum(float(r["gap_percent"]) for r in all_records) / expected_count
    batch_size = baseline["paper_protocol"].get("original_batch_size", 1)
    expected_batches = expected_count // batch_size
    if len(all_batch_timings) != expected_batches:
        raise ValueError("full-set native-batch timing count mismatch")
    total_time = sum(float(row["runtime_seconds"]) for row in all_batch_timings)
    mean_time = total_time / expected_batches
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
        "batch_size": batch_size,
        "num_batches": expected_batches,
        "obj_mean_independent_objective": mean_objective,
        "drop_mean_per_instance_gap_percent": mean_drop,
        "time_mean_batch_seconds": mean_time,
        "time_total_seconds": total_time,
        "time_mean_single_instance_seconds": mean_time if batch_size == 1 else None,
        "drop_definition": (
            "mean_i((independent_objective_i-reference_objective_i)"
            "/reference_objective_i*100)"),
        "timing_semantics": baseline["timing_semantics"],
        "consistency_identity": baseline,
        "chunks": chunk_descriptions,
        "created_at": utc_now(),
    }
