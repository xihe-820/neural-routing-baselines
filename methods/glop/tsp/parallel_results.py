"""Strict result contract for the GLOP TSP Parallel Table evaluator."""
from __future__ import annotations

import json
import math
from pathlib import Path

from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from methods.glop.paper_protocol import formal_protocol
from methods.glop.paper_results import utc_now, write_json


SCHEMA_VERSION = "glop-tsp-parallel-results-v1"
ALLOWED_BATCH_SIZES = (16, 128)
EXPECTED_INSTANCE_COUNTS = {100: 1280, 500: 128, 1000: 128}
VALIDATED_RECORDS_FILE = "validated_records.jsonl"
BATCH_TIMINGS_FILE = "batch_timings.jsonl"
METADATA_FILE = "metadata.json"
SUMMARY_FILE = "summary.json"
TIMING_SEMANTICS = (
    "RTX4090 CUDA-synchronized wall-clock seconds for native vectorized original-"
    "instance batches; total includes one shared RI-order generation cost and every "
    "formal batch call; each batch call includes random insertion, candidate "
    "construction, top-level transforms, all official reconnect revisions, local "
    "augmentation/pruning/final selection, selected-tour D2H transfer and exact node "
    "decode; excludes checkpoint/model and dataset loading, warm-up, provenance, "
    "independent/ML4CO-Kit validation, artifact I/O and summary aggregation"
)


def parallel_scope(problem_size, protocol_name, batch_size):
    """Return the unchanged frozen BS1 protocol plus the only parallel override."""
    size = int(problem_size)
    batch = int(batch_size)
    if size not in EXPECTED_INSTANCE_COUNTS:
        raise ValueError("parallel GLOP scope is TSP100/500/1000 only")
    if batch not in ALLOWED_BATCH_SIZES:
        raise ValueError("parallel GLOP batch size must be 16 or 128")
    protocol = formal_protocol("TSP", size, protocol_name)
    if protocol_name not in ("official_standard", "official_more"):
        raise ValueError("parallel GLOP protocol must be official_standard or official_more")
    if protocol["original_batch_size"] != 1:
        raise ValueError("frozen GLOP paper protocol no longer has BS1 semantics")
    count = EXPECTED_INSTANCE_COUNTS[size]
    if count % batch:
        raise ValueError("formal dataset count is not divisible by batch size")
    return {
        "paper_protocol": protocol,
        "parallel_original_instance_batch_size": batch,
        "instance_count": count,
        "batch_count": count // batch,
    }


def exact_batch_ranges(instance_count, batch_size):
    count = int(instance_count)
    batch = int(batch_size)
    if count <= 0 or batch not in ALLOWED_BATCH_SIZES or count % batch:
        raise ValueError("parallel fullset requires nonempty exact native batches")
    return [(start, start + batch) for start in range(0, count, batch)]


def _finite_number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def write_jsonl(path, rows):
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    temporary.replace(target)


def summarize_parallel(records, batch_timings, *, scope, shared_setup_seconds):
    """Fail closed over coverage and compute Parallel Table values."""
    count = scope["instance_count"]
    batch_size = scope["parallel_original_instance_batch_size"]
    ranges = exact_batch_ranges(count, batch_size)
    if len(records) != count:
        raise ValueError("parallel records do not cover the formal fullset")
    indices = [row.get("dataset_instance_index") for row in records]
    if indices != list(range(count)):
        raise ValueError("parallel records have missing, duplicate, or reordered indices")
    for expected_index, row in enumerate(records):
        if (row.get("evidence_status") != "KIT_VALIDATED" or
                row.get("independent_feasible") is not True or
                row.get("reported_objective_agrees") is not True or
                row.get("kit_feasible") is not True or
                row.get("kit_objective_agrees") is not True):
            raise ValueError("parallel record is not independently and Kit validated")
        if (row.get("batch_index") != expected_index // batch_size or
                not isinstance(row.get("canonical_solution"), list) or
                not row["canonical_solution"]):
            raise ValueError("parallel record has invalid batch or solution provenance")
        independent = _finite_number(
            row.get("independent_objective"), "independent_objective")
        reported = _finite_number(
            row.get("reported_objective"), "reported_objective")
        kit = _finite_number(row.get("kit_objective"), "kit_objective")
        reference = _finite_number(
            row.get("reference_objective"), "reference_objective")
        if reference <= 0:
            raise ValueError("reference_objective must be positive")
        gap = _finite_number(row.get("gap_percent"), "gap_percent")
        expected_gap = (independent - reference) / reference * 100.0
        if (not objective_agrees(reported, independent) or
                not objective_agrees(kit, independent) or
                not math.isclose(gap, expected_gap,
                                 rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("parallel record objective or per-instance gap mismatch")
    if len(batch_timings) != len(ranges):
        raise ValueError("parallel timing count differs from exact batch count")
    solve_total = 0.0
    shared = _finite_number(
        shared_setup_seconds, "shared_ri_order_generation_seconds",
        nonnegative=True)
    for batch_index, ((start, stop), row) in enumerate(zip(ranges, batch_timings)):
        if (row.get("batch_index") != batch_index or
                row.get("dataset_index_start") != start or
                row.get("dataset_index_stop_exclusive") != stop or
                row.get("original_instance_count") != batch_size):
            raise ValueError("parallel timing row does not describe its exact native batch")
        solve_total += _finite_number(
            row.get("solver_runtime_seconds"), "solver_runtime_seconds",
            nonnegative=True)
        charged = _finite_number(
            row.get("shared_ri_order_generation_seconds_charged"),
            "shared_ri_order_generation_seconds_charged", nonnegative=True)
        expected_charge = shared if batch_index == 0 else 0.0
        if not math.isclose(charged, expected_charge,
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("shared RI setup must be charged exactly once to batch zero")
    total = solve_total + shared
    objective = sum(_finite_number(
        row["independent_objective"], "independent_objective")
                    for row in records) / count
    drop = sum(_finite_number(row["gap_percent"], "gap_percent")
               for row in records) / count
    return {
        "status": "PAPER_READY",
        "problem": "TSP",
        "problem_size": scope["paper_protocol"]["problem_size"],
        "protocol": scope["paper_protocol"]["official_protocol_name"],
        "batch_size": batch_size,
        "instance_count": count,
        "batch_count": len(ranges),
        "obj_mean_independent_objective": objective,
        "drop_mean_per_instance_gap_percent": drop,
        "total_runtime_seconds": total,
        "time_mean_batch_seconds": total / len(ranges),
        "solver_batch_runtime_sum_seconds": solve_total,
        "shared_ri_order_generation_seconds": shared,
        "drop_definition": "mean_i((objective_i-reference_i)/reference_i*100)",
        "timing_semantics": TIMING_SEMANTICS,
    }


def finalize_parallel(output_dir, *, metadata, records, batch_timings,
                      shared_setup_seconds):
    output = Path(output_dir)
    scope = metadata["parallel_scope"]
    summary = summarize_parallel(
        records, batch_timings, scope=scope,
        shared_setup_seconds=shared_setup_seconds)
    records_path = output / VALIDATED_RECORDS_FILE
    timings_path = output / BATCH_TIMINGS_FILE
    write_jsonl(records_path, records)
    write_jsonl(timings_path, batch_timings)
    summary.update({
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "GLOP TSP Parallel Table full-set result",
        "project_commit": metadata["project"]["commit"],
        "official_glop_commit": metadata["upstream"]["commit"],
        "dataset": metadata["dataset"],
        "revisers": metadata["assets"]["revisers"],
        "gpu": metadata["environment"]["gpu"],
        "validated_records_sha256": sha256_file(records_path),
        "batch_timings_sha256": sha256_file(timings_path),
        "created_at": utc_now(),
    })
    write_json(output / SUMMARY_FILE, summary)
    metadata.update(
        state="PAPER_READY", completed_records=len(records),
        completed_batches=len(batch_timings),
        validated_records_sha256=summary["validated_records_sha256"],
        batch_timings_sha256=summary["batch_timings_sha256"],
        summary_sha256=sha256_file(output / SUMMARY_FILE),
        completed_at=utc_now())
    write_json(output / METADATA_FILE, metadata)
    return summary
