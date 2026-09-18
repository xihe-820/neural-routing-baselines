"""Fail-closed artifacts and paper results for formal NeuOpt TSP100 runs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev

from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from common.provenance import normalize_git_repository_identity
from methods.neuopt.tsp.compat import PINNED_TSP_STEP_SHA256
from methods.neuopt.tsp.config import supported_config
from methods.neuopt.tsp.decode import decode_successor
from methods.neuopt.tsp.paper_protocol import (
    FORMAL_BATCH_SIZES, UPSTREAM_COMMIT, formal_protocol, protocol_fingerprint,
)


SCHEMA_VERSION = "neuopt-tsp100-production-v1"
SUMMARY_SCHEMA_VERSION = "neuopt-tsp100-summary-v1"
PAPER_RESULT_SCHEMA_VERSION = "neuopt-tsp100-paper-result-v1"
PAPER_MATRIX_SCHEMA_VERSION = "neuopt-tsp100-paper-matrix-v1"
METADATA_FILE = "metadata.json"
RECORDS_FILE = "validated_records.jsonl"
BATCH_TIMINGS_FILE = "batch_timings.jsonl"
SUMMARY_FILE = "summary.json"
TIMED_REPLAY = {
    "timed_rollout_record": False,
    "evidence_replay_record": True,
    "rng_state_restored": True,
    "timed_replay_official_objective_exact": True,
    "timed_replay_rng_after_exact": True,
    "cuda_synchronized_before_timing": True,
    "cuda_synchronized_after_timing": True,
    "evidence_replay_excluded_from_runtime": True,
}
TIMING_SEMANTICS = (
    "solver-only wall-clock seconds for one native original-instance batch; CUDA "
    "synchronized immediately before and after the official record=False rollout; "
    "excludes untimed record=True evidence replay, model/checkpoint loading, dataset "
    "parsing, native adapter construction, warm-up, resume RNG replay, decoding, "
    "validation, and artifact I/O; table Time is mean native batch latency and Total "
    "is the sum of native batch runtimes"
)
WARMUP_POLICY = {"batches": 1, "rng_state_restored": True}
RNG_POLICY = (
    "seed once after model setup; warm-up state restored; each timed record=False "
    "batch is replayed record=True from its saved RNG start; resumed completed batch "
    "prefix is replayed record=False to restore the sequential stream"
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
        "independent_reference_objective", "benchmark_reference_agrees",
        "gap_percent", "independent_feasible", "reported_objective_agrees",
        "kit_feasible", "kit_objective", "kit_objective_agrees",
        "constraint_details", "source_coordinate_dtype", "model_input_dtype",
        "model_input_dtype_cast", "evidence_status",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"NeuOpt TSP production record missing fields: {missing}")
    index = record["dataset_instance_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("dataset_instance_index must be a nonnegative integer")
    for field in ("batch_index", "position_in_batch"):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"record {index} has invalid {field}")
    tour, successor = record["canonical_solution"], record["successor"]
    if (not isinstance(tour, list) or len(tour) != 101 or tour[0] != tour[-1] or
            not isinstance(successor, list) or len(successor) != 100):
        raise ValueError(f"record {index} lacks an actual TSP100 solution")
    decoded, _ = decode_successor(successor, problem_size=100)
    if decoded != tour:
        raise ValueError(f"record {index} successor and canonical tour disagree")
    independent = _finite(record["independent_objective"], "independent_objective")
    reference = _finite(record["reference_objective"], "reference_objective", positive=True)
    for field in ("reported_objective", "timed_official_objective",
                  "replay_official_objective", "official_recomputed_objective",
                  "kit_objective", "kit_reference_objective",
                  "independent_reference_objective"):
        _finite(record[field], field, nonnegative=True)
    if record["timed_official_objective"] != record["replay_official_objective"]:
        raise ValueError(f"record {index} replay objective differs from timed objective")
    if record["reported_objective"] != record["timed_official_objective"]:
        raise ValueError(f"record {index} reported objective is not the timed objective")
    for field in ("reported_objective", "official_recomputed_objective", "kit_objective"):
        if not objective_agrees(record[field], independent):
            raise ValueError(f"record {index} {field} disagrees with independent objective")
    if not objective_agrees(record["kit_reference_objective"], reference):
        raise ValueError(f"record {index} Kit reference disagrees with benchmark reference")
    if (not objective_agrees(record["independent_reference_objective"], reference) or
            record["benchmark_reference_agrees"] is not True):
        raise ValueError(f"record {index} benchmark reference failed independent agreement")
    expected_gap = (independent - reference) / reference * 100.0
    if not math.isclose(float(record["gap_percent"]), expected_gap,
                        rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"record {index} gap is not its per-instance percentage gap")
    if any(record[field] is not True for field in (
            "independent_feasible", "reported_objective_agrees",
            "kit_feasible", "kit_objective_agrees")):
        raise ValueError(f"record {index} failed a validation gate")
    if record["evidence_status"] != "KIT_VALIDATED":
        raise ValueError(f"record {index} is not KIT_VALIDATED")
    if (not isinstance(record["source_coordinate_dtype"], str) or
            record["model_input_dtype"] != "float32" or
            record["model_input_dtype_cast"] not in (True, False)):
        raise ValueError(f"record {index} coordinate dtype provenance is invalid")
    return index


def validate_batch_timing(timing, *, batch_size):
    required = ("batch_index", "dataset_indices", "batch_size", "runtime_seconds",
                "timed_replay", "rng_before_sha256", "rng_after_sha256",
                "native_vectorized_batch")
    missing = [field for field in required if field not in timing]
    if missing:
        raise ValueError(f"NeuOpt TSP batch timing missing fields: {missing}")
    batch_index = timing["batch_index"]
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise ValueError("batch_index must be a nonnegative integer")
    if timing["batch_size"] != batch_size or batch_size not in FORMAL_BATCH_SIZES:
        raise ValueError("batch timing has a wrong or mixed native batch size")
    indices = timing["dataset_indices"]
    if (not isinstance(indices, list) or len(indices) != batch_size or
            any(isinstance(i, bool) or not isinstance(i, int) for i in indices) or
            len(indices) != len(set(indices))):
        raise ValueError("batch timing does not cover one complete native batch")
    _finite(timing["runtime_seconds"], "runtime_seconds", nonnegative=True)
    if timing["timed_replay"] != TIMED_REPLAY:
        raise ValueError("batch timing record/replay provenance mismatch")
    for field in ("rng_before_sha256", "rng_after_sha256"):
        value = timing[field]
        if (not isinstance(value, str) or len(value) != 64 or
                any(character not in "0123456789abcdef" for character in value)):
            raise ValueError("batch timing RNG fingerprint is invalid")
    expected_native = {
        "one_official_rollout_call": True,
        "original_instances_in_call": batch_size,
        "bs1_loop_emulation": False,
    }
    if timing["native_vectorized_batch"] != expected_native:
        raise ValueError("batch timing is not a true native vectorized batch")
    return batch_index


def validate_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError("NeuOpt TSP production identity is missing")
    protocol = identity.get("paper_protocol", {})
    expected_protocol = formal_protocol(
        identity.get("problem_size"), budget=protocol.get("budget"),
        batch_size=protocol.get("original_batch_size"), T_max=protocol.get("T_max"),
        d2a=protocol.get("D2A"), stall_limit=protocol.get("stall_limit"),
        k=protocol.get("k"))
    if protocol != expected_protocol:
        raise ValueError("NeuOpt TSP protocol is not an exact frozen formal setting")
    if identity.get("protocol_fingerprint") != protocol_fingerprint(protocol):
        raise ValueError("NeuOpt TSP protocol fingerprint mismatch")
    config = supported_config(identity.get("problem_size"))
    dataset = identity.get("dataset", {})
    checkpoint = identity.get("checkpoint", {})
    try:
        upstream_url = normalize_git_repository_identity(
            identity.get("upstream", {}).get("url", ""))
        project_url = normalize_git_repository_identity(
            identity.get("project", {}).get("url", ""))
    except ValueError as exc:
        raise ValueError("NeuOpt TSP repository URL provenance mismatch") from exc
    checkpoint_path = Path(str(checkpoint.get("path", ""))).resolve()
    upstream_checkout = Path(str(identity.get("upstream_checkout_path", ""))).resolve()
    expected_checkpoint_path = upstream_checkout / config["checkpoint_relative_path"]
    if (identity.get("method") != "NeuOpt" or identity.get("variant") != "NeuOpt-GIRE" or
            identity.get("problem") != "TSP" or
            identity.get("upstream", {}).get("commit") != UPSTREAM_COMMIT or
            identity.get("upstream", {}).get("dirty") is not False or
            identity.get("project", {}).get("dirty") is not False or
            upstream_url != "github.com/yining043/NeuOpt" or
            project_url != "github.com/xihe-820/neural-routing-baselines" or
            checkpoint.get("filename") != "tsp100.pt" or
            checkpoint.get("relative_path") != config["checkpoint_relative_path"] or
            checkpoint_path != expected_checkpoint_path or
            checkpoint.get("sha256") != config["checkpoint_sha256"] or
            dataset.get("filename") != config["dataset_filename"] or
            dataset.get("sha256") != config["dataset_sha256"] or
            dataset.get("count") != config["dataset_count"] or
            identity.get("timing_semantics") != TIMING_SEMANTICS):
        raise ValueError("NeuOpt TSP identity/assets/provenance mismatch")
    batch_size = protocol["original_batch_size"]
    decoder = identity.get("decoder_compatibility", {})
    if (decoder.get("original_batch_size") != batch_size or
            decoder.get("D2A") != 1 or decoder.get("val_m") != 1 or
            decoder.get("internal_decoder_batch_size") != batch_size or
            decoder.get("bs1_shape_shim") is not (batch_size == 1) or
            decoder.get("official_source_modified") is not False or
            decoder.get("action_reward_logits_rng_budget_changed") is not False):
        raise ValueError("NeuOpt TSP decoder compatibility provenance mismatch")
    record = identity.get("record_compatibility", {})
    if (record.get("record_step_shim") is not True or
            record.get("pinned_step_sha256") != PINNED_TSP_STEP_SHA256 or
            record.get("timed_rollout_shim_active") is not False or
            record.get("evidence_replay_shim_active") is not True or
            record.get("official_source_modified") is not False or
            record.get("action_reward_logits_rng_budget_changed") is not False):
        raise ValueError("NeuOpt TSP replay-only compatibility provenance mismatch")
    tensorboard = identity.get("tensorboard_compatibility", {})
    available = tensorboard.get("tensorboard_logger_available")
    shim = tensorboard.get("tensorboard_logger_import_shim")
    if (available not in (True, False) or shim is not (not available) or
            tensorboard.get("official_source_modified") is not False):
        raise ValueError("NeuOpt TSP TensorBoard compatibility provenance mismatch")
    environment = identity.get("environment", {})
    if ("RTX 4090" not in str(environment.get("gpu")) or
            not str(environment.get("device", "")).startswith("cuda")):
        raise ValueError("NeuOpt TSP production requires RTX 4090 CUDA")
    scope = identity.get("scope")
    expected_count = batch_size if scope == "preflight" else config["dataset_count"]
    expected = list(range(expected_count))
    if (scope not in ("preflight", "fullset") or
            identity.get("expected_indices") != expected or expected_count % batch_size or
            identity.get("native_batch_count") != expected_count // batch_size):
        raise ValueError("NeuOpt TSP scope/index/native-batch identity mismatch")
    if identity.get("native_batching") != {
            "one_official_rollout_call_per_batch": True,
            "bs1_loop_emulation": False,
            "partial_batches": False,
    }:
        raise ValueError("NeuOpt TSP native batching provenance mismatch")
    adapter = identity.get("adapter_mapping", {})
    if (adapter.get("problem_size") != 100 or
            adapter.get("source_shape") != [batch_size, 100, 2] or
            not isinstance(adapter.get("source_dtype"), str) or
            adapter.get("model_input_dtype") != "float32" or
            adapter.get("dtype_cast") not in (True, False) or
            adapter.get("coordinate_transformation") !=
            "none; no normalization, regeneration, or node reordering"):
        raise ValueError("NeuOpt TSP adapter/native-input provenance mismatch")
    if (identity.get("warmup") != WARMUP_POLICY or
            identity.get("rng_policy") != RNG_POLICY):
        raise ValueError("NeuOpt TSP warm-up/RNG policy mismatch")
    if identity.get("artifact_class") != "formal_production_separate_from_calibration":
        raise ValueError("NeuOpt TSP production/calibration artifact isolation is missing")
    return identity


def _validate_progress(identity, records, timings, *, require_complete=False):
    validate_identity(identity)
    batch_size = identity["paper_protocol"]["original_batch_size"]
    expected = identity["expected_indices"]
    indices = [validate_record(record) for record in records]
    for position, record in enumerate(records):
        if (record["batch_index"] != position // batch_size or
                record["position_in_batch"] != position % batch_size):
            raise ValueError("NeuOpt TSP record batch/position mapping mismatch")
    if len(indices) != len(set(indices)) or indices != expected[:len(indices)]:
        raise ValueError("NeuOpt TSP records are not the exact ordered dataset prefix")
    timing_indices = []
    prior_rng_after = None
    for expected_batch, timing in enumerate(timings):
        if validate_batch_timing(timing, batch_size=batch_size) != expected_batch:
            raise ValueError("NeuOpt TSP timings are not an ordered batch prefix")
        current = expected[expected_batch * batch_size:(expected_batch + 1) * batch_size]
        if timing["dataset_indices"] != current:
            raise ValueError("NeuOpt TSP timing-to-dataset mapping mismatch")
        if prior_rng_after is not None and timing["rng_before_sha256"] != prior_rng_after:
            raise ValueError("NeuOpt TSP sequential RNG fingerprints are discontinuous")
        prior_rng_after = timing["rng_after_sha256"]
        timing_indices.extend(current)
    if indices != timing_indices:
        raise ValueError("NeuOpt TSP records and timings cover different instances")
    if require_complete and indices != expected:
        raise ValueError("NeuOpt TSP artifact is incomplete")
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
                metadata.get("state") != "IN_PROGRESS" or
                metadata.get("resume_fingerprint") != fingerprint or
                metadata.get("resume_identity") != identity):
            raise ValueError("resume refused: existing output is not the exact IN_PROGRESS run")
        records, timings = _read_jsonl(records_path), _read_jsonl(timings_path)
        batch_size = identity["paper_protocol"]["original_batch_size"]
        committed = len(timings) * batch_size
        if len(records) == committed + batch_size:
            records = records[:committed]
            _write_jsonl(records_path, records)
        completed = _validate_progress(identity, records, timings)
        if (metadata.get("completed_batches") != completed or
                metadata.get("completed_records") != len(records)):
            metadata["completed_batches"] = completed
            metadata["completed_records"] = len(records)
            _write_json(metadata_path, metadata)
        return metadata, records, timings
    if output_dir.exists():
        raise ValueError("new production output directory must not already exist")
    output_dir.mkdir(parents=True)
    _write_jsonl(records_path, [])
    _write_jsonl(timings_path, [])
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "NeuOpt TSP100 formal production run",
        "artifact_class": "formal_production_separate_from_calibration",
        "state": "IN_PROGRESS", "paper_ready": False,
        "resume_identity": identity, "resume_fingerprint": fingerprint,
        "records_file": RECORDS_FILE, "batch_timings_file": BATCH_TIMINGS_FILE,
        "completed_records": 0, "completed_batches": 0,
    }
    _write_json(metadata_path, metadata)
    return metadata, [], []


def append_batch(output_dir, records, timing):
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / METADATA_FILE).read_text())
    if metadata.get("state") != "IN_PROGRESS":
        raise ValueError("cannot append to a non-active NeuOpt TSP run")
    identity = metadata["resume_identity"]
    current_records = _read_jsonl(output_dir / RECORDS_FILE)
    current_timings = _read_jsonl(output_dir / BATCH_TIMINGS_FILE)
    completed = _validate_progress(identity, current_records, current_timings)
    batch_size = identity["paper_protocol"]["original_batch_size"]
    if len(records) != batch_size or validate_batch_timing(
            timing, batch_size=batch_size) != completed:
        raise ValueError("append is not exactly the next complete native batch")
    expected = identity["expected_indices"][completed * batch_size:(completed + 1) * batch_size]
    if [validate_record(record) for record in records] != expected:
        raise ValueError("appended NeuOpt TSP record indices are wrong")
    if timing["dataset_indices"] != expected:
        raise ValueError("appended NeuOpt TSP timing indices are wrong")
    if (current_timings and timing["rng_before_sha256"] !=
            current_timings[-1]["rng_after_sha256"]):
        raise ValueError("appended NeuOpt TSP RNG stream is discontinuous")
    _write_jsonl(output_dir / RECORDS_FILE, current_records + records)
    _write_jsonl(output_dir / BATCH_TIMINGS_FILE, current_timings + [timing])
    metadata["completed_records"] = len(current_records) + batch_size
    metadata["completed_batches"] = completed + 1
    _write_json(output_dir / METADATA_FILE, metadata)


def mark_failed(output_dir, *, failure_type, message):
    output_dir = Path(output_dir)
    metadata_path = output_dir / METADATA_FILE
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("state") != "IN_PROGRESS":
        return
    metadata.update({
        "state": "FAILED", "paper_ready": False,
        "failure": {"type": str(failure_type), "message": str(message)},
    })
    _write_json(metadata_path, metadata)


def _statistics(identity, records, timings):
    _validate_progress(identity, records, timings, require_complete=True)
    objectives = [float(record["independent_objective"]) for record in records]
    gaps = [float(record["gap_percent"]) for record in records]
    runtimes = [float(timing["runtime_seconds"]) for timing in timings]
    total = sum(runtimes)
    batch_size = identity["paper_protocol"]["original_batch_size"]
    return {
        "instance_count": len(records), "num_native_batches": len(timings),
        "batch_size": batch_size,
        "obj_mean_independent_objective": mean(objectives),
        "drop_mean_per_instance_gap_percent": mean(gaps),
        "drop_definition": (
            "mean_i((independent_objective_i-reference_objective_i)/"
            "reference_objective_i*100); not gap of means"
        ),
        "total_native_batch_solver_seconds": total,
        "time_mean_native_batch_latency_seconds": mean(runtimes),
        "median_native_batch_latency_seconds": median(runtimes),
        "min_native_batch_latency_seconds": min(runtimes),
        "max_native_batch_latency_seconds": max(runtimes),
        "std_native_batch_latency_seconds": pstdev(runtimes),
        "throughput_instances_per_second": len(records) / total if total else None,
        "time_table_semantics": (
            "mean per-instance solver latency" if batch_size == 1
            else f"mean native {batch_size}-instance batch latency; not divided by batch size"
        ),
        "all_independently_feasible": True,
        "all_kit_validated": True,
    }


def finalize_run(output_dir):
    output_dir = Path(output_dir)
    metadata_path = output_dir / METADATA_FILE
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("state") != "IN_PROGRESS":
        raise ValueError("only an IN_PROGRESS NeuOpt TSP run can be finalized")
    identity = metadata["resume_identity"]
    records = _read_jsonl(output_dir / RECORDS_FILE)
    timings = _read_jsonl(output_dir / BATCH_TIMINGS_FILE)
    statistics = _statistics(identity, records, timings)
    paper_ready = identity["scope"] == "fullset"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "artifact_type": "NeuOpt TSP100 formal production summary",
        "status": "PAPER_READY" if paper_ready else "PREFLIGHT_VALIDATED",
        "scope": identity["scope"], "paper_ready": paper_ready,
        "resume_identity": identity, **statistics,
    }
    _write_json(output_dir / SUMMARY_FILE, summary)
    metadata.update({
        "state": "KIT_VALIDATED", "paper_ready": paper_ready,
        "completed_records": len(records), "completed_batches": len(timings),
        "validated_records_sha256": sha256_file(output_dir / RECORDS_FILE),
        "batch_timings_sha256": sha256_file(output_dir / BATCH_TIMINGS_FILE),
        "summary_sha256": sha256_file(output_dir / SUMMARY_FILE),
    })
    _write_json(metadata_path, metadata)
    return summary


def load_run(directory, *, require_fullset=False):
    directory = Path(directory)
    metadata = json.loads((directory / METADATA_FILE).read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("state") != "KIT_VALIDATED":
        raise ValueError("NeuOpt TSP run is not finalized KIT evidence")
    identity = validate_identity(metadata.get("resume_identity"))
    if metadata.get("resume_fingerprint") != json_fingerprint(identity):
        raise ValueError("NeuOpt TSP run fingerprint mismatch")
    records_path, timings_path = directory / RECORDS_FILE, directory / BATCH_TIMINGS_FILE
    summary_path = directory / SUMMARY_FILE
    if (metadata.get("validated_records_sha256") != sha256_file(records_path) or
            metadata.get("batch_timings_sha256") != sha256_file(timings_path) or
            metadata.get("summary_sha256") != sha256_file(summary_path)):
        raise ValueError("NeuOpt TSP finalized artifact hash mismatch")
    records, timings = _read_jsonl(records_path), _read_jsonl(timings_path)
    expected_statistics = _statistics(identity, records, timings)
    summary = json.loads(summary_path.read_text())
    expected_ready = identity["scope"] == "fullset"
    if (metadata.get("paper_ready") is not expected_ready or
            metadata.get("completed_records") != len(records) or
            metadata.get("completed_batches") != len(timings) or
            summary.get("schema_version") != SUMMARY_SCHEMA_VERSION or
            summary.get("scope") != identity["scope"] or
            summary.get("paper_ready") is not expected_ready or
            summary.get("resume_identity") != identity or
            any(summary.get(key) != value for key, value in expected_statistics.items())):
        raise ValueError("NeuOpt TSP summary/metadata does not reproduce")
    if require_fullset and not expected_ready:
        raise ValueError("paper result requires a complete 1280-instance fullset")
    return metadata, records, timings, summary


def build_paper_result(run_dir):
    metadata, _, _, summary = load_run(run_dir, require_fullset=True)
    identity = metadata["resume_identity"]
    protocol = identity["paper_protocol"]
    return {
        "schema_version": PAPER_RESULT_SCHEMA_VERSION,
        "artifact_type": "NeuOpt TSP100 paper result",
        "status": "PAPER_READY", "paper_ready": True,
        "problem": "TSP", "problem_size": 100,
        "budget": protocol["budget"], "T_max": protocol["T_max"],
        "D2A": 1, "batch_size": protocol["original_batch_size"],
        "Obj": summary["obj_mean_independent_objective"],
        "Drop_percent": summary["drop_mean_per_instance_gap_percent"],
        "Time_seconds": summary["time_mean_native_batch_latency_seconds"],
        "Total_seconds": summary["total_native_batch_solver_seconds"],
        "instance_count": summary["instance_count"],
        "native_batch_count": summary["num_native_batches"],
        "time_semantics": summary["time_table_semantics"],
        "source_run": str(Path(run_dir).resolve()),
        "source_metadata_sha256": sha256_file(Path(run_dir) / METADATA_FILE),
        "source_summary_sha256": sha256_file(Path(run_dir) / SUMMARY_FILE),
        "project": identity["project"], "upstream": identity["upstream"],
        "environment": identity["environment"],
        "checkpoint_sha256": identity["checkpoint"]["sha256"],
        "dataset_sha256": identity["dataset"]["sha256"],
    }


def write_new_json(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f"refusing to overwrite existing result: {path}")
    _write_json(path, value)
    return path


def read_paper_result(path):
    result = json.loads(Path(path).read_text())
    if (result.get("schema_version") != PAPER_RESULT_SCHEMA_VERSION or
            result.get("status") != "PAPER_READY" or result.get("paper_ready") is not True or
            result.get("problem") != "TSP" or result.get("problem_size") != 100 or
            result.get("D2A") != 1 or result.get("batch_size") not in FORMAL_BATCH_SIZES):
        raise ValueError(f"invalid NeuOpt TSP100 paper result: {path}")
    protocol = formal_protocol(
        100, budget=result.get("budget"), batch_size=result.get("batch_size"),
        T_max=result.get("T_max"))
    for field in ("Obj", "Drop_percent", "Time_seconds", "Total_seconds"):
        _finite(result.get(field), field, nonnegative=field != "Drop_percent")
    if result.get("instance_count") != 1280 or result.get("native_batch_count") != 1280 // protocol["original_batch_size"]:
        raise ValueError("NeuOpt TSP100 paper result has incomplete native-batch coverage")
    expected = build_paper_result(result.get("source_run"))
    if result != expected:
        raise ValueError("NeuOpt TSP100 paper result does not reproduce from its source run")
    return result


def build_result_matrix(result_paths):
    results = [read_paper_result(path) for path in result_paths]
    expected = {(budget, batch_size) for budget in ("fewer", "more")
                for batch_size in FORMAL_BATCH_SIZES}
    keyed = {(row["budget"], row["batch_size"]): row for row in results}
    if len(results) != 6 or set(keyed) != expected:
        raise ValueError("NeuOpt TSP100 result matrix requires exactly fewer/more x BS1/16/128")
    provenance = [
        (row["project"], row["upstream"], row["environment"],
         row["checkpoint_sha256"], row["dataset_sha256"])
        for row in results
    ]
    if any(value != provenance[0] for value in provenance[1:]):
        raise ValueError("NeuOpt TSP100 paper matrix mixes provenance or assets")
    return {
        "schema_version": PAPER_MATRIX_SCHEMA_VERSION,
        "artifact_type": "NeuOpt TSP100 complete/parallel paper result matrix",
        "status": "PAPER_READY", "paper_ready": True,
        "main_complete_results_source": "BS1",
        "parallel_batch_sizes": list(FORMAL_BATCH_SIZES),
        "rows": [keyed[(budget, batch_size)] for budget in ("fewer", "more")
                 for batch_size in FORMAL_BATCH_SIZES],
    }
