"""Strict NeuOpt calibration artifacts and report aggregation."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median

from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from methods.neuopt.cvrp.config import supported_config
from methods.neuopt.cvrp.paper_protocol import (
    LEGACY_CALIBRATION_TARGETS, LEGACY_MANUSCRIPT_CANDIDATE_T, UPSTREAM_COMMIT,
    legacy_calibration_protocol, protocol_fingerprint,
)


SCHEMA_VERSION = "neuopt-paper-calibration-v1"
RECORDS_FILE = "validated_records.jsonl"
METADATA_FILE = "metadata.json"
TIMING_SEMANTICS = (
    "single-original-instance NeuOpt solver wall-clock seconds; CUDA synchronized "
    "immediately before and after the complete official record=False rollout; excludes "
    "the untimed record=True evidence replay, model and checkpoint loading, dataset "
    "parsing, adapter preparation, warm-up, successor decoding, independent/Kit "
    "validation, provenance, artifact I/O, and aggregation"
)


def json_fingerprint(value):
    import hashlib
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(destination)


def _finite(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def validate_record(record):
    required = (
        "dataset_instance_index", "instance_id", "canonical_solution",
        "reported_objective", "independent_objective", "reference_objective",
        "timed_official_objective", "replay_official_objective",
        "official_recomputed_objective", "kit_reference_objective",
        "gap_percent", "runtime_seconds", "independent_feasible",
        "reported_objective_agrees", "kit_feasible", "kit_objective",
        "kit_objective_agrees", "selected_d2a_candidate", "evidence_status",
        "timed_replay",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"NeuOpt calibration record missing fields: {missing}")
    index = record["dataset_instance_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("dataset_instance_index must be a nonnegative integer")
    if not isinstance(record["canonical_solution"], list) or not record["canonical_solution"]:
        raise ValueError(f"record {index} has no canonical solution")
    independent = _finite(record["independent_objective"], "independent_objective")
    reference = _finite(record["reference_objective"], "reference_objective")
    if reference <= 0:
        raise ValueError("reference_objective must be positive")
    for field in ("reported_objective", "timed_official_objective",
                  "replay_official_objective", "official_recomputed_objective",
                  "kit_objective", "kit_reference_objective", "runtime_seconds"):
        _finite(record[field], field, nonnegative=True)
    expected_gap = (independent - reference) / reference * 100.0
    if not math.isclose(float(record["gap_percent"]), expected_gap,
                        rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"record {index} gap is not its per-instance percentage gap")
    gates = ("independent_feasible", "reported_objective_agrees",
             "kit_feasible", "kit_objective_agrees")
    if any(record[field] is not True for field in gates):
        raise ValueError(f"record {index} failed a correctness gate")
    if (record["timed_official_objective"] !=
            record["replay_official_objective"]):
        raise ValueError(f"record {index} replay objective differs from timed objective")
    if record["reported_objective"] != record["timed_official_objective"]:
        raise ValueError(f"record {index} reported objective disagrees with timed objective")
    for field in ("reported_objective", "timed_official_objective",
                  "replay_official_objective", "official_recomputed_objective",
                  "kit_objective"):
        if not objective_agrees(record[field], independent):
            raise ValueError(f"record {index} {field} disagrees with independent objective")
    if not objective_agrees(record["kit_reference_objective"], reference):
        raise ValueError(f"record {index} Kit reference disagrees with prepared reference")
    if record["evidence_status"] != "KIT_VALIDATED":
        raise ValueError(f"record {index} is not KIT_VALIDATED")
    replay = record["timed_replay"]
    expected_replay = {
        "timed_rollout_record": False,
        "evidence_replay_record": True,
        "rng_state_restored": True,
        "timed_replay_official_objective_exact": True,
        "timed_replay_rng_after_exact": True,
    }
    if replay != expected_replay:
        raise ValueError(f"record {index} timed/evidence replay provenance mismatch")
    return index


def write_artifact(output_dir, resume_identity, records):
    output_dir = Path(output_dir)
    if ((output_dir / METADATA_FILE).exists() or
            (output_dir / RECORDS_FILE).exists()):
        raise ValueError("NeuOpt calibration output directory already contains evidence")
    expected = resume_identity["subset"]["dataset_indices"]
    indices = [validate_record(record) for record in records]
    if len(indices) != len(set(indices)) or indices != expected:
        raise ValueError("records do not exactly match the ordered calibration subset")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / RECORDS_FILE
    temporary = records_path.with_name(records_path.name + ".tmp")
    with temporary.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    temporary.replace(records_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "NeuOpt BS1 runtime calibration candidate",
        "state": "KIT_VALIDATED",
        "resume_identity": resume_identity,
        "resume_fingerprint": json_fingerprint(resume_identity),
        "records_file": RECORDS_FILE,
        "validated_records_sha256": sha256_file(records_path),
        "completed_records": len(records),
        "paper_ready": False,
        "final_T_status": "PENDING_CALIBRATION",
    }
    _write_json(output_dir / METADATA_FILE, metadata)
    return metadata


def _read_records(path):
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


def load_artifact(directory):
    directory = Path(directory)
    metadata = json.loads((directory / METADATA_FILE).read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported NeuOpt calibration schema in {directory}")
    if metadata.get("state") != "KIT_VALIDATED" or metadata.get("paper_ready") is not False:
        raise ValueError(f"NeuOpt candidate {directory} is not calibration-only KIT evidence")
    identity = metadata.get("resume_identity")
    if metadata.get("resume_fingerprint") != json_fingerprint(identity):
        raise ValueError(f"NeuOpt calibration fingerprint mismatch in {directory}")
    if not isinstance(identity, dict):
        raise ValueError(f"NeuOpt calibration identity is missing in {directory}")
    protocol = identity.get("paper_protocol", {})
    expected_protocol = legacy_calibration_protocol(
        identity.get("problem_size"), T_max=protocol.get("T_max"),
        d2a=protocol.get("D2A"), stall_limit=protocol.get("stall_limit"),
        k=protocol.get("k"))
    if protocol != expected_protocol or identity.get("protocol_fingerprint") != protocol_fingerprint(protocol):
        raise ValueError(f"NeuOpt formal protocol mismatch in {directory}")
    size_config = supported_config(identity.get("problem_size"))
    if (identity.get("method") != "NeuOpt" or
            identity.get("variant") != "NeuOpt-GIRE" or
            identity.get("problem") != "CVRP" or
            identity.get("upstream", {}).get("commit") != UPSTREAM_COMMIT or
            identity.get("checkpoint", {}).get("sha256") != size_config["checkpoint_sha256"] or
            identity.get("dataset", {}).get("sha256") != size_config["dataset_sha256"] or
            identity.get("dataset", {}).get("count") != size_config["dataset_count"]):
        raise ValueError(f"mixed or non-pinned NeuOpt identity/assets in {directory}")
    if (identity.get("original_batch_size") != 1 or
            identity.get("timing_semantics") != TIMING_SEMANTICS):
        raise ValueError(f"NeuOpt timing is not paper single-instance timing in {directory}")
    if identity.get("project", {}).get("dirty") or identity.get("upstream", {}).get("dirty"):
        raise ValueError(f"NeuOpt calibration provenance is dirty in {directory}")
    tensorboard = identity.get("tensorboard_compatibility", {})
    if tensorboard.get("official_source_modified") is not False:
        raise ValueError(f"NeuOpt TensorBoard compatibility provenance mismatch in {directory}")
    compatibility = identity.get("bs1_compatibility", {})
    if (compatibility.get("bs1_shape_shim") is not False or
            compatibility.get("original_batch_size") != 1 or
            compatibility.get("val_m") != 5 or
            compatibility.get("internal_decoder_batch_size") != 5 or
            compatibility.get("historical_d2a1_shape_shim_available") is not True or
            compatibility.get("official_source_modified") is not False or
            compatibility.get("action_reward_logits_rng_budget_changed") is not False):
        raise ValueError(f"NeuOpt BS1 compatibility provenance mismatch in {directory}")
    environment = identity.get("environment", {})
    if ("RTX 4090" not in str(environment.get("gpu")) or
            not str(environment.get("device", "")).startswith("cuda")):
        raise ValueError(f"NeuOpt calibration was not measured on RTX 4090 CUDA in {directory}")
    records_path = directory / metadata["records_file"]
    if metadata.get("validated_records_sha256") != sha256_file(records_path):
        raise ValueError(f"NeuOpt validated record hash mismatch in {directory}")
    records = _read_records(records_path)
    indices = [validate_record(record) for record in records]
    expected_indices = identity.get("subset", {}).get("dataset_indices")
    if (indices != expected_indices or len(indices) != len(set(indices)) or
            metadata.get("completed_records") != len(records)):
        raise ValueError(f"NeuOpt calibration subset coverage mismatch in {directory}")
    return metadata, records


def _same_protocol_identity(metadata):
    identity = metadata["resume_identity"]
    return {
        "method": identity["method"], "variant": identity["variant"],
        "problem": identity["problem"], "problem_size": identity["problem_size"],
        "paper_protocol": identity["paper_protocol"],
        "project": identity["project"], "upstream": identity["upstream"],
        "checkpoint_sha256": identity["checkpoint"]["sha256"],
        "dataset_sha256": identity["dataset"]["sha256"],
        "dataset_count": identity["dataset"]["count"],
        "subset": identity["subset"], "environment": identity["environment"],
        "timing_semantics": identity["timing_semantics"],
    }


def summarize_same_protocol(directories):
    """Combine repeats/chunks only when T, D2A, data and assets all match."""
    baseline = None
    records = []
    for directory in directories:
        metadata, current = load_artifact(directory)
        identity = _same_protocol_identity(metadata)
        if baseline is None:
            baseline = identity
        elif identity != baseline:
            raise ValueError("mixed T, D2A, checkpoint, dataset, subset, or provenance")
        records.extend(current)
    if baseline is None:
        raise ValueError("at least one NeuOpt calibration artifact is required")
    return baseline, records


def _calibration_identity(metadata):
    result = _same_protocol_identity(metadata)
    protocol = dict(result["paper_protocol"])
    protocol.pop("T_max")
    protocol.pop("budget_status")
    result["paper_protocol"] = protocol
    return result


def build_calibration_report(directories, *, problem_size):
    """Compare T candidates without choosing a margin or freezing a paper budget."""
    size = int(problem_size)
    baseline = None
    candidates = []
    seen_t = set()
    for directory in directories:
        metadata, records = load_artifact(directory)
        identity = metadata["resume_identity"]
        if identity["problem_size"] != size:
            raise ValueError("calibration report cannot mix problem sizes")
        comparable = _calibration_identity(metadata)
        if baseline is None:
            baseline = comparable
        elif comparable != baseline:
            raise ValueError("calibration candidates mix D2A, checkpoint, dataset, subset, or provenance")
        T_max = identity["paper_protocol"]["T_max"]
        if T_max in seen_t:
            raise ValueError(f"duplicate calibration candidate T={T_max}")
        seen_t.add(T_max)
        runtimes = [float(record["runtime_seconds"]) for record in records]
        objectives = [float(record["independent_objective"]) for record in records]
        targets = LEGACY_CALIBRATION_TARGETS[size]
        candidates.append({
            "T_max": T_max,
            "D2A": identity["paper_protocol"]["D2A"],
            "original_batch_size": 1,
            "count": len(records),
            "mean_runtime_seconds": mean(runtimes),
            "median_runtime_seconds": median(runtimes),
            "min_runtime_seconds": min(runtimes),
            "max_runtime_seconds": max(runtimes),
            "mean_independent_objective": mean(objectives),
            "median_independent_objective": median(objectives),
            "min_independent_objective": min(objectives),
            "max_independent_objective": max(objectives),
            "all_feasible": True,
            "fewer_target_seconds": targets["fewer_seconds"],
            "delta_from_fewer_seconds": mean(runtimes) - targets["fewer_seconds"],
            "above_fewer_target": mean(runtimes) > targets["fewer_seconds"],
            "more_target_seconds": targets["more_seconds"],
            "delta_from_more_seconds": mean(runtimes) - targets["more_seconds"],
            "above_more_target": mean(runtimes) > targets["more_seconds"],
            "artifact": str(Path(directory).resolve()),
        })
    missing = sorted(set(LEGACY_MANUSCRIPT_CANDIDATE_T) - seen_t)
    if missing:
        raise ValueError(f"calibration report is missing manuscript candidates: {missing}")
    candidates.sort(key=lambda item: item["T_max"])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "NeuOpt BS1 runtime calibration report",
        "status": "PENDING_USER_SELECTION",
        "problem": "CVRP", "problem_size": size,
        "targets": dict(LEGACY_CALIBRATION_TARGETS[size]),
        "selection_rule": (
            "user selects the measured stable candidate closest to and above each target; "
            "no fixed percentage margin is encoded"
        ),
        "candidates": candidates,
        "final_T_fewer": None, "final_T_more": None,
        "manuscript_protocol_update": "PENDING_CALIBRATION",
        "paper_ready": False,
    }


def write_calibration_report(path, report):
    _write_json(path, report)
    return Path(path)
