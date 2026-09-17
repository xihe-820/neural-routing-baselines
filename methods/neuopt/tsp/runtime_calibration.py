#!/usr/bin/env python3
"""Validate NeuOpt TSP100 candidates and build a non-freezing calibration report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from common.objective_agreement import OBJECTIVE_ATOL, OBJECTIVE_RTOL, objective_agrees
from common.provenance import normalize_git_repository_identity
from methods.neuopt.tsp.compat import PINNED_TSP_STEP_SHA256
from methods.neuopt.tsp.config import supported_config
from methods.neuopt.tsp.decode import decode_successor
from methods.neuopt.tsp.paper_protocol import (
    CALIBRATION_INDICES, CALIBRATION_TARGETS, FIRST_ROUND_T_VALUES,
    UPSTREAM_COMMIT, UPSTREAM_URL, calibration_protocol, protocol_fingerprint,
)


SCHEMA_VERSION = "neuopt-tsp100-calibration-v1"
REPORT_SCHEMA_VERSION = "neuopt-tsp100-calibration-report-v1"
METADATA_FILE = "metadata.json"
RECORDS_FILE = "validated_records.jsonl"
SUMMARY_FILE = "summary.json"
TIMING_SEMANTICS = (
    "BS1 solver-only wall-clock seconds; CUDA synchronized immediately before and "
    "after the complete official agent.rollout(record=False); excludes checkpoint/model "
    "loading, dataset parsing, adapter preparation, warm-up, record=True evidence replay, "
    "successor extraction/decoding, independent validation, ML4CO-Kit validation, "
    "provenance, artifact I/O, and aggregation"
)
WARMUP_POLICY = {
    "instances": 1,
    "dataset_indices": [0],
    "record": False,
    "included_in_timing": False,
    "rng_state_restored": True,
}
RNG_POLICY = (
    "seed Python, NumPy, torch, and all CUDA generators with 6666 before model setup; "
    "restore the pre-warmup state; process indices 0..19 sequentially; for each instance "
    "restore the pre-timed-rollout state before record=True replay and require identical "
    "post-rollout RNG state"
)


def _json_fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def _finite(value, field, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{field} has an invalid value")
    return result


def validate_record(record):
    required = (
        "index", "instance_id", "runtime_seconds", "successor", "canonical_solution",
        "official_objective", "replay_official_objective",
        "official_recomputed_objective", "independent_objective",
        "reference_objective", "gap_percent", "feasible", "kit_feasible",
        "kit_objective", "kit_reference_objective", "objective_agreement",
        "constraint_details", "timed_replay", "evidence_status",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"NeuOpt TSP calibration record missing fields: {missing}")
    index = record["index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("record index must be a nonnegative integer")
    runtime = _finite(record["runtime_seconds"], "runtime_seconds", positive=True)
    del runtime
    for field in ("official_objective", "replay_official_objective",
                  "official_recomputed_objective", "independent_objective",
                  "reference_objective", "kit_objective", "kit_reference_objective"):
        if _finite(record[field], field, positive=True) <= 0:
            raise AssertionError("unreachable")
    successor = record["successor"]
    canonical = record["canonical_solution"]
    if (not isinstance(successor, list) or sorted(successor) != list(range(100)) or
            not isinstance(canonical, list) or len(canonical) != 101 or
            canonical[0] != 0 or canonical[-1] != 0 or
            sorted(canonical[:-1]) != list(range(100))):
        raise ValueError(f"record {index} does not contain an exact Hamiltonian cycle")
    decoded, _ = decode_successor(successor, problem_size=100)
    if decoded != canonical:
        raise ValueError(f"record {index} canonical tour does not match its successor")
    if record["official_objective"] != record["replay_official_objective"]:
        raise ValueError(f"record {index} replay objective differs from timed objective")
    independent = float(record["independent_objective"])
    for field in ("official_objective", "official_recomputed_objective", "kit_objective"):
        if not objective_agrees(record[field], independent):
            raise ValueError(f"record {index} {field} disagrees with independent objective")
    if not objective_agrees(record["kit_reference_objective"], record["reference_objective"]):
        raise ValueError(f"record {index} Kit reference disagrees with benchmark reference")
    expected_gap = (independent - float(record["reference_objective"])) / float(
        record["reference_objective"]) * 100.0
    if not math.isclose(float(record["gap_percent"]), expected_gap,
                        rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"record {index} gap is not its per-instance percentage gap")
    if record["feasible"] is not True or record["kit_feasible"] is not True:
        raise ValueError(f"record {index} failed feasibility")
    expected_agreement = {
        "timed_replay_exact": True,
        "official_independent": True,
        "official_recomputed_independent": True,
        "kit_independent": True,
        "rtol": OBJECTIVE_RTOL,
        "atol": OBJECTIVE_ATOL,
    }
    if record["objective_agreement"] != expected_agreement:
        raise ValueError(f"record {index} objective agreement provenance mismatch")
    expected_replay = {
        "timed_rollout_record": False,
        "evidence_replay_record": True,
        "rng_state_restored": True,
        "timed_replay_official_objective_exact": True,
        "timed_replay_rng_after_exact": True,
        "cuda_synchronized_before_timing": True,
        "cuda_synchronized_after_timing": True,
        "evidence_replay_excluded_from_runtime": True,
    }
    if record["timed_replay"] != expected_replay:
        raise ValueError(f"record {index} timed/replay provenance mismatch")
    if record["evidence_status"] != "KIT_VALIDATED":
        raise ValueError(f"record {index} is not KIT_VALIDATED")
    return index


def validate_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError("NeuOpt TSP calibration identity is missing")
    protocol = identity.get("paper_protocol", {})
    expected_protocol = calibration_protocol(
        identity.get("problem_size"), T_max=protocol.get("T_max"),
        batch_size=protocol.get("original_batch_size"), d2a=protocol.get("D2A"),
        stall_limit=protocol.get("stall_limit"), k=protocol.get("k"))
    if protocol != expected_protocol:
        raise ValueError("NeuOpt TSP calibration protocol mismatch")
    if identity.get("protocol_fingerprint") != protocol_fingerprint(protocol):
        raise ValueError("NeuOpt TSP protocol fingerprint mismatch")
    config = supported_config(identity.get("problem_size"))
    project = identity.get("project", {})
    upstream = identity.get("upstream", {})
    checkpoint = identity.get("checkpoint", {})
    dataset = identity.get("dataset", {})
    if (identity.get("method") != "NeuOpt" or identity.get("variant") != "NeuOpt-GIRE" or
            identity.get("problem") != "TSP" or project.get("dirty") is not False or
            upstream.get("commit") != UPSTREAM_COMMIT or upstream.get("dirty") is not False):
        raise ValueError("NeuOpt TSP project/upstream provenance mismatch or dirty source")
    if normalize_git_repository_identity(upstream.get("url", "")) != \
            normalize_git_repository_identity(UPSTREAM_URL):
        raise ValueError("NeuOpt TSP upstream URL mismatch")
    if (checkpoint.get("sha256") != config["checkpoint_sha256"] or
            Path(checkpoint.get("path", "")).name != "tsp100.pt"):
        raise ValueError("NeuOpt TSP checkpoint mismatch")
    if (dataset.get("sha256") != config["dataset_sha256"] or
            dataset.get("filename") != config["dataset_filename"] or
            dataset.get("count") != config["dataset_count"]):
        raise ValueError("NeuOpt TSP dataset mismatch")
    if identity.get("sample_indices") != list(CALIBRATION_INDICES):
        raise ValueError("NeuOpt TSP calibration must use exact dataset indices 0..19")
    if identity.get("timing_semantics") != TIMING_SEMANTICS:
        raise ValueError("NeuOpt TSP timing semantics mismatch")
    if identity.get("warmup_policy") != WARMUP_POLICY or identity.get("rng_policy") != RNG_POLICY:
        raise ValueError("NeuOpt TSP warm-up/RNG policy mismatch")
    compatibility = identity.get("decoder_compatibility", {})
    if (compatibility.get("bs1_shape_shim") is not True or
            compatibility.get("original_batch_size") != 1 or
            compatibility.get("val_m") != 1 or
            compatibility.get("internal_decoder_batch_size") != 1 or
            compatibility.get("official_source_modified") is not False or
            compatibility.get("action_reward_logits_rng_budget_changed") is not False):
        raise ValueError("NeuOpt TSP BS1 decoder compatibility provenance mismatch")
    record_compatibility = identity.get("record_compatibility", {})
    if (record_compatibility.get("record_step_shim") is not True or
            record_compatibility.get("pinned_step_sha256") != PINNED_TSP_STEP_SHA256 or
            record_compatibility.get("timed_rollout_shim_active") is not False or
            record_compatibility.get("evidence_replay_shim_active") is not True or
            record_compatibility.get("official_source_modified") is not False or
            record_compatibility.get("action_reward_logits_rng_budget_changed") is not False):
        raise ValueError("NeuOpt TSP record=True replay compatibility provenance mismatch")
    if identity.get("tensorboard_compatibility", {}).get("official_source_modified") is not False:
        raise ValueError("NeuOpt TensorBoard compatibility provenance mismatch")
    environment = identity.get("environment", {})
    if ("RTX 4090" not in str(environment.get("gpu")) or
            not str(environment.get("device", "")).startswith("cuda") or
            not environment.get("torch") or not environment.get("torch_cuda_build")):
        raise ValueError("NeuOpt TSP calibration requires RTX 4090 CUDA provenance")
    return identity


def _summary(identity, records):
    runtimes = [float(row["runtime_seconds"]) for row in records]
    objectives = [float(row["independent_objective"]) for row in records]
    gaps = [float(row["gap_percent"]) for row in records]
    return {
        "problem": "TSP", "problem_size": 100,
        "T_max": identity["paper_protocol"]["T_max"], "D2A": 1,
        "original_batch_size": 1, "n": len(records),
        "mean_runtime_seconds": mean(runtimes),
        "median_runtime_seconds": median(runtimes),
        "std_runtime_seconds": pstdev(runtimes),
        "min_runtime_seconds": min(runtimes),
        "max_runtime_seconds": max(runtimes),
        "mean_independent_objective": mean(objectives),
        "mean_per_instance_gap_percent": mean(gaps),
        "all_feasible": all(row["feasible"] is True for row in records),
        "all_objective_agreement_pass": all(
            all(value is True for key, value in row["objective_agreement"].items()
                if key not in ("rtol", "atol")) for row in records),
        "all_kit_validation_pass": all(
            row["kit_feasible"] is True and row["objective_agreement"]["kit_independent"] is True
            for row in records),
        "targets": dict(CALIBRATION_TARGETS),
        "runtime_minus_fewer": mean(runtimes) - CALIBRATION_TARGETS["fewer_seconds"],
        "runtime_minus_more": mean(runtimes) - CALIBRATION_TARGETS["more_seconds"],
        "paper_ready": False, "final_T": None,
    }


def write_candidate(output_dir, identity, records):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("calibration output directory already exists; refusing overwrite")
    validate_identity(identity)
    indices = [validate_record(record) for record in records]
    if indices != list(CALIBRATION_INDICES) or len(indices) != len(set(indices)):
        raise ValueError("candidate records do not exactly cover ordered indices 0..19")
    output_dir.mkdir(parents=True)
    records_path = output_dir / RECORDS_FILE
    with records_path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    summary = _summary(identity, records)
    summary_path = output_dir / SUMMARY_FILE
    _write_json(summary_path, summary)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "NeuOpt TSP100 BS1 runtime calibration candidate",
        "state": "KIT_VALIDATED", "paper_ready": False,
        "final_T_status": "PENDING_USER_REVIEW",
        "identity": identity, "identity_fingerprint": _json_fingerprint(identity),
        "records_file": RECORDS_FILE,
        "validated_records_sha256": sha256_file(records_path),
        "summary_file": SUMMARY_FILE, "summary_sha256": sha256_file(summary_path),
        "completed_records": len(records),
    }
    _write_json(output_dir / METADATA_FILE, metadata)
    return metadata, summary


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


def load_candidate(directory):
    directory = Path(directory)
    metadata = json.loads((directory / METADATA_FILE).read_text())
    if (metadata.get("schema_version") != SCHEMA_VERSION or
            metadata.get("state") != "KIT_VALIDATED" or
            metadata.get("paper_ready") is not False or
            metadata.get("final_T_status") != "PENDING_USER_REVIEW"):
        raise ValueError(f"invalid NeuOpt TSP calibration metadata in {directory}")
    identity = validate_identity(metadata.get("identity"))
    if metadata.get("identity_fingerprint") != _json_fingerprint(identity):
        raise ValueError(f"identity fingerprint mismatch in {directory}")
    records_path = directory / metadata.get("records_file", "")
    summary_path = directory / metadata.get("summary_file", "")
    if metadata.get("validated_records_sha256") != sha256_file(records_path):
        raise ValueError(f"validated record hash mismatch in {directory}")
    if metadata.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError(f"candidate summary hash mismatch in {directory}")
    records = _read_records(records_path)
    indices = [validate_record(row) for row in records]
    if (indices != list(CALIBRATION_INDICES) or
            metadata.get("completed_records") != len(records)):
        raise ValueError(f"calibration subset coverage mismatch in {directory}")
    summary = json.loads(summary_path.read_text())
    if summary != _summary(identity, records):
        raise ValueError(f"candidate summary does not reproduce records in {directory}")
    return metadata, records, summary


def _comparable_identity(identity):
    result = dict(identity)
    protocol = dict(result["paper_protocol"])
    protocol.pop("T_max")
    protocol.pop("first_round_candidate")
    result["paper_protocol"] = protocol
    result.pop("protocol_fingerprint")
    return result


def build_calibration_report(candidate_dirs):
    baseline = None
    candidates = []
    seen = set()
    for directory in candidate_dirs:
        metadata, _, summary = load_candidate(directory)
        identity = metadata["identity"]
        comparable = _comparable_identity(identity)
        if baseline is None:
            baseline = comparable
        elif comparable != baseline:
            raise ValueError("calibration candidates mix project, assets, subset, or protocol")
        T_max = identity["paper_protocol"]["T_max"]
        if T_max in seen:
            raise ValueError(f"duplicate calibration candidate T={T_max}")
        seen.add(T_max)
        candidates.append({**summary, "artifact_dir": str(Path(directory).resolve())})
    missing = sorted(set(FIRST_ROUND_T_VALUES) - seen)
    if missing:
        raise ValueError(f"calibration report is missing first-round candidates: {missing}")
    candidates.sort(key=lambda row: row["T_max"])
    t1 = next(row for row in candidates if row["T_max"] == 1)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "NeuOpt TSP100 BS1 runtime calibration report",
        "status": "PENDING_USER_REVIEW", "problem": "TSP", "problem_size": 100,
        "targets": dict(CALIBRATION_TARGETS), "sample_indices": list(CALIBRATION_INDICES),
        "minimum_valid_T": 1,
        "minimum_budget_note": "no positive integer T can produce a smaller search budget than T=1",
        "t1_runtime_exceeds_fewer_target": (
            t1["mean_runtime_seconds"] > CALIBRATION_TARGETS["fewer_seconds"]),
        "selection_policy": (
            "human review selects T_fewer and T_more after comparing measured BS1 runtimes; "
            "no percentage margin or automatic selection is encoded"
        ),
        "candidates": candidates,
        "final_T_fewer": None, "final_T_more": None,
        "automatic_freeze": False, "paper_ready": False,
    }


def write_calibration_report(path, report):
    path = Path(path)
    if path.exists():
        raise ValueError("calibration report already exists; refusing overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, report)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_calibration_report(args.candidate_dirs)
    write_calibration_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
