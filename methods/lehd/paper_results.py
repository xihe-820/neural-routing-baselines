"""Atomic LEHD BS1 artifacts, exact resume identity, and aggregation."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

from common.hashing import sha256_file


SCHEMA = "lehd-paper-production.v1"
TERMINAL_STATES = {"KIT_VALIDATED", "PAPER_READY"}
TIMING_SEMANTICS = (
    "CUDA-synchronized wall time around one pinned official LEHD _test_one_batch at BS=1, "
    "including greedy construction, exactly RRC_budget official reconstruction loops, "
    "official internal objective/search operations and logging; excluding dataset/checkpoint/"
    "model loading, adapter, isolated untimed warm-up tester, solution CPU clone, independent "
    "validation, ML4CO-Kit validation, and artifact I/O"
)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_jsonl(path: Path, values):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def capture_rng_state(torch):
    numpy_state = np.random.get_state()
    encode = lambda value: base64.b64encode(value.cpu().numpy().tobytes()).decode()
    return {
        "python": random.getstate(),
        "numpy": {
            "kind": numpy_state[0], "keys": numpy_state[1].tolist(),
            "position": numpy_state[2], "has_gauss": numpy_state[3],
            "cached": numpy_state[4],
        },
        "torch_cpu": encode(torch.get_rng_state()),
        "torch_cuda": [encode(value) for value in torch.cuda.get_rng_state_all()],
    }


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def restore_rng_state(state, torch):
    random.setstate(_tuples(state["python"]))
    value = state["numpy"]
    np.random.set_state((
        value["kind"], np.asarray(value["keys"], dtype=np.uint32),
        value["position"], value["has_gauss"], value["cached"],
    ))
    decode = lambda text: torch.from_numpy(
        np.frombuffer(base64.b64decode(text), dtype=np.uint8).copy())
    torch.set_rng_state(decode(state["torch_cpu"]))
    torch.cuda.set_rng_state_all([decode(item) for item in state["torch_cuda"]])


def initialize(output_dir: Path, identity: dict, *, resume: bool):
    output_dir = Path(output_dir)
    metadata_path = output_dir / "metadata.json"
    expected = fingerprint(identity)
    existed = output_dir.exists()
    if existed and not resume:
        raise ValueError("output directory already exists; pass --resume/--skip-existing")
    if existed and resume and not metadata_path.is_file():
        raise ValueError("resume refused: existing LEHD output lacks metadata.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("schema") != SCHEMA or metadata.get("identity") != identity
                or metadata.get("identity_fingerprint") != expected):
            raise ValueError("resume refused: LEHD run identity differs")
        records = read_jsonl(output_dir / "validated_records.jsonl")
        timings = read_jsonl(output_dir / "batch_timings.jsonl")
        if records:
            if metadata.get("validated_records_sha256") != sha256_file(
                    output_dir / "validated_records.jsonl"):
                raise ValueError("resume refused: LEHD validated records hash mismatch")
            if metadata.get("batch_timings_sha256") != sha256_file(
                    output_dir / "batch_timings.jsonl"):
                raise ValueError("resume refused: LEHD timing hash mismatch")
        indices = [row["dataset_instance_index"] for row in records]
        if indices != identity["indices"][:len(records)]:
            raise ValueError("resume refused: LEHD records are not an ordered prefix")
        if [row["dataset_instance_index"] for row in timings] != indices:
            raise ValueError("resume refused: LEHD timing and record indices differ")
        checkpoint = None
        if records:
            checkpoint_path = output_dir / "checkpoint_state.json"
            if metadata.get("checkpoint_state_sha256") != sha256_file(checkpoint_path):
                raise ValueError("resume refused: LEHD RNG checkpoint hash mismatch")
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint.get("completed_records") != len(records):
                raise ValueError("LEHD RNG checkpoint record count mismatch")
        if metadata.get("state") in TERMINAL_STATES:
            summary_path = output_dir / "summary.json"
            if metadata.get("summary_sha256") != sha256_file(summary_path):
                raise ValueError("resume refused: LEHD summary hash mismatch")
        return metadata, records, timings, checkpoint
    metadata = {
        "schema": SCHEMA, "state": "RUNNING", "created_at": _utc(),
        "identity": identity, "identity_fingerprint": expected,
        "validated_records_file": "validated_records.jsonl",
        "batch_timings_file": "batch_timings.jsonl",
        "summary_file": "summary.json", "completed_records": 0,
    }
    _atomic_json(metadata_path, metadata)
    _atomic_jsonl(output_dir / "validated_records.jsonl", [])
    _atomic_jsonl(output_dir / "batch_timings.jsonl", [])
    return metadata, [], [], None


def append(output_dir: Path, metadata: dict, records: list, timings: list,
           record: dict, timing: dict, rng_state: dict):
    expected_index = metadata["identity"]["indices"][len(records)]
    if record["dataset_instance_index"] != expected_index:
        raise ValueError("LEHD append is not the next expected dataset index")
    records.append(record)
    timings.append(timing)
    _atomic_jsonl(output_dir / "validated_records.jsonl", records)
    _atomic_jsonl(output_dir / "batch_timings.jsonl", timings)
    checkpoint_path = output_dir / "checkpoint_state.json"
    _atomic_json(checkpoint_path, {
        "schema": SCHEMA, "completed_records": len(records), "rng_state": rng_state,
    })
    metadata["completed_records"] = len(records)
    metadata["updated_at"] = _utc()
    metadata["validated_records_sha256"] = sha256_file(
        output_dir / "validated_records.jsonl")
    metadata["batch_timings_sha256"] = sha256_file(output_dir / "batch_timings.jsonl")
    metadata["checkpoint_state_sha256"] = sha256_file(checkpoint_path)
    _atomic_json(output_dir / "metadata.json", metadata)


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}")
    return value


def finalize(output_dir: Path, metadata: dict, records: list, timings: list):
    identity = metadata["identity"]
    if [row["dataset_instance_index"] for row in records] != identity["indices"]:
        raise ValueError("cannot finalize incomplete LEHD run")
    if len(timings) != len(records):
        raise ValueError("cannot finalize LEHD run with missing timings")
    for row in records:
        if (row.get("evidence_status") != "KIT_VALIDATED"
                or row.get("independent_feasible") is not True
                or row.get("kit_feasible") is not True
                or row.get("official_vs_independent", {}).get("pass") is not True
                or row.get("independent_vs_kit", {}).get("pass") is not True):
            raise ValueError("LEHD record failed validation gate")
    objectives = np.asarray([
        _finite(row["independent_objective"], "objective") for row in records])
    refs = np.asarray([_finite(row["reference_objective"], "reference") for row in records])
    gaps = np.asarray([_finite(row["gap_percent"], "gap") for row in records])
    runtimes = np.asarray([_finite(row["runtime_seconds"], "runtime") for row in timings])
    full = (identity["scope"] == "fullset" and identity["offset"] == 0
            and len(records) == identity["dataset"]["count"])
    status = "PAPER_READY" if full else "KIT_VALIDATED"
    protocol = identity["protocol"]
    summary = {
        "schema": SCHEMA, "status": status, "method": "LEHD",
        "problem": protocol["problem"],
        "problem_size": protocol["actual_problem_size"],
        "trained_on_size": protocol["trained_on_size"],
        "protocol_label": protocol["protocol_label"],
        "RRC_budget": protocol["RRC_budget"], "count": len(records),
        "protocol_fingerprint": identity["protocol_fingerprint"],
        "dataset_sha256": identity["dataset"]["sha256"],
        "checkpoint_sha256": identity["checkpoint"]["sha256"],
        "upstream_commit": identity["upstream"]["commit"],
        "project_commit": identity["project"]["commit"],
        "source_provenance_fingerprint": fingerprint(identity["source_files"]),
        "full_dataset_complete": full,
        "paper_ready": status == "PAPER_READY",
        "mean_objective": float(objectives.mean()),
        "mean_reference_objective": float(refs.mean()),
        "mean_instance_gap_percent": float(gaps.mean()),
        "mean_batch_runtime_seconds": float(runtimes.mean()),
        "total_runtime_seconds": float(runtimes.sum()),
        "timing_semantics": TIMING_SEMANTICS,
        "validated_count": len(records), "failed_count": 0,
        "identity": identity,
        "records_sha256": sha256_file(output_dir / "validated_records.jsonl"),
        "batch_timings_sha256": sha256_file(output_dir / "batch_timings.jsonl"),
    }
    _atomic_json(output_dir / "summary.json", summary)
    metadata["state"] = status
    metadata["completed_records"] = len(records)
    metadata["summary_sha256"] = sha256_file(output_dir / "summary.json")
    metadata["updated_at"] = _utc()
    _atomic_json(output_dir / "metadata.json", metadata)
    return summary
