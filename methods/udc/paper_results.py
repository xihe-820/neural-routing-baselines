"""Fail-closed UDC production artifacts, summaries, and exact RNG resume state."""
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

SCHEMA = "udc-paper-production.v1"
METADATA_FILE = "metadata.json"
RECORDS_FILE = "validated_records.jsonl"
SUMMARY_FILE = "summary.json"
CHECKPOINT_FILE = "checkpoint_state.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _jsonl_bytes(rows) -> bytes:
    return b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"),
                                allow_nan=False) + "\n").encode()
                    for row in rows)


def atomic_jsonl(path: Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_jsonl_bytes(rows))
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
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


def _number(value, name, *, positive=False, nonnegative=False) -> float:
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


def validate_production_record(record: dict, *, problem: str, size: int,
                               alpha: int, x_stages: int) -> int:
    required = {
        "dataset_instance_index", "instance_id", "selected_solution",
        "canonical_solution", "budget_label", "best_alpha", "official_objective",
        "independent_objective", "reference_objective", "drop_percent",
        "runtime_seconds", "timing_semantics", "completed_x_stages",
        "independent_feasible", "kit_feasible", "internal_vs_independent",
        "independent_vs_kit", "evidence_status",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"UDC production record missing fields: {missing}")
    index = record["dataset_instance_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("UDC dataset index must be a nonnegative integer")
    if record.get("problem", "").lower() != problem or record.get("size") != size:
        raise ValueError(f"UDC record {index} problem/size mismatch")
    if record.get("alpha") != alpha or record.get("x") != x_stages:
        raise ValueError(f"UDC record {index} budget mismatch")
    if record["completed_x_stages"] != x_stages:
        raise ValueError(f"UDC record {index} did not complete exact x stages")
    if not isinstance(record["selected_solution"], list) or not record["selected_solution"]:
        raise ValueError(f"UDC record {index} lacks selected actual solution")
    if not isinstance(record["canonical_solution"], list) or not record["canonical_solution"]:
        raise ValueError(f"UDC record {index} lacks canonical solution")
    if problem == "cvrp":
        flags = record.get("solution_flag")
        if not isinstance(flags, list) or len(flags) != len(record["selected_solution"]):
            raise ValueError(f"UDC CVRP record {index} lacks matching solution_flag")
    objective = _number(record["independent_objective"], "independent_objective")
    reference = _number(record["reference_objective"], "reference_objective", positive=True)
    _number(record["official_objective"], "official_objective")
    _number(record["runtime_seconds"], "runtime_seconds", nonnegative=True)
    drop = _number(record["drop_percent"], "drop_percent")
    expected_drop = (objective - reference) / reference * 100.0
    if not math.isclose(drop, expected_drop, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"UDC record {index} drop is not its instance-wise drop")
    if (record["independent_feasible"] is not True or record["kit_feasible"] is not True
            or record.get("internal_vs_independent", {}).get("pass") is not True
            or record.get("independent_vs_kit", {}).get("pass") is not True
            or record["evidence_status"] != "KIT_VALIDATED"):
        raise ValueError(f"UDC record {index} failed independent/Kit validation")
    if not isinstance(record["timing_semantics"], str) or not record["timing_semantics"]:
        raise ValueError(f"UDC record {index} lacks timing semantics")
    return index


def summarize_records(records: list[dict], identity: dict) -> dict:
    problem = identity["problem"]
    size = identity["size"]
    budget = identity["budget"]
    expected_count = identity["dataset"]["count"]
    indices = [validate_production_record(
        record, problem=problem, size=size, alpha=budget["alpha"],
        x_stages=budget["x"]) for record in records]
    if any(record["budget_label"] != budget["label"] for record in records):
        raise ValueError("UDC production record budget label mismatch")
    if indices != list(range(expected_count)):
        raise ValueError("UDC production records are not the complete ordered dataset")
    objectives = np.asarray([r["independent_objective"] for r in records], dtype=np.float64)
    references = np.asarray([r["reference_objective"] for r in records], dtype=np.float64)
    drops = np.asarray([r["drop_percent"] for r in records], dtype=np.float64)
    runtimes = np.asarray([r["runtime_seconds"] for r in records], dtype=np.float64)
    return {
        "schema": SCHEMA, "artifact_type": "UDC formal paper production summary",
        "state": "KIT_VALIDATED", "method": "UDC", "problem": problem.upper(),
        "size": size, "budget_label": budget["label"], "alpha": budget["alpha"],
        "x": budget["x"], "count": expected_count,
        "dataset_sha256": identity["dataset"]["sha256"],
        "mean_objective": float(objectives.mean()),
        "mean_reference_objective": float(references.mean()),
        "mean_instance_drop_percent": float(drops.mean()),
        "mean_runtime_seconds": float(runtimes.mean()),
        "objective_std": float(objectives.std()), "drop_std": float(drops.std()),
        "runtime_std": float(runtimes.std()), "validated_count": len(records),
        "failed_count": 0, "hardware": identity["environment"]["gpu_name"],
        "batch_size": budget["batch_size"], "seed": budget["seed"],
        "timing_semantics": identity["timing_semantics"],
        "resume_identity": identity, "resume_fingerprint": fingerprint(identity),
    }


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_bytes(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


def _lists_to_tuples(value):
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value


def capture_rng_state(torch) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"bit_generator": numpy_state[0],
                  "keys": numpy_state[1].astype(np.uint32).tolist(),
                  "position": int(numpy_state[2]),
                  "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "torch_cpu_b64": _encode_bytes(torch.get_rng_state().cpu().numpy().tobytes()),
        "torch_cuda_b64": [_encode_bytes(state.cpu().numpy().tobytes())
                           for state in torch.cuda.get_rng_state_all()],
    }


def restore_rng_state(state: dict, torch):
    random.setstate(_lists_to_tuples(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["bit_generator"],
                         np.asarray(numpy_state["keys"], dtype=np.uint32),
                         int(numpy_state["position"]), int(numpy_state["has_gauss"]),
                         float(numpy_state["cached_gaussian"])))
    cpu = np.frombuffer(_decode_bytes(state["torch_cpu_b64"]), dtype=np.uint8).copy()
    torch.set_rng_state(torch.from_numpy(cpu))
    cuda = [torch.from_numpy(np.frombuffer(_decode_bytes(value), dtype=np.uint8).copy())
            for value in state["torch_cuda_b64"]]
    torch.cuda.set_rng_state_all(cuda)


def initialize_or_resume(output_dir: Path, identity: dict) -> tuple[dict, list[dict], dict | None]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / METADATA_FILE
    records_path = output_dir / RECORDS_FILE
    checkpoint_path = output_dir / CHECKPOINT_FILE
    expected_fingerprint = fingerprint(identity)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("schema") != SCHEMA
                or metadata.get("resume_fingerprint") != expected_fingerprint
                or metadata.get("resume_identity") != identity):
            raise ValueError("resume refused: UDC production identity differs")
        if metadata.get("state") == "FAILED":
            raise ValueError("resume refused: UDC production run is FAILED")
        if metadata.get("state") == "KIT_VALIDATED":
            records = read_jsonl(records_path)
            summary_path = output_dir / SUMMARY_FILE
            if (metadata.get("validated_records_sha256") != sha256_file(records_path)
                    or not summary_path.is_file()
                    or metadata.get("summary_sha256") != sha256_file(summary_path)):
                raise ValueError("finalized UDC records or summary changed")
            return metadata, records, None
    else:
        if any(output_dir.iterdir()):
            raise ValueError("UDC output directory is nonempty without metadata")
        metadata = {"schema": SCHEMA, "artifact_type": "UDC formal paper production",
                    "state": "IN_PROGRESS", "created_at": utc_now(),
                    "resume_identity": identity,
                    "resume_fingerprint": expected_fingerprint,
                    "records_file": RECORDS_FILE, "checkpoint_file": CHECKPOINT_FILE}
        atomic_json(metadata_path, metadata)
        atomic_jsonl(records_path, [])
    records = read_jsonl(records_path)
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
    if checkpoint is not None:
        if checkpoint.get("resume_fingerprint") != expected_fingerprint:
            raise ValueError("resume refused: UDC checkpoint identity differs")
        committed = checkpoint.get("next_instance_index")
        if not isinstance(committed, int) or committed < 0:
            raise ValueError("UDC checkpoint next index is invalid")
        if len(records) == committed + 1:
            prefix = records[:committed]
            prefix_bytes = _jsonl_bytes(prefix)
            if hashlib.sha256(prefix_bytes).hexdigest() != checkpoint.get("records_sha256"):
                raise ValueError("UDC interrupted record prefix differs from checkpoint")
            atomic_jsonl(records_path, prefix)
            records = prefix
        if len(records) != committed or sha256_file(records_path) != checkpoint.get("records_sha256"):
            raise ValueError("UDC checkpoint and records are inconsistent")
    elif records:
        raise ValueError("UDC records exist without an RNG checkpoint")
    return metadata, records, checkpoint


def commit_record(output_dir: Path, identity: dict, records: list[dict],
                  record: dict, rng_state: dict):
    expected_index = len(records)
    validate_production_record(
        record, problem=identity["problem"], size=identity["size"],
        alpha=identity["budget"]["alpha"], x_stages=identity["budget"]["x"])
    if record["dataset_instance_index"] != expected_index:
        raise ValueError("UDC record commit is not the next ordered dataset index")
    updated = [*records, record]
    records_path = Path(output_dir) / RECORDS_FILE
    atomic_jsonl(records_path, updated)
    checkpoint = {"schema": SCHEMA, "resume_fingerprint": fingerprint(identity),
                  "next_instance_index": len(updated),
                  "records_sha256": sha256_file(records_path),
                  "rng_state_after_committed_prefix": rng_state,
                  "updated_at": utc_now()}
    atomic_json(Path(output_dir) / CHECKPOINT_FILE, checkpoint)
    return updated, checkpoint


def initialize_rng_checkpoint(output_dir: Path, identity: dict, rng_state: dict) -> dict:
    output_dir = Path(output_dir)
    records_path = output_dir / RECORDS_FILE
    if read_jsonl(records_path):
        raise ValueError("initial UDC RNG checkpoint requires an empty record prefix")
    checkpoint = {"schema": SCHEMA, "resume_fingerprint": fingerprint(identity),
                  "next_instance_index": 0,
                  "records_sha256": sha256_file(records_path),
                  "rng_state_after_committed_prefix": rng_state,
                  "updated_at": utc_now()}
    atomic_json(output_dir / CHECKPOINT_FILE, checkpoint)
    return checkpoint


def finalize_run(output_dir: Path, identity: dict, records: list[dict]) -> dict:
    output_dir = Path(output_dir)
    summary = summarize_records(records, identity)
    records_path = output_dir / RECORDS_FILE
    summary.update({"validated_records_sha256": sha256_file(records_path),
                    "completed_at": utc_now()})
    atomic_json(output_dir / SUMMARY_FILE, summary)
    metadata_path = output_dir / METADATA_FILE
    metadata = json.loads(metadata_path.read_text())
    metadata.update({"state": "KIT_VALIDATED", "completed_records": len(records),
                     "validated_records_sha256": sha256_file(records_path),
                     "summary_sha256": sha256_file(output_dir / SUMMARY_FILE),
                     "completed_at": utc_now()})
    atomic_json(metadata_path, metadata)
    return summary
