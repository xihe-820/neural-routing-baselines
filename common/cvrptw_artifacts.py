"""Artifact and gate helpers shared by RF-TE, CaDA, and MoSES(CaDA)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from common.cvrptw_formal import DROP_DEFINITION, TIMING_SEMANTICS
from common.hashing import sha256_file
from common.paper_results import json_fingerprint, read_jsonl, utc_now, write_json

SCHEMA = "remaining-cvrptw-paper-v1"
RECORDS = "validated_records.jsonl"
TIMINGS = "timings.jsonl"


def validate_record(record: dict, *, problem_size=None) -> int:
    required = (
        "dataset_instance_index", "instance_id", "raw_official_action",
        "canonical_solution", "selected_candidate", "official_reward",
        "reported_objective", "independent_objective", "kit_objective",
        "reference_objective", "instance_drop_percent", "runtime_seconds",
        "independent_feasible", "kit_feasible", "reported_objective_agrees",
        "kit_objective_agrees", "route_count", "route_loads",
        "route_timelines", "status",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"CVRPTW record missing fields: {missing}")
    index = record["dataset_instance_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("dataset_instance_index must be a nonnegative integer")
    for field in ("official_reward", "reported_objective", "independent_objective",
                  "kit_objective", "reference_objective", "instance_drop_percent",
                  "runtime_seconds"):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"record {index} has invalid {field}")
    if record["reference_objective"] <= 0 or record["runtime_seconds"] < 0:
        raise ValueError(f"record {index} has invalid reference/runtime")
    expected = ((record["independent_objective"] - record["reference_objective"])
                / record["reference_objective"] * 100.0)
    if not math.isclose(record["instance_drop_percent"], expected,
                        rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"record {index} Drop is not per-instance Drop")
    for field in ("independent_feasible", "kit_feasible",
                  "reported_objective_agrees", "kit_objective_agrees"):
        if record[field] is not True:
            raise ValueError(f"record {index} failed {field}")
    if record["status"] != "KIT_VALIDATED":
        raise ValueError(f"record {index} is not KIT_VALIDATED")
    canonical = record["canonical_solution"]
    raw = record["raw_official_action"]
    if (not isinstance(canonical, list) or not isinstance(raw, list) or
            len(canonical) < 3 or canonical[0] != 0 or canonical[-1] != 0):
        raise ValueError(f"record {index} has invalid canonical/raw action representation")
    candidate = record["selected_candidate"]
    if not isinstance(candidate, dict) or any(
            isinstance(candidate.get(field), bool) or
            not isinstance(candidate.get(field), int) or candidate[field] < 0
            for field in ("augmentation_index", "start_index", "flat_index")):
        raise ValueError(f"record {index} has invalid selected candidate provenance")
    if problem_size is not None:
        customers = [node for node in canonical if node != 0]
        if sorted(customers) != list(range(1, problem_size + 1)):
            raise ValueError(f"record {index} canonical solution is not exact customer coverage")
        if (candidate["augmentation_index"] >= 8 or
                candidate["start_index"] >= problem_size or
                candidate["flat_index"] !=
                candidate["augmentation_index"] * problem_size + candidate["start_index"]):
            raise ValueError(f"record {index} candidate indices violate Aug8/POMO semantics")
    return index


def initialize(output_dir, identity: dict):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    fingerprint = json_fingerprint(identity)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("schema") != SCHEMA or metadata.get("resume_fingerprint") != fingerprint:
            raise ValueError("resume refused: artifact schema/provenance/configuration differs")
    else:
        metadata = {
            "schema": SCHEMA, "artifact_type": "remaining CVRPTW evaluation",
            "state": "INFERENCE_IN_PROGRESS", "resume_identity": identity,
            "resume_fingerprint": fingerprint, "records_file": RECORDS,
            "timings_file": TIMINGS, "created_at": utc_now(), "completed_records": 0,
        }
        write_json(metadata_path, metadata)
    state = metadata.get("state")
    if state in ("KIT_VALIDATED", "PAPER_READY"):
        records_path = output / RECORDS
        timings_path = output / TIMINGS
        if (metadata.get("validated_records_sha256") != sha256_file(records_path) or
                metadata.get("timings_sha256") != sha256_file(timings_path)):
            raise ValueError("resume refused: finalized artifact hashes changed")
    records = read_jsonl(output / RECORDS) if (output / RECORDS).exists() else []
    completed = set()
    for row in records:
        index = validate_record(row, problem_size=identity["problem_size"])
        if index in completed or index not in identity["chunk"]["expected_indices"]:
            raise ValueError("existing artifact has duplicate or out-of-scope index")
        completed.add(index)
    return metadata, completed


def append(output_dir, record: dict):
    validate_record(record)
    output = Path(output_dir)
    with (output / RECORDS).open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    timing = {"dataset_instance_index": record["dataset_instance_index"],
              "runtime_seconds": record["runtime_seconds"]}
    with (output / TIMINGS).open("a") as stream:
        stream.write(json.dumps(timing, sort_keys=True, separators=(",", ":")) + "\n")


def finalize(output_dir, *, paper_ready: bool = False):
    output = Path(output_dir)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    records = read_jsonl(output / RECORDS)
    problem_size = metadata["resume_identity"]["problem_size"]
    indices = [validate_record(row, problem_size=problem_size) for row in records]
    expected = metadata["resume_identity"]["chunk"]["expected_indices"]
    if len(indices) != len(set(indices)) or set(indices) != set(expected):
        raise ValueError("artifact has duplicate, missing, or extra dataset indices")
    expected_count = metadata["resume_identity"]["dataset"]["count"]
    if paper_ready and set(indices) != set(range(expected_count)):
        raise ValueError("PAPER_READY requires exact full-dataset coverage")
    timings = read_jsonl(output / TIMINGS)
    if ([row.get("dataset_instance_index") for row in timings] != indices or
            any(row.get("runtime_seconds") != record["runtime_seconds"]
                for row, record in zip(timings, records))):
        raise ValueError("timings.jsonl differs from validated records")
    count = len(records)
    summary = {
        "schema": SCHEMA, "state": "PAPER_READY" if paper_ready else "KIT_VALIDATED",
        "method": metadata["resume_identity"]["method"],
        "problem": "CVRPTW", "problem_size": metadata["resume_identity"]["problem_size"],
        "count": count, "validated_count": count, "failed_count": 0,
        "mean_independent_objective": sum(row["independent_objective"] for row in records) / count,
        "mean_instance_drop_percent": sum(row["instance_drop_percent"] for row in records) / count,
        "mean_runtime_seconds": sum(row["runtime_seconds"] for row in records) / count,
        "drop_definition": DROP_DEFINITION, "timing_semantics": TIMING_SEMANTICS,
        "dataset_sha256": metadata["resume_identity"]["dataset"]["sha256"],
        "checkpoint_sha256": metadata["resume_identity"]["checkpoint"]["sha256"],
        "records_sha256": sha256_file(output / RECORDS),
    }
    write_json(output / "summary.json", summary)
    metadata.update(state=summary["state"], completed_records=count,
                    validated_records_sha256=summary["records_sha256"],
                    timings_sha256=sha256_file(output / TIMINGS), completed_at=utc_now())
    write_json(metadata_path, metadata)
    return summary


def require_our5_gate(path, *, method: str, problem_size: int,
                      dataset_sha256: str, checkpoint_sha256: str):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    identity = metadata.get("resume_identity", {})
    expected_indices = list(range(5))
    records = read_jsonl(path / RECORDS)
    record_indices = []
    try:
        record_indices = [validate_record(row, problem_size=problem_size) for row in records]
    except ValueError:
        record_indices = []
    checks = {
        "schema": metadata.get("schema") == SCHEMA,
        "state": metadata.get("state") == "KIT_VALIDATED",
        "summary_state": summary.get("state") == "KIT_VALIDATED",
        "method": identity.get("method") == method,
        "problem_size": identity.get("problem_size") == problem_size,
        "scope": identity.get("scope") == "our_5",
        "indices": identity.get("chunk", {}).get("expected_indices") == expected_indices,
        "dataset": identity.get("dataset", {}).get("sha256") == dataset_sha256,
        "checkpoint": identity.get("checkpoint", {}).get("sha256") == checkpoint_sha256,
        "records_hash": metadata.get("validated_records_sha256") == sha256_file(path / RECORDS),
        "record_semantics": record_indices == expected_indices,
        "summary_count": summary.get("count") == 5 and summary.get("validated_count") == 5,
        "summary_dataset": summary.get("dataset_sha256") == dataset_sha256,
        "summary_checkpoint": summary.get("checkpoint_sha256") == checkpoint_sha256,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"our_5 production gate failed closed: {failed}")
    return checks


def require_our2_gate(path, *, method: str, problem_size: int,
                      dataset_sha256: str, checkpoint_sha256: str):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    identity = metadata.get("resume_identity", {})
    records = read_jsonl(path / RECORDS)
    indices = []
    try:
        indices = [validate_record(row, problem_size=problem_size) for row in records]
    except ValueError:
        indices = []
    checks = {
        "schema": metadata.get("schema") == SCHEMA,
        "state": metadata.get("state") == "KIT_VALIDATED",
        "method": identity.get("method") == method,
        "problem_size": identity.get("problem_size") == problem_size,
        "scope": identity.get("scope") == "our_2",
        "indices": indices == [0, 1],
        "dataset": identity.get("dataset", {}).get("sha256") == dataset_sha256,
        "checkpoint": identity.get("checkpoint", {}).get("sha256") == checkpoint_sha256,
        "records_hash": metadata.get("validated_records_sha256") == sha256_file(path / RECORDS),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"our_2 gate failed closed: {failed}")
    return metadata, records


def require_our5_prefix_matches_our2(our5_path, our2_metadata, our2_records):
    our5_path = Path(our5_path)
    our5_metadata = json.loads((our5_path / "metadata.json").read_text())
    identity5 = our5_metadata["resume_identity"]
    identity2 = our2_metadata["resume_identity"]
    for field in ("method", "variant", "problem", "problem_size", "project",
                  "upstream", "checkpoint", "dataset", "protocol", "environment",
                  "source_provenance", "timing_semantics"):
        if identity5.get(field) != identity2.get(field):
            raise ValueError(f"our_5 differs from our_2 provenance/protocol field {field}")
    records5 = read_jsonl(our5_path / RECORDS)
    if len(records5) < 2:
        raise ValueError("our_5 does not contain its first two records")
    stable_fields = (
        "dataset_instance_index", "instance_id", "raw_official_action",
        "canonical_solution", "selected_candidate", "official_reward",
        "reported_objective", "independent_objective", "kit_objective",
        "reference_objective", "instance_drop_percent", "independent_feasible",
        "kit_feasible", "reported_objective_agrees", "kit_objective_agrees",
        "route_count", "route_loads", "route_timelines", "adapter_mapping", "status",
    )
    for index in range(2):
        differences = [field for field in stable_fields
                       if records5[index].get(field) != our2_records[index].get(field)]
        if differences:
            raise ValueError(
                f"our_5 first-two semantic replay differs at index {index}: {differences}")
    return True
