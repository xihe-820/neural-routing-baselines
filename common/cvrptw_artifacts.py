"""Batch-aware artifact and gate helpers for remaining CVRPTW baselines."""
from __future__ import annotations

import json
import math
from pathlib import Path

from common.cvrptw_formal import DROP_DEFINITION, TIMING_SEMANTICS
from common.hashing import sha256_file
from common.paper_results import json_fingerprint, read_jsonl, utc_now, write_json

SCHEMA = "remaining-cvrptw-paper-v2"
RECORDS = "validated_records.jsonl"
TIMINGS = "batch_timings.jsonl"


def _batch_size(identity: dict) -> int:
    value = identity.get("protocol", {}).get(
        "original_instance_batch_size", identity.get("batch_size", 1))
    if isinstance(value, bool) or value not in (1, 10):
        raise ValueError("formal original-instance batch size must be 1 or 10")
    return value


def scope_instance_count(scope: str, batch_size: int, dataset_count: int) -> int:
    """Return an exact number of instances comprising only complete native batches."""
    if batch_size not in (1, 10):
        raise ValueError("formal original-instance batch size must be 1 or 10")
    counts = {
        "preflight": batch_size,
        "small_gate": 2 * batch_size,
        "validation_gate": 5 * batch_size,
        "production": dataset_count,
    }
    # These old names remain valid only for the historical batch-one flow.
    if batch_size == 1:
        counts.update(our_2=2, our_5=5)
    if scope not in counts:
        raise ValueError(f"scope {scope!r} is not valid for batch size {batch_size}")
    count = counts[scope]
    if count <= 0 or count % batch_size:
        raise ValueError("scope would contain a partial native inference batch")
    return count


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


def validate_batch_timing(row: dict, *, batch_size: int) -> int:
    required = ("batch_index", "dataset_indices", "batch_size", "runtime_seconds")
    if any(field not in row for field in required):
        raise ValueError("batch timing row is incomplete")
    batch_index = row["batch_index"]
    runtime = row["runtime_seconds"]
    if (isinstance(batch_index, bool) or not isinstance(batch_index, int) or
            batch_index < 0 or row["batch_size"] != batch_size or
            not isinstance(row["dataset_indices"], list) or
            len(row["dataset_indices"]) != batch_size or
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in row["dataset_indices"]) or
            isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or
            not math.isfinite(runtime) or runtime < 0):
        raise ValueError("batch timing row has invalid fields")
    return batch_index


def _load_progress(output: Path, identity: dict):
    records = read_jsonl(output / RECORDS) if (output / RECORDS).exists() else []
    timings = read_jsonl(output / TIMINGS) if (output / TIMINGS).exists() else []
    batch_size = _batch_size(identity)
    expected = identity["chunk"]["expected_indices"]
    if expected != list(range(len(expected))) or len(expected) % batch_size:
        raise ValueError("artifact expected indices must be a complete contiguous prefix")
    indices = [validate_record(row, problem_size=identity["problem_size"])
               for row in records]
    if indices != expected[:len(indices)]:
        raise ValueError("existing artifact is not an exact dataset-index prefix")
    if len(records) != len(timings) * batch_size:
        raise ValueError("existing artifact contains a partial/corrupt native batch")
    for batch_index, timing in enumerate(timings):
        if validate_batch_timing(timing, batch_size=batch_size) != batch_index:
            raise ValueError("batch timings are not a contiguous ordered prefix")
        first = batch_index * batch_size
        batch_indices = expected[first:first + batch_size]
        if timing["dataset_indices"] != batch_indices:
            raise ValueError("batch timing dataset indices differ from formal chunks")
        for position, record in enumerate(records[first:first + batch_size]):
            if (record.get("batch_index", batch_index) != batch_index or
                    record.get("position_in_batch", position) != position or
                    record["runtime_seconds"] != timing["runtime_seconds"]):
                raise ValueError("validated record and native-batch timing disagree")
    return records, timings


def initialize(output_dir, identity: dict):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    batch_size = _batch_size(identity)
    protocol_batch = identity.get("protocol", {}).get(
        "original_instance_batch_size", batch_size)
    if protocol_batch != batch_size:
        raise ValueError("identity and protocol batch sizes differ")
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
            "batch_timings_file": TIMINGS, "created_at": utc_now(),
            "completed_records": 0, "completed_batches": 0,
        }
        write_json(metadata_path, metadata)
    state = metadata.get("state")
    if state in ("KIT_VALIDATED", "PAPER_READY"):
        if (metadata.get("validated_records_sha256") != sha256_file(output / RECORDS) or
                metadata.get("batch_timings_sha256") != sha256_file(output / TIMINGS)):
            raise ValueError("resume refused: finalized artifact hashes changed")
    records, timings = _load_progress(output, identity)
    if (metadata.get("completed_records") not in (0, len(records)) or
            metadata.get("completed_batches") not in (0, len(timings))):
        raise ValueError("artifact metadata progress differs from batch files")
    return metadata, {row["dataset_instance_index"] for row in records}


def append_batch(output_dir, records: list[dict], timing: dict):
    output = Path(output_dir)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != SCHEMA or metadata.get("state") != "INFERENCE_IN_PROGRESS":
        raise ValueError("cannot append to a non-active CVRPTW artifact")
    identity = metadata["resume_identity"]
    existing_records, existing_timings = _load_progress(output, identity)
    batch_size = _batch_size(identity)
    batch_index = len(existing_timings)
    expected = identity["chunk"]["expected_indices"]
    expected_indices = expected[len(existing_records):len(existing_records) + batch_size]
    if len(records) != batch_size or len(expected_indices) != batch_size:
        raise ValueError("append must contain one complete native inference batch")
    if validate_batch_timing(timing, batch_size=batch_size) != batch_index:
        raise ValueError("appended timing has the wrong batch index")
    if timing["dataset_indices"] != expected_indices:
        raise ValueError("appended timing has the wrong dataset indices")
    normalized = []
    for position, record in enumerate(records):
        record = dict(record)
        record.setdefault("batch_index", batch_index)
        record.setdefault("position_in_batch", position)
        record.setdefault("runtime_seconds", timing["runtime_seconds"])
        if (validate_record(record, problem_size=identity["problem_size"]) !=
                expected_indices[position] or record["batch_index"] != batch_index or
                record["position_in_batch"] != position or
                record["runtime_seconds"] != timing["runtime_seconds"]):
            raise ValueError("appended record does not match its complete native batch")
        normalized.append(record)
    with (output / RECORDS).open("a") as stream:
        for record in normalized:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    with (output / TIMINGS).open("a") as stream:
        stream.write(json.dumps(timing, sort_keys=True, separators=(",", ":"),
                                allow_nan=False) + "\n")
    metadata.update(completed_records=len(existing_records) + batch_size,
                    completed_batches=batch_index + 1)
    write_json(metadata_path, metadata)


def append(output_dir, record: dict):
    """Historical batch-one API retained for CaDA and existing callers."""
    output = Path(output_dir)
    metadata = json.loads((output / "metadata.json").read_text())
    identity = metadata["resume_identity"]
    if _batch_size(identity) != 1:
        raise ValueError("append(record) is only valid for batch size 1")
    _, timings = _load_progress(output, identity)
    timing = {
        "batch_index": len(timings),
        "dataset_indices": [record["dataset_instance_index"]],
        "batch_size": 1,
        "runtime_seconds": record["runtime_seconds"],
    }
    append_batch(output, [record], timing)


def finalize(output_dir, *, paper_ready: bool = False):
    output = Path(output_dir)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    identity = metadata["resume_identity"]
    records, timings = _load_progress(output, identity)
    expected = identity["chunk"]["expected_indices"]
    indices = [row["dataset_instance_index"] for row in records]
    if indices != expected:
        raise ValueError("artifact has duplicate, missing, extra, or reordered dataset indices")
    expected_count = identity["dataset"]["count"]
    if paper_ready and indices != list(range(expected_count)):
        raise ValueError("PAPER_READY requires exact full-dataset coverage")
    if not records or not timings:
        raise ValueError("cannot finalize an empty artifact")
    batch_size = _batch_size(identity)
    runtimes = [row["runtime_seconds"] for row in timings]
    count = len(records)
    mean_runtime = sum(runtimes) / len(runtimes)
    summary = {
        "schema": SCHEMA, "state": "PAPER_READY" if paper_ready else "KIT_VALIDATED",
        "method": identity["method"], "problem": "CVRPTW",
        "problem_size": identity["problem_size"],
        "count": count, "num_instances": count, "validated_count": count,
        "failed_count": 0, "batch_size": batch_size, "num_batches": len(timings),
        "mean_independent_objective": sum(row["independent_objective"] for row in records) / count,
        "mean_instance_drop_percent": sum(row["instance_drop_percent"] for row in records) / count,
        "mean_batch_runtime_seconds": mean_runtime,
        "mean_runtime_seconds": mean_runtime,
        "mean_runtime_seconds_semantics": "mean native inference-batch latency; never divided by batch size",
        "total_runtime_seconds": sum(runtimes),
        "drop_definition": DROP_DEFINITION, "timing_semantics": TIMING_SEMANTICS,
        "dataset_sha256": identity["dataset"]["sha256"],
        "checkpoint_sha256": identity["checkpoint"]["sha256"],
        "records_sha256": sha256_file(output / RECORDS),
        "batch_timings_sha256": sha256_file(output / TIMINGS),
    }
    write_json(output / "summary.json", summary)
    metadata.update(state=summary["state"], completed_records=count,
                    completed_batches=len(timings),
                    validated_records_sha256=summary["records_sha256"],
                    batch_timings_sha256=summary["batch_timings_sha256"],
                    completed_at=utc_now())
    write_json(metadata_path, metadata)
    return summary


def _require_gate(path, *, method: str, problem_size: int, dataset_sha256: str,
                  checkpoint_sha256: str, batch_size: int, scope: str,
                  expected_batches: int):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    identity = metadata.get("resume_identity", {})
    expected_indices = list(range(batch_size * expected_batches))
    records = read_jsonl(path / RECORDS)
    timings = read_jsonl(path / TIMINGS)
    try:
        record_indices = [validate_record(row, problem_size=problem_size) for row in records]
        timing_indices = [validate_batch_timing(row, batch_size=batch_size)
                          for row in timings]
    except ValueError:
        record_indices, timing_indices = [], []
    checks = {
        "schema": metadata.get("schema") == SCHEMA,
        "state": metadata.get("state") == "KIT_VALIDATED",
        "summary_state": summary.get("state") == "KIT_VALIDATED",
        "method": identity.get("method") == method,
        "problem_size": identity.get("problem_size") == problem_size,
        "scope": identity.get("scope") == scope,
        "batch_size": (identity.get("protocol", {}).get("original_instance_batch_size", 1)
                       == batch_size and summary.get("batch_size") == batch_size),
        "indices": identity.get("chunk", {}).get("expected_indices") == expected_indices,
        "dataset": identity.get("dataset", {}).get("sha256") == dataset_sha256,
        "checkpoint": identity.get("checkpoint", {}).get("sha256") == checkpoint_sha256,
        "records_hash": metadata.get("validated_records_sha256") == sha256_file(path / RECORDS),
        "timings_hash": metadata.get("batch_timings_sha256") == sha256_file(path / TIMINGS),
        "record_semantics": record_indices == expected_indices,
        "timing_semantics": timing_indices == list(range(expected_batches)),
        "summary_count": (summary.get("num_instances") == len(expected_indices) and
                          summary.get("num_batches") == expected_batches),
        "summary_dataset": summary.get("dataset_sha256") == dataset_sha256,
        "summary_checkpoint": summary.get("checkpoint_sha256") == checkpoint_sha256,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{scope} gate failed closed: {failed}")
    return metadata, records


def require_small_gate(path, *, method: str, problem_size: int,
                       dataset_sha256: str, checkpoint_sha256: str,
                       batch_size: int):
    return _require_gate(
        path, method=method, problem_size=problem_size,
        dataset_sha256=dataset_sha256, checkpoint_sha256=checkpoint_sha256,
        batch_size=batch_size, scope="small_gate", expected_batches=2)


def require_validation_gate(path, *, method: str, problem_size: int,
                            dataset_sha256: str, checkpoint_sha256: str,
                            batch_size: int):
    return _require_gate(
        path, method=method, problem_size=problem_size,
        dataset_sha256=dataset_sha256, checkpoint_sha256=checkpoint_sha256,
        batch_size=batch_size, scope="validation_gate", expected_batches=5)


def _prefix_matches(larger_path, smaller_metadata, smaller_records,
                    *, larger_scope: str, prefix_label: str):
    larger_path = Path(larger_path)
    larger_metadata = json.loads((larger_path / "metadata.json").read_text())
    identity_larger = larger_metadata["resume_identity"]
    identity_smaller = smaller_metadata["resume_identity"]
    for field in ("method", "variant", "problem", "problem_size", "project",
                  "upstream", "checkpoint", "dataset", "protocol", "environment",
                  "source_provenance", "timing_semantics"):
        if identity_larger.get(field) != identity_smaller.get(field):
            raise ValueError(
                f"{larger_scope} differs from {prefix_label} provenance/protocol field {field}")
    larger_records = read_jsonl(larger_path / RECORDS)
    if len(larger_records) < len(smaller_records):
        raise ValueError(f"{larger_scope} does not contain its {prefix_label} prefix")
    stable_fields = (
        "dataset_instance_index", "instance_id", "raw_official_action",
        "canonical_solution", "selected_candidate", "official_reward",
        "reported_objective", "independent_objective", "kit_objective",
        "reference_objective", "instance_drop_percent", "independent_feasible",
        "kit_feasible", "reported_objective_agrees", "kit_objective_agrees",
        "route_count", "route_loads", "route_timelines", "adapter_mapping", "status",
    )
    for index, smaller in enumerate(smaller_records):
        differences = [field for field in stable_fields
                       if larger_records[index].get(field) != smaller.get(field)]
        if differences:
            raise ValueError(
                f"{larger_scope} semantic replay differs at index {index}: {differences}")
    return True


def require_validation_prefix_matches_small(validation_path, small_metadata, small_records):
    return _prefix_matches(
        validation_path, small_metadata, small_records,
        larger_scope="validation_gate", prefix_label="small_gate")


# Historical batch-one gate APIs retained for CaDA and old commands.
def require_our5_gate(path, *, method: str, problem_size: int,
                      dataset_sha256: str, checkpoint_sha256: str):
    return _require_gate(
        path, method=method, problem_size=problem_size,
        dataset_sha256=dataset_sha256, checkpoint_sha256=checkpoint_sha256,
        batch_size=1, scope="our_5", expected_batches=5)[0]


def require_our2_gate(path, *, method: str, problem_size: int,
                      dataset_sha256: str, checkpoint_sha256: str):
    return _require_gate(
        path, method=method, problem_size=problem_size,
        dataset_sha256=dataset_sha256, checkpoint_sha256=checkpoint_sha256,
        batch_size=1, scope="our_2", expected_batches=2)


def require_our5_prefix_matches_our2(our5_path, our2_metadata, our2_records):
    return _prefix_matches(
        our5_path, our2_metadata, our2_records,
        larger_scope="our_5", prefix_label="our_2")
