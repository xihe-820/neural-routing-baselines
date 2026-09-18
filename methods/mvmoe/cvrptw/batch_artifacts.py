"""Complete-native-batch resume and timing contract for MVMoE CVRPTW."""
from __future__ import annotations

import json
import math
from pathlib import Path

from common.hashing import sha256_file
from common.paper_results import (METADATA_FILE, RECORDS_FILE, finalize_chunk,
                                  initialize_chunk, read_jsonl, validate_record,
                                  write_json)

BATCH_TIMINGS_FILE = "batch_timings.jsonl"


def _batch_size(identity):
    value = identity.get("paper_protocol", {}).get("original_batch_size")
    if value not in (1, 10):
        raise ValueError("MVMoE artifact requires formal original batch size 1 or 10")
    return value


def validate_batch_timing(row, *, batch_size):
    runtime = row.get("runtime_seconds")
    if (isinstance(row.get("batch_index"), bool) or
            not isinstance(row.get("batch_index"), int) or row["batch_index"] < 0 or
            row.get("batch_size") != batch_size or
            not isinstance(row.get("dataset_indices"), list) or
            len(row["dataset_indices"]) != batch_size or
            any(isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in row["dataset_indices"]) or
            isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or
            not math.isfinite(runtime) or runtime < 0):
        raise ValueError("invalid MVMoE native-batch timing row")
    return row["batch_index"]


def load_batch_progress(output_dir, identity):
    output = Path(output_dir)
    expected = identity["chunk"]["expected_indices"]
    batch_size = _batch_size(identity)
    if (not expected or expected != list(range(expected[0], expected[0] + len(expected))) or
            expected[0] % batch_size or len(expected) % batch_size):
        raise ValueError("MVMoE chunk must align to complete formal native batches")
    records = read_jsonl(output / RECORDS_FILE) if (output / RECORDS_FILE).exists() else []
    timings = (read_jsonl(output / BATCH_TIMINGS_FILE)
               if (output / BATCH_TIMINGS_FILE).exists() else [])
    indices = [validate_record(record) for record in records]
    if indices != expected[:len(indices)] or len(records) != len(timings) * batch_size:
        raise ValueError("MVMoE resume contains a partial, reordered, or corrupt batch")
    for local_batch_index, timing in enumerate(timings):
        if validate_batch_timing(timing, batch_size=batch_size) != local_batch_index:
            raise ValueError("MVMoE batch timings are not an ordered chunk-local prefix")
        first = local_batch_index * batch_size
        expected_indices = expected[first:first + batch_size]
        if timing["dataset_indices"] != expected_indices:
            raise ValueError("MVMoE batch timing indices differ from chunk identity")
        for position, record in enumerate(records[first:first + batch_size]):
            if (record.get("batch_index") != local_batch_index or
                    record.get("position_in_batch") != position or
                    record["runtime_seconds"] != timing["runtime_seconds"]):
                raise ValueError("MVMoE records disagree with native-batch timing")
    return records, timings


def initialize_batch_chunk(output_dir, identity):
    metadata, _ = initialize_chunk(output_dir, identity)
    records, timings = load_batch_progress(output_dir, identity)
    if metadata.get("batch_timings_file", BATCH_TIMINGS_FILE) != BATCH_TIMINGS_FILE:
        raise ValueError("MVMoE artifact has an unexpected batch-timing file")
    finalized_hash = metadata.get("batch_timings_sha256")
    if finalized_hash is not None and finalized_hash != sha256_file(
            Path(output_dir) / BATCH_TIMINGS_FILE):
        raise ValueError("MVMoE finalized native-batch timings changed")
    return metadata, {record["dataset_instance_index"] for record in records}, len(timings)


def append_batch_records(output_dir, records, timing):
    output = Path(output_dir)
    metadata = json.loads((output / METADATA_FILE).read_text())
    if metadata.get("state") != "INFERENCE_IN_PROGRESS":
        raise ValueError("cannot append to a finalized MVMoE chunk")
    identity = metadata["resume_identity"]
    old_records, old_timings = load_batch_progress(output, identity)
    batch_size = _batch_size(identity)
    expected = identity["chunk"]["expected_indices"]
    expected_indices = expected[len(old_records):len(old_records) + batch_size]
    if len(records) != batch_size or len(expected_indices) != batch_size:
        raise ValueError("MVMoE append must contain one complete native batch")
    if (validate_batch_timing(timing, batch_size=batch_size) != len(old_timings) or
            timing["dataset_indices"] != expected_indices):
        raise ValueError("MVMoE appended timing does not match the next native batch")
    normalized = []
    for position, record in enumerate(records):
        record = dict(record)
        if (validate_record(record) != expected_indices[position] or
                record.get("batch_index") != len(old_timings) or
                record.get("position_in_batch") != position or
                record["runtime_seconds"] != timing["runtime_seconds"]):
            raise ValueError("MVMoE appended record does not match its native batch")
        normalized.append(record)
    with (output / RECORDS_FILE).open("a") as stream:
        for record in normalized:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    with (output / BATCH_TIMINGS_FILE).open("a") as stream:
        stream.write(json.dumps(timing, sort_keys=True, separators=(",", ":"),
                                allow_nan=False) + "\n")


def finalize_batch_chunk(output_dir):
    output = Path(output_dir)
    metadata = json.loads((output / METADATA_FILE).read_text())
    records, timings = load_batch_progress(output, metadata["resume_identity"])
    if len(records) != len(metadata["resume_identity"]["chunk"]["expected_indices"]):
        raise ValueError("MVMoE batch chunk is incomplete")
    metadata = finalize_chunk(output)
    metadata.update(
        batch_timings_file=BATCH_TIMINGS_FILE,
        completed_batches=len(timings),
        batch_timings_sha256=sha256_file(output / BATCH_TIMINGS_FILE),
    )
    write_json(output / METADATA_FILE, metadata)
    return metadata
