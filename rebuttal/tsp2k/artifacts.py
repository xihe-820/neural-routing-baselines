"""Hash-bound measurements with exact coverage; partial/failed runs cannot aggregate."""
import json
import math
from pathlib import Path
import statistics

from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from methods.glop.paper_results import fingerprint, write_json
from rebuttal.tsp2k.config import COUNT, DATASET, UPSTREAMS, protocol

SCHEMA = "tsp2k-runtime-quality.v1"


def number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"non-finite/non-numeric {name}")
    if positive and value <= 0:
        raise ValueError(f"non-positive {name}")
    return float(value)


def indices_for(phase, count=None):
    if phase not in ("fullset", "calibration"):
        raise ValueError("invalid phase")
    if phase == "fullset":
        if count not in (None, COUNT):
            raise ValueError("fullset must cover exactly 64 instances")
        return list(range(COUNT))
    count = 2 if count is None else count
    if type(count) is not int or count < 1 or count > COUNT:
        raise ValueError("invalid calibration count")
    return list(range(count))


def check_identity(identity):
    config = protocol(identity["method"], identity["budget_value"])
    if any(identity.get(k) != v for k, v in config.items()):
        raise ValueError("budget/protocol fingerprint mismatch")
    dataset = identity["dataset"]
    if (dataset.get("count") != COUNT or Path(dataset.get("path", "")).name != DATASET or
            len(dataset.get("sha256", "")) != 64 or
            len(dataset.get("reference_objectives_sha256", "")) != 64 or
            any(c not in "0123456789abcdef" for c in dataset["sha256"])):
        raise ValueError("dataset identity mismatch")
    if (identity["project"].get("dirty") is not False or
            identity.get("project_commit") != identity["project"].get("commit") or
            len(identity["project_commit"]) != 40 or
            identity["upstream"].get("dirty") is not False or
            identity.get("upstream_commit") != UPSTREAMS[identity["method"]][1] or
            identity["upstream"].get("commit") != identity["upstream_commit"] or
            "RTX 4090" not in identity.get("gpu", "") or
            not identity.get("checkpoint_sha256") or
            not identity.get("source_provenance")):
        raise ValueError("missing/dirty/unapproved execution provenance")
    environment = identity.get("environment", {})
    if (environment.get("gpu") != identity["gpu"] or
            not str(environment.get("device", "")).startswith("cuda") or
            not all(environment.get(key) for key in ("python", "torch", "numpy"))):
        raise ValueError("missing or conflicting actual environment provenance")
    for digest in identity["checkpoint_sha256"].values():
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid checkpoint SHA256")
    expected = indices_for(identity["phase"], identity["requested_count"])
    if identity.get("dataset_indices") != expected:
        raise ValueError("dataset index identity mismatch")
    method = identity["method"]
    timing = identity["timing_indices"]
    if method == "SIL":
        if not timing or timing != list(range(len(timing))) or len(timing) > len(expected):
            raise ValueError("SIL timing must be an exact fixed dataset prefix")
        if identity["actual_quality_batch_size"] != len(expected):
            raise ValueError("SIL must use one real author-style quality batch")
    elif timing or identity["actual_quality_batch_size"] != 1:
        raise ValueError("BS1 batch identity mismatch")
    return config


def check_record(row):
    if type(row.get("instance_index")) is not int:
        raise ValueError("invalid instance index")
    if (row.get("feasible") is not True or row.get("kit_feasible") is not True or
            row.get("official_objective_agrees") is not True or
            row.get("kit_objective_agrees") is not True):
        raise ValueError("validation gate failed")
    tour = row.get("canonical_solution")
    if not isinstance(tour, list) or any(type(v) is not int for v in tour):
        raise ValueError("missing or invalid actual node tour")
    order = tour[:-1] if len(tour) == 2001 and tour[0] == tour[-1] else tour
    if sorted(order) != list(range(2000)):
        raise ValueError("missing or invalid actual node tour")
    obj = number(row.get("objective"), "objective", positive=True)
    ref = number(row.get("reference_objective"), "reference", positive=True)
    gap = number(row.get("gap_percent"), "gap")
    official = number(row.get("official_objective"), "official", positive=True)
    kit = number(row.get("kit_objective"), "Kit", positive=True)
    if (not objective_agrees(gap, (obj-ref)/ref*100) or
            not objective_agrees(official, obj) or not objective_agrees(kit, obj)):
        raise ValueError("objective/reference/gap agreement failed")
    if row.get("runtime_seconds") is not None:
        number(row["runtime_seconds"], "runtime", positive=True)


def summarize(identity, records):
    method, phase = identity["method"], identity["phase"]
    config = check_identity(identity)
    expected = indices_for(phase, identity["requested_count"])
    quality = [r for r in records if r.get("source") == "quality"]
    timing = [r for r in records if r.get("source") == "timing"]
    if len(quality) + len(timing) != len(records):
        raise ValueError("unknown record source")
    for rows, required in ((quality, expected), (timing, identity["timing_indices"])):
        if [r.get("instance_index") for r in rows] != required:
            raise ValueError("missing, reordered, duplicate or extra instance")
        for row in rows:
            check_record(row)
    if method == "GLOP":
        for row in quality:
            parts = row.get("runtime_components", {})
            solve = number(parts.get("per_instance_solve_seconds"), "solver time", positive=True)
            shared = number(parts.get("shared_ri_order_generation_seconds"), "shared RI time")
            if shared < 0 or (row["instance_index"] != 0 and shared != 0):
                raise ValueError("shared RI order timing must only charge index zero")
            if not objective_agrees(row["runtime_seconds"], solve + shared):
                raise ValueError("runtime component sum mismatch")
    measured = timing if method == "SIL" else quality
    if method != "SIL" and timing:
        raise ValueError("BS1 method must use its measured quality records")
    if method == "SIL" and any(r["runtime_seconds"] is not None for r in quality):
        raise ValueError("author batch quality must not be reported as BS1 timing")
    runtimes = [number(r.get("runtime_seconds"), "runtime", positive=True)
                for r in measured]
    if not runtimes:
        raise ValueError("no measured BS1 timing")
    return {
        **config, "phase": phase, "status": (
            "MEASURED_FULLSET" if phase == "fullset" else "MEASURED_TIMING_SUBSET"),
        "paper_ready": False, "dataset_sha256": identity["dataset"]["sha256"],
        "dataset_indices": expected,
        "mean_objective": statistics.fmean(r["objective"] for r in quality),
        "mean_drop_percent": statistics.fmean(r["gap_percent"] for r in quality),
        "mean_time_seconds": statistics.fmean(runtimes),
        "total_time_seconds": sum(runtimes), "quality_instance_count": len(quality),
        "timing_sample_count": len(measured),
        "quality_batch_size": identity["actual_quality_batch_size"],
        "quality_source": "inference_records.jsonl:quality",
        "time_source": "inference_records.jsonl:" + ("timing" if method == "SIL" else "quality"),
        "validation_status": "KIT_VALIDATED", "reuse_status": "NEW_MEASUREMENT",
        "identity_fingerprint": fingerprint(identity),
    }


class Writer:
    def __init__(self, directory, identity):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.identity, self.rows = identity, []
        self.metadata = {"schema": SCHEMA, "state": "IN_PROGRESS",
                         "identity": identity, "identity_fingerprint": fingerprint(identity)}
        self.save()

    def save(self):
        write_json(self.directory / "metadata.json", self.metadata)

    def append(self, row):
        check_record(row)
        with (self.directory / "inference_records.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.rows.append(row)

    def fail(self, error):
        self.metadata.update(state="FAILED", error=str(error))
        self.save()

    def finish(self):
        summary = summarize(self.identity, self.rows)
        summary["artifact_path"] = str(self.directory)
        write_json(self.directory / "summary.json", summary)
        self.metadata.update(state="KIT_VALIDATED",
                             records_sha256=sha256_file(self.directory / "inference_records.jsonl"),
                             summary_sha256=sha256_file(self.directory / "summary.json"))
        self.save()
        return summary


def read_artifact(directory):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    identity = meta["identity"]
    if (meta.get("schema") != SCHEMA or meta.get("state") != "KIT_VALIDATED" or
            meta.get("identity_fingerprint") != fingerprint(identity)):
        raise ValueError("invalid measurement metadata/fingerprint/state")
    for filename, field in (("inference_records.jsonl", "records_sha256"),
                            ("summary.json", "summary_sha256")):
        if sha256_file(directory / filename) != meta.get(field):
            raise ValueError("measurement artifact hash mismatch")
    records = [json.loads(line) for line in
               (directory / "inference_records.jsonl").read_text().splitlines()]
    expected = summarize(identity, records)
    summary = json.loads((directory / "summary.json").read_text())
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError("measurement summary differs from actual records")
    return identity, records, summary
