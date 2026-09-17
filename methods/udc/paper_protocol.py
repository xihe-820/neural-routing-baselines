"""Frozen UDC paper budgets, formal scope, and production provenance gates."""
from __future__ import annotations

import json
from pathlib import Path

from common.hashing import sha256_file
from methods.udc.protocol import (FORMAL_SIZES, OFFICIAL_COMMIT,
                                  SCALE_DATASET_FILENAMES, S3_IMPLEMENTATION_COMMIT,
                                  _git)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "manifests/paper_budgets.json"
REGISTRY_SHA256 = "60c7f630dca7a7b09c28042830f16320bfd8cdcdded52b9e9e442c36886a2cf4"
PRODUCTION_BASE_COMMIT = "99b85d299987d7275c45f75d08dad038759d10ab"
S4_EVIDENCE_REVISION = PRODUCTION_BASE_COMMIT[:8]
PILOT_INDICES = (0, 1, 2)
SAFETY_CELLS = (("tsp", 10000, "more"), ("cvrp", 2000, "more"))
PROVENANCE_LABEL = "PROJECT-SELECTED PAPER BUDGETS"

PRODUCTION_TRACKED_ALLOWLIST = {
    "docs/BASELINE_PAPER_BUDGETS.md",
    "docs/SERVER_RUNBOOK.md",
    "manifests/paper_budgets.json",
    "methods/udc/build_paper_table.py",
    "methods/udc/paper_production.py",
    "methods/udc/paper_protocol.py",
    "methods/udc/paper_results.py",
    "methods/udc/s3_eval.py",
    "tests/test_udc_production.py",
}

EXPECTED_BUDGETS = {
    ("tsp", "fewer"): {"alpha": 50, "x": 2, "sub_size": 100,
                         "configured_pomo": 2, "effective_pomo": 2},
    ("tsp", "more"): {"alpha": 50, "x": 50, "sub_size": 100,
                        "configured_pomo": 2, "effective_pomo": 2},
    ("cvrp", "fewer"): {"alpha": 50, "x": 50, "sub_size": 100,
                          "configured_pomo": 10, "effective_pomo": 1},
    ("cvrp", "more"): {"alpha": 50, "x": 250, "sub_size": 100,
                         "configured_pomo": 10, "effective_pomo": 1},
}


def formal_cells():
    return tuple((problem, size, label)
                 for problem, sizes in FORMAL_SIZES.items()
                 for size in sizes for label in ("fewer", "more"))


def load_budget_registry(path: Path = REGISTRY_PATH) -> dict:
    path = Path(path)
    actual = sha256_file(path)
    if actual != REGISTRY_SHA256:
        raise ValueError("UDC paper budget registry SHA256 mismatch")
    raw = json.loads(path.read_text())
    if (raw.get("schema") != "paper-budgets.v1" or raw.get("method") != "UDC"
            or raw.get("provenance_label") != PROVENANCE_LABEL):
        raise ValueError("UDC paper budget registry header differs from frozen schema")
    entries = raw.get("entries")
    if not isinstance(entries, list) or len(entries) != 4:
        raise ValueError("UDC paper budget registry must contain exactly four entries")
    observed = {}
    required = {
        "method", "problem", "label", "alpha", "x", "sub_size",
        "configured_pomo", "effective_pomo", "seed", "batch_size",
        "config_origin", "selection_origin", "official_source_commit",
        "project_budget_rationale", "pilot_artifact", "freeze_commit", "freeze_date",
    }
    for entry in entries:
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise ValueError("UDC paper budget registry entry is incomplete")
        key = (str(entry["problem"]).lower(), entry["label"])
        if key in observed or key not in EXPECTED_BUDGETS:
            raise ValueError("UDC paper budget registry has an unexpected or duplicate entry")
        expected = EXPECTED_BUDGETS[key]
        for field, value in expected.items():
            if entry.get(field) != value:
                raise ValueError(f"UDC frozen budget changed: {key} {field}")
        if (entry["method"] != "UDC" or entry["seed"] != 1234
                or entry["batch_size"] != 1
                or entry["selection_origin"] != PROVENANCE_LABEL
                or entry["official_source_commit"] != OFFICIAL_COMMIT):
            raise ValueError(f"UDC frozen budget provenance changed: {key}")
        observed[key] = entry
    if set(observed) != set(EXPECTED_BUDGETS):
        raise ValueError("UDC paper budget registry scope differs from four frozen budgets")
    return {"path": str(path.resolve()), "sha256": actual,
            "raw": raw, "entries": observed}


def budget_config(problem: str, label: str, path: Path = REGISTRY_PATH) -> dict:
    key = (problem.lower(), label)
    registry = load_budget_registry(path)
    if key not in registry["entries"]:
        raise ValueError(f"outside frozen UDC paper budgets: {key}")
    return dict(registry["entries"][key])


def production_project_gate(root: Path) -> dict:
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    dirty = bool(_git(root, "status", "--porcelain=v1").stdout.strip())
    ancestor = _git(root, "merge-base", "--is-ancestor",
                    PRODUCTION_BASE_COMMIT, "HEAD", ok=(0, 1)).returncode == 0
    changed = [line for line in _git(
        root, "diff", "--name-only", f"{PRODUCTION_BASE_COMMIT}..HEAD"
    ).stdout.splitlines() if line]
    outside = sorted(set(changed) - PRODUCTION_TRACKED_ALLOWLIST)
    result = {
        "head": head, "dirty": dirty,
        "s4_implementation_base_commit": PRODUCTION_BASE_COMMIT,
        "base_is_ancestor": ancestor, "base_to_head_changed_paths": changed,
        "allowed_paths": sorted(PRODUCTION_TRACKED_ALLOWLIST),
        "disallowed_changed_paths": outside,
    }
    result["pass"] = not dirty and ancestor and not outside
    return result


def s4_gate(path: Path) -> dict:
    path = Path(path)
    if (path.name != "s4_scale_preflight"
            or path.parent.name != S4_EVIDENCE_REVISION):
        raise ValueError("S4 evidence path is not authoritative 99b85d29 evidence")
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("schema") != "udc_s4_scale_summary.v1":
        raise ValueError("S4 summary schema mismatch")
    rows = summary.get("rows")
    expected = {(problem, size) for problem, sizes in FORMAL_SIZES.items()
                for size in sizes}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("S4 summary does not contain all ten formal cells")
    observed = set()
    artifacts = []
    for row in rows:
        key = (row.get("problem"), row.get("size"))
        if key not in expected or key in observed or row.get("status") != "PASS":
            raise ValueError("S4 formal cell is missing, duplicated, or not PASS")
        observed.add(key)
        problem, size = key
        if Path(row.get("dataset", "")).name != SCALE_DATASET_FILENAMES[key]:
            raise ValueError("S4 dataset identity differs from formal mapping")
        directory = path / f"{problem}{size}"
        metadata_path, record_path = directory / "metadata.json", directory / "record.json"
        metadata = json.loads(metadata_path.read_text())
        record = json.loads(record_path.read_text())
        if (metadata.get("status") != "PASS" or record.get("status") != "PASS"
                or metadata.get("record_sha256") != sha256_file(record_path)):
            raise ValueError("S4 per-size artifact integrity failed")
        dataset_sha = row.get("dataset_sha256")
        if (not isinstance(dataset_sha, str) or len(dataset_sha) != 64
                or metadata.get("dataset", {}).get("sha256") != dataset_sha
                or record.get("dataset_sha256") != dataset_sha):
            raise ValueError("S4 dataset SHA provenance mismatch")
        for name in ("project_pre", "project_post"):
            gate = metadata.get(name, {})
            if gate.get("head") != PRODUCTION_BASE_COMMIT or gate.get("pass") is not True:
                raise ValueError("S4 project provenance mismatch")
        for name in ("official_pre", "official_post"):
            gate = metadata.get(name, {})
            if gate.get("head") != OFFICIAL_COMMIT or gate.get("pass") is not True:
                raise ValueError("S4 official provenance mismatch")
        artifacts.append({"problem": problem, "size": size,
                          "metadata_sha256": sha256_file(metadata_path),
                          "record_sha256": sha256_file(record_path)})
    if observed != expected:
        raise ValueError("S4 formal scope mismatch")
    return {"path": str(path.resolve()), "summary_path": str(summary_path.resolve()),
            "summary_sha256": sha256_file(summary_path), "artifacts": artifacts,
            "s3_implementation_commit": S3_IMPLEMENTATION_COMMIT,
            "s4_implementation_commit": PRODUCTION_BASE_COMMIT, "pass": True}
