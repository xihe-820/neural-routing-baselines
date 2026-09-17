"""Frozen UDC Stage S3 protocol and fail-closed provenance helpers."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from common.hashing import sha256_file

FORMAL_PIPELINE_BASE_COMMIT = "15d02f5b95ecb418c19f12dacbe6a2acc7516fed"
OFFICIAL_COMMIT = "274df3c4975384592b60fe7f79fbb2441ce11c15"
SEED = 1234
ALPHA = 50
S1_SHA256 = "fb92785051c8f317e62c642477bb0745fb83f2a781c544dce5bff370f8d34f0a"
S2_SCRIPT_SHA256 = "c5987d75c70f38d992ad866731baa9c1230d045fb028389cdc0ca8e46da93047"
DATASET_FILENAMES = {
    "tsp": "tsp500_concorde_16.546.pkl",
    "cvrp": "cvrp500_hgs-300s_37.154.pkl",
}
SPECS = {
    "tsp": {"x": 2, "configured_pomo": 2, "effective_pomo": 2},
    "cvrp": {"x": 50, "configured_pomo": 10, "effective_pomo": 1},
}
S3_TRACKED_ALLOWLIST = {
    "methods/udc/adapter.py", "methods/udc/protocol.py",
    "methods/udc/s3_eval.py", "tests/test_udc_s3.py",
    "docs/SERVER_RUNBOOK.md",
}


def _git(root: Path, *args: str, ok=(0,)) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False)
    if result.returncode not in ok:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def project_gate(root: Path) -> dict:
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    dirty = bool(_git(root, "status", "--porcelain=v1").stdout.strip())
    ancestor = _git(root, "merge-base", "--is-ancestor",
                    FORMAL_PIPELINE_BASE_COMMIT, "HEAD", ok=(0, 1)).returncode == 0
    changed = [line for line in _git(
        root, "diff", "--name-only", f"{FORMAL_PIPELINE_BASE_COMMIT}..HEAD"
    ).stdout.splitlines() if line]
    outside = sorted(set(changed) - S3_TRACKED_ALLOWLIST)
    result = {"head": head, "dirty": dirty,
              "formal_pipeline_base_commit": FORMAL_PIPELINE_BASE_COMMIT,
              "base_is_ancestor": ancestor, "base_to_head_changed_paths": changed,
              "allowed_paths": sorted(S3_TRACKED_ALLOWLIST),
              "disallowed_changed_paths": outside}
    result["pass"] = not dirty and ancestor and not outside
    return result


def official_gate(root: Path) -> dict:
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    dirty = bool(_git(root, "status", "--porcelain=v1").stdout.strip())
    return {"head": head, "expected": OFFICIAL_COMMIT, "dirty": dirty,
            "pass": head == OFFICIAL_COMMIT and not dirty}


def discover_datasets(root: Path) -> dict[str, Path]:
    result = {}
    for family, filename in DATASET_FILENAMES.items():
        matches = sorted(path.resolve() for path in root.rglob(filename)
                         if path.is_file())
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {filename} below {root}; found {matches}")
        result[family] = matches[0]
    return result


def s1_gate(path: Path) -> dict:
    if path.name != "s1_audit_v5.json":
        raise ValueError("S1 evidence filename must be s1_audit_v5.json")
    actual = sha256_file(path)
    if actual != S1_SHA256:
        raise ValueError("S1 authoritative evidence SHA256 mismatch")
    return {"path": str(path.resolve()), "sha256": actual, "pass": True}


def s2_gate(path: Path) -> dict:
    if path.name != "s2_official_smoke_v3" or not path.is_dir():
        raise ValueError("S2 evidence must be the authoritative s2_official_smoke_v3 directory")
    metadata_path = path / "metadata.json"
    raw = json.loads(metadata_path.read_text())
    checks = {
        "schema": raw.get("schema") == "udc_s2_official_smoke.v1",
        "ready": raw.get("ready_for_stage_s3_ml4co_adapter") == "YES",
        "script": raw.get("script_sha256") == S2_SCRIPT_SHA256,
        "project_pre": raw.get("project_pre", {}).get("head") == FORMAL_PIPELINE_BASE_COMMIT
                       and raw.get("project_pre", {}).get("pass") is True,
        "project_post": raw.get("project_post", {}).get("head") == FORMAL_PIPELINE_BASE_COMMIT
                        and raw.get("project_post", {}).get("pass") is True,
        "official_pre": raw.get("official_pre", {}).get("head") == OFFICIAL_COMMIT
                        and raw.get("official_pre", {}).get("pass") is True,
        "official_post": raw.get("official_post", {}).get("head") == OFFICIAL_COMMIT
                         and raw.get("official_post", {}).get("pass") is True,
        "tsp": raw.get("tsp", {}).get("final_pass") is True,
        "cvrp": raw.get("cvrp", {}).get("final_pass") is True,
    }
    failed = sorted(key for key, value in checks.items() if value is not True)
    if failed:
        raise ValueError(f"S2 authoritative evidence failed closed: {failed}")
    return {"path": str(path.resolve()), "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": sha256_file(metadata_path), "checks": checks, "pass": True}

