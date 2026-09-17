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
S3_IMPLEMENTATION_COMMIT = "df082675e7c5fa64d3ce20174207da33917a5b50"
S3_SCRIPT_SHA256 = "5fedf4c1270246b17981b076a73c5046e4ce2a29387aeb08e187781ce60aa750"
DATASET_FILENAMES = {
    "tsp": "tsp500_concorde_16.546.pkl",
    "cvrp": "cvrp500_hgs-300s_37.154.pkl",
}
SPECS = {
    "tsp": {"x": 2, "configured_pomo": 2, "effective_pomo": 2},
    "cvrp": {"x": 50, "configured_pomo": 10, "effective_pomo": 1},
}
FORMAL_SIZES = {
    "tsp": (100, 500, 1000, 2000, 5000, 10000),
    "cvrp": (200, 500, 1000, 2000),
}
SCALE_DATASET_FILENAMES = {
    ("tsp", 100): "tsp100_concorde_7.756.pkl",
    ("tsp", 500): "tsp500_concorde_16.546.pkl",
    ("tsp", 1000): "tsp1000_concorde_23.118.pkl",
    ("tsp", 2000): "tsp2000_lkh_500_32.436.pkl",
    ("tsp", 5000): "tsp5000_lkh_500_50.968.pkl",
    ("tsp", 10000): "tsp10000_lkh_500_71.782.pkl",
    ("cvrp", 200): "cvrp200_hgs-60s_19.630.pkl",
    ("cvrp", 500): "cvrp500_hgs-300s_37.154.pkl",
    ("cvrp", 1000): "cvrp1000_hgs-360s_41.171.pkl",
    ("cvrp", 2000): "cvrp2000_hgs-360s_57.181.pkl",
}
S3_TRACKED_ALLOWLIST = {
    "methods/udc/adapter.py", "methods/udc/protocol.py",
    "methods/udc/s3_eval.py", "tests/test_udc_s3.py",
    "docs/SERVER_RUNBOOK.md",
}
S4_TRACKED_ALLOWLIST = {
    "methods/udc/adapter.py", "methods/udc/protocol.py",
    "methods/udc/s3_eval.py", "methods/udc/s4_scale_preflight.py",
    "tests/test_udc_s4.py", "docs/SERVER_RUNBOOK.md",
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


def s4_project_gate(root: Path) -> dict:
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    dirty = bool(_git(root, "status", "--porcelain=v1").stdout.strip())
    ancestor = _git(root, "merge-base", "--is-ancestor",
                    S3_IMPLEMENTATION_COMMIT, "HEAD", ok=(0, 1)).returncode == 0
    changed = [line for line in _git(
        root, "diff", "--name-only", f"{S3_IMPLEMENTATION_COMMIT}..HEAD"
    ).stdout.splitlines() if line]
    outside = sorted(set(changed) - S4_TRACKED_ALLOWLIST)
    result = {"head": head, "dirty": dirty,
              "s3_implementation_base_commit": S3_IMPLEMENTATION_COMMIT,
              "base_is_ancestor": ancestor, "base_to_head_changed_paths": changed,
              "allowed_paths": sorted(S4_TRACKED_ALLOWLIST),
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


def scale_dataset_filename(problem: str, size: int) -> str:
    key = (problem.lower(), int(size))
    if key not in SCALE_DATASET_FILENAMES:
        raise ValueError(f"outside formal UDC scale scope: {key}")
    return SCALE_DATASET_FILENAMES[key]


def discover_scale_dataset(root: Path, problem: str, size: int) -> Path:
    filename = scale_dataset_filename(problem, size)
    matches = sorted(path.resolve() for path in root.rglob(filename)
                     if path.is_file())
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {filename} below {root}; found {matches}")
    return matches[0]


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


def s3_gate(path: Path) -> dict:
    if (path.name != "our_5" or path.parent.name != "s3_ml4co_adapter"
            or path.parent.parent.name != S3_IMPLEMENTATION_COMMIT[:8]):
        raise ValueError("S3 evidence path is not authoritative df082675/our_5")
    metadata_path = path / "metadata.json"
    raw = json.loads(metadata_path.read_text())
    checks = {
        "state": raw.get("state") == "KIT_VALIDATED",
        "count": raw.get("count") == 5,
        "ready": raw.get("ready_for_stage_s4_scale_preflight") == "YES",
        "script_sha256": raw.get("script_sha256") == S3_SCRIPT_SHA256,
        "project_head": raw.get("project_post", {}).get("head") == S3_IMPLEMENTATION_COMMIT,
        "project_pass": raw.get("project_post", {}).get("pass") is True,
        "official_head": raw.get("official_post", {}).get("head") == OFFICIAL_COMMIT,
        "official_pass": raw.get("official_post", {}).get("pass") is True,
    }
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if failed:
        raise ValueError(f"S3 authoritative evidence failed closed: {failed}")
    return {"path": str(path.resolve()), "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "script_sha256": raw["script_sha256"], "checks": checks, "pass": True}
