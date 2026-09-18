#!/usr/bin/env python3
"""Fail-closed source, asset, environment, and pinned-dataset audit for three baselines."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.cvrptw_formal import DATASETS
from common.hashing import sha256_file
from common.provenance import git_provenance, normalize_git_repository_identity
from common.paper_results import utc_now, write_json
from methods.symnco.cvrptw.config import (
    CHECKPOINTS as SYMNCO_CHECKPOINTS,
    CHECKPOINT_HASHES as SYMNCO_HASHES,
    HISTORICAL_RESULTS as SYMNCO_HISTORICAL_RESULTS,
    SOURCE_SNAPSHOT as SYMNCO_SOURCE,
    validate_historical_report,
    validate_snapshot as validate_symnco_snapshot,
)

METHODS = {
    "rfte": {
        "repo": "https://github.com/ai4co/routefinder", "commit": "fe0e45b6df118af03c5f42db8b93a351f7629131",
        "upstream": "routefinder", "checkpoints": {50: "checkpoints/50/rf-transformer.ckpt", 100: "checkpoints/100/rf-transformer.ckpt"},
        "hashes": {},
    },
    "cada": {
        "repo": "https://github.com/CIAM-Group/CaDA", "commit": "b9868e1e09b3a1a3754960d830e801e51c7cb38d",
        "upstream": "CaDA", "checkpoints": {50: "50/result/2024-1111-1139/checkpoint-300.pt", 100: "100/result/2024-1121-1355/checkpoint-300.pt"},
        "hashes": {},
    },
    "moses_cada": {
        "repo": "https://github.com/panyxy/moses_vrp", "commit": "e301478b7a5df6d7b0b10a0543f4dee5e3c027a8",
        "upstream": "moses_vrp", "checkpoints": {50: "pretrained_moses_model/cada/50/multilora_denseroute_sigmoid.ckpt", 100: "pretrained_moses_model/cada/100/multilora_denseroute_sigmoid.ckpt"},
        "hashes": {50: "1aa499f3fce5d3412c2544c9632bbb9709309a7299b4298b535fa7e9014ef803", 100: "2eac9b038ae4655581aa73e4dbe8ad529aefd1963368c9a92d254b6269f8aabf"},
    },
    "symnco": {
        "source_kind": "snapshot", "upstream": None,
        "checkpoints": SYMNCO_CHECKPOINTS, "hashes": SYMNCO_HASHES,
    },
}


def packages():
    names = ("torch", "numpy", "torch-geometric", "rl4co", "tensordict", "torchrl",
             "lightning", "pytorch-lightning", "einops", "entmax")
    values = {}
    for name in names:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symnco-source", type=Path, default=SYMNCO_SOURCE)
    parser.add_argument("--include-symnco", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS),
                        default=["rfte", "cada", "moses_cada"])
    args = parser.parse_args()
    report = {"schema": "remaining-cvrptw-audit-v1", "created_at": utc_now(),
              "environment": {"python": sys.version, "executable": sys.executable,
                              "platform": platform.platform(), "packages": packages()},
              "project": git_provenance(ROOT), "datasets": {}, "methods": {}}
    failed = []
    if report["project"]["dirty"]:
        failed.append("project_dirty")
    selected_methods = list(args.methods)
    if args.include_symnco and "symnco" not in selected_methods:
        selected_methods.append("symnco")
    dataset_sizes = (50, 100, 200) if "symnco" in selected_methods else (50, 100)
    for size in dataset_sizes:
        cfg = DATASETS[size]
        path = (args.dataset_root / cfg["filename"]).resolve()
        row = {"path": str(path), "exists": path.is_file(), "expected": cfg}
        if path.is_file():
            row.update(size_bytes=path.stat().st_size, sha256=sha256_file(path))
        row["identity_pass"] = (row["exists"] and path.name == cfg["filename"] and
                                row.get("sha256") == cfg["sha256"])
        if not row["identity_pass"]:
            failed.append(f"dataset_{size}")
        report["datasets"][str(size)] = row
    for name, cfg in METHODS.items():
        if name not in selected_methods:
            continue
        upstream = (args.symnco_source.resolve() if name == "symnco" else
                    (args.upstream_root / cfg["upstream"]).resolve())
        row = {"upstream_path": str(upstream), "checkpoints": {}}
        try:
            if name == "symnco":
                identity = validate_symnco_snapshot(upstream)
                row["upstream"], row["upstream_pass"] = identity, True
            else:
                identity = git_provenance(upstream)
                row["upstream"] = identity
                row["upstream_pass"] = (
                    identity["commit"] == cfg["commit"] and not identity["dirty"] and
                    normalize_git_repository_identity(identity["url"]) ==
                    normalize_git_repository_identity(cfg["repo"]))
        except Exception as exc:
            row.update(upstream_pass=False, upstream_error=f"{type(exc).__name__}: {exc}")
        if not row["upstream_pass"]:
            failed.append(f"{name}_upstream")
        for size, relative in cfg["checkpoints"].items():
            path = (Path(relative).resolve() if name == "symnco" else
                    (args.checkpoint_root / cfg["upstream"] / relative).resolve())
            asset = {"path": str(path), "exists": path.is_file(),
                     "expected_sha256": cfg["hashes"].get(size)}
            if path.is_file():
                asset.update(size_bytes=path.stat().st_size, sha256=sha256_file(path),
                             lfs_pointer=path.read_bytes()[:40].startswith(b"version https://git-lfs"))
            asset["identity_pass"] = bool(
                asset["exists"] and not asset.get("lfs_pointer", False) and
                (asset["expected_sha256"] is None or asset.get("sha256") == asset["expected_sha256"]))
            if not asset["identity_pass"]:
                failed.append(f"{name}_checkpoint_{size}")
            row["checkpoints"][str(size)] = asset
        if name == "symnco":
            row["historical_e1"] = {}
            for size, value in SYMNCO_HISTORICAL_RESULTS.items():
                path = Path(value).resolve()
                evidence = {"path": str(path), "exists": path.is_file()}
                try:
                    report_payload = json.loads(path.read_text())
                    validate_historical_report(
                        report_payload, problem_size=size,
                        dataset_count=DATASETS[size]["count"])
                    evidence.update(identity_pass=True, sha256=sha256_file(path),
                                    rows=len(report_payload["rows"]))
                except Exception as exc:
                    evidence.update(identity_pass=False,
                                    error=f"{type(exc).__name__}: {exc}")
                    failed.append(f"symnco_historical_e1_{size}")
                row["historical_e1"][str(size)] = evidence
        report["methods"][name] = row
    report["failed_checks"] = failed
    report["status"] = "PASS" if not failed else "FAIL"
    write_json(args.output, report)
    print(args.output)
    if failed:
        raise SystemExit(f"remaining CVRPTW audit failed closed: {failed}")


if __name__ == "__main__":
    main()
