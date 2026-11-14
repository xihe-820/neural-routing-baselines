"""Read-only server inventory producing an explicit, hash-bound asset lock."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from common.hashing import sha256_file
from common.provenance import git_provenance, normalize_git_repository_identity
from methods.glop.paper_protocol import REVISER_ASSETS
from methods.glop.paper_results import fingerprint, write_json
from methods.glop.runtime import verify_file
from methods.udc.s3_eval import CHECKPOINTS, FAMILY_DIRS, checkpoint_gate
from rebuttal.tsp2k.config import BASE_COMMITS, COUNT, DATASET, UPSTREAMS


def checkout(path, method):
    value = git_provenance(path)
    repo, commit = UPSTREAMS[method]
    if (value["dirty"] or value["commit"] != commit or
            normalize_git_repository_identity(value["url"]) != "github.com/" + repo):
        raise ValueError(f"{method} source identity/cleanliness mismatch: {path}")
    return value


def file_identity(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"missing asset: {path}")
    with path.open("rb") as stream:
        header = stream.read(128)
    if header.startswith(b"version https://git-lfs"):
        raise ValueError(f"missing/binary asset is LFS pointer: {path}")
    return {"path": str(path), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size}


def discover_files(roots):
    found = {}
    for root in roots:
        for base, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv"}]
            for name in names:
                path = Path(base, name)
                found.setdefault(name, set()).add(path.resolve())
    return found


def unique_bytes(paths, description, suffix=None):
    values = [Path(p) for p in paths
              if suffix is None or str(p).endswith(suffix)]
    identities = [file_identity(p) for p in sorted(set(values))]
    if not identities or len({p["sha256"] for p in identities}) != 1:
        raise ValueError(f"missing or conflicting {description}: {identities}")
    return {**identities[0], "identical_locations": [p["path"] for p in identities]}


def project(path, method):
    evidence = git_provenance(path)
    if evidence["dirty"]:
        raise ValueError(f"rebuttal checkout must be reviewed/committed and clean: {path}")
    subprocess.run(["git", "-C", str(path), "merge-base", "--is-ancestor",
                    BASE_COMMITS[method], "HEAD"], check=True)
    return {"path": str(Path(path).resolve()), **evidence}


def verify_lock(lock, method, root):
    if lock.get("schema") != "tsp2k-asset-lock.v1":
        raise ValueError("unexpected lock schema")
    current = project(root, method)
    if current != lock["projects"][method]:
        raise ValueError("reviewed project differs from asset lock")
    spec = lock["methods"][method]
    if checkout(spec["upstream_path"], method) != spec["upstream"]:
        raise ValueError("upstream changed after audit")
    for asset in [lock["dataset"], *spec["assets"]]:
        if sha256_file(asset["path"]) != asset["sha256"]:
            raise ValueError("dataset/checkpoint bytes changed after asset audit")
    if lock["dataset"]["count"] != COUNT or Path(lock["dataset"]["path"]).name != DATASET:
        raise ValueError("dataset identity differs from TSP2K benchmark")
    if method == "UDC":
        from methods.udc.protocol import s1_gate, s2_gate, s3_gate
        from methods.udc.paper_protocol import load_budget_registry, s4_gate
        load_budget_registry()
        gates = spec["evidence"]
        for name, gate in (("s1", s1_gate), ("s2", s2_gate), ("s3", s3_gate), ("s4", s4_gate)):
            if gate(Path(gates[name]["path"])) != gates[name]:
                raise ValueError("UDC authoritative evidence changed after audit")
        checkpoint_gate(Path(spec["supplemental_root"]), "tsp")
    if "reference_lock" in lock:
        verify_reference_lock(lock)
    return spec


def verify_reference_lock(lock):
    reference = lock["reference_lock"]
    if sha256_file(reference["path"]) != reference["sha256"]:
        raise ValueError("original asset lock bytes changed")
    old = json.loads(Path(reference["path"]).read_text())
    expected = {"GLOP": "14a3ebd6d3d2a6c7d371487bc1895a1ebb639fe7",
                "UDC": "14a3ebd6d3d2a6c7d371487bc1895a1ebb639fe7",
                "SIL": "c51d1edae0fbaadf7d8af1993da6ca3517404c9a"}
    if old.get("schema") != "tsp2k-asset-lock.v1" or any(
            old["projects"][method]["commit"] != commit for method, commit in expected.items()):
        raise ValueError("reference lock is not the reviewed original experiment")
    for key in ("sha256", "count", "reference_objectives_sha256"):
        if old["dataset"][key] != lock["dataset"][key]:
            raise ValueError(f"original dataset/reference identity changed: {key}")
    for method in expected:
        original_source, new_source = (old["methods"][method]["upstream"],
                                       lock["methods"][method]["upstream"])
        same_source = (
            original_source.get("dirty") is False and new_source.get("dirty") is False and
            original_source["commit"] == new_source["commit"] and
            normalize_git_repository_identity(original_source["url"]) ==
            normalize_git_repository_identity(new_source["url"]))
        if (not same_source or
                [a["sha256"] for a in old["methods"][method]["assets"]] !=
                [a["sha256"] for a in lock["methods"][method]["assets"]]):
            raise ValueError(f"original {method} official source/checkpoint identity changed")
    if (lock["projects"]["UDC"]["commit"] == expected["UDC"] or
            lock["projects"]["SIL"]["commit"] == expected["SIL"]):
        raise ValueError("extended experiment requires both new reviewed worktree commits")
    return old


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, action="append", required=True)
    parser.add_argument("--glop-udc-root", type=Path, required=True)
    parser.add_argument("--sil-root", type=Path, required=True)
    for name in ("glop", "udc", "sil"):
        parser.add_argument(f"--{name}-upstream", type=Path)
    parser.add_argument("--sil-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-lock", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("audit output exists; retain historical lock")
    roots = [p.resolve() for p in args.search_root]
    if any(not p.is_dir() for p in roots):
        raise ValueError("search roots must exist")
    files = discover_files(roots)
    dataset = unique_bytes(files.get(DATASET, []), "TSP2K dataset")
    import ml4co_kit as kit
    import numpy as np
    wrapper = kit.TSPWrapper()
    wrapper.from_pickle(dataset["path"])
    tasks = wrapper.task_list
    if (len(tasks) != COUNT or any(type(t) is not kit.TSPTask or
            np.asarray(t.points).shape != (2000, 2) or
            not np.isfinite(t.points).all() for t in tasks)):
        raise ValueError("TSP2K requires exact 64 ML4CO TSPTask [2000,2]")
    refs = [float(t.evaluate(t.ref_sol)) for t in tasks]
    if not np.isfinite(refs).all() or min(refs) <= 0:
        raise ValueError("invalid benchmark reference")
    dataset.update(count=COUNT, reference_objectives_sha256=fingerprint(refs),
                   source_dtypes=sorted({str(t.points.dtype) for t in tasks}))
    methods, projects = {}, {}
    expected_dirs = {"GLOP": "GLOP", "UDC": "NCO_code", "SIL": "SIL"}
    for method, name in expected_dirs.items():
        # os.walk prunes .git: source checkouts are discovered from known source files.
        marker = {"GLOP": "main.py", "UDC": "README.md", "SIL": "README.md"}[method]
        candidates = sorted({p.parent for p in files.get(marker, [])
                             if p.parent.name == name and (p.parent / ".git").exists()})
        explicit = getattr(args, method.lower() + "_upstream")
        if explicit is not None:
            candidates = [explicit.resolve()]
        valid = []
        for candidate in candidates:
            try:
                valid.append((candidate, checkout(candidate, method)))
            except (ValueError, RuntimeError):
                continue
        if len(valid) != 1:
            raise ValueError(f"need one exact clean {method} checkout; found {valid}")
        upstream_path, provenance = valid[0]
        spec = {"upstream_path": str(upstream_path), "upstream": provenance, "assets": []}
        root = args.sil_root if method == "SIL" else args.glop_udc_root
        projects[method] = project(root, method)
        if method == "GLOP":
            asset_roots = set()
            for size in (100, 50, 20):
                registered = REVISER_ASSETS[size]
                asset = unique_bytes(files.get("epoch-299.pt", []),
                                     f"Reviser {size}", registered["path"])
                path = Path(asset["path"])
                spec["assets"].append(asset)
                args_asset = file_identity(path.parent / "args.json")
                verify_file(path, registered)
                verify_file(Path(args_asset["path"]), registered,
                            sha_key="args_sha256", size_key="args_size_bytes")
                spec["assets"].append(args_asset)
                asset_roots.add(path.parents[2])
            if len(asset_roots) != 1:
                raise ValueError("reviser assets must share their official asset root")
            spec["asset_root"] = str(asset_roots.pop())
        elif method == "UDC":
            asset = unique_bytes(files.get(CHECKPOINTS["tsp"]["conquering"][0], []),
                                 "UDC supplemental conquering",
                                 FAMILY_DIRS["tsp"] + "/" + CHECKPOINTS["tsp"]["conquering"][0])
            supplemental = Path(asset["path"]).parents[1]
            paths = checkpoint_gate(supplemental, "tsp")
            spec.update(supplemental_root=str(supplemental),
                        assets=[file_identity(v["path"]) for v in paths.values()])
            from methods.udc.protocol import s1_gate, s2_gate, s3_gate
            from methods.udc.paper_protocol import s4_gate, load_budget_registry
            load_budget_registry()
            evidence = {}
            for name, gate, marker, dirname in (
                ("s1", s1_gate, "s1_audit_v5.json", None),
                ("s2", s2_gate, "metadata.json", "s2_official_smoke_v3"),
                ("s3", s3_gate, "metadata.json", "our_5"),
                ("s4", s4_gate, "summary.json", "s4_scale_preflight")):
                passed = []
                for candidate in files.get(marker, []):
                    if dirname is not None and candidate.parent.name != dirname:
                        continue
                    try:
                        passed.append(gate(candidate if name == "s1" else candidate.parent))
                    except (ValueError, KeyError, OSError):
                        continue
                if len(passed) != 1:
                    raise ValueError(f"need exact authoritative UDC {name} evidence: {passed}")
                evidence[name] = passed[0]
            spec["evidence"] = evidence
        else:
            spec["assets"] = [unique_bytes(
                [args.sil_checkpoint] if args.sil_checkpoint is not None else
                files.get("checkpoint-tsp1k.pt", []),
                                           "released SIL TSP1K checkpoint")]
            spec["checkpoint"] = spec["assets"][0]["path"]
            spec["release_identity_note"] = (
                "existing user asset; hash locked by this audit; release/provenance "
                "must be reviewed against prior SIL checkpoint audit before execution")
        methods[method] = spec
    historical = {"GLOP": [], "SIL_quality": [], "SIL_timing": []}
    for path in files.get("summary.json", []):
        try:
            summary = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if summary.get("problem") != "TSP" or summary.get("problem_size") != 2000:
            continue
        if summary.get("method") == "GLOP" and summary.get("status") == "PAPER_READY":
            historical["GLOP"].append(str(path))
        if summary.get("method") == "SIL" and summary.get("budget") == 10:
            historical["SIL_quality"].append(str(path.parent))
    for path in files.get("fewer.json", []):
        try:
            summary = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if (summary.get("method"), summary.get("problem"), summary.get("problem_size"),
                summary.get("budget")) == ("SIL", "TSP", 2000, 10):
            historical["SIL_timing"].append(str(path))
    result = {
        "schema": "tsp2k-asset-lock.v1", "dataset": dataset, "methods": methods,
        "projects": projects, "search_roots": list(map(str, roots)),
        "environment": {"python": sys.version, "executable": sys.executable,
                        "packages": {name: importlib.metadata.version(name)
                                     for name in ("torch", "numpy", "ml4co_kit")}},
        "gpu_status": "PENDING_RUNTIME_PREFLIGHT",
        "historical_reuse": "PENDING_EXPLICIT_ARTIFACT_AUDIT",
        "historical_candidates": historical,
    }
    if args.reference_lock is not None:
        result["reference_lock"] = {"path": str(args.reference_lock.resolve()),
                                    "sha256": sha256_file(args.reference_lock)}
        verify_reference_lock(result)
        # Inventory only: no model, solver, forward or training is invoked.
        import torch
        from common.provenance import environment_provenance
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
                "RTX 4090" not in torch.cuda.get_device_name(0)):
            raise ValueError("extended inventory requires one visible RTX 4090")
        result["environment"] = environment_provenance(torch.device("cuda:0"))
        result["gpu_status"] = "VERIFIED_RTX4090_INVENTORY"
        result["dataset"]["reference_objectives"] = refs
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
