"""Explicit fullset sources, immutable historical Git snapshots and CPU revalidation."""
import argparse
import ast
from functools import lru_cache
import json
from pathlib import Path
import subprocess

import numpy as np

from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from methods.glop.paper_results import fingerprint, write_json
from rebuttal.tsp2k.artifacts import read_artifact
from rebuttal.tsp2k.audit import checkout
from rebuttal.tsp2k.config import COUNT, DATASET, SWEEPS

SCHEMA = "tsp2k-fullset-manifest.v1"
LEGACY_COMMITS = {
    "GLOP": "14a3ebd6d3d2a6c7d371487bc1895a1ebb639fe7",
    "SIL": "c51d1edae0fbaadf7d8af1993da6ca3517404c9a",
}


def exact_sha(value, length):
    if not isinstance(value, str) or len(value) != length or any(
            c not in "0123456789abcdef" for c in value):
        raise ValueError("expected exact lowercase Git/SHA256 identity")
    return value


@lru_cache(maxsize=4096)
def git_bytes(repository, commit, path):
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError("invalid source path")
    return subprocess.run(["git", "-C", str(repository), "show", f"{commit}:{path}"],
                          check=True, capture_output=True).stdout


def verify_sources(identity, repository):
    commit = exact_sha(identity["project_commit"], 40)
    folders = ("rebuttal/tsp2k/", "common/", f"methods/{identity['method'].lower()}/",
               "problems/tsp/")
    paths = subprocess.run(["git", "-C", str(repository), "ls-tree", "-r", "--name-only", commit],
                           check=True, capture_output=True, text=True).stdout.splitlines()
    expected = {p for p in paths if p.endswith(".py") and p.startswith(folders)}
    sources = identity["source_provenance"]
    observed = [s["path"] for s in sources]
    if set(observed) != expected or len(observed) != len(expected) or not expected:
        raise ValueError("source provenance omits, duplicates or adds executable files")
    import hashlib
    for source in sources:
        digest = hashlib.sha256(git_bytes(str(repository), commit, source["path"])).hexdigest()
        if digest != source["sha256"]:
            raise ValueError(f"source hash differs from execution commit: {source['path']}")
    if identity["method"] == "UDC":
        tree = ast.parse(git_bytes(str(repository), commit, "rebuttal/tsp2k/config.py"))
        sweep = next(n.value for n in tree.body if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == "UDC_SWEEP" for t in n.targets))
        if identity["budget_value"] not in ast.literal_eval(sweep):
            raise ValueError("recorded source commit did not support this UDC budget")


def verify_assets(identity):
    reference = identity["asset_lock"]
    if sha256_file(reference["path"]) != reference["sha256"]:
        raise ValueError("recorded asset lock bytes differ")
    lock = json.loads(Path(reference["path"]).read_text())
    method = identity["method"]
    spec = lock["methods"][method]
    if (lock.get("schema") != "tsp2k-asset-lock.v1" or
            identity["project"] != lock["projects"][method] or
            identity["dataset"] != lock["dataset"] or identity["asset_spec"] != spec or
            identity["upstream"] != spec["upstream"]):
        raise ValueError("artifact identity differs from its exact original asset lock")
    if checkout(spec["upstream_path"], method) != spec["upstream"]:
        raise ValueError("pinned official checkout differs")
    expected_hashes = {Path(a["path"]).name + ":" + str(i): a["sha256"]
                       for i, a in enumerate(spec["assets"])}
    if identity["checkpoint_sha256"] != expected_hashes:
        raise ValueError("checkpoint identity differs from original lock")
    for asset in [lock["dataset"], *spec["assets"]]:
        if sha256_file(asset["path"]) != asset["sha256"]:
            raise ValueError("dataset/checkpoint bytes differ from original lock")
    if method == "GLOP":
        from methods.glop.paper_protocol import REVISER_ASSETS
        from methods.glop.runtime import verify_file
        for size in (100, 50, 20):
            registered = REVISER_ASSETS[size]
            verify_file(Path(spec["asset_root"]) / registered["path"], registered)
            verify_file(Path(spec["asset_root"]) / registered["args_path"], registered,
                        sha_key="args_sha256", size_key="args_size_bytes")
    elif method == "UDC":
        from rebuttal.tsp2k.audit import verify_lock
        verify_lock(lock, "UDC", Path(lock["projects"]["UDC"]["path"]))
    return lock


def load_tasks(dataset):
    import ml4co_kit as kit
    wrapper = kit.TSPWrapper()
    wrapper.from_pickle(dataset["path"])
    tasks = wrapper.task_list
    if (len(tasks) != COUNT or Path(dataset["path"]).name != DATASET or
            any(type(t) is not kit.TSPTask or np.asarray(t.points).shape != (2000, 2)
                or not np.isfinite(t.points).all() for t in tasks)):
        raise ValueError("not exact TSP2K ML4CO benchmark tasks")
    refs = [float(t.evaluate(t.ref_sol)) for t in tasks]
    if fingerprint(refs) != dataset["reference_objectives_sha256"]:
        raise ValueError("all 64 benchmark references differ from recorded identity")
    if "reference_objectives" in dataset and dataset["reference_objectives"] != refs:
        raise ValueError("locked reference vector differs")
    return tasks


def revalidate(identity, records, tasks):
    from rebuttal.tsp2k.runner import validated
    from rebuttal.tsp2k.artifacts import check_record
    for row in records:
        index = row["instance_index"]
        actual = validated(tasks[index], index, row["canonical_solution"],
                           row["official_objective"])
        check_record(actual)
        for field in ("objective", "reference_objective", "kit_objective", "gap_percent"):
            if not objective_agrees(actual[field], row[field]):
                raise ValueError(f"decoded tour/reference revalidation differs: {index} {field}")
    if identity["phase"] == "fullset":
        quality = [r for r in records if r["source"] == "quality"]
        if fingerprint([r["reference_objective"] for r in quality]) != (
                identity["dataset"]["reference_objectives_sha256"]):
            raise ValueError("recorded fullset reference vector differs")


def verify_evidence(directory, *, method, budget, commit, repository,
                    metadata_sha=None, tasks_cache=None):
    directory = Path(directory).resolve()
    digest = sha256_file(directory / "metadata.json")
    if metadata_sha is not None and digest != exact_sha(metadata_sha, 64):
        raise ValueError("manifest metadata SHA256 mismatch")
    identity, records, summary = read_artifact(directory)
    if (identity["method"] != method or identity["budget_value"] != budget or
            identity["project_commit"] != exact_sha(commit, 40) or
            identity.get("evidence_kind") != "SERVER_MEASUREMENT"):
        raise ValueError("method/budget/approved commit or measured evidence mismatch")
    verify_sources(identity, repository)
    lock = verify_assets(identity)
    if method == "UDC":
        from methods.udc.paper_production import PRODUCTION_TIMING, CUDA_ALLOCATOR_PROVENANCE
        if identity["timing"] != {"semantics": PRODUCTION_TIMING, "warmup_instances": 0,
                                  "cuda_allocator": CUDA_ALLOCATOR_PROVENANCE}:
            raise ValueError("UDC timing/memory protocol differs from production")
        if "reference_lock" in lock and identity["environment"] != lock["environment"]:
            raise ValueError("UDC runtime environment differs from its new audit lock")
        for row in records:
            native = row["native_evidence"]
            if (native["x"] != budget or len(native["completed_stages"]) != budget or
                    (native["alpha"], native["configured_pomo"], native["effective_pomo"],
                     native["seed"]) != (50, 2, 2, 1234) or
                    native["status"] != "KIT_VALIDATED" or
                    native["runtime_seconds"] != row["runtime_seconds"] or
                    native["timing_semantics"] != PRODUCTION_TIMING):
                raise ValueError("UDC actual stage count/config/timing differs from request")
    elif method == "SIL":
        from methods.sil.paper_results import TIMING_SEMANTICS
        from rebuttal.tsp2k.runner import SIL_WARMUP_POLICY
        if identity["timing"] != {"semantics": TIMING_SEMANTICS, "warmup": SIL_WARMUP_POLICY}:
            raise ValueError("SIL quality/timing warmup or latency semantics differ")
    else:
        from methods.glop.paper_results import TSP_TIMING_SEMANTICS
        if identity["timing"] != {"semantics": TSP_TIMING_SEMANTICS,
                                  "warmup_instances": min(2, identity["requested_count"])}:
            raise ValueError("GLOP timing/warmup semantics differ")
    cache = {} if tasks_cache is None else tasks_cache
    key = (identity["dataset"]["path"], identity["dataset"]["sha256"])
    if key not in cache:
        cache[key] = load_tasks(identity["dataset"])
    revalidate(identity, records, cache[key])
    if method == "SIL" and identity["phase"] == "fullset":
        if (identity["actual_quality_batch_size"] != 64 or
                identity["timing_indices"] != [0, 1, 2]):
            raise ValueError("SIL fullset needs native BS64 quality and fixed three BS1 timings")
    return identity, records, summary


def verify_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected explicit artifact manifest schema")
    approved = manifest["approved_projects"]
    if any(approved.get(m) != c for m, c in LEGACY_COMMITS.items()):
        raise ValueError("GLOP/SIL approved execution commits differ from frozen fullsets")
    for method in SWEEPS:
        exact_sha(approved[method], 40)
    entries = manifest["artifacts"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest artifacts must be nonempty")
    seen, directories, verified, cache = set(), [], [], {}
    for entry in entries:
        method, budget = entry["method"], entry["budget_value"]
        key = (method, budget)
        if method not in SWEEPS or type(budget) is not int or budget not in SWEEPS[method] or key in seen:
            raise ValueError("duplicate or invalid manifest method/budget")
        if entry["expected_project_commit"] != approved[method]:
            raise ValueError("same-method commit differs from explicit approval")
        directory = Path(entry["path"])
        if not directory.is_absolute():
            raise ValueError("manifest artifact path must be absolute")
        identity, records, summary = verify_evidence(
            directory, method=method, budget=budget, commit=approved[method],
            repository=manifest["source_repository"],
            metadata_sha=entry["metadata_sha256"], tasks_cache=cache)
        if (identity["phase"] != "fullset" or summary["status"] != "MEASURED_FULLSET" or
                summary["validation_status"] != "KIT_VALIDATED"):
            raise ValueError("manifest includes non-fullset evidence")
        if method == "UDC":
            from rebuttal.tsp2k.udc_extended import stage_gate
            if identity.get("experiment_stage") != "fullset":
                raise ValueError("extended UDC fullset lacks gated stage provenance")
            calibration = identity["stage_evidence"]["calibration"]["path"]
            actual_gate = stage_gate("fullset", identity["udc_evidence_root"],
                                     identity["asset_lock"]["path"], calibration)
            if actual_gate != identity["stage_evidence"]:
                raise ValueError("UDC fullset stage evidence changed")
        seen.add(key)
        directories.append(directory)
        verified.append({"path": str(directory), "method": method, "budget_value": budget,
                         "project_commit": identity["project_commit"],
                         "upstream_commit": identity["upstream_commit"],
                         "metadata_sha256": sha256_file(directory / "metadata.json"),
                         "asset_lock": identity["asset_lock"],
                         "checkpoint_sha256": identity["checkpoint_sha256"],
                         "source_provenance_sha256": fingerprint(identity["source_provenance"]),
                         "environment_sha256": fingerprint(identity["environment"]),
                         "cpu_revalidation": "PASS"})
    from rebuttal.tsp2k.aggregate import aggregate
    report = aggregate(directories, require_fullset=True)
    # Stricter explicit mode additionally binds environment and original lock per method.
    for method in SWEEPS:
        group = [r for r in verified if r["method"] == method]
        if len({(r["asset_lock"]["sha256"], r["environment_sha256"]) for r in group}) != 1:
            raise ValueError(f"{method} mixes environment or asset locks")
    report["artifact_manifest"] = {"path": str(path), "sha256": sha256_file(path)}
    verification = {"schema": "tsp2k-fullset-verification.v1", "status": "PASS",
                    "artifact_manifest": report["artifact_manifest"], "approved_projects": approved,
                    "dataset_sha256": report["dataset_sha256"],
                    "points": verified,
                    "all_64_references_and_decoded_tours_revalidated": True}
    return report, verification


def build_manifest(*, glop_root, sil_root, udc_root, udc_budgets, udc_commit,
                   source_repository):
    if (not udc_budgets or len(set(udc_budgets)) != len(udc_budgets) or
            any(type(b) is not int or b not in SWEEPS["UDC"] for b in udc_budgets)):
        raise ValueError("invalid explicit calibrated UDC selection")
    approved = {**LEGACY_COMMITS, "UDC": exact_sha(udc_commit, 40)}
    entries = []
    for method, root, budgets in (("GLOP", glop_root, SWEEPS["GLOP"]),
                                  ("SIL", sil_root, SWEEPS["SIL"]),
                                  ("UDC", udc_root, udc_budgets)):
        for budget in budgets:
            directory = (Path(root) / str(budget)).resolve()
            entries.append({"path": str(directory), "method": method,
                            "budget_value": budget, "expected_project_commit": approved[method],
                            "metadata_sha256": sha256_file(directory / "metadata.json")})
    return {"schema": SCHEMA, "approved_projects": approved,
            "source_repository": str(Path(source_repository).resolve()), "artifacts": entries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    for name in ("glop-root", "sil-root", "udc-root", "source-repository", "output"):
        build.add_argument("--" + name, type=Path, required=True)
    build.add_argument("--udc-budgets", type=int, nargs="+", required=True)
    build.add_argument("--udc-commit", required=True)
    find = commands.add_parser("find-reference-lock")
    find.add_argument("--search-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "find-reference-lock":
        found = []
        for path in args.search_root.rglob("asset_lock.json"):
            raw = json.loads(path.read_text())
            projects = raw.get("projects", {})
            if (raw.get("schema") == "tsp2k-asset-lock.v1" and
                    projects.get("GLOP", {}).get("commit") == LEGACY_COMMITS["GLOP"] and
                    projects.get("UDC", {}).get("commit") == LEGACY_COMMITS["GLOP"] and
                    projects.get("SIL", {}).get("commit") == LEGACY_COMMITS["SIL"]):
                found.append(path.resolve())
        if len(set(found)) != 1:
            raise ValueError(f"need one exact original lock; explicit candidates={found}")
        print(found[0])
        return
    if args.output.exists():
        raise ValueError("explicit artifact manifest already exists")
    result = build_manifest(glop_root=args.glop_root, sil_root=args.sil_root,
        udc_root=args.udc_root, udc_budgets=args.udc_budgets, udc_commit=args.udc_commit,
        source_repository=args.source_repository)
    # Full verification occurs before any final report/plot can be emitted.
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
