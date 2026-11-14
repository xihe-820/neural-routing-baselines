"""Read-only historical artifact audit; raw evidence, not printed numbers, is required."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from common.hashing import sha256_file
from common.provenance import normalize_git_repository_identity
from methods.glop.paper_results import write_json
from rebuttal.tsp2k.config import BASE_COMMITS, COUNT, UPSTREAMS


def revalidate_records(records, dataset):
    import ml4co_kit as kit
    from rebuttal.tsp2k.runner import validated
    wrapper = kit.TSPWrapper()
    wrapper.from_pickle(dataset["path"])
    if len(wrapper.task_list) != COUNT:
        raise ValueError("historical dataset count changed")
    for row in records:
        index = row["dataset_instance_index"]
        canonical = row.get("canonical_solution")
        if canonical is None:
            raise ValueError("historical record lacks an actual tour")
        official = row.get("reported_objective", row.get("official_objective"))
        result = validated(wrapper.task_list[index], index, canonical, official)
        from rebuttal.tsp2k.artifacts import check_record
        check_record(result)
        from common.objective_agreement import objective_agrees
        if (not objective_agrees(result["objective"], row["independent_objective"]) or
                not objective_agrees(result["reference_objective"], row["reference_objective"])):
            raise ValueError("historical objective/reference differs from actual decoded tour")


def source_gate(root, source_files, commit, method):
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("historical evidence lacks executable source hashes")
    required = ({"methods/glop/tsp/paper_eval.py", "methods/glop/runtime.py",
                 "methods/glop/paper_protocol.py"} if method == "GLOP" else
                {"methods/sil/config.py", "methods/sil/runtime.py",
                 "methods/sil/author_batch_eval.py"})
    actual_paths = {v["path"] for v in source_files}
    if not required.issubset(actual_paths):
        raise ValueError("historical evidence lacks required solver source binding")
    for item in source_files:
        path = item["path"]
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("invalid historical source path")
        recorded = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{path}"],
                                  capture_output=True, check=True).stdout
        approved = subprocess.run(["git", "-C", str(root), "show",
                                  f"{BASE_COMMITS[method]}:{path}"],
                                 capture_output=True, check=True).stdout
        if hashlib.sha256(recorded).hexdigest() != item["sha256"] or recorded != approved:
            raise ValueError(f"historical source differs from reviewed implementation: {path}")


def audit_glop(summary_path, lock, root):
    from methods.glop.paper_protocol import formal_protocol
    from methods.glop.paper_results import summarize_chunks
    old = json.loads(Path(summary_path).read_text())
    if old.get("status") != "PAPER_READY" or old.get("method") != "GLOP":
        raise ValueError("historical GLOP summary is not PAPER_READY")
    chunks = [Path(row["path"]) for row in old["chunks"]]
    recomputed = summarize_chunks(chunks)
    for key in ("consistency_identity", "instance_count", "obj_mean_independent_objective",
                "drop_mean_per_instance_gap_percent", "time_mean_single_instance_seconds"):
        if old.get(key) != recomputed.get(key):
            raise ValueError("historical GLOP summary disagrees with underlying chunks")
    identity = recomputed["consistency_identity"]
    if (identity["paper_protocol"] != formal_protocol("TSP", 2000, "official_standard") or
            identity["dataset"]["sha256"] != lock["dataset"]["sha256"] or
            identity["dataset"]["count"] != COUNT or
            identity["upstream"]["commit"] != UPSTREAMS["GLOP"][1]):
        raise ValueError("historical GLOP protocol/dataset/source differs")
    source_gate(root, identity["source_provenance"], identity["project"]["commit"], "GLOP")
    expected_hashes = {a["sha256"] for a in lock["methods"]["GLOP"]["assets"]}
    actual = {r[k] for r in identity["assets"]["revisers"]
              for k in ("checkpoint_sha256", "args_sha256")}
    if actual != expected_hashes:
        raise ValueError("historical reviser checkpoint/args differ")
    for chunk in chunks:
        records = [json.loads(line) for line in
                   (chunk / "validated_records.jsonl").read_text().splitlines()]
        revalidate_records(records, lock["dataset"])
    return {"budget_value": 1, "quality_instance_count": COUNT,
            "timing_sample_count": COUNT, "identity": identity,
            "summary_sha256": sha256_file(summary_path),
            "raw_chunk_paths": list(map(str, chunks))}


def audit_sil(quality_dir, timing_path, lock, root):
    from common.protocol_rebind import verify_quality_artifact, verify_timing_artifact
    from methods.sil.config import resolve_author_batch_config
    from methods.sil.paper_results import TIMING_SEMANTICS
    spec = lock["methods"]["SIL"]
    quality = verify_quality_artifact(
        quality_dir, method="SIL", problem="tsp", problem_size=2000,
        protocol_label="fewer", budget_key="budget", budget=10,
        expected_protocol=resolve_author_batch_config("tsp", 2000, "fewer"),
        expected_count=COUNT, expected_batch_size=64, expected_project_commit=None,
        upstream_commit=UPSTREAMS["SIL"][1], dataset_filename=lock["dataset"]["path"].split("/")[-1],
        checkpoint_filename="checkpoint-tsp1k.pt")
    if (quality["dataset"]["sha256"] != lock["dataset"]["sha256"] or
            quality["checkpoint"]["sha256"] != spec["assets"][0]["sha256"]):
        raise ValueError("historical SIL asset identity differs")
    timing = verify_timing_artifact(
        timing_path, method="SIL", problem="tsp", problem_size=2000,
        protocol_label="fewer", budget_key="budget", budget=10, expected_count=3,
        expected_project_commit=None, upstream_commit=UPSTREAMS["SIL"][1],
        dataset_sha256=lock["dataset"]["sha256"],
        checkpoint_sha256=spec["assets"][0]["sha256"])
    if timing["summary"].get("timing_semantics") != TIMING_SEMANTICS:
        raise ValueError("historical SIL BS1 timing semantics mismatch")
    for summary in (quality["summary"], timing["summary"]):
        provenance = summary["project"]
        if len(provenance["commit"]) != 40:
            raise ValueError("historical source commit must be exact")
        source_gate(root, summary["source_files"], provenance["commit"], "SIL")
        if normalize_git_repository_identity(summary["upstream"]["url"]) != "github.com/CIAM-Group/SIL":
            raise ValueError("historical upstream remote differs")
    revalidate_records(quality["records"], lock["dataset"])
    revalidate_records(timing["summary"]["validation_records"], lock["dataset"])
    return {"budget_value": 10, "quality_instance_count": COUNT, "timing_sample_count": 3,
            "quality_batch_size": 64, "timing_batch_size": 1,
            "quality_hashes": quality["hashes"], "timing_sha256": timing["sha256"],
            "quality_path": str(Path(quality_dir).resolve()),
            "timing_path": str(Path(timing_path).resolve())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-lock", type=Path, required=True)
    parser.add_argument("--method", choices=["GLOP", "SIL"])
    parser.add_argument("--all-candidates", action="store_true")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--quality-dir", type=Path)
    parser.add_argument("--timing", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("reuse audit output exists")
    lock = json.loads(args.asset_lock.read_text())
    if args.all_candidates:
        candidates = lock.get("historical_candidates", {})
        reports = {}
        for method in ("GLOP", "SIL"):
            try:
                if method == "GLOP":
                    summaries = candidates.get("GLOP", [])
                    if len(summaries) != 1:
                        raise ValueError(f"need one explicit GLOP standard source; candidates={summaries}")
                    reports[method] = audit_glop(summaries[0], lock, lock["projects"][method]["path"])
                else:
                    quality, timing = (candidates.get("SIL_quality", []),
                                       candidates.get("SIL_timing", []))
                    if len(quality) != 1 or len(timing) != 1:
                        raise ValueError(f"need exact SIL PRC10 sources; quality={quality}, timing={timing}")
                    reports[method] = audit_sil(quality[0], timing[0], lock, lock["projects"][method]["path"])
                reports[method]["status"] = "VERIFIED_HISTORICAL_REUSE"
            except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
                reports[method] = {"status": "PENDING", "reuse_status": "NEEDS_RERUN", "reason": str(exc)}
        write_json(args.output, reports)
        print(json.dumps(reports, indent=2, allow_nan=False))
        return
    if args.method is None:
        parser.error("specify --all-candidates or --method with exact source paths")
    root = lock["projects"][args.method]["path"]
    try:
        result = (audit_glop(args.summary, lock, root) if args.method == "GLOP" else
                  audit_sil(args.quality_dir, args.timing, lock, root))
        result.update(status="VERIFIED_HISTORICAL_REUSE",
                      interpretation="eligible raw evidence; preserve original execution commit",
                      method=args.method)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        result = {"status": "PENDING", "reuse_status": "NEEDS_RERUN",
                  "method": args.method, "reason": str(exc)}
    write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
