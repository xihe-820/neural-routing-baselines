#!/usr/bin/env python3
"""UDC fixed pilot, production safety gate, resumable fullset, and aggregation CLI."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import random
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from methods.udc.adapter import adapt_cvrp_task, adapt_tsp_task
from methods.udc.paper_protocol import (PILOT_INDICES, REGISTRY_PATH,
                                        SAFETY_CELLS, budget_config,
                                        load_budget_registry,
                                        production_project_gate, s4_gate)
from methods.udc.paper_results import (CHECKPOINT_FILE, METADATA_FILE,
                                       RECORDS_FILE, atomic_json,
                                       capture_rng_state, commit_record,
                                       finalize_run, fingerprint,
                                       initialize_or_resume,
                                       initialize_rng_checkpoint, read_jsonl,
                                       restore_rng_state, utc_now,
                                       validate_production_record)
from methods.udc.protocol import (FORMAL_SIZES, OFFICIAL_COMMIT, SEED,
                                  discover_scale_dataset, official_gate,
                                  s3_gate)
from methods.udc.s3_eval import checkpoint_gate, load_models, solve_cvrp, solve_tsp

PRODUCTION_TIMING = (
    "paper_production: BS=1 per-instance solver wall-clock; includes alpha=50 "
    "initial solution generation, all x refinement stages, official algorithm-internal "
    "CPU/GPU work and CVRP route_ranking2 NumPy work; excludes model/checkpoint load, "
    "dataset parsing, adapter preprocessing, independent/Kit validation, JSON/artifact "
    "I/O and aggregation; CUDA synchronized immediately before and after solver"
)
PILOT_TIMING = "budget_freeze_pilot_only; " + PRODUCTION_TIMING
SAFETY_TIMING = "production_safety_gate_only; " + PRODUCTION_TIMING
INTER_INSTANCE_MEMORY_HYGIENE = {
    "python_gc_collect": True,
    "torch_cuda_empty_cache": True,
    "placement": "immediately before each solver interval",
    "included_in_solver_timing": False,
    "rng_effect": "none",
}


def ensure_external_output(path: Path, project_root: Path):
    try:
        path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return
    raise ValueError("UDC production evidence must live outside the project repository")


def exact_gpu(torch) -> dict:
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "RTX 4090" not in torch.cuda.get_device_name(0)):
        raise ValueError("UDC formal production requires exactly one visible RTX 4090")
    return {"gpu_name": torch.cuda.get_device_name(0),
            "visible_device_count": torch.cuda.device_count(),
            "torch": torch.__version__, "numpy": np.__version__,
            "python": sys.version, "platform": platform.platform()}


def seed_once(torch):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def prepare_solver_memory(torch):
    """Release unreachable objects and unused allocator cache before timing."""
    gc.collect()
    torch.cuda.empty_cache()


def load_formal_dataset(path: Path, problem: str, size: int):
    import ml4co_kit as kit
    wrapper = kit.TSPWrapper() if problem == "tsp" else kit.CVRPWrapper()
    wrapper.from_pickle(path)
    expected = kit.TSPTask if problem == "tsp" else kit.CVRPTask
    if not wrapper.task_list or any(type(task) is not expected for task in wrapper.task_list):
        raise ValueError("formal dataset contains an empty or non-exact ML4CO task list")
    if any(np.asarray(task.points).shape != (size, 2) for task in wrapper.task_list):
        raise ValueError("formal dataset task shape differs from requested size")
    names = [str(task.name) for task in wrapper.task_list]
    if len(names) != len(set(names)):
        raise ValueError("formal dataset contains duplicate instance names")
    return wrapper.task_list, kit, {
        "path": str(path.resolve()), "filename": path.name,
        "sha256": sha256_file(path), "count": len(wrapper.task_list),
        "task_class": f"{expected.__module__}.{expected.__name__}",
        "indices": {"first": 0, "last": len(wrapper.task_list) - 1,
                    "order": "0..N-1 without shuffle"},
        "instance_names_sha256": fingerprint(names), "instance_names": names,
    }


def audit_adapter(task, problem: str, size: int):
    if problem == "tsp":
        adapt_tsp_task(task, size=size)
    else:
        adapt_cvrp_task(task, size=size)


def verify_loaded_environment(env, budget):
    if (env.problem_size != budget["sub_size"]
            or env.pomo_size != budget["configured_pomo"]):
        raise ValueError("loaded UDC environment differs from frozen sub-size/POMO")


def common_gates(args, problem: str) -> dict:
    project = production_project_gate(args.project_root)
    official = official_gate(args.official_root)
    if project["pass"] is not True or official["pass"] is not True:
        raise ValueError("project or official source provenance failed closed")
    registry = load_budget_registry(args.registry)
    return {
        "project": project, "official": official,
        "s3": s3_gate(args.s3_evidence), "s4": s4_gate(args.s4_evidence),
        "registry": {"path": registry["path"], "sha256": registry["sha256"]},
        "checkpoints": checkpoint_gate(args.supplemental_root, problem),
    }


def solve_record(task, problem, size, budget, env, harness, torch, device,
                 dataset_index, timing_semantics):
    solve = solve_tsp if problem == "tsp" else solve_cvrp
    raw = solve(task, env, harness, torch, device, dataset_index,
                problem_size=size, timing_semantics=timing_semantics,
                capture_memory=False, x_stages=budget["x"])
    best = raw["best_alpha"]
    if problem == "tsp":
        selected = raw["final_solution_population"][best]
        canonical = raw["solution"]
        official = raw["udc_internal_objective"]
    else:
        selected = raw["solution"]
        canonical = raw["canonical_solution"]
        official = raw["udc_internal_best"]
    record = {
        "dataset_instance_index": dataset_index, "instance_id": raw["instance_id"],
        "problem": problem, "size": size, "budget_label": budget["label"],
        "alpha": budget["alpha"], "x": budget["x"], "best_alpha": best,
        "completed_x_stages": len(raw["completed_stages"]),
        "selected_solution": selected, "canonical_solution": canonical,
        "official_objective": official,
        "independent_objective": raw["independent_objective"],
        "reference_objective": raw["reference_objective"],
        "drop_percent": raw["instance_drop_percent"],
        "runtime_seconds": raw["runtime_seconds"],
        "timing_semantics": timing_semantics,
        "independent_feasible": raw["independent_feasible"],
        "kit_feasible": raw["kit_feasible"],
        "internal_vs_independent": raw["internal_vs_independent"],
        "independent_vs_kit": raw["independent_vs_kit"],
        "kit_objective": raw["kit_objective"],
        "input_semantics": raw["input_semantics"],
        "rng_before_solve": raw["rng_before_solve"],
        "rng_after_solve": raw["rng_after_solve"],
        "evidence_status": "KIT_VALIDATED" if raw["status"] == "KIT_VALIDATED" else "FAILED",
    }
    if problem == "cvrp":
        record.update({"solution_flag": raw["solution_flag"],
                       "decoded_routes": raw["decoded_routes"],
                       "route_count": raw["route_count"],
                       "route_loads": raw["route_demands"],
                       "max_route_load": raw["max_route_load"]})
    validate_production_record(record, problem=problem, size=size,
                               alpha=budget["alpha"], x_stages=budget["x"])
    return record


def one_shot_run(args, *, problem, size, label, indices, timing_semantics):
    import torch
    output_dir = args.output_root.resolve()
    ensure_external_output(output_dir, args.project_root)
    budget = budget_config(problem, label, args.registry)
    gates = common_gates(args, problem)
    environment = exact_gpu(torch)
    dataset_path = discover_scale_dataset(args.dataset_root, problem, size)
    tasks, _kit, dataset = load_formal_dataset(dataset_path, problem, size)
    if any(index < 0 or index >= len(tasks) for index in indices):
        raise ValueError("requested pilot/safety index is outside the formal dataset")
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite existing evidence: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata = {"schema": "udc-paper-one-shot.v1", "state": "IN_PROGRESS",
                "created_at": utc_now(), "problem": problem, "size": size,
                "budget": budget, "indices": list(indices), "dataset": dataset,
                "environment": environment, "gates": gates,
                "timing_semantics": timing_semantics,
                "inter_instance_memory_hygiene": INTER_INSTANCE_MEMORY_HYGIENE,
                "script_sha256": sha256_file(Path(__file__))}
    atomic_json(output_dir / METADATA_FILE, metadata)
    try:
        for index in indices:
            audit_adapter(tasks[index], problem, size)
        seed_once(torch)
        env, harness, loaded, device = load_models(
            problem, args.official_root, args.supplemental_root, torch,
            problem_size=size)
        verify_loaded_environment(env, budget)
        metadata["loaded_checkpoints"] = loaded
        records = []
        for index in indices:
            prepare_solver_memory(torch)
            records.append(solve_record(
                tasks[index], problem, size, budget, env, harness,
                torch, device, index, timing_semantics))
        metadata.update({"state": "KIT_VALIDATED", "completed_records": len(records),
                         "project_post": production_project_gate(args.project_root),
                         "official_post": official_gate(args.official_root)})
        if (metadata["project_post"]["pass"] is not True
                or metadata["official_post"]["pass"] is not True):
            raise RuntimeError("repository provenance changed during one-shot run")
        atomic_json(output_dir / "records.json", records)
        metadata["records_sha256"] = sha256_file(output_dir / "records.json")
    except Exception as exc:
        is_oom = ("cuda" in str(exc).lower() and "out of memory" in str(exc).lower())
        metadata.update({"state": "FAILED",
                         "status": "CUDA_OOM" if is_oom else "FAILED",
                         "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()})
        atomic_json(output_dir / METADATA_FILE, metadata)
        raise
    atomic_json(output_dir / METADATA_FILE, metadata)
    return metadata, records


def _pilot_run_path(root, problem, label):
    return Path(root) / "budget_freeze" / "pilot_runs" / problem / label


def _pilot_summary(records, problem, label, budget):
    objectives = [row["independent_objective"] for row in records]
    runtimes = [row["runtime_seconds"] for row in records]
    return {"problem": problem, "budget_label": label, "alpha": budget["alpha"],
            "x": budget["x"], "indices": list(PILOT_INDICES),
            "mean_objective": float(np.mean(objectives)),
            "mean_runtime_seconds": float(np.mean(runtimes)),
            "completed_x_stages": [row["completed_x_stages"] for row in records],
            "all_kit_validated": all(row["evidence_status"] == "KIT_VALIDATED"
                                     for row in records)}


def finalize_pilot_if_complete(args):
    freeze_root = args.output_root / "budget_freeze"
    runs = {}
    run_project_heads = set()
    for problem in ("tsp", "cvrp"):
        for label in ("fewer", "more"):
            directory = _pilot_run_path(args.output_root, problem, label)
            metadata_path, records_path = directory / METADATA_FILE, directory / "records.json"
            if not metadata_path.is_file() or not records_path.is_file():
                return None
            metadata = json.loads(metadata_path.read_text())
            if (metadata.get("state") != "KIT_VALIDATED"
                    or metadata.get("records_sha256") != sha256_file(records_path)):
                raise ValueError("pilot run artifact is incomplete or changed")
            project_gate = metadata.get("gates", {}).get("project", {})
            project_post = metadata.get("project_post", {})
            official_source = metadata.get("gates", {}).get("official", {})
            official_post = metadata.get("official_post", {})
            if (project_gate.get("pass") is not True
                    or project_post.get("pass") is not True
                    or project_post.get("head") != project_gate.get("head")
                    or official_source.get("pass") is not True
                    or official_source.get("head") != OFFICIAL_COMMIT
                    or official_post.get("pass") is not True
                    or official_post.get("head") != OFFICIAL_COMMIT):
                raise ValueError("pilot run repository provenance failed")
            run_project_heads.add(project_gate.get("head"))
            records = json.loads(records_path.read_text())
            budget = budget_config(problem, label, args.registry)
            if (not isinstance(records, list) or len(records) != len(PILOT_INDICES)
                    or metadata.get("problem") != problem
                    or metadata.get("size") != 500
                    or metadata.get("budget") != budget
                    or metadata.get("indices") != list(PILOT_INDICES)
                    or metadata.get("completed_records") != len(PILOT_INDICES)):
                raise ValueError("pilot run metadata differs from the fixed pilot protocol")
            for row in records:
                validate_production_record(row, problem=problem, size=500,
                                           alpha=budget["alpha"], x_stages=budget["x"])
                if row["budget_label"] != label:
                    raise ValueError("UDC pilot record budget label mismatch")
            if [row["dataset_instance_index"] for row in records] != list(PILOT_INDICES):
                raise ValueError("pilot indices differ from fixed 0,1,2")
            runs[(problem, label)] = {
                "artifact_path": str(directory.resolve()),
                "metadata_sha256": sha256_file(metadata_path),
                "records_sha256": sha256_file(records_path),
                "summary": _pilot_summary(records, problem, label, budget),
                "records": records,
            }
    project = production_project_gate(args.project_root)
    if project["pass"] is not True or run_project_heads != {project["head"]}:
        raise ValueError("pilot runs were not produced from one current project commit")
    pilot = {"schema": "udc-budget-freeze-pilot.v1", "created_at": utc_now(),
             "provenance_label": "PROJECT-SELECTED PAPER BUDGETS",
             "project_head": project["head"],
             "registry_sha256": load_budget_registry(args.registry)["sha256"],
             "fixed_indices": list(PILOT_INDICES),
             "runs": {f"{p}_{b}": value for (p, b), value in runs.items()}}
    atomic_json(freeze_root / "pilot.json", pilot)
    checks = {}
    decisions = {}
    for problem in ("tsp", "cvrp"):
        fewer = runs[(problem, "fewer")]["summary"]
        more = runs[(problem, "more")]["summary"]
        checks[problem] = {
            "all_semantics_pass": fewer["all_kit_validated"] and more["all_kit_validated"],
            "exact_fewer_stage_count": all(value == fewer["x"]
                                             for value in fewer["completed_x_stages"]),
            "exact_more_stage_count": all(value == more["x"]
                                            for value in more["completed_x_stages"]),
            "more_has_larger_fixed_x": more["x"] > fewer["x"],
            "more_mean_runtime_exceeds_fewer": (
                more["mean_runtime_seconds"] > fewer["mean_runtime_seconds"]),
            "more_mean_objective_nonworsening": (
                more["mean_objective"] <= fewer["mean_objective"]),
        }
        decisions[problem] = {"fewer": {"alpha": fewer["alpha"], "x": fewer["x"]},
                              "more": {"alpha": more["alpha"], "x": more["x"]}}
    passed = all(all(values.values()) for values in checks.values())
    decision = {"schema": "udc-budget-freeze-decision.v1", "created_at": utc_now(),
                "status": "FROZEN" if passed else "BLOCKED",
                "provenance_label": "PROJECT-SELECTED PAPER BUDGETS",
                "project_head": project["head"],
                "registry_sha256": pilot["registry_sha256"],
                "pilot_sha256": sha256_file(freeze_root / "pilot.json"),
                "checks": checks, "budgets": decisions,
                "note": ("Budgets were fixed before this pilot; the pilot verifies them "
                         "and does not select or tune x.")}
    atomic_json(freeze_root / "decision.json", decision)
    return decision


def freeze_decision_gate(output_root: Path, registry_path: Path) -> dict:
    freeze_root = Path(output_root) / "budget_freeze"
    pilot_path, decision_path = freeze_root / "pilot.json", freeze_root / "decision.json"
    pilot, decision = json.loads(pilot_path.read_text()), json.loads(decision_path.read_text())
    if (decision.get("status") != "FROZEN"
            or decision.get("pilot_sha256") != sha256_file(pilot_path)
            or decision.get("registry_sha256") != load_budget_registry(registry_path)["sha256"]
            or pilot.get("registry_sha256") != decision["registry_sha256"]):
        raise ValueError("UDC budget freeze evidence failed closed")
    if (not isinstance(decision.get("project_head"), str)
            or pilot.get("project_head") != decision["project_head"]):
        raise ValueError("UDC budget freeze project commit mismatch")
    if not all(all(values.values()) for values in decision.get("checks", {}).values()):
        raise ValueError("UDC budget freeze decision contains a failed check")
    recomputed_checks = {}
    for problem in ("tsp", "cvrp"):
        summaries = {}
        for label in ("fewer", "more"):
            key = f"{problem}_{label}"
            run = pilot.get("runs", {}).get(key, {})
            directory = Path(run.get("artifact_path", ""))
            metadata_path, records_path = directory / METADATA_FILE, directory / "records.json"
            if (run.get("metadata_sha256") != sha256_file(metadata_path)
                    or run.get("records_sha256") != sha256_file(records_path)):
                raise ValueError("UDC pilot run artifact hash integrity failed")
            metadata = json.loads(metadata_path.read_text())
            records = json.loads(records_path.read_text())
            budget = budget_config(problem, label, registry_path)
            if (metadata.get("state") != "KIT_VALIDATED"
                    or metadata.get("records_sha256") != sha256_file(records_path)
                    or [row.get("dataset_instance_index") for row in records]
                    != list(PILOT_INDICES)):
                raise ValueError("UDC pilot run is incomplete")
            if (metadata.get("gates", {}).get("project", {}).get("head")
                    != decision["project_head"]
                    or metadata.get("gates", {}).get("official", {}).get("head")
                    != OFFICIAL_COMMIT):
                raise ValueError("UDC pilot repository provenance mismatch")
            for row in records:
                validate_production_record(row, problem=problem, size=500,
                                           alpha=budget["alpha"], x_stages=budget["x"])
            if run.get("records") != records:
                raise ValueError("UDC pilot.json records differ from run artifact")
            summary = _pilot_summary(records, problem, label, budget)
            stored = run.get("summary", {})
            if (summary != stored):
                raise ValueError("UDC pilot summary differs from validated records")
            summaries[label] = summary
        fewer, more = summaries["fewer"], summaries["more"]
        recomputed_checks[problem] = {
            "all_semantics_pass": fewer["all_kit_validated"] and more["all_kit_validated"],
            "exact_fewer_stage_count": all(value == fewer["x"]
                                             for value in fewer["completed_x_stages"]),
            "exact_more_stage_count": all(value == more["x"]
                                            for value in more["completed_x_stages"]),
            "more_has_larger_fixed_x": more["x"] > fewer["x"],
            "more_mean_runtime_exceeds_fewer": (
                more["mean_runtime_seconds"] > fewer["mean_runtime_seconds"]),
            "more_mean_objective_nonworsening": (
                more["mean_objective"] <= fewer["mean_objective"]),
        }
    if decision.get("checks") != recomputed_checks:
        raise ValueError("UDC budget freeze checks differ from pilot recomputation")
    expected_budgets = {
        problem: {label: {"alpha": budget_config(problem, label, registry_path)["alpha"],
                          "x": budget_config(problem, label, registry_path)["x"]}
                  for label in ("fewer", "more")}
        for problem in ("tsp", "cvrp")}
    if decision.get("budgets") != expected_budgets:
        raise ValueError("UDC budget freeze decision differs from registry")
    return {"pilot_path": str(pilot_path.resolve()), "pilot_sha256": sha256_file(pilot_path),
            "decision_path": str(decision_path.resolve()),
            "decision_sha256": sha256_file(decision_path),
            "project_head": decision["project_head"], "pass": True}


def run_pilot(args):
    if args.problem not in ("tsp", "cvrp") or args.budget not in ("fewer", "more"):
        raise ValueError("pilot requires --problem and --budget")
    args.output_root = args.output_root.resolve()
    run_args = argparse.Namespace(**vars(args))
    run_args.output_root = _pilot_run_path(args.output_root, args.problem, args.budget)
    one_shot_run(run_args, problem=args.problem, size=500, label=args.budget,
                 indices=PILOT_INDICES, timing_semantics=PILOT_TIMING)
    decision = finalize_pilot_if_complete(args)
    print(decision if decision else "PILOT_RUN_PASS; remaining fixed pilot runs required")


def run_safety(args):
    cell = (args.problem, args.size, args.budget)
    if cell not in SAFETY_CELLS:
        raise ValueError(f"safety-gate must be one of {SAFETY_CELLS}")
    freeze = freeze_decision_gate(args.output_root, args.registry)
    if freeze["project_head"] != production_project_gate(args.project_root)["head"]:
        raise ValueError("safety gate project commit differs from budget freeze commit")
    run_args = argparse.Namespace(**vars(args))
    run_args.output_root = (args.output_root / "production_safety_gate" /
                            f"{args.problem}{args.size}" / args.budget)
    metadata, records = one_shot_run(
        run_args, problem=args.problem, size=args.size, label=args.budget,
        indices=(0,), timing_semantics=SAFETY_TIMING)
    metadata["budget_freeze"] = freeze
    metadata["status"] = "PASS"
    metadata["exact_completed_stage_count"] = (
        records[0]["completed_x_stages"] == metadata["budget"]["x"])
    if metadata["exact_completed_stage_count"] is not True:
        raise RuntimeError("safety gate did not complete exact x stages")
    atomic_json(run_args.output_root / METADATA_FILE, metadata)
    print(run_args.output_root / METADATA_FILE)


def safety_gate_evidence(output_root: Path, freeze: dict) -> dict:
    rows = []
    for problem, size, label in SAFETY_CELLS:
        directory = Path(output_root) / "production_safety_gate" / f"{problem}{size}" / label
        metadata_path, records_path = directory / METADATA_FILE, directory / "records.json"
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("status") != "PASS" or metadata.get("state") != "KIT_VALIDATED"
                or metadata.get("exact_completed_stage_count") is not True
                or metadata.get("records_sha256") != sha256_file(records_path)
                or metadata.get("budget_freeze", {}).get("decision_sha256")
                != freeze["decision_sha256"]):
            raise ValueError("UDC largest-size more safety gate failed closed")
        rows.append({"problem": problem, "size": size, "budget": label,
                     "metadata_sha256": sha256_file(metadata_path),
                     "records_sha256": sha256_file(records_path)})
    return {"required_cells": rows, "pass": True}


def production_identity(args, problem, size, budget, dataset, environment, gates,
                        freeze, safety):
    return {
        "schema": "udc-paper-production-identity.v1", "method": "UDC",
        "problem": problem, "size": size, "budget": budget,
        "project": gates["project"], "official": gates["official"],
        "s3": gates["s3"], "s4": gates["s4"], "registry": gates["registry"],
        "budget_freeze": freeze, "safety_gates": safety,
        "checkpoints": gates["checkpoints"], "dataset": dataset,
        "environment": environment, "batch_size": 1, "seed_once": SEED,
        "rng_policy": ("seed Python/NumPy/Torch CPU/CUDA once before model construction; "
                       "process indices 0..N-1 continuously; restore serialized RNG state "
                       "on resume; never reseed per instance"),
        "inter_instance_memory_hygiene": INTER_INSTANCE_MEMORY_HYGIENE,
        "timing_semantics": PRODUCTION_TIMING,
        "source": {"script": str(Path(__file__).resolve()),
                   "script_sha256": sha256_file(Path(__file__)),
                   "results_sha256": sha256_file(ROOT / "methods/udc/paper_results.py")},
    }


def run_production(args):
    import torch
    problem, size, label = args.problem, args.size, args.budget
    if problem not in FORMAL_SIZES or size not in FORMAL_SIZES[problem]:
        raise ValueError("production problem/size is outside formal UDC scope")
    if label not in ("fewer", "more"):
        raise ValueError("production requires --budget fewer or more")
    ensure_external_output(args.output_root, args.project_root)
    freeze = freeze_decision_gate(args.output_root, args.registry)
    if freeze["project_head"] != production_project_gate(args.project_root)["head"]:
        raise ValueError("production project commit differs from budget freeze commit")
    safety = safety_gate_evidence(args.output_root, freeze)
    budget = budget_config(problem, label, args.registry)
    gates = common_gates(args, problem)
    environment = exact_gpu(torch)
    dataset_path = discover_scale_dataset(args.dataset_root, problem, size)
    tasks, _kit, dataset = load_formal_dataset(dataset_path, problem, size)
    output_dir = args.output_root / f"{problem}{size}" / label
    identity = production_identity(args, problem, size, budget, dataset, environment,
                                   gates, freeze, safety)
    metadata, records, checkpoint = initialize_or_resume(output_dir, identity)
    if metadata["state"] == "KIT_VALIDATED":
        print(output_dir / "summary.json")
        return
    seed_once(torch)
    env, harness, loaded, device = load_models(
        problem, args.official_root, args.supplemental_root, torch,
        problem_size=size)
    verify_loaded_environment(env, budget)
    if loaded != gates["checkpoints"]:
        raise RuntimeError("loaded checkpoint evidence differs from gated checkpoints")
    if checkpoint is None:
        checkpoint = initialize_rng_checkpoint(
            output_dir, identity, capture_rng_state(torch))
    else:
        restore_rng_state(checkpoint["rng_state_after_committed_prefix"], torch)
    try:
        for index in range(len(records), len(tasks)):
            audit_adapter(tasks[index], problem, size)
            prepare_solver_memory(torch)
            record = solve_record(tasks[index], problem, size, budget, env, harness,
                                  torch, device, index, PRODUCTION_TIMING)
            records, checkpoint = commit_record(
                output_dir, identity, records, record, capture_rng_state(torch))
            print(json.dumps({"problem": problem, "size": size, "budget": label,
                              "dataset_instance_index": index,
                              "runtime_seconds": record["runtime_seconds"],
                              "validated_records": len(records)}, sort_keys=True), flush=True)
        project_post = production_project_gate(args.project_root)
        official_post = official_gate(args.official_root)
        if project_post["pass"] is not True or official_post["pass"] is not True:
            raise RuntimeError("repository provenance changed during production")
        summary = finalize_run(output_dir, identity, records)
        metadata = json.loads((output_dir / METADATA_FILE).read_text())
        metadata.update({"project_post": project_post, "official_post": official_post})
        atomic_json(output_dir / METADATA_FILE, metadata)
        print(json.dumps(summary, sort_keys=True))
    except Exception as exc:
        metadata = json.loads((output_dir / METADATA_FILE).read_text())
        is_oom = ("cuda" in str(exc).lower() and "out of memory" in str(exc).lower())
        metadata.update({"state": "FAILED", "status": "FAILED",
                         "failure_kind": "CUDA_OOM" if is_oom else type(exc).__name__,
                         "failed_at_index": len(records),
                         "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc(), "failed_at": utc_now()})
        atomic_json(output_dir / METADATA_FILE, metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "safety-gate", "production", "aggregate"),
                        required=True)
    parser.add_argument("--problem", choices=("tsp", "cvrp"))
    parser.add_argument("--size", type=int)
    parser.add_argument("--budget", choices=("fewer", "more"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--supplemental-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--s3-evidence", type=Path)
    parser.add_argument("--s4-evidence", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.registry = args.registry.resolve()
    if args.mode == "aggregate":
        from methods.udc.build_paper_table import build_and_write
        build_and_write(args.output_root, args.output_root)
        return
    required = ("project_root", "official_root", "supplemental_root", "dataset_root",
                "s3_evidence", "s4_evidence")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"mode {args.mode} requires: {', '.join(missing)}")
    for name in required:
        setattr(args, name, getattr(args, name).resolve())
    if args.mode == "pilot":
        run_pilot(args)
    elif args.mode == "safety-gate":
        if args.size is None:
            parser.error("safety-gate requires --size")
        run_safety(args)
    else:
        if args.size is None:
            parser.error("production requires --size")
        run_production(args)


if __name__ == "__main__":
    main()
