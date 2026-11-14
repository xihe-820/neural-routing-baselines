"""Each invocation owns exactly one solver's imports, RNG stream and budget."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from common.provenance import environment_provenance, source_provenance
from methods.glop.paper_results import fingerprint
from problems.tsp.validate import validate as validate_tsp
from rebuttal.tsp2k.artifacts import Writer, indices_for
from rebuttal.tsp2k.audit import verify_lock
from rebuttal.tsp2k.config import COUNT, SWEEPS, protocol

ROOT = Path(__file__).resolve().parents[2]

SIL_WARMUP_POLICY = {
    "quality": {"batches": 0},
    "timing": {"batches": 1, "original_instance_batch_size": 1,
               "rng_state_restored": True, "excluded_from_timing": True},
}


def validated(task, index, canonical, official, runtime=None, **extra):
    canonical = np.asarray(canonical, dtype=np.int64).tolist()
    checked = validate_tsp(np.asarray(task.points), canonical)
    independent = checked["independent_objective"]
    kit_feasible = bool(task.check_constraints(np.asarray(canonical)))
    kit_obj = float(task.evaluate(np.asarray(canonical)))
    reference = float(task.evaluate(task.ref_sol))
    return {
        "instance_index": index, "canonical_solution": canonical,
        "objective": independent, "official_objective": official,
        "reference_objective": reference,
        "gap_percent": (independent-reference)/reference*100 if independent is not None else None,
        "runtime_seconds": runtime, "feasible": bool(checked["feasible"]),
        "kit_feasible": kit_feasible, "kit_objective": kit_obj,
        "official_objective_agrees": independent is not None and objective_agrees(official, independent),
        "kit_objective_agrees": independent is not None and objective_agrees(kit_obj, independent),
        **extra,
    }


def glop_execute(tasks, config, spec, torch, device, indices, writer):
    from methods.glop.runtime import (activate_upstream, load_revisers,
        make_shared_tsp_orders, official_seeded_setup, run_warmup_isolated,
        random_insertion_identity)
    from methods.glop.tsp.paper_eval import _solve_one
    from methods.glop.paper_results import tsp_runtime_accounting
    activate_upstream(spec["upstream_path"])
    from utils.functions import load_model, load_problem, reconnect
    from utils.insertion import random_insertion_parallel
    insertion = random_insertion_identity()
    revisers, assets = official_seeded_setup(
        torch, config["seed"], lambda: load_revisers(
            spec["asset_root"], config, device=device, torch=torch, load_model=load_model))
    started = time.perf_counter()
    orders = make_shared_tsp_orders(torch, problem_size=2000, width=1)
    shared_seconds = time.perf_counter() - started
    writer.metadata["runtime_setup"] = {
        "revisers": assets, "insertion": insertion,
        "shared_orders_sha256": fingerprint([v.tolist() for v in orders]),
        "shared_ri_order_generation_seconds": shared_seconds}
    writer.save()

    def solve(index, timed):
        return _solve_one(
            np.asarray(tasks[index].points, dtype=np.float32),
            protocol=config, orders=orders, revisers=revisers, device=device,
            torch=torch, reconnect=reconnect, load_problem=load_problem,
            random_insertion_parallel=random_insertion_parallel, timed=timed)

    run_warmup_isolated(torch, device,
                        lambda: [solve(i, False) for i in indices[:2]])
    for index in indices:
        canonical, official, decoding, seconds = solve(index, True)
        runtime, components = tsp_runtime_accounting(index, seconds, shared_seconds)
        writer.append(validated(tasks[index], index, canonical, official, runtime,
                                source="quality", runtime_components=components,
                                decoding=decoding))


def udc_execute(tasks, config, spec, torch, device, indices, writer):
    from methods.udc.paper_production import (cuda_allocator_gate, seed_once,
        prepare_solver_memory, verify_loaded_environment, PRODUCTION_TIMING)
    from methods.udc.s3_eval import load_models, solve_tsp
    cuda_allocator_gate()
    seed_once(torch)
    env, harness, assets, device = load_models(
        "tsp", Path(spec["upstream_path"]), Path(spec["supplemental_root"]),
        torch, problem_size=2000)
    verify_loaded_environment(env, config)
    writer.metadata["runtime_setup"] = {"loaded_checkpoints": assets}
    writer.save()
    # Production performs no warm-up and never reseeds individual instances.
    for index in indices:
        prepare_solver_memory(torch)
        row = solve_tsp(tasks[index], env, harness, torch, device, index,
                        problem_size=2000, timing_semantics=PRODUCTION_TIMING,
                        capture_memory=False, x_stages=config["x"])
        if (row["status"] != "KIT_VALIDATED" or
                len(row["completed_stages"]) != config["x"]):
            raise RuntimeError("UDC solver/semantic gate failed")
        writer.append(validated(
            tasks[index], index, row["solution"], row["udc_internal_objective"],
            row["runtime_seconds"], source="quality", native_evidence=row))


def sil_execute(tasks, config, spec, torch, device, indices, writer):
    from methods.sil.runtime import (build_tester, run_rng_preserving_warmup,
                                    solve_batch, solve_one)
    from methods.sil.author_batch_eval import prepare_batch, validate_batch
    budget = config["budget"]

    def tester():
        result, actual = build_tester(
            problem="tsp", problem_size=2000, budget_label="fewer",
            upstream=Path(spec["upstream_path"]), checkpoint=Path(spec["checkpoint"]),
            device=device, torch=torch, rebuttal_prc_budget=budget)
        if actual != config:
            raise ValueError("constructed SIL config differs from integer PRC request")
        return result

    # Separate official tester construction resets the same official seed for each path.
    for role, selected in (("quality", indices), ("timing", writer.identity["timing_indices"])):
        current = tester()
        if role == "timing":
            warm = prepare_batch("tsp", [tasks[0]], 2000,
                                 device=device, torch=torch)
            run_rng_preserving_warmup(
                lambda: solve_one(current, problem="tsp", problem_size=2000,
                    inject=warm["inject"], device=device, torch=torch,
                    timed=False), torch=torch)
        groups = [selected] if role == "quality" else [[i] for i in selected]
        for group in groups:
            prepared = prepare_batch("tsp", [tasks[i] for i in group], 2000,
                                     device=device, torch=torch)
            if role == "quality":
                solved = solve_batch(current, problem="tsp", problem_size=2000,
                    batch_size=len(group), inject=prepared["inject"],
                    device=device, torch=torch, timed=False)
                solutions, objectives = solved["solutions"], solved["official_objectives"]
                runtime = None
            else:
                solved = solve_one(current, problem="tsp", problem_size=2000,
                    inject=prepared["inject"], device=device, torch=torch, timed=True)
                solutions = np.asarray([solved["solution"]])
                objectives = np.asarray([solved["official_objective"]])
                runtime = solved["runtime_seconds"]
            native = validate_batch("tsp", [tasks[i] for i in group], prepared,
                solutions, objectives, 2000, dataset_offset=group[0])
            for index, row in zip(group, native):
                writer.append(validated(tasks[index], index, row["canonical_solution"],
                    row["official_objective"], runtime, source=role,
                    native_evidence=row, solution_capture=solved["capture"]))


def main(argv=None, *, method=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if method is None:
        parser.add_argument("--method", choices=list(SWEEPS), required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--phase", choices=["calibration", "fullset"], required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--timing-count", type=int, default=3)
    parser.add_argument("--asset-lock", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dump-config", action="store_true")
    args = parser.parse_args(argv)
    method = method or args.method
    config = protocol(method, args.budget)
    indices = indices_for(args.phase, args.count)
    if type(args.timing_count) is not int or not 1 <= args.timing_count <= COUNT:
        raise ValueError("invalid timing sample count")
    timing_indices = list(range(min(args.timing_count, len(indices)))) if method == "SIL" else []
    if args.dump_config:
        print(json.dumps({**config, "phase": args.phase, "dataset_indices": indices,
                          "timing_indices": timing_indices}, indent=2, allow_nan=False))
        return 0
    if args.asset_lock is None or args.output_dir is None:
        parser.error("execution requires --asset-lock and --output-dir")
    output = args.output_dir.resolve()
    if output == ROOT or ROOT in output.parents or output.exists():
        raise ValueError("new output must be outside source checkout and must not exist")
    lock = json.loads(args.asset_lock.read_text())
    spec = verify_lock(lock, method, ROOT)
    if method == "UDC":
        from methods.udc.paper_production import cuda_allocator_gate
        cuda_allocator_gate()  # Must precede importing Torch.
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            "RTX 4090" not in torch.cuda.get_device_name(0)):
        raise ValueError("exactly one visible RTX 4090 is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    import ml4co_kit as kit
    wrapper = kit.TSPWrapper()
    wrapper.from_pickle(lock["dataset"]["path"])
    tasks = wrapper.task_list
    if len(tasks) != COUNT or any(type(t) is not kit.TSPTask or
            np.asarray(t.points).shape != (2000, 2) or
            not np.isfinite(t.points).all() for t in tasks):
        raise ValueError("dataset no longer has exact 64 TSP2K ML4CO tasks")
    refs = [float(t.evaluate(t.ref_sol)) for t in tasks]
    if (not np.isfinite(refs).all() or min(refs) <= 0 or
            fingerprint(refs) != lock["dataset"]["reference_objectives_sha256"]):
        raise ValueError("benchmark reference identity mismatch")
    sources = source_provenance([
        p for folder in ("rebuttal/tsp2k", "common", f"methods/{method.lower()}", "problems/tsp")
        for p in (ROOT / folder).rglob("*.py")], root=ROOT)
    timing = {}
    if method == "GLOP":
        from methods.glop.paper_results import TSP_TIMING_SEMANTICS
        timing["semantics"] = TSP_TIMING_SEMANTICS
        timing["warmup_instances"] = min(2, len(indices))
    elif method == "UDC":
        from methods.udc.paper_production import PRODUCTION_TIMING, CUDA_ALLOCATOR_PROVENANCE
        timing.update(semantics=PRODUCTION_TIMING, warmup_instances=0,
                      cuda_allocator=CUDA_ALLOCATOR_PROVENANCE)
    else:
        from methods.sil.paper_results import TIMING_SEMANTICS
        timing.update(semantics=TIMING_SEMANTICS, warmup=SIL_WARMUP_POLICY)
    identity = {
        **config, "phase": args.phase, "requested_count": len(indices),
        "dataset_indices": indices, "timing_indices": timing_indices,
        "actual_quality_batch_size": len(indices) if method == "SIL" else 1,
        "dataset": lock["dataset"], "project": lock["projects"][method],
        "project_commit": lock["projects"][method]["commit"],
        "upstream_commit": spec["upstream"]["commit"], "upstream": spec["upstream"],
        "checkpoint_sha256": {Path(a["path"]).name + ":" + str(i): a["sha256"]
                              for i, a in enumerate(spec["assets"])},
        "asset_lock": {"path": str(args.asset_lock.resolve()), "sha256": sha256_file(args.asset_lock)},
        "asset_spec": spec, "environment": environment_provenance(device),
        "gpu": torch.cuda.get_device_name(device),
        "source_provenance": sources, "timing": timing,
        "evidence_kind": "SERVER_MEASUREMENT",
    }
    writer = Writer(output, identity)
    try:
        {"GLOP": glop_execute, "UDC": udc_execute, "SIL": sil_execute}[method](
            tasks, config["full_solver_config"], spec, torch, device, indices, writer)
        verify_lock(lock, method, ROOT)
        summary = writer.finish()
    except Exception as exc:
        writer.fail(exc)
        raise
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
