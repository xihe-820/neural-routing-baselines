#!/usr/bin/env python3
"""Formal NeuOpt TSP100 production for frozen fewer/more and native BS1/16/128."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from common.provenance import (environment_provenance, git_provenance,
                               normalize_git_repository_identity,
                               source_provenance)
from methods.neuopt.cvrp.compat import ensure_tensorboard_logger
from methods.neuopt.tsp.adapter import adapt_batch
from methods.neuopt.tsp.compat import (configure_decoder,
                                       record_compatibility_provenance,
                                       replay_record_compatibility)
from methods.neuopt.tsp.config import supported_config
from methods.neuopt.tsp.decode import decode_successor, extract_final_best
from methods.neuopt.tsp.paper_protocol import (
    D2A, GRAPH_SIZE, K, SEED, STALL_LIMIT, UPSTREAM_COMMIT, UPSTREAM_URL,
    formal_protocol, protocol_fingerprint,
)
from methods.neuopt.tsp.production_results import (
    RNG_POLICY, TIMING_SEMANTICS, WARMUP_POLICY, append_batch, finalize_run,
    initialize_or_resume, mark_failed,
)
from methods.neuopt.tsp.validate_with_kit import validate_task
from problems.tsp.validate import validate as validate_tsp


def cuda_device(value, torch):
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal NeuOpt TSP100 production requires CUDA")
    gpu = torch.cuda.get_device_name(device)
    if "RTX 4090" not in gpu:
        raise RuntimeError(f"formal NeuOpt TSP100 production requires RTX 4090; observed {gpu!r}")
    return device


def seed_inference(torch):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng(torch):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state, torch):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def rng_states_equal(left, right, torch):
    left_numpy, right_numpy = left["numpy"], right["numpy"]
    numpy_equal = (
        left_numpy[0] == right_numpy[0]
        and np.array_equal(left_numpy[1], right_numpy[1])
        and left_numpy[2:] == right_numpy[2:]
    )
    return (
        left["python"] == right["python"] and numpy_equal
        and torch.equal(left["torch"], right["torch"])
        and len(left["cuda"]) == len(right["cuda"])
        and all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))
    )


def rng_state_fingerprint(state):
    digest = hashlib.sha256()
    digest.update(repr(state["python"]).encode())
    numpy_state = state["numpy"]
    digest.update(str(numpy_state[0]).encode())
    digest.update(np.asarray(numpy_state[1]).tobytes())
    digest.update(repr(numpy_state[2:]).encode())
    digest.update(state["torch"].detach().cpu().numpy().tobytes())
    for cuda_state in state["cuda"]:
        digest.update(cuda_state.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def official_option_args(*, config, checkpoint, device, batch_size):
    args = [
        "--problem", "tsp", "--graph_size", str(GRAPH_SIZE), "--eval_only",
        "--init_val_met", config["init_val_met"], "--seed", str(config["seed"]),
        "--val_m", str(config["val_m"]), "--stall_limit", str(config["stall_limit"]),
        "--k", str(config["k"]), "--T_max", str(config["T_max"]),
        "--val_size", str(batch_size), "--val_batch_size", str(batch_size),
        "--load_path", str(Path(checkpoint).resolve()), "--no_tb", "--no_saving",
        "--no_DDP", "--no_progress_bar",
    ]
    if device.type == "cpu":
        args.append("--no_cuda")
    return args


def solve_batch(agent, problem, native, *, config, batch_size, device, torch,
                timed, record):
    if tuple(native["coordinates"].shape) != (batch_size, GRAPH_SIZE, 2):
        raise ValueError("native input is not the declared complete TSP100 batch")
    if timed:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
    with torch.no_grad():
        output = agent.rollout(
            problem, T=config["T_max"], val_m=config["val_m"],
            stall_limit=config["stall_limit"], batch=native,
            record=record, show_bar=False)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started if timed else None
    return output, elapsed


def timed_rollout_with_evidence_replay(agent, problem, native, *, config,
                                       batch_size, device, torch):
    """Time one native record-free batch, then replay it from the exact RNG start."""
    before = capture_rng(torch)
    timed_output, elapsed = solve_batch(
        agent, problem, native, config=config, batch_size=batch_size,
        device=device, torch=torch, timed=True, record=False)
    timed_after = capture_rng(torch)
    restore_rng(before, torch)
    with replay_record_compatibility(problem) as compatibility:
        replay_output, _ = solve_batch(
            agent, problem, native, config=config, batch_size=batch_size,
            device=device, torch=torch, timed=False, record=True)
    replay_after = capture_rng(torch)
    if not torch.equal(timed_output[0], replay_output[0]):
        raise RuntimeError("record=True replay objective differs from timed record=False objective")
    if not rng_states_equal(timed_after, replay_after, torch):
        raise RuntimeError("record=True replay consumed different RNG than timed rollout")
    return timed_output[0], replay_output, elapsed, {
        "timed_rollout_record": False,
        "evidence_replay_record": True,
        "rng_state_restored": True,
        "timed_replay_official_objective_exact": True,
        "timed_replay_rng_after_exact": True,
        "cuda_synchronized_before_timing": True,
        "cuda_synchronized_after_timing": True,
        "evidence_replay_excluded_from_runtime": True,
    }, compatibility, {
        "before_sha256": rng_state_fingerprint(before),
        "after_sha256": rng_state_fingerprint(timed_after),
    }


def _task_name(task, index):
    value = getattr(task, "name", None)
    return str(value) if value is not None else f"tsp100-{index}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=["preflight", "fullset"], required=True)
    parser.add_argument("--problem-size", type=int, choices=[100], default=100)
    parser.add_argument("--budget", choices=["fewer", "more"], required=True)
    parser.add_argument("--T-max", dest="T_max", type=int, required=True)
    parser.add_argument("--batch-size", type=int, choices=[1, 16, 128], required=True)
    parser.add_argument("--d2a", "--val-m", dest="val_m", type=int, default=D2A)
    parser.add_argument("--stall-limit", type=int, default=STALL_LIMIT)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--warmup-batches", type=int, choices=[1], default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    expected = supported_config(args.problem_size)
    protocol = formal_protocol(
        args.problem_size, budget=args.budget, batch_size=args.batch_size,
        T_max=args.T_max, d2a=args.val_m, stall_limit=args.stall_limit, k=args.k)
    if args.dataset.name != expected["dataset_filename"]:
        raise ValueError(f"dataset filename must be exactly {expected['dataset_filename']}")
    if args.checkpoint.name != "tsp100.pt":
        raise ValueError("checkpoint filename must be exactly tsp100.pt")
    expected_checkpoint_path = (
        args.upstream.resolve() / expected["checkpoint_relative_path"])
    if args.checkpoint.resolve() != expected_checkpoint_path:
        raise ValueError("checkpoint must be the pinned upstream pre-trained/tsp100.pt path")
    dataset_hash, checkpoint_hash = sha256_file(args.dataset), sha256_file(args.checkpoint)
    if dataset_hash != expected["dataset_sha256"]:
        raise ValueError("dataset is not the pinned TSP100 Concorde benchmark")
    if checkpoint_hash != expected["checkpoint_sha256"]:
        raise ValueError("checkpoint is not the pinned official NeuOpt TSP100 checkpoint")
    upstream = git_provenance(args.upstream)
    if (upstream["commit"] != UPSTREAM_COMMIT or upstream["dirty"] or
            normalize_git_repository_identity(upstream["url"]) !=
            normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("official NeuOpt checkout identity/cleanliness mismatch")
    project = git_provenance(ROOT)
    if project["dirty"]:
        raise ValueError("formal NeuOpt TSP100 production requires a clean project checkout")

    import ml4co_kit as kit
    wrapper = kit.TSPWrapper()
    wrapper.from_pickle(args.dataset)
    if len(wrapper.task_list) != expected["dataset_count"]:
        raise ValueError("TSP100 dataset count does not match the pinned identity")
    count = args.batch_size if args.scope == "preflight" else expected["dataset_count"]
    indices = list(range(count))
    tasks = [wrapper.task_list[index] for index in indices]
    for index, task in zip(indices, tasks):
        points = np.asarray(task.points)
        if type(task) is not kit.TSPTask or points.shape != (100, 2) or not np.isfinite(points).all():
            raise ValueError(f"dataset index {index} is not an exact finite ML4CO TSP100 task")

    import torch
    device = cuda_device(args.device, torch)
    seed_inference(torch)
    environment = environment_provenance(device)
    for module_name in list(sys.modules):
        if module_name == "problems" or module_name.startswith("problems."):
            del sys.modules[module_name]
    sys.path.insert(0, str(args.upstream.resolve()))
    tensorboard_compatibility = ensure_tensorboard_logger()
    from options import get_options
    from problems.problem_tsp import TSP
    from nets.graph_layers import kopt_Decoder
    decoder_compatibility = configure_decoder(
        kopt_Decoder, original_batch_size=args.batch_size, val_m=protocol["val_m"])
    from agent.ppo import PPO

    option_args = official_option_args(
        config=protocol, checkpoint=args.checkpoint, device=device,
        batch_size=args.batch_size)
    opts = get_options(option_args)
    opts.device, opts.use_cuda, opts.distributed, opts.world_size = device, True, False, 1
    problem = TSP(
        p_size=GRAPH_SIZE, init_val_met=opts.init_val_met,
        with_assert=opts.use_assert, DUMMY_RATE=opts.dummy_rate, k=opts.k,
        with_bonus=not opts.wo_bonus, with_regular=not opts.wo_regular)
    if problem.size != GRAPH_SIZE:
        raise ValueError("official NeuOpt problem construction is not TSP100")
    record_compatibility = record_compatibility_provenance(problem)
    agent = PPO(problem, opts)
    if any(hasattr(agent, name) for name in ("critic", "optimizer", "lr_scheduler")):
        raise RuntimeError("eval_only PPO unexpectedly constructed training components")
    agent.load(str(args.checkpoint.resolve()))
    agent.eval()
    problem.eval()

    task_batches = [tasks[start:start + args.batch_size]
                    for start in range(0, len(tasks), args.batch_size)]
    index_batches = [indices[start:start + args.batch_size]
                     for start in range(0, len(indices), args.batch_size)]
    if any(len(batch) != args.batch_size for batch in task_batches):
        raise ValueError("partial native batch is forbidden")

    def native_for(batch):
        points = np.stack([np.asarray(task.points) for task in batch], axis=0)
        return adapt_batch(points, problem_size=GRAPH_SIZE, device=device)

    _, adapter_mapping = native_for(task_batches[0])
    sources = source_provenance([
        Path(__file__), Path(__file__).with_name("production_results.py"),
        Path(__file__).with_name("paper_protocol.py"),
        Path(__file__).with_name("config.py"), Path(__file__).with_name("compat.py"),
        Path(__file__).with_name("adapter.py"), Path(__file__).with_name("decode.py"),
        Path(__file__).with_name("validate_with_kit.py"),
        ROOT / "methods/neuopt/cvrp/compat.py",
        ROOT / "problems/tsp/validate.py", ROOT / "problems/tsp/objective.py",
        ROOT / "common/objective_agreement.py", ROOT / "common/provenance.py",
    ], root=ROOT)
    identity = {
        "method": "NeuOpt", "variant": "NeuOpt-GIRE", "problem": "TSP",
        "problem_size": 100, "paper_protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "scope": args.scope, "expected_indices": indices,
        "native_batch_count": len(task_batches),
        "artifact_class": "formal_production_separate_from_calibration",
        "project": project, "upstream": upstream,
        "upstream_checkout_path": str(args.upstream.resolve()),
        "checkpoint": {"path": str(args.checkpoint.resolve()),
                       "filename": args.checkpoint.name,
                       "relative_path": expected["checkpoint_relative_path"],
                       "sha256": checkpoint_hash},
        "dataset": {"path": str(args.dataset.resolve()), "filename": args.dataset.name,
                    "sha256": dataset_hash, "count": len(wrapper.task_list),
                    "reference_source": "ML4CO TSPTask.ref_sol evaluated by TSPTask.evaluate"},
        "adapter_mapping": adapter_mapping,
        "native_batching": {
            "one_official_rollout_call_per_batch": True,
            "bs1_loop_emulation": False,
            "partial_batches": False,
        },
        "warmup": dict(WARMUP_POLICY), "rng_policy": RNG_POLICY,
        "environment": environment,
        "tensorboard_compatibility": tensorboard_compatibility,
        "decoder_compatibility": decoder_compatibility,
        "record_compatibility": record_compatibility,
        "source_provenance": sources, "timing_semantics": TIMING_SEMANTICS,
    }
    metadata, prior_records, prior_timings = initialize_or_resume(args.output_dir, identity)

    try:
        state_before_warmup = capture_rng(torch)
        for batch in task_batches[:args.warmup_batches]:
            native, _ = native_for(batch)
            solve_batch(
                agent, problem, native, config=protocol, batch_size=args.batch_size,
                device=device, torch=torch, timed=False, record=False)
        restore_rng(state_before_warmup, torch)

        completed_batches = len(prior_timings)
        for batch_index in range(completed_batches):
            if rng_state_fingerprint(capture_rng(torch)) != prior_timings[batch_index]["rng_before_sha256"]:
                raise RuntimeError("resume RNG state before prefix batch does not match evidence")
            native, _ = native_for(task_batches[batch_index])
            output, _ = solve_batch(
                agent, problem, native, config=protocol, batch_size=args.batch_size,
                device=device, torch=torch, timed=False, record=False)
            saved = torch.tensor([
                record["timed_official_objective"]
                for record in prior_records[batch_index * args.batch_size:
                                            (batch_index + 1) * args.batch_size]
            ], dtype=output[0].dtype, device=output[0].device)
            if not torch.equal(output[0], saved):
                raise RuntimeError("resume RNG prefix replay does not reproduce saved objectives")
            if rng_state_fingerprint(capture_rng(torch)) != prior_timings[batch_index]["rng_after_sha256"]:
                raise RuntimeError("resume RNG state after prefix batch does not match evidence")

        for batch_index in range(completed_batches, len(task_batches)):
            batch, current_indices = task_batches[batch_index], index_batches[batch_index]
            native, current_mapping = native_for(batch)
            if current_mapping != adapter_mapping:
                raise ValueError("dataset batches do not share one exact adapter mapping")
            (timed_tensor, replay_output, elapsed, timed_replay,
             current_compatibility, rng_fingerprints) = (
                timed_rollout_with_evidence_replay(
                    agent, problem, native, config=protocol, batch_size=args.batch_size,
                    device=device, torch=torch))
            if current_compatibility != record_compatibility:
                raise RuntimeError("record=True compatibility provenance changed during production")
            replay_tensor, successor_tensor = extract_final_best(
                replay_output, batch_size=args.batch_size, val_m=1)
            if not torch.equal(timed_tensor, replay_tensor):
                raise RuntimeError("selected replay successors differ from timed objectives")
            official_orders = problem.get_order(
                successor_tensor, return_solution=True).detach().cpu().tolist()
            official_recomputed = problem.get_costs(
                native, successor_tensor, get_context=False,
                check_full_feasibility=True).detach().cpu()
            records = []
            for position, (task, dataset_index) in enumerate(zip(batch, current_indices)):
                successor = successor_tensor[position].detach().cpu().numpy()
                canonical, decode_info = decode_successor(successor, problem_size=100)
                if official_orders[position] != decode_info["internal_order"]:
                    raise ValueError("independent successor traversal disagrees with official get_order")
                timed_objective = float(timed_tensor[position].detach().cpu())
                replay_objective = float(replay_tensor[position].detach().cpu())
                recomputed = float(official_recomputed[position])
                if timed_objective != replay_objective or not objective_agrees(
                        timed_objective, recomputed):
                    raise RuntimeError("official timed/replay/recomputed objective mismatch")
                raw_points = np.asarray(task.points)
                validation = validate_tsp(raw_points, canonical)
                independent = validation["independent_objective"]
                if not validation["feasible"] or not objective_agrees(
                        timed_objective, independent):
                    raise RuntimeError(f"independent validation failed at index {dataset_index}")
                reference = float(task.evaluate(task.ref_sol))
                reference_validation = validate_tsp(raw_points, task.ref_sol)
                independent_reference = reference_validation["independent_objective"]
                if (not reference_validation["feasible"] or
                        not objective_agrees(reference, independent_reference)):
                    raise RuntimeError(
                        f"benchmark reference validation failed at index {dataset_index}")
                kit_validation = validate_task(
                    task, canonical, independent_objective=independent,
                    reference_objective=reference, kit_module=kit)
                records.append({
                    "dataset_instance_index": dataset_index,
                    "instance_id": _task_name(task, dataset_index),
                    "batch_index": batch_index, "position_in_batch": position,
                    "successor": successor.tolist(), "canonical_solution": canonical,
                    "reported_objective": timed_objective,
                    "timed_official_objective": timed_objective,
                    "replay_official_objective": replay_objective,
                    "official_recomputed_objective": recomputed,
                    "independent_objective": independent,
                    "reference_objective": reference,
                    "independent_reference_objective": independent_reference,
                    "benchmark_reference_agrees": True,
                    "gap_percent": (independent - reference) / reference * 100.0,
                    "independent_feasible": True, "reported_objective_agrees": True,
                    **kit_validation,
                    "constraint_details": validation["constraint_details"],
                    "source_coordinate_dtype": str(raw_points.dtype),
                    "model_input_dtype": current_mapping["model_input_dtype"],
                    "model_input_dtype_cast": current_mapping["dtype_cast"],
                    "evidence_status": "KIT_VALIDATED",
                })
            timing = {
                "batch_index": batch_index, "dataset_indices": current_indices,
                "batch_size": args.batch_size, "runtime_seconds": elapsed,
                "timed_replay": timed_replay,
                "rng_before_sha256": rng_fingerprints["before_sha256"],
                "rng_after_sha256": rng_fingerprints["after_sha256"],
                "native_vectorized_batch": {
                    "one_official_rollout_call": True,
                    "original_instances_in_call": args.batch_size,
                    "bs1_loop_emulation": False,
                },
            }
            append_batch(args.output_dir, records, timing)
            print(json.dumps({
                "method": "NeuOpt", "problem": "TSP", "problem_size": 100,
                "budget": args.budget, "T_max": args.T_max,
                "batch_size": args.batch_size, "batch_index": batch_index,
                "dataset_indices": current_indices, "runtime_seconds": elapsed,
                "validated_instances": len(records),
            }, sort_keys=True), flush=True)
        if git_provenance(args.upstream) != upstream or git_provenance(ROOT) != project:
            raise RuntimeError("project or pinned upstream provenance changed during production")
        finalize_run(args.output_dir)
    except torch.cuda.OutOfMemoryError as exc:
        mark_failed(args.output_dir, failure_type="CUDA_OUT_OF_MEMORY", message=str(exc))
        raise RuntimeError(
            f"native BS{args.batch_size} CUDA OOM; batch splitting is forbidden") from exc
    except Exception as exc:
        mark_failed(args.output_dir, failure_type=type(exc).__name__, message=str(exc))
        raise

    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
