#!/usr/bin/env python3
"""Run one NeuOpt TSP100 BS1 timing-calibration candidate on RTX 4090."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from common.objective_agreement import OBJECTIVE_ATOL, OBJECTIVE_RTOL, objective_agrees
from common.provenance import (environment_provenance, git_provenance,
                               normalize_git_repository_identity, source_provenance)
from methods.neuopt.cvrp.compat import (configure_production_decoder,
                                       ensure_tensorboard_logger)
from methods.neuopt.tsp.adapter import adapt_batch
from methods.neuopt.tsp.config import supported_config
from methods.neuopt.tsp.compat import replay_record_compatibility
from methods.neuopt.tsp.decode import decode_successor, extract_final_best
from methods.neuopt.tsp.paper_protocol import (
    CALIBRATION_INDICES, D2A, GRAPH_SIZE, K, ORIGINAL_BATCH_SIZE, SEED,
    STALL_LIMIT, UPSTREAM_COMMIT, UPSTREAM_URL, calibration_protocol,
    protocol_fingerprint,
)
from methods.neuopt.tsp.runtime_calibration import (
    RNG_POLICY, TIMING_SEMANTICS, WARMUP_POLICY, write_candidate,
)
from methods.neuopt.tsp.validate_with_kit import validate_task
from problems.tsp.validate import validate as validate_tsp


def positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def cuda_device(value, torch):
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("NeuOpt TSP timing calibration requires CUDA")
    gpu = torch.cuda.get_device_name(device)
    if "RTX 4090" not in gpu:
        raise RuntimeError(f"NeuOpt TSP timing calibration requires RTX 4090; observed {gpu!r}")
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


def official_option_args(*, config, checkpoint, device):
    args = [
        "--problem", "tsp", "--graph_size", str(GRAPH_SIZE), "--eval_only",
        "--init_val_met", config["init_val_met"], "--seed", str(config["seed"]),
        "--val_m", str(config["val_m"]), "--stall_limit", str(config["stall_limit"]),
        "--k", str(config["k"]), "--T_max", str(config["T_max"]),
        "--val_size", "1", "--val_batch_size", "1",
        "--load_path", str(Path(checkpoint).resolve()), "--no_tb", "--no_saving",
        "--no_DDP", "--no_progress_bar",
    ]
    if device.type == "cpu":
        args.append("--no_cuda")
    return args


def solve_one(agent, problem, native, *, config, device, torch, timed, record):
    if tuple(native["coordinates"].shape) != (1, GRAPH_SIZE, 2):
        raise ValueError("NeuOpt TSP calibration requires one native TSP100 instance")
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
                                       device, torch):
    before = capture_rng(torch)
    timed_output, elapsed = solve_one(
        agent, problem, native, config=config, device=device, torch=torch,
        timed=True, record=False)
    timed_after = capture_rng(torch)
    restore_rng(before, torch)
    with replay_record_compatibility(problem) as record_compatibility:
        replay_output, _ = solve_one(
            agent, problem, native, config=config, device=device, torch=torch,
            timed=False, record=True)
    replay_after = capture_rng(torch)
    if not torch.equal(timed_output[0], replay_output[0]):
        raise RuntimeError(
            "record=True evidence replay objective differs from timed record=False objective")
    if not rng_states_equal(timed_after, replay_after, torch):
        raise RuntimeError(
            "record=True evidence replay consumed different RNG than timed record=False rollout")
    return timed_output[0], replay_output, elapsed, {
        "timed_rollout_record": False,
        "evidence_replay_record": True,
        "rng_state_restored": True,
        "timed_replay_official_objective_exact": True,
        "timed_replay_rng_after_exact": True,
        "cuda_synchronized_before_timing": True,
        "cuda_synchronized_after_timing": True,
        "evidence_replay_excluded_from_runtime": True,
    }, record_compatibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, choices=[100], default=100)
    parser.add_argument("--T-max", dest="T_max", type=positive_integer, required=True)
    parser.add_argument("--batch-size", type=int, choices=[1], default=1)
    parser.add_argument("--d2a", "--val-m", dest="val_m", type=int, choices=[1], default=D2A)
    parser.add_argument("--stall-limit", type=int, choices=[STALL_LIMIT], default=STALL_LIMIT)
    parser.add_argument("--k", type=int, choices=[K], default=K)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise ValueError("calibration output directory already exists; refusing overwrite")
    config = supported_config(args.problem_size)
    protocol = calibration_protocol(
        args.problem_size, T_max=args.T_max, batch_size=args.batch_size,
        d2a=args.val_m, stall_limit=args.stall_limit, k=args.k)
    if args.dataset.name != config["dataset_filename"]:
        raise ValueError(f"dataset filename must be exactly {config['dataset_filename']}")
    dataset_hash = sha256_file(args.dataset)
    checkpoint_hash = sha256_file(args.checkpoint)
    if dataset_hash != config["dataset_sha256"]:
        raise ValueError("dataset is not the pinned TSP100 paper benchmark")
    if (args.checkpoint.name != "tsp100.pt" or
            checkpoint_hash != config["checkpoint_sha256"]):
        raise ValueError("checkpoint is not the pinned official NeuOpt TSP100 artifact")

    upstream = git_provenance(args.upstream)
    if (upstream["commit"] != UPSTREAM_COMMIT or upstream["dirty"] or
            normalize_git_repository_identity(upstream["url"]) !=
            normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("official NeuOpt checkout identity/cleanliness mismatch")
    project = git_provenance(ROOT)
    if project["dirty"]:
        raise ValueError("NeuOpt TSP calibration requires a clean project checkout")

    import ml4co_kit as kit
    wrapper = kit.TSPWrapper()
    wrapper.from_pickle(args.dataset)
    if len(wrapper.task_list) != config["dataset_count"]:
        raise ValueError("TSP100 dataset count does not match pinned identity")
    tasks = [wrapper.task_list[index] for index in CALIBRATION_INDICES]
    for task in tasks:
        if type(task) is not kit.TSPTask or np.asarray(task.points).shape != (100, 2):
            raise ValueError("calibration subset contains a non-exact ML4CO TSP100 task")
        if not np.isfinite(np.asarray(task.points)).all():
            raise ValueError("calibration subset contains non-finite coordinates")

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
    decoder_compatibility = configure_production_decoder(
        kopt_Decoder, original_batch_size=ORIGINAL_BATCH_SIZE, val_m=D2A)
    from agent.ppo import PPO

    option_args = official_option_args(
        config=protocol, checkpoint=args.checkpoint, device=device)
    opts = get_options(option_args)
    opts.device, opts.use_cuda, opts.distributed, opts.world_size = device, True, False, 1
    problem = TSP(
        p_size=GRAPH_SIZE, init_val_met=opts.init_val_met,
        with_assert=opts.use_assert, DUMMY_RATE=opts.dummy_rate, k=opts.k,
        with_bonus=not opts.wo_bonus, with_regular=not opts.wo_regular)
    if problem.size != GRAPH_SIZE:
        raise ValueError("official TSP problem construction is not TSP100")
    agent = PPO(problem, opts)
    if any(hasattr(agent, name) for name in ("critic", "optimizer", "lr_scheduler")):
        raise RuntimeError("eval_only PPO unexpectedly constructed training components")
    agent.load(str(args.checkpoint.resolve()))
    agent.eval()
    problem.eval()

    def native_for(task):
        return adapt_batch(
            np.asarray(task.points)[None, :, :], problem_size=GRAPH_SIZE,
            device=device)

    warmup_state = capture_rng(torch)
    warmup_native, adapter_mapping = native_for(tasks[0])
    solve_one(
        agent, problem, warmup_native, config=protocol, device=device,
        torch=torch, timed=False, record=False)
    restore_rng(warmup_state, torch)

    records = []
    for index, task in zip(CALIBRATION_INDICES, tasks):
        native, current_mapping = native_for(task)
        if current_mapping != adapter_mapping:
            raise ValueError("calibration tasks do not share one adapter identity")
        timed_tensor, replay_output, elapsed, timed_replay, record_compatibility = (
            timed_rollout_with_evidence_replay(
                agent, problem, native, config=protocol, device=device, torch=torch))
        replay_tensor, successor_tensor = extract_final_best(
            replay_output, batch_size=1, val_m=1)
        if not torch.equal(timed_tensor, replay_tensor):
            raise RuntimeError("selected replay successor differs from timed objective")
        successor = successor_tensor[0].detach().cpu().numpy()
        canonical, decode_info = decode_successor(successor, problem_size=GRAPH_SIZE)
        official_order = problem.get_order(
            successor_tensor, return_solution=True)[0].detach().cpu().tolist()
        if official_order != decode_info["internal_order"]:
            raise ValueError("independent successor traversal disagrees with official get_order")
        official = float(timed_tensor[0].detach().cpu())
        replay = float(replay_tensor[0].detach().cpu())
        official_recomputed = float(problem.get_costs(
            native, successor_tensor, get_context=False,
            check_full_feasibility=True)[0].detach().cpu())
        validation = validate_tsp(np.asarray(task.points), canonical)
        independent = validation["independent_objective"]
        if (official != replay or not validation["feasible"] or
                not objective_agrees(official, independent) or
                not objective_agrees(official_recomputed, independent)):
            raise RuntimeError(f"independent objective validation failed at index {index}")
        reference = float(task.evaluate(task.ref_sol))
        kit_validation = validate_task(
            task, canonical, independent_objective=independent,
            reference_objective=reference, kit_module=kit)
        records.append({
            "index": index, "instance_id": task.name,
            "runtime_seconds": elapsed,
            "successor": successor.tolist(), "canonical_solution": canonical,
            "official_objective": official,
            "replay_official_objective": replay,
            "official_recomputed_objective": official_recomputed,
            "independent_objective": independent,
            "reference_objective": reference,
            "gap_percent": (independent - reference) / reference * 100.0,
            "feasible": True,
            **kit_validation,
            "objective_agreement": {
                "timed_replay_exact": True,
                "official_independent": True,
                "official_recomputed_independent": True,
                "kit_independent": True,
                "rtol": OBJECTIVE_RTOL, "atol": OBJECTIVE_ATOL,
            },
            "constraint_details": validation["constraint_details"],
            "timed_replay": timed_replay,
            "evidence_status": "KIT_VALIDATED",
        })
        if index == CALIBRATION_INDICES[0]:
            shared_record_compatibility = record_compatibility
        elif record_compatibility != shared_record_compatibility:
            raise RuntimeError("record=True replay compatibility provenance changed across instances")
        print(json.dumps({
            "problem": "TSP", "problem_size": 100, "T_max": args.T_max,
            "index": index, "runtime_seconds": elapsed,
            "official_objective": official,
            "independent_objective": independent,
            "gap_percent": records[-1]["gap_percent"],
            "status": "KIT_VALIDATED",
        }, sort_keys=True), flush=True)

    sources = source_provenance([
        Path(__file__), Path(__file__).with_name("config.py"),
        Path(__file__).with_name("compat.py"),
        Path(__file__).with_name("adapter.py"), Path(__file__).with_name("decode.py"),
        Path(__file__).with_name("paper_protocol.py"),
        Path(__file__).with_name("runtime_calibration.py"),
        Path(__file__).with_name("validate_with_kit.py"),
        ROOT / "methods/neuopt/cvrp/compat.py",
        ROOT / "problems/tsp/validate.py", ROOT / "problems/tsp/objective.py",
        ROOT / "common/objective_agreement.py", ROOT / "common/provenance.py",
    ], root=ROOT)
    identity = {
        "method": "NeuOpt", "variant": "NeuOpt-GIRE", "problem": "TSP",
        "problem_size": 100, "paper_protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "project": project, "upstream": upstream,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": checkpoint_hash},
        "dataset": {"path": str(args.dataset.resolve()),
                    "filename": args.dataset.name, "sha256": dataset_hash,
                    "count": len(wrapper.task_list)},
        "sample_indices": list(CALIBRATION_INDICES),
        "adapter_mapping": adapter_mapping,
        "environment": environment,
        "tensorboard_compatibility": tensorboard_compatibility,
        "decoder_compatibility": decoder_compatibility,
        "record_compatibility": shared_record_compatibility,
        "timing_semantics": TIMING_SEMANTICS,
        "warmup_policy": dict(WARMUP_POLICY), "rng_policy": RNG_POLICY,
        "source_provenance": sources,
    }
    _, summary = write_candidate(args.output_dir, identity, records)
    print(json.dumps(summary, sort_keys=True))
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
