#!/usr/bin/env python3
"""Final NeuOpt-GIRE CVRP production evaluator for BS1 and true BS100."""
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
from common.objective_agreement import objective_agrees
from common.provenance import (environment_provenance, git_provenance,
                               normalize_git_repository_identity,
                               source_provenance)
from methods.neuopt.cvrp.adapter import adapt_batch
from methods.neuopt.cvrp.compat import (configure_production_decoder,
                                       ensure_tensorboard_logger)
from methods.neuopt.cvrp.config import supported_config
from methods.neuopt.cvrp.decode import decode_successor, extract_final_best
from methods.neuopt.cvrp.paper_protocol import (
    FORMAL_D2A, FORMAL_K, FORMAL_STALL_LIMIT, SEED, UPSTREAM_COMMIT,
    UPSTREAM_URL, paper_protocol, protocol_fingerprint,
)
from methods.neuopt.cvrp.production_results import (
    TIMING_SEMANTICS, append_batch, finalize_chunk, initialize_or_resume,
)
from problems.cvrp.validate import validate


def cuda_device(value, torch):
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal NeuOpt paper evaluation requires CUDA")
    gpu = torch.cuda.get_device_name(device)
    if "RTX 4090" not in gpu:
        raise RuntimeError(f"formal NeuOpt timing requires RTX 4090; observed {gpu!r}")
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


def official_option_args(*, problem_size, config, checkpoint, device, batch_size):
    size = supported_config(problem_size)
    args = [
        "--problem", "cvrp", "--graph_size", str(problem_size),
        "--dummy_rate", str(size["dummy_rate"]), "--eval_only",
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
    if native["coordinates"].shape[0] != batch_size:
        raise ValueError("native input does not contain the declared original batch size")
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
    """Time record-free batch inference, then replay from its exact RNG start."""
    before = capture_rng(torch)
    timed_output, elapsed = solve_batch(
        agent, problem, native, config=config, batch_size=batch_size,
        device=device, torch=torch, timed=True, record=False)
    timed_after = capture_rng(torch)
    restore_rng(before, torch)
    replay_output, _ = solve_batch(
        agent, problem, native, config=config, batch_size=batch_size,
        device=device, torch=torch, timed=False, record=True)
    replay_after = capture_rng(torch)
    if not torch.equal(timed_output[0], replay_output[0]):
        raise RuntimeError(
            "record=True evidence replay objective differs from timed record=False objective")
    if not rng_states_equal(timed_after, replay_after, torch):
        raise RuntimeError(
            "record=True evidence replay consumed different RNG than timed record=False rollout")
    return timed_output[0], replay_output, elapsed, {
        "timed_rollout_record": False, "evidence_replay_record": True,
        "rng_state_restored": True,
        "timed_replay_official_objective_exact": True,
        "timed_replay_rng_after_exact": True,
        "cuda_synchronized_before_timing": True,
        "cuda_synchronized_after_timing": True,
        "evidence_replay_excluded_from_runtime": True,
    }


def _load_prepared(args, expected):
    metadata_path = args.input_metadata or args.input.with_suffix(args.input.suffix + ".json")
    prepared = json.loads(metadata_path.read_text())
    input_hash = sha256_file(args.input)
    if prepared.get("format") != "neuopt-cvrp-neutral-input-v1":
        raise ValueError("unsupported prepared NeuOpt input metadata")
    if prepared.get("problem_size") != args.problem_size:
        raise ValueError("prepared NeuOpt problem_size mismatch")
    if prepared.get("dataset_sha256") != expected["dataset_sha256"]:
        raise ValueError("prepared input is not the pinned official dataset")
    if prepared.get("input_npz_sha256") != input_hash:
        raise ValueError("prepared NeuOpt NPZ hash mismatch")
    with np.load(args.input, allow_pickle=False) as data:
        arrays = {
            "depots": data["depots"], "points": data["points"],
            "demands": data["raw_demands"], "capacities": data["raw_capacities"],
            "indices": data["dataset_indices"], "references": data["reference_objectives"],
        }
    all_indices = [int(value) for value in arrays["indices"]]
    if prepared.get("dataset_indices") != all_indices:
        raise ValueError("prepared metadata and NPZ dataset indices differ")
    index_to_position = {index: position for position, index in enumerate(all_indices)}
    if len(index_to_position) != len(all_indices):
        raise ValueError("prepared input contains duplicate dataset indices")
    requested = list(range(args.offset, args.offset + args.count))
    try:
        positions = [index_to_position[index] for index in requested]
    except KeyError as exc:
        raise ValueError(f"prepared input is missing dataset index {exc.args[0]}") from exc
    names = prepared.get("instance_names", [])
    if len(names) != len(all_indices):
        raise ValueError("prepared metadata instance names do not match NPZ")
    if not np.all(arrays["capacities"][positions] == expected["capacity"]):
        raise ValueError("prepared capacities do not match the pinned dataset")
    return prepared, metadata_path, input_hash, arrays, positions, requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-metadata", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, choices=[50, 100], required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--d2a", "--val-m", dest="val_m", type=int, default=FORMAL_D2A)
    parser.add_argument("--T-max", dest="T_max", type=int, choices=[20, 50], required=True)
    parser.add_argument("--batch-size", type=int, choices=[1, 100], required=True)
    parser.add_argument("--stall-limit", type=int, default=FORMAL_STALL_LIMIT)
    parser.add_argument("--k", type=int, default=FORMAL_K)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, required=True,
                        help="number of original benchmark instances")
    parser.add_argument("--warmup-batches", type=int, choices=range(0, 6), default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.offset < 0 or args.count <= 0:
        parser.error("offset must be nonnegative and count positive")
    if args.count % args.batch_size:
        parser.error("count must form complete original-instance batches")

    expected = supported_config(args.problem_size)
    protocol = paper_protocol(
        args.problem_size, T_max=args.T_max, batch_size=args.batch_size,
        d2a=args.val_m, stall_limit=args.stall_limit, k=args.k)
    prepared, metadata_path, input_hash, arrays, positions, indices = _load_prepared(
        args, expected)
    dataset_hash = sha256_file(args.dataset)
    checkpoint_hash = sha256_file(args.checkpoint)
    if dataset_hash != expected["dataset_sha256"]:
        raise ValueError("Kit dataset is not the pinned official dataset")
    if checkpoint_hash != expected["checkpoint_sha256"]:
        raise ValueError("checkpoint is not the pinned official NeuOpt artifact")
    upstream = git_provenance(args.upstream)
    if (upstream["commit"] != UPSTREAM_COMMIT or upstream["dirty"] or
            normalize_git_repository_identity(upstream["url"]) !=
            normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("official NeuOpt checkout identity/cleanliness mismatch")
    project = git_provenance(ROOT)
    if project["dirty"]:
        raise ValueError("formal NeuOpt production requires a clean project checkout")

    import ml4co_kit as kit
    kit_wrapper = kit.CVRPWrapper()
    kit_wrapper.from_pickle(args.dataset)
    if len(kit_wrapper.task_list) != expected["dataset_count"]:
        raise ValueError("Kit dataset count does not match the pinned identity")

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
    from problems.problem_cvrp import CVRP
    from nets.graph_layers import kopt_Decoder
    decoder_compatibility = configure_production_decoder(
        kopt_Decoder, original_batch_size=args.batch_size, val_m=protocol["val_m"])
    from agent.ppo import PPO

    option_args = official_option_args(
        problem_size=args.problem_size, config=protocol, checkpoint=args.checkpoint,
        device=device, batch_size=args.batch_size)
    opts = get_options(option_args)
    opts.device, opts.use_cuda, opts.distributed, opts.world_size = device, True, False, 1
    problem = CVRP(
        p_size=args.problem_size, init_val_met=opts.init_val_met,
        with_assert=opts.use_assert, DUMMY_RATE=opts.dummy_rate, k=opts.k,
        with_bonus=not opts.wo_bonus, with_regular=not opts.wo_regular)
    if problem.dummy_size != expected["dummy_size"] or problem.size != expected["sequence_length"]:
        raise ValueError("official problem construction contradicts pinned size settings")
    agent = PPO(problem, opts)
    if any(hasattr(agent, name) for name in ("critic", "optimizer", "lr_scheduler")):
        raise RuntimeError("eval_only PPO unexpectedly constructed training components")
    agent.load(str(args.checkpoint.resolve()))
    agent.eval()
    problem.eval()

    def native_for(batch_positions):
        return adapt_batch(
            arrays["depots"][batch_positions], arrays["points"][batch_positions],
            arrays["demands"][batch_positions], arrays["capacities"][batch_positions],
            problem_size=args.problem_size, device=device)[0]

    batch_positions = [
        positions[start:start + args.batch_size]
        for start in range(0, len(positions), args.batch_size)
    ]
    batch_indices = [
        indices[start:start + args.batch_size]
        for start in range(0, len(indices), args.batch_size)
    ]
    sources = source_provenance([
        Path(__file__), Path(__file__).with_name("paper_protocol.py"),
        Path(__file__).with_name("production_results.py"),
        Path(__file__).with_name("adapter.py"), Path(__file__).with_name("decode.py"),
        Path(__file__).with_name("config.py"), Path(__file__).with_name("compat.py"),
        ROOT / "problems/cvrp/validate.py", ROOT / "problems/cvrp/objective.py",
        ROOT / "common/objective_agreement.py", ROOT / "common/provenance.py",
    ], root=ROOT)
    identity = {
        "method": "NeuOpt", "variant": "NeuOpt-GIRE", "problem": "CVRP",
        "problem_size": args.problem_size, "paper_protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "project": project, "upstream": upstream,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": checkpoint_hash},
        "dataset": {"path": str(args.dataset.resolve()), "sha256": dataset_hash,
                    "count": expected["dataset_count"]},
        "prepared_input": {"path": str(args.input.resolve()), "sha256": input_hash,
                           "metadata_path": str(metadata_path.resolve())},
        "chunk": {"offset": args.offset, "count": args.count,
                  "expected_indices": indices},
        "warmup": {"batches": args.warmup_batches, "rng_state_restored": True},
        "rng_policy": (
            "seed once after model setup per chunk; warm-up state restored; resumed completed "
            "batch prefix replayed untimed exactly once to restore the sequential stream"
        ),
        "environment": environment,
        "tensorboard_compatibility": tensorboard_compatibility,
        "decoder_compatibility": decoder_compatibility,
        "source_provenance": sources, "timing_semantics": TIMING_SEMANTICS,
    }
    metadata, prior_records, prior_timings = initialize_or_resume(args.output_dir, identity)
    if metadata["state"] == "KIT_VALIDATED":
        print(args.output_dir / "summary.json")
        return

    state_before_warmup = capture_rng(torch)
    for current_positions in batch_positions[:args.warmup_batches]:
        solve_batch(
            agent, problem, native_for(current_positions), config=protocol,
            batch_size=args.batch_size, device=device, torch=torch,
            timed=False, record=False)
    restore_rng(state_before_warmup, torch)

    completed_batches = len(prior_timings)
    for batch_index in range(completed_batches):
        output, _ = solve_batch(
            agent, problem, native_for(batch_positions[batch_index]), config=protocol,
            batch_size=args.batch_size, device=device, torch=torch,
            timed=False, record=False)
        saved = torch.tensor([
            record["timed_official_objective"]
            for record in prior_records[batch_index * args.batch_size:
                                        (batch_index + 1) * args.batch_size]
        ], dtype=output[0].dtype, device=output[0].device)
        if not torch.equal(output[0], saved):
            raise RuntimeError("resume RNG prefix replay does not reproduce saved objectives")

    for batch_index in range(completed_batches, len(batch_positions)):
        current_positions = batch_positions[batch_index]
        current_indices = batch_indices[batch_index]
        native = native_for(current_positions)
        timed_tensor, replay_output, elapsed, timed_replay = (
            timed_rollout_with_evidence_replay(
                agent, problem, native, config=protocol, batch_size=args.batch_size,
                device=device, torch=torch))
        replay_tensor, successor_tensor = extract_final_best(
            replay_output, batch_size=args.batch_size, val_m=1)
        if not torch.equal(timed_tensor, replay_tensor):
            raise RuntimeError("selected replay successors differ from timed official objectives")
        official_orders = problem.get_order(
            successor_tensor, return_solution=True).detach().cpu().tolist()
        official_recomputed = problem.get_costs(
            native, successor_tensor, get_context=False,
            check_full_feasibility=True).detach().cpu()
        records = []
        for position_in_batch, (position, dataset_index) in enumerate(
                zip(current_positions, current_indices)):
            successor = successor_tensor[position_in_batch].detach().cpu().numpy()
            canonical, decode_info = decode_successor(
                successor, problem_size=args.problem_size)
            if official_orders[position_in_batch] != decode_info["internal_order"]:
                raise ValueError("independent successor traversal disagrees with official get_order")
            timed_objective = float(timed_tensor[position_in_batch].detach().cpu())
            replay_objective = float(replay_tensor[position_in_batch].detach().cpu())
            recomputed = float(official_recomputed[position_in_batch])
            if timed_objective != replay_objective or not objective_agrees(
                    timed_objective, recomputed):
                raise RuntimeError("official objective/replay correspondence failed")
            validation = validate(
                arrays["depots"][position], arrays["points"][position],
                arrays["demands"][position], arrays["capacities"][position], canonical)
            independent = validation["independent_objective"]
            if not validation["feasible"] or not objective_agrees(timed_objective, independent):
                raise RuntimeError(f"independent validation failed at dataset index {dataset_index}")
            task = kit_wrapper.task_list[dataset_index]
            if type(task) is not kit.CVRPTask or task.name != prepared["instance_names"][position]:
                raise ValueError("Kit task identity does not match prepared input")
            solution_array = np.asarray(canonical, dtype=np.int64)
            kit_feasible = bool(task.check_constraints(solution_array))
            kit_objective = float(task.evaluate(solution_array))
            if not kit_feasible or not objective_agrees(kit_objective, independent):
                raise RuntimeError(f"Kit validation failed at dataset index {dataset_index}")
            reference = float(arrays["references"][position])
            kit_reference = float(task.evaluate(task.ref_sol))
            if not objective_agrees(reference, kit_reference):
                raise ValueError("prepared reference objective does not match exact Kit task")
            records.append({
                "dataset_instance_index": dataset_index,
                "instance_id": prepared["instance_names"][position],
                "batch_index": batch_index, "position_in_batch": position_in_batch,
                "canonical_solution": canonical, "successor": successor.tolist(),
                "reported_objective": timed_objective,
                "timed_official_objective": timed_objective,
                "replay_official_objective": replay_objective,
                "official_recomputed_objective": recomputed,
                "independent_objective": independent, "reference_objective": reference,
                "kit_reference_objective": kit_reference,
                "gap_percent": (independent - reference) / reference * 100.0,
                "independent_feasible": True, "reported_objective_agrees": True,
                "kit_feasible": True, "kit_objective": kit_objective,
                "kit_objective_agrees": True,
                "constraint_details": validation["constraint_details"],
                "evidence_status": "KIT_VALIDATED",
            })
        timing = {
            "batch_index": batch_index, "dataset_indices": current_indices,
            "batch_size": args.batch_size, "runtime_seconds": elapsed,
            "timed_replay": timed_replay,
        }
        append_batch(args.output_dir, records, timing)
        print(json.dumps({
            "method": "NeuOpt", "problem_size": args.problem_size,
            "D2A": 1, "T_max": args.T_max, "batch_size": args.batch_size,
            "batch_index": batch_index, "dataset_indices": current_indices,
            "runtime_seconds": elapsed, "validated_instances": len(records),
        }, sort_keys=True), flush=True)

    finalize_chunk(args.output_dir)
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
