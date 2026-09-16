#!/usr/bin/env python3
"""BS1 NeuOpt-GIRE CVRP paper correctness and runtime calibration runner."""
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
from methods.neuopt.cvrp.compat import (ensure_bs1_decoder_compatibility,
                                       ensure_tensorboard_logger)
from methods.neuopt.cvrp.config import supported_config
from methods.neuopt.cvrp.decode import (decode_successor,
                                        extract_final_best_d2a)
from methods.neuopt.cvrp.paper_protocol import (
    FORMAL_D2A, FORMAL_K, FORMAL_STALL_LIMIT, SEED, UPSTREAM_COMMIT,
    UPSTREAM_URL, paper_protocol, protocol_fingerprint,
)
from methods.neuopt.cvrp.paper_results import TIMING_SEMANTICS, write_artifact
from problems.cvrp.validate import validate


def cuda_device(value, torch):
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal NeuOpt paper calibration requires CUDA")
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
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state, torch):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def official_option_args(*, problem_size, config, checkpoint, device):
    size = supported_config(problem_size)
    args = [
        "--problem", "cvrp", "--graph_size", str(problem_size),
        "--dummy_rate", str(size["dummy_rate"]), "--eval_only",
        "--init_val_met", config["init_val_met"], "--seed", str(config["seed"]),
        "--val_m", str(config["val_m"]),
        "--stall_limit", str(config["stall_limit"]),
        "--k", str(config["k"]), "--T_max", str(config["T_max"]),
        "--val_size", "1", "--val_batch_size", "1",
        "--load_path", str(Path(checkpoint).resolve()), "--no_tb", "--no_saving",
        "--no_DDP", "--no_progress_bar", "--record",
    ]
    if device.type == "cpu":
        args.append("--no_cuda")
    return args


def solve_one(agent, problem, native, *, config, device, torch, timed):
    if native["coordinates"].shape[0] != 1:
        raise ValueError("formal NeuOpt timing requires original batch size exactly one")
    if timed:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
    with torch.no_grad():
        output = agent.rollout(
            problem, T=config["T_max"], val_m=config["val_m"],
            stall_limit=config["stall_limit"], batch=native,
            record=True, show_bar=False)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started if timed else None
    return output, elapsed


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
            "indices": data["dataset_indices"],
            "references": data["reference_objectives"],
        }
    indices = [int(value) for value in arrays["indices"]]
    requested = list(range(args.offset, args.offset + args.count))
    positions = []
    for index in requested:
        if index not in indices:
            raise ValueError(f"prepared input does not contain requested dataset index {index}")
        positions.append(indices.index(index))
    if prepared.get("dataset_indices") != indices:
        raise ValueError("prepared metadata and NPZ dataset indices differ")
    names = prepared.get("instance_names", [])
    if len(names) != len(indices):
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
    parser.add_argument("--d2a", "--val-m", dest="val_m", type=int,
                        default=FORMAL_D2A)
    parser.add_argument("--T-max", dest="T_max", type=int, required=True)
    parser.add_argument("--stall-limit", type=int, default=FORMAL_STALL_LIMIT)
    parser.add_argument("--k", type=int, default=FORMAL_K)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--warmup-instances", type=int, choices=range(0, 6), default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.offset < 0 or args.count <= 0:
        parser.error("offset must be nonnegative and count positive")

    expected = supported_config(args.problem_size)
    protocol = paper_protocol(
        args.problem_size, T_max=args.T_max, d2a=args.val_m,
        stall_limit=args.stall_limit, k=args.k)
    prepared, metadata_path, input_hash, arrays, positions, indices = _load_prepared(
        args, expected)
    dataset_hash = sha256_file(args.dataset)
    if dataset_hash != expected["dataset_sha256"]:
        raise ValueError("Kit dataset is not the pinned official dataset")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != expected["checkpoint_sha256"]:
        raise ValueError("checkpoint is not the pinned official NeuOpt artifact")
    upstream = git_provenance(args.upstream)
    if (upstream["commit"] != UPSTREAM_COMMIT or upstream["dirty"] or
            normalize_git_repository_identity(upstream["url"]) !=
            normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("official NeuOpt checkout identity/cleanliness mismatch")
    project = git_provenance(ROOT)
    if project["dirty"]:
        raise ValueError("formal NeuOpt calibration requires a clean project checkout")

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
    bs1_compatibility = ensure_bs1_decoder_compatibility(kopt_Decoder)
    from agent.ppo import PPO

    option_args = official_option_args(
        problem_size=args.problem_size, config=protocol,
        checkpoint=args.checkpoint, device=device)
    opts = get_options(option_args)
    opts.device = device
    opts.use_cuda = True
    opts.distributed = False
    opts.world_size = 1
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

    def native_at(position):
        return adapt_batch(
            arrays["depots"][position:position + 1],
            arrays["points"][position:position + 1],
            arrays["demands"][position:position + 1],
            arrays["capacities"][position:position + 1],
            problem_size=args.problem_size, device=device)[0]

    state_before_warmup = capture_rng(torch)
    for position in positions[:min(args.warmup_instances, len(positions))]:
        solve_one(agent, problem, native_at(position), config=protocol,
                  device=device, torch=torch, timed=False)
    restore_rng(state_before_warmup, torch)

    records = []
    for position, dataset_index in zip(positions, indices):
        native = native_at(position)
        output, elapsed = solve_one(
            agent, problem, native, config=protocol,
            device=device, torch=torch, timed=True)
        reported_tensor, successor_tensor, selected_candidate = extract_final_best_d2a(
            output, problem=problem, native_batch=native,
            batch_size=1, val_m=protocol["val_m"])
        successor = successor_tensor[0].detach().cpu().numpy()
        canonical, decode_info = decode_successor(successor, problem_size=args.problem_size)
        official_order = problem.get_order(successor_tensor, return_solution=True)[0].detach().cpu().tolist()
        if official_order != decode_info["internal_order"]:
            raise ValueError("independent successor traversal disagrees with official get_order")
        official_recomputed = float(problem.get_costs(
            native, successor_tensor, get_context=False,
            check_full_feasibility=True)[0].detach().cpu())
        reported = float(reported_tensor[0].detach().cpu())
        if not objective_agrees(reported, official_recomputed):
            raise ValueError("selected D2A successor does not match the official objective")
        validation = validate(
            arrays["depots"][position], arrays["points"][position],
            arrays["demands"][position], arrays["capacities"][position], canonical)
        independent = validation["independent_objective"]
        if not validation["feasible"] or not objective_agrees(reported, independent):
            raise RuntimeError(f"independent validation failed at dataset index {dataset_index}")
        task = kit_wrapper.task_list[dataset_index]
        if type(task) is not kit.CVRPTask:
            raise ValueError("Kit returned a non-exact CVRPTask")
        if task.name != prepared["instance_names"][position]:
            raise ValueError("Kit task name does not match the prepared input identity")
        solution_array = np.asarray(canonical, dtype=np.int64)
        kit_feasible = bool(task.check_constraints(solution_array))
        kit_objective = float(task.evaluate(solution_array))
        kit_agrees = objective_agrees(kit_objective, independent)
        if not kit_feasible or not kit_agrees:
            raise RuntimeError(f"Kit validation failed at dataset index {dataset_index}")
        reference = float(arrays["references"][position])
        kit_reference = float(task.evaluate(task.ref_sol))
        if not objective_agrees(reference, kit_reference):
            raise ValueError("prepared reference objective does not match the exact Kit task")
        record = {
            "dataset_instance_index": dataset_index,
            "instance_id": prepared["instance_names"][position],
            "canonical_solution": canonical,
            "successor": successor.tolist(),
            "selected_d2a_candidate": int(selected_candidate[0].detach().cpu()),
            "reported_objective": reported,
            "official_recomputed_objective": official_recomputed,
            "independent_objective": independent,
            "reference_objective": reference,
            "kit_reference_objective": kit_reference,
            "gap_percent": (independent - reference) / reference * 100.0,
            "runtime_seconds": elapsed,
            "runtime_semantics": TIMING_SEMANTICS,
            "independent_feasible": True,
            "reported_objective_agrees": True,
            "kit_feasible": True,
            "kit_objective": kit_objective,
            "kit_objective_agrees": True,
            "constraint_details": validation["constraint_details"],
            "evidence_status": "KIT_VALIDATED",
        }
        records.append(record)
        print(json.dumps({
            "method": "NeuOpt", "problem": "CVRP", "size": args.problem_size,
            "D2A": protocol["D2A"], "T_max": protocol["T_max"],
            "original_batch_size": 1, "dataset_instance_index": dataset_index,
            "runtime_seconds": elapsed, "objective": independent,
            "feasible": True, "kit_feasible": True,
        }, sort_keys=True, allow_nan=False), flush=True)

    sources = source_provenance([
        Path(__file__), Path(__file__).with_name("paper_protocol.py"),
        Path(__file__).with_name("paper_results.py"),
        Path(__file__).with_name("adapter.py"),
        Path(__file__).with_name("decode.py"), Path(__file__).with_name("config.py"),
        Path(__file__).with_name("compat.py"), ROOT / "problems/cvrp/validate.py",
        ROOT / "problems/cvrp/objective.py", ROOT / "common/objective_agreement.py",
        ROOT / "common/provenance.py",
    ], root=ROOT)
    resume_identity = {
        "method": "NeuOpt", "variant": "NeuOpt-GIRE", "problem": "CVRP",
        "problem_size": args.problem_size, "paper_protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "original_batch_size": 1,
        "project": project, "upstream": upstream,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": checkpoint_hash},
        "dataset": {"path": str(args.dataset.resolve()), "sha256": dataset_hash,
                    "count": expected["dataset_count"]},
        "prepared_input": {"path": str(args.input.resolve()), "sha256": input_hash,
                           "metadata_path": str(metadata_path.resolve())},
        "subset": {"offset": args.offset, "count": args.count,
                   "dataset_indices": indices},
        "warmup": {"instances": args.warmup_instances,
                   "rng_state_restored": True,
                   "policy": "prefix instances; untimed; RNG state restored before evidence"},
        "environment": environment,
        "tensorboard_compatibility": tensorboard_compatibility,
        "bs1_compatibility": bs1_compatibility,
        "source_provenance": sources,
        "timing_semantics": TIMING_SEMANTICS,
    }
    write_artifact(args.output_dir, resume_identity, records)
    print(args.output_dir / "metadata.json")


if __name__ == "__main__":
    main()
