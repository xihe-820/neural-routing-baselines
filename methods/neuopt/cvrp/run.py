#!/usr/bin/env python3
"""Run the pinned official NeuOpt CVRP50/100 rollout and capture actual solutions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
                               normalize_git_repository_identity, source_provenance)
from common.result_schema import make_run_metadata, new_result, write_result_bundle
from methods.neuopt.cvrp.adapter import adapt_batch
from methods.neuopt.cvrp.compat import ensure_tensorboard_logger
from methods.neuopt.cvrp.config import supported_config
from methods.neuopt.cvrp.decode import decode_successor, extract_final_best
from problems.cvrp.validate import validate

UPSTREAM_URL = "https://github.com/yining043/NeuOpt"
UPSTREAM_COMMIT = "ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b"
SEED = 6666
T_MAX = 1000
VAL_M = 1
STALL_LIMIT = 10
K = 4


def _device(value, torch):
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, choices=[50, 100], required=True)
    parser.add_argument("--input-metadata", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = supported_config(args.problem_size)

    metadata_path = args.input_metadata or args.input.with_suffix(args.input.suffix + ".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format") != "neuopt-cvrp-neutral-input-v1":
        raise ValueError("unsupported prepared input metadata")
    if metadata.get("problem_size") != args.problem_size:
        raise ValueError("prepared input problem_size does not match requested problem_size")
    if metadata.get("input_npz_sha256") != sha256_file(args.input):
        raise ValueError("prepared NPZ hash does not match metadata")
    if metadata.get("dataset_sha256") != config["dataset_sha256"]:
        raise ValueError("prepared input is not from the pinned official dataset")
    upstream = git_provenance(args.upstream)
    if (upstream["commit"] != UPSTREAM_COMMIT or upstream["dirty"] or
            normalize_git_repository_identity(upstream["url"]) !=
            normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("official NeuOpt checkout identity/cleanliness does not match the pin")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != config["checkpoint_sha256"]:
        raise ValueError("NeuOpt checkpoint SHA256 does not match the pinned official artifact")

    import torch
    device = _device(args.device, torch)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with np.load(args.input, allow_pickle=False) as data:
        depots, points = data["depots"], data["points"]
        demands, capacities = data["raw_demands"], data["raw_capacities"]
        indices, references = data["dataset_indices"], data["reference_objectives"]
    if not np.all(capacities == config["capacity"]):
        raise ValueError("prepared capacities do not match the pinned size configuration")
    native, mapping = adapt_batch(depots, points, demands, capacities,
                                  problem_size=args.problem_size, device="cpu")
    batch_size = len(points)

    # This repository and NeuOpt both use a top-level ``problems`` package.
    # The independent validator function is already bound above; clear only the
    # module cache before importing the read-only upstream package by its native name.
    for module_name in list(sys.modules):
        if module_name == "problems" or module_name.startswith("problems."):
            del sys.modules[module_name]
    sys.path.insert(0, str(args.upstream.resolve()))
    tensorboard_compatibility = ensure_tensorboard_logger()
    from options import get_options
    from problems.problem_cvrp import CVRP
    from agent.ppo import PPO
    option_args = [
        "--problem", "cvrp", "--graph_size", str(args.problem_size),
        "--dummy_rate", str(config["dummy_rate"]), "--eval_only",
        "--init_val_met", "random", "--seed", str(SEED),
        "--val_m", str(VAL_M), "--stall_limit", str(STALL_LIMIT),
        "--k", str(K), "--T_max", str(T_MAX),
        "--val_size", str(batch_size), "--val_batch_size", str(batch_size),
        "--load_path", str(args.checkpoint.resolve()), "--no_tb", "--no_saving",
        "--no_DDP", "--no_progress_bar", "--record",
    ]
    if device.type == "cpu":
        option_args.append("--no_cuda")
    opts = get_options(option_args)
    opts.device = device
    opts.use_cuda = device.type == "cuda"
    opts.distributed = False
    opts.world_size = 1
    problem = CVRP(
        p_size=args.problem_size, init_val_met=opts.init_val_met,
        with_assert=opts.use_assert, DUMMY_RATE=opts.dummy_rate, k=opts.k,
        with_bonus=not opts.wo_bonus, with_regular=not opts.wo_regular,
    )
    if (problem.dummy_size != config["dummy_size"] or
            problem.size != config["sequence_length"]):
        raise ValueError("official Problem construction contradicts the pinned size table")
    agent = PPO(problem, opts)
    if any(hasattr(agent, name) for name in ("critic", "optimizer", "lr_scheduler")):
        raise RuntimeError("eval_only PPO unexpectedly constructed training components")
    agent.load(str(args.checkpoint.resolve()))
    agent.eval()
    problem.eval()

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    with torch.no_grad():
        rollout_output = agent.rollout(problem, T=T_MAX, val_m=VAL_M,
                                       stall_limit=STALL_LIMIT, batch=native,
                                       record=True, show_bar=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    finished_at = datetime.now(timezone.utc).isoformat()
    reported_tensor, successor_tensor = extract_final_best(
        rollout_output, batch_size=batch_size, val_m=VAL_M
    )
    native_device = {key: value.to(device) for key, value in native.items()}
    recomputed = problem.get_costs(native_device, successor_tensor,
                                   get_context=False, check_full_feasibility=True)
    official_orders = problem.get_order(successor_tensor, return_solution=True)

    project = git_provenance(ROOT)
    adapter_sources = source_provenance([
        Path(__file__), Path(__file__).with_name("adapter.py"),
        Path(__file__).with_name("decode.py"), Path(__file__).with_name("config.py"),
        ROOT / "problems/cvrp/validate.py", ROOT / "common/objective_agreement.py",
        ROOT / "common/result_schema.py", ROOT / "common/provenance.py",
    ], root=ROOT)
    environment = environment_provenance(device)
    rows = []
    for i in range(batch_size):
        successor = successor_tensor[i].detach().cpu().numpy()
        canonical, decode_info = decode_successor(successor, problem_size=args.problem_size)
        official_order = official_orders[i].detach().cpu().tolist()
        if decode_info["internal_order"] != official_order:
            raise ValueError("independent successor traversal disagrees with official get_order")
        reported = float(reported_tensor[i].detach().cpu())
        official_recomputed = float(recomputed[i].detach().cpu())
        if not objective_agrees(reported, official_recomputed):
            raise ValueError("recorded final solution_best does not match official best objective")
        validation = validate(depots[i], points[i], demands[i], capacities[i], canonical)
        independent = validation["independent_objective"]
        abs_error = abs(reported - independent) if independent is not None else None
        rel_error = abs_error / abs(independent) if independent not in (None, 0) else None
        agrees = objective_agrees(reported, independent) if independent is not None else False
        reference = float(references[i])
        passed = validation["feasible"] and agrees
        rows.append(new_result(
            method="NeuOpt", variant="GIRE", problem="CVRP", problem_size=args.problem_size,
            instance_id=metadata["instance_names"][i],
            project_repo_commit=project["commit"], project_repo_dirty=project["dirty"],
            upstream_url=upstream["url"], upstream_commit=upstream["commit"],
            upstream_dirty=upstream["dirty"], checkpoint_path=str(args.checkpoint.resolve()),
            checkpoint_sha256=checkpoint_hash, dataset_path=metadata["dataset_path"],
            dataset_sha256=metadata["dataset_sha256"], dataset_instance_index=int(indices[i]),
            adapter_provenance={"sources": adapter_sources, "mapping": mapping},
            inference_config={
                "problem_size": args.problem_size, "dummy_rate": config["dummy_rate"],
                "dummy_size": config["dummy_size"], "sequence_length": config["sequence_length"],
                "eval_only": True, "init_val_met": "random", "seed": SEED,
                "val_m": VAL_M, "stall_limit": STALL_LIMIT, "k": K, "T_max": T_MAX,
                "with_bonus": True, "with_regular": True, "training": False,
                "backward": False, "optimizer_created": False,
                "protobuf_workaround": False, "configuration_class": "official_inference_smoke",
                "configuration_evidence": config["inference_evidence"],
                **tensorboard_compatibility,
            },
            selection_metadata={
                **decode_info, "successor": successor.tolist(),
                "official_get_order_matches": True,
                "official_recomputed_objective": official_recomputed,
                "objective_source": "rollout out[0] == final obj_history[:, -1, 1] and final solution_best_history[-1]",
            },
            canonical_solution=canonical, reported_objective=reported,
            independent_objective=independent, objective_abs_error=abs_error,
            objective_rel_error=rel_error, reported_objective_agrees=agrees,
            reference_objective=reference,
            gap_percent=((independent - reference) / reference * 100) if independent is not None else None,
            independent_feasible=validation["feasible"],
            constraint_details=validation["constraint_details"],
            runtime_seconds=runtime / batch_size,
            runtime_semantics="amortized 1000-step batch rollout seconds; engineering smoke; NOT PAPER-COMPARABLE",
            environment=environment,
            evidence_status="LOCAL_VERIFIED_INDEPENDENT" if passed else "FAILED",
            error=None if passed else "independent feasibility or reported-objective agreement failed",
        ))
    run_metadata = make_run_metadata(
        started_at=started_at, finished_at=finished_at,
        total_runtime_seconds=runtime, batch_size=batch_size,
        problem_size=args.problem_size, training=False, backward=False,
        optimizer_created=False, T_max=T_MAX, val_m=VAL_M,
        stall_limit=STALL_LIMIT, k=K, init_val_met="random",
        tensorboard_compatibility=tensorboard_compatibility,
        seed_side_effects={
            "source": "NeuOpt/run.py __main__", "python_random": SEED,
            "numpy": SEED, "torch": SEED, "torch_cuda_all": SEED,
            "cudnn_deterministic": True, "cudnn_benchmark": False,
        },
        solution_history_length=len(rollout_output[3][1]),
        runtime_class="engineering smoke; NOT PAPER-COMPARABLE",
    )
    write_result_bundle(args.output, rows, run_metadata=run_metadata)
    print(args.output)


if __name__ == "__main__":
    main()
