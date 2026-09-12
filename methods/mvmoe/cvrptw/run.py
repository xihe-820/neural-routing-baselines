#!/usr/bin/env python3
"""Direct official MVMoE/4E CVRPTW50 rollout with solution capture."""
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
from methods.mvmoe.cvrptw.adapter import adapt_batch
from methods.mvmoe.cvrptw.config import (CAPACITY, CHECKPOINT_SHA256, DATASET_SHA256,
                                        PROBLEM_SIZE, require_problem_size)
from methods.mvmoe.cvrptw.decode import select_best_candidates
from problems.cvrptw.validate import validate

UPSTREAM_URL = "https://github.com/RoyalSkye/Routing-MVMoE"
UPSTREAM_COMMIT = "af29e5af0595f94f3ecc3bc46d72df1089a62682"

MODEL_CONFIG = {
    "embedding_dim": 128, "sqrt_embedding_dim": 128 ** 0.5,
    "encoder_layer_num": 6, "decoder_layer_num": 1, "qkv_dim": 16,
    "head_num": 8, "logit_clipping": 10, "ff_hidden_dim": 512,
    "num_experts": 4, "eval_type": "argmax", "norm": "instance",
    "norm_loc": "norm_last", "expert_loc": ["Enc0", "Enc1", "Enc2", "Enc3", "Enc4", "Enc5", "Dec"],
    "problem": "Train_ALL", "topk": 2, "routing_level": "node",
    "routing_method": "input_choice",
}


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
    parser.add_argument("--problem-size", type=int, choices=[PROBLEM_SIZE], required=True)
    parser.add_argument("--input-metadata", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aug-factor", type=int, choices=[1, 8], required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    require_problem_size(args.problem_size)
    metadata_path = args.input_metadata or args.input.with_suffix(args.input.suffix + ".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format") != "mvmoe-cvrptw-input-v1":
        raise ValueError("unsupported prepared input metadata")
    if metadata.get("problem_size") != PROBLEM_SIZE:
        raise ValueError("prepared input is not CVRPTW50")
    if metadata.get("input_npz_sha256") != sha256_file(args.input):
        raise ValueError("prepared NPZ hash does not match metadata")
    if metadata.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError("prepared input is not from the pinned official CVRPTW50 dataset")
    upstream = git_provenance(args.upstream)
    if (upstream["commit"] != UPSTREAM_COMMIT or
            normalize_git_repository_identity(upstream["url"]) !=
            normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("unexpected MVMoE upstream identity")
    if upstream["dirty"]:
        raise ValueError("official MVMoE checkout must be clean")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("unexpected MVMoE/4E n50 checkpoint SHA256")

    import torch
    device = _device(args.device, torch)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    # Match Routing-MVMoE/utils.py:seed_everything without importing its extras.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    with np.load(args.input, allow_pickle=False) as data:
        depots, points = data["depots"], data["points"]
        demands, capacities = data["raw_demands"], data["raw_capacities"]
        time_windows, service_times = data["time_windows"], data["service_times"]
        time_tolerances = data["time_tolerances"]
        indices, references = data["dataset_indices"], data["reference_objectives"]
    if not np.all(capacities == CAPACITY):
        raise ValueError("prepared capacities do not match CVRPTW50")
    native, depot_window, mapping = adapt_batch(
        depots, points, demands, capacities, time_windows, service_times,
        problem_size=PROBLEM_SIZE, device=device)
    batch_size = len(points)

    sys.path.insert(0, str(args.upstream.resolve()))
    from envs.VRPTWEnv import VRPTWEnv
    from models.MOEModel import MOEModel
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("problem") != "Train_ALL" or checkpoint.get("epoch") != 5000:
        raise ValueError("checkpoint metadata does not match official 4E n50 artifact")
    config = dict(MODEL_CONFIG, device=device)
    model = MOEModel(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    env = VRPTWEnv(problem_size=PROBLEM_SIZE, pomo_size=PROBLEM_SIZE,
                   loc_scaler=None, device=device)
    # Official Tester._solve_cvrptwlib overrides the depot horizon before load.
    # We additionally preserve the benchmark lower bound; adapter rejects mixed batches.
    env.depot_start, env.depot_end = depot_window

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    with torch.no_grad():
        env.load_problems(batch_size, problems=native, aug_factor=args.aug_factor)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    finished_at = datetime.now(timezone.utc).isoformat()
    selections = select_best_candidates(
        reward.detach().cpu().numpy(), env.selected_node_list.detach().cpu().numpy(),
        aug_factor=args.aug_factor, batch_size=batch_size)

    project = git_provenance(ROOT)
    adapter_sources = source_provenance([
        Path(__file__), Path(__file__).with_name("config.py"),
        Path(__file__).with_name("adapter.py"), Path(__file__).with_name("decode.py"),
        ROOT / "problems/cvrptw/validate.py", ROOT / "problems/cvrp/validate.py",
        ROOT / "common/objective_agreement.py", ROOT / "common/result_schema.py",
        ROOT / "common/provenance.py",
    ], root=ROOT)
    environment = environment_provenance(device)
    rows = []
    for i, selection in enumerate(selections):
        tolerance = float(time_tolerances[i])
        validation = validate(
            depots[i], points[i], demands[i], capacities[i], time_windows[i],
            service_times[i], selection["canonical_solution"], speed=1.0,
            start_time=depot_window[0], time_tolerance=tolerance,
            capacity_tolerance=tolerance)
        independent = validation["independent_objective"]
        reported = selection["reported_objective"]
        abs_error = abs(reported - independent) if independent is not None else None
        rel_error = abs_error / abs(independent) if independent not in (None, 0) else None
        reported_agrees = objective_agrees(reported, independent) if independent is not None else False
        reference = float(references[i])
        independent_pass = validation["feasible"] and reported_agrees
        rows.append(new_result(
            method="MVMoE", variant="MOE/4E", problem="CVRPTW", problem_size=PROBLEM_SIZE,
            instance_id=metadata["instance_names"][i],
            project_repo_commit=project["commit"], project_repo_dirty=project["dirty"],
            upstream_url=upstream["url"], upstream_commit=upstream["commit"],
            upstream_dirty=upstream["dirty"], checkpoint_path=str(args.checkpoint.resolve()),
            checkpoint_sha256=checkpoint_hash, dataset_path=metadata["dataset_path"],
            dataset_sha256=metadata["dataset_sha256"], dataset_instance_index=int(indices[i]),
            adapter_provenance={"sources": adapter_sources, "mapping": mapping},
            inference_config={
                "problem_size": PROBLEM_SIZE, "pomo_size": PROBLEM_SIZE,
                "aug_factor": args.aug_factor, "eval_type": "argmax", "seed": args.seed,
                "model_type": "MOE", "model": MODEL_CONFIG, "fine_tune_epochs": 0,
                "loc_scaler": None, "speed": 1.0,
                "depot_time_window": list(depot_window), "time_tolerance": tolerance,
                "configuration_class": "official_search_decode_smoke"
                if args.aug_factor == 8 else "reduced_debug",
            },
            selection_metadata={k: v for k, v in selection.items()
                                if k not in ("canonical_solution", "reported_objective")},
            canonical_solution=selection["canonical_solution"],
            reported_objective=reported, independent_objective=independent,
            objective_abs_error=abs_error, objective_rel_error=rel_error,
            reported_objective_agrees=reported_agrees, reference_objective=reference,
            gap_percent=((independent - reference) / reference * 100)
            if independent is not None else None,
            independent_feasible=validation["feasible"],
            constraint_details=validation["constraint_details"],
            runtime_seconds=runtime / batch_size,
            runtime_semantics="amortized_batch_runtime_seconds; not single-instance latency or paper-comparable",
            environment=environment,
            evidence_status="LOCAL_VERIFIED_INDEPENDENT" if independent_pass else "FAILED",
            error=None if independent_pass else
            "independent feasibility or reported-objective agreement failed",
        ))
    run_metadata = make_run_metadata(
        started_at=started_at, finished_at=finished_at,
        total_runtime_seconds=runtime, batch_size=batch_size,
        aug_factor=args.aug_factor, problem_size=PROBLEM_SIZE,
        selected_node_list_shape=list(env.selected_node_list.shape),
        reward_shape=list(reward.shape), depot_time_window=list(depot_window), speed=1.0,
        training=False, backward=False, optimizer_created=False, fine_tune_epochs=0,
        seed_side_effects={
            "source": "Routing-MVMoE/utils.py:seed_everything",
            "python_random": args.seed, "numpy": args.seed, "torch": args.seed,
            "torch_cuda_all": args.seed, "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
        runtime_class="engineering smoke; NOT PAPER-COMPARABLE",
    )
    write_result_bundle(args.output, rows, run_metadata=run_metadata)
    print(args.output)


if __name__ == "__main__":
    main()
