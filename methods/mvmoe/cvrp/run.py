#!/usr/bin/env python3
"""Direct official MVMoE/4E CVRP50 rollout with canonical solution capture."""
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
from common.provenance import environment_provenance, git_provenance, source_provenance
from common.result_schema import new_result, write_result_bundle
from methods.mvmoe.cvrp.adapter import adapt_batch
from methods.mvmoe.cvrp.decode import select_best_candidates
from problems.cvrp.validate import validate

UPSTREAM_URL = "https://github.com/RoyalSkye/Routing-MVMoE"
UPSTREAM_COMMIT = "af29e5af0595f94f3ecc3bc46d72df1089a62682"
CHECKPOINT_SHA256 = "3417f302fbddf232fd19a2a886cd1c7f44290b6d8c7280fcb0ae3777eeed3192"
DATASET_SHA256 = "eea12fbefe9c1bcc008d56ecfc1c50dadd64ac774f3547774c9fade8a7baa6c2"

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
    parser.add_argument("--input-metadata", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aug-factor", type=int, choices=[1, 8], required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metadata_path = args.input_metadata or args.input.with_suffix(args.input.suffix + ".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format") != "mvmoe-cvrp50-input-v1":
        raise ValueError("unsupported prepared input metadata")
    if metadata.get("input_npz_sha256") != sha256_file(args.input):
        raise ValueError("prepared NPZ hash does not match metadata")
    if metadata.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError("prepared input is not from the pinned official CVRP50 dataset")
    upstream = git_provenance(args.upstream)
    if upstream["commit"] != UPSTREAM_COMMIT or upstream["url"].removesuffix(".git") != UPSTREAM_URL:
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

    with np.load(args.input, allow_pickle=False) as data:
        depots, points = data["depots"], data["points"]
        demands, capacities = data["raw_demands"], data["raw_capacities"]
        indices, references = data["dataset_indices"], data["reference_objectives"]
    native, mapping = adapt_batch(depots, points, demands, capacities, device=device)
    batch_size = len(points)

    sys.path.insert(0, str(args.upstream.resolve()))
    from envs.CVRPEnv import CVRPEnv
    from models.MOEModel import MOEModel
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("problem") != "Train_ALL" or checkpoint.get("epoch") != 5000:
        raise ValueError("checkpoint metadata does not match official 4E n50 artifact")
    config = dict(MODEL_CONFIG, device=device)
    model = MOEModel(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    env = CVRPEnv(problem_size=50, pomo_size=50, device=device)

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
    selections = select_best_candidates(reward.detach().cpu().numpy(),
                                        env.selected_node_list.detach().cpu().numpy(),
                                        aug_factor=args.aug_factor, batch_size=batch_size)

    project = git_provenance(ROOT)
    adapter_sources = source_provenance([
        Path(__file__), Path(__file__).with_name("adapter.py"),
        Path(__file__).with_name("decode.py"), ROOT / "problems/cvrp/validate.py",
        ROOT / "common/result_schema.py", ROOT / "common/provenance.py",
    ], root=ROOT)
    environment = environment_provenance(device)
    rows = []
    for i, selection in enumerate(selections):
        validation = validate(depots[i], points[i], demands[i], capacities[i],
                              selection["canonical_solution"])
        independent = validation["independent_objective"]
        reported = selection["reported_objective"]
        abs_error = abs(reported - independent) if independent is not None else None
        rel_error = abs_error / abs(independent) if independent not in (None, 0) else None
        reference = float(references[i])
        rows.append(new_result(
            method="MVMoE", variant="MOE/4E", problem="CVRP", problem_size=50,
            instance_id=metadata["instance_names"][i],
            project_repo_commit=project["commit"], project_repo_dirty=project["dirty"],
            upstream_url=upstream["url"], upstream_commit=upstream["commit"],
            upstream_dirty=upstream["dirty"], checkpoint_path=str(args.checkpoint.resolve()),
            checkpoint_sha256=checkpoint_hash, dataset_path=metadata["dataset_path"],
            dataset_sha256=metadata["dataset_sha256"], dataset_instance_index=int(indices[i]),
            adapter_provenance={"sources": adapter_sources, "mapping": mapping},
            inference_config={"problem_size": 50, "pomo_size": 50,
                              "aug_factor": args.aug_factor, "eval_type": "argmax",
                              "seed": args.seed, "model_type": "MOE",
                              "model": MODEL_CONFIG, "fine_tune_epochs": 0,
                              "loc_scaler": None,
                              "configuration_class": "official" if args.aug_factor == 8 else "reduced_debug"},
            selection_metadata={k: v for k, v in selection.items()
                                if k not in ("canonical_solution", "reported_objective")},
            canonical_solution=selection["canonical_solution"],
            reported_objective=reported, independent_objective=independent,
            objective_abs_error=abs_error, objective_rel_error=rel_error,
            reference_objective=reference,
            gap_percent=((independent - reference) / reference * 100) if independent is not None else None,
            independent_feasible=validation["feasible"],
            constraint_details=validation["constraint_details"],
            runtime_seconds=runtime / batch_size, environment=environment,
            evidence_status="LOCAL_VERIFIED_INDEPENDENT" if validation["feasible"] else "FAILED",
            error=None if validation["feasible"] else "independent CVRP validation failed",
        ))
    write_result_bundle(args.output, rows, run_metadata={
        "started_at": datetime.now(timezone.utc).isoformat(), "total_runtime_seconds": runtime,
        "batch_size": batch_size, "aug_factor": args.aug_factor,
        "selected_node_list_shape": list(env.selected_node_list.shape),
        "reward_shape": list(reward.shape), "training": False, "backward": False,
        "optimizer_created": False, "fine_tune_epochs": 0,
    })
    print(args.output)


if __name__ == "__main__":
    main()
