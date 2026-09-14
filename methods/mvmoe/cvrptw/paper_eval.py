#!/usr/bin/env python3
"""Paper evaluation of official MVMoE/4E on CVRPTW50/100, batch one."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from common.paper_results import (TIMING_SEMANTICS, append_record, finalize_chunk,
                                  initialize_chunk)
from common.provenance import (environment_provenance, git_provenance,
                               normalize_git_repository_identity, source_provenance)
from methods.mvmoe.cvrptw.adapter import adapt_batch
from methods.mvmoe.cvrptw.config import SUPPORTED_SIZES, get_size_config
from methods.mvmoe.cvrptw.decode import select_best_candidates
from methods.mvmoe.paper_config import (MODEL_CONFIG, UPSTREAM_COMMIT, UPSTREAM_URL,
                                        paper_inference_config)
from methods.mvmoe.paper_runtime import (compact_constraint_details, cuda_device,
                                         seed_official_inference, solve_one)
from problems.cvrptw.validate import validate


def _slice_native(arrays, index, *, problem_size, device):
    return adapt_batch(
        arrays["depots"][index:index + 1], arrays["points"][index:index + 1],
        arrays["demands"][index:index + 1], arrays["capacities"][index:index + 1],
        arrays["time_windows"][index:index + 1],
        arrays["service_times"][index:index + 1],
        problem_size=problem_size, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-metadata", type=Path)
    parser.add_argument("--problem-size", type=int, choices=SUPPORTED_SIZES, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-instances", type=int, choices=range(0, 6), default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    expected = get_size_config(args.problem_size)
    metadata_path = args.input_metadata or args.input.with_suffix(args.input.suffix + ".json")
    prepared = json.loads(metadata_path.read_text())
    input_hash = sha256_file(args.input)
    if prepared.get("format") != "mvmoe-cvrptw-input-v1":
        raise ValueError("unsupported prepared input metadata")
    if prepared.get("problem_size") != args.problem_size:
        raise ValueError("prepared input problem_size mismatch")
    if prepared.get("input_npz_sha256") != input_hash:
        raise ValueError("prepared NPZ hash does not match metadata")
    if (prepared.get("dataset_sha256") != expected["dataset_sha256"] or
            prepared.get("dataset_count") != expected["dataset_count"]):
        raise ValueError(f"prepared input is not the pinned CVRPTW{args.problem_size} dataset")
    upstream = git_provenance(args.upstream)
    if (upstream["commit"] != UPSTREAM_COMMIT or upstream["dirty"] or
            normalize_git_repository_identity(upstream["url"]) !=
            normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("official MVMoE checkout identity/cleanliness mismatch")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != expected["checkpoint_sha256"]:
        raise ValueError("unexpected official MVMoE checkpoint SHA256")

    import torch
    device = cuda_device(args.device, torch)
    seed_official_inference(torch)
    project = git_provenance(ROOT)
    if project["dirty"]:
        raise ValueError("formal paper evaluation requires a clean project checkout")
    environment = environment_provenance(device)
    protocol = paper_inference_config(args.problem_size, problem="CVRPTW")

    with np.load(args.input, allow_pickle=False) as data:
        arrays = {
            "depots": data["depots"], "points": data["points"],
            "demands": data["raw_demands"], "capacities": data["raw_capacities"],
            "time_windows": data["time_windows"], "service_times": data["service_times"],
            "time_tolerances": data["time_tolerances"],
            "indices": data["dataset_indices"],
            "references": data["reference_objectives"],
        }
    count = len(arrays["points"])
    indices = [int(value) for value in arrays["indices"]]
    if count == 0 or indices != list(range(indices[0], indices[0] + count)):
        raise ValueError("prepared chunk indices must be nonempty and contiguous")
    if prepared.get("dataset_indices") != indices or len(prepared.get("instance_names", [])) != count:
        raise ValueError("prepared metadata does not match NPZ instance identities")
    if not np.all(arrays["capacities"] == expected["capacity"]):
        raise ValueError("prepared capacities do not match pinned dataset")

    sources = source_provenance([
        Path(__file__), Path(__file__).with_name("adapter.py"),
        Path(__file__).with_name("decode.py"), Path(__file__).with_name("config.py"),
        ROOT / "methods/mvmoe/paper_config.py", ROOT / "methods/mvmoe/paper_runtime.py",
        ROOT / "problems/cvrptw/validate.py", ROOT / "problems/cvrp/validate.py",
        ROOT / "problems/cvrp/objective.py", ROOT / "common/objective_agreement.py",
        ROOT / "common/paper_results.py", ROOT / "common/provenance.py",
    ], root=ROOT)
    resume_identity = {
        "method": "MVMoE", "variant": "MVMoE/4E", "problem": "CVRPTW",
        "problem_size": args.problem_size, "paper_protocol": protocol,
        "project": project, "upstream": upstream,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": checkpoint_hash},
        "dataset": {"path": prepared["dataset_path"], "sha256": prepared["dataset_sha256"],
                    "count": expected["dataset_count"]},
        "prepared_input": {"path": str(args.input.resolve()), "sha256": input_hash,
                           "metadata_path": str(metadata_path.resolve())},
        "chunk": {"offset": indices[0], "count": count, "expected_indices": indices},
        "warmup": {"instances": args.warmup_instances,
                   "policy": "first chunk instances, then rerun formally; excluded from timing"},
        "environment": environment, "source_provenance": sources,
        "timing_semantics": TIMING_SEMANTICS,
    }
    _, completed = initialize_chunk(args.output_dir, resume_identity)
    if completed == set(indices):
        finalize_chunk(args.output_dir)
        print(args.output_dir / "metadata.json")
        return

    sys.path.insert(0, str(args.upstream.resolve()))
    from envs.VRPTWEnv import VRPTWEnv
    from models.MOEModel import MOEModel
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("problem") != "Train_ALL" or checkpoint.get("epoch") != 5000:
        raise ValueError("checkpoint metadata does not match official MVMoE/4E")
    model = MOEModel(**dict(MODEL_CONFIG, device=device)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    env = VRPTWEnv(problem_size=args.problem_size, pomo_size=args.problem_size,
                   loc_scaler=None, device=device)

    for local_index in range(min(args.warmup_instances, count)):
        native, depot_window, _ = _slice_native(
            arrays, local_index, problem_size=args.problem_size, device=device)
        env.depot_start, env.depot_end = depot_window
        solve_one(model, env, native, selector=select_best_candidates,
                  problem_size=args.problem_size, device=device, torch=torch, timed=False)

    for local_index, dataset_index in enumerate(indices):
        if dataset_index in completed:
            continue
        native, depot_window, _ = _slice_native(
            arrays, local_index, problem_size=args.problem_size, device=device)
        env.depot_start, env.depot_end = depot_window
        selection, runtime = solve_one(
            model, env, native, selector=select_best_candidates,
            problem_size=args.problem_size, device=device, torch=torch, timed=True)
        tolerance = float(arrays["time_tolerances"][local_index])
        validation = validate(
            arrays["depots"][local_index], arrays["points"][local_index],
            arrays["demands"][local_index], arrays["capacities"][local_index],
            arrays["time_windows"][local_index], arrays["service_times"][local_index],
            selection["canonical_solution"], speed=1.0, start_time=depot_window[0],
            time_tolerance=tolerance, capacity_tolerance=tolerance)
        independent = validation["independent_objective"]
        reported = selection["reported_objective"]
        agrees = independent is not None and objective_agrees(reported, independent)
        reference = float(arrays["references"][local_index])
        passed = validation["feasible"] and agrees
        record = {
            "dataset_instance_index": dataset_index,
            "instance_id": prepared["instance_names"][local_index],
            "canonical_solution": selection["canonical_solution"],
            "reported_objective": reported,
            "independent_objective": independent,
            "reference_objective": reference,
            "gap_percent": ((independent - reference) / reference * 100.0)
            if independent is not None else None,
            "runtime_seconds": runtime,
            "independent_feasible": bool(validation["feasible"]),
            "reported_objective_agrees": bool(agrees),
            "selection": {key: value for key, value in selection.items()
                          if key not in ("canonical_solution", "reported_objective")},
            "constraint_details": compact_constraint_details(
                validation["constraint_details"], problem="CVRPTW"),
            "depot_time_window": list(depot_window),
            "time_tolerance": tolerance,
            "evidence_status": "INDEPENDENT_VERIFIED" if passed else "FAILED",
        }
        if not passed:
            raise RuntimeError(f"independent validation failed at dataset index {dataset_index}")
        append_record(args.output_dir, record)
        print(json.dumps({
            "method": "MVMoE", "problem": "CVRPTW", "size": args.problem_size,
            "original_batch_size": 1, "pomo_size": args.problem_size, "augmentation": 8,
            "checkpoint_sha256": checkpoint_hash, "dataset_sha256": prepared["dataset_sha256"],
            "gpu": environment["gpu"], "dataset_instance_index": dataset_index,
            "objective": independent, "reference": reference,
            "gap_percent": record["gap_percent"], "runtime_seconds": runtime,
            "feasible": validation["feasible"],
        }, sort_keys=True, allow_nan=False), flush=True)
    finalize_chunk(args.output_dir)
    print(args.output_dir / "metadata.json")


if __name__ == "__main__":
    main()
