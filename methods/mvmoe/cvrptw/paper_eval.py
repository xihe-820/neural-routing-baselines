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
from methods.mvmoe.cvrptw.paper_protocol import (SCALED_PROTOCOL_FIELDS,
                                                 scaled_paper_inference_config)
from methods.mvmoe.cvrptw.scaling import (assert_continuous_env,
                                          scale_prepared_instance)
from methods.mvmoe.paper_config import MODEL_CONFIG, UPSTREAM_COMMIT, UPSTREAM_URL
from methods.mvmoe.paper_runtime import (compact_constraint_details, cuda_device,
                                         seed_official_inference, solve_one)
from problems.cvrptw.validate import validate


def require_scaled_artifact_path(output_dir, problem_size):
    """Keep canonical scaled chunks out of the verified unscaled directories."""
    expected = f"cvrptw{int(problem_size)}_scaled"
    if expected not in Path(output_dir).resolve().parts:
        raise ValueError(
            f"scaled CVRPTW formal output must be under an {expected} directory")


def _slice_native(arrays, index, *, problem_size, device):
    """Scale one original row before the untimed model-input adaptation."""
    scaled = scale_prepared_instance(arrays, index)
    native, depot_window, mapping = adapt_batch(
        scaled["depot"][None, :], scaled["points"][None, :, :],
        scaled["raw_demands"][None, :],
        np.asarray([scaled["raw_capacity"]], dtype=np.float32),
        scaled["time_windows"][None, :, :],
        scaled["service_times"][None, :],
        problem_size=problem_size, device=device)
    mapping = dict(mapping)
    mapping.update({
        "coordinate_scaling": "coordinates / per-instance scaler",
        "time_window_scaling": "time_windows / per-instance scaler",
        "service_time_scaling": "service_times / per-instance scaler",
        "scaler": scaled["scaler"],
        "loc_scaler": None,
        "distance_rounding": False,
    })
    return scaled, native, depot_window, mapping


def validate_scaled_solution(arrays, index, scaled, route, depot_window):
    """Validate one route in both domains and enforce unit conversion."""
    tolerance = float(arrays["time_tolerances"][index])
    scaled_validation = validate(
        scaled["depot"], scaled["points"], scaled["raw_demands"],
        scaled["raw_capacity"], scaled["time_windows"],
        scaled["service_times"], route, speed=1.0,
        start_time=depot_window[0],
        time_tolerance=tolerance / scaled["scaler"],
        capacity_tolerance=tolerance)
    original_validation = validate(
        arrays["depots"][index], arrays["points"][index],
        arrays["demands"][index], arrays["capacities"][index],
        arrays["time_windows"][index], arrays["service_times"][index],
        route, speed=1.0,
        start_time=float(arrays["time_windows"][index, 0, 0]),
        time_tolerance=tolerance, capacity_tolerance=tolerance)
    scaled_objective = scaled_validation["independent_objective"]
    original_objective = original_validation["independent_objective"]
    if scaled_objective is None or original_objective is None:
        raise RuntimeError("independent validation could not score scaled formal route")
    scaled_times_s = scaled_objective * scaled["scaler"]
    cross_domain_agrees = objective_agrees(scaled_times_s, original_objective)
    if (not scaled_validation["feasible"] or
            not original_validation["feasible"] or
            not cross_domain_agrees):
        raise RuntimeError("scaled/original formal route correctness gate failed")
    return {
        "scaled_validation": scaled_validation,
        "original_validation": original_validation,
        "scaled_route_objective": scaled_objective,
        "original_objective": original_objective,
        "scaled_objective_times_s": scaled_times_s,
        "scaled_to_original_objective_agrees": cross_domain_agrees,
    }


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
    require_scaled_artifact_path(args.output_dir, args.problem_size)

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
    protocol = scaled_paper_inference_config(args.problem_size)
    observed_scaling_protocol = {
        key: protocol.get(key) for key in SCALED_PROTOCOL_FIELDS}
    if observed_scaling_protocol != SCALED_PROTOCOL_FIELDS:
        raise RuntimeError("canonical scaled CVRPTW paper protocol identity mismatch")

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
        Path(__file__).with_name("scaling.py"),
        Path(__file__).with_name("paper_protocol.py"),
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
    assert_continuous_env(env)

    for local_index in range(min(args.warmup_instances, count)):
        _, native, depot_window, _ = _slice_native(
            arrays, local_index, problem_size=args.problem_size, device=device)
        env.depot_start, env.depot_end = depot_window
        assert_continuous_env(env)
        solve_one(model, env, native, selector=select_best_candidates,
                  problem_size=args.problem_size, device=device, torch=torch, timed=False)

    for local_index, dataset_index in enumerate(indices):
        if dataset_index in completed:
            continue
        scaled, native, depot_window, mapping = _slice_native(
            arrays, local_index, problem_size=args.problem_size, device=device)
        env.depot_start, env.depot_end = depot_window
        assert_continuous_env(env)
        selection, runtime = solve_one(
            model, env, native, selector=select_best_candidates,
            problem_size=args.problem_size, device=device, torch=torch, timed=True)
        domain_result = validate_scaled_solution(
            arrays, local_index, scaled, selection["canonical_solution"], depot_window)
        validation = domain_result["original_validation"]
        independent = domain_result["original_objective"]
        scaled_reported = selection["reported_objective"]
        scaled_reported_agrees = objective_agrees(
            scaled_reported, domain_result["scaled_route_objective"])
        reported = scaled_reported * scaled["scaler"]
        agrees = scaled_reported_agrees and objective_agrees(reported, independent)
        reference = float(arrays["references"][local_index])
        passed = validation["feasible"] and agrees
        tolerance = float(arrays["time_tolerances"][local_index])
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
            "scaled_reported_objective": scaled_reported,
            "scaled_reported_objective_agrees": bool(scaled_reported_agrees),
            "scaled_route_objective": domain_result["scaled_route_objective"],
            "scaled_objective_times_s": domain_result["scaled_objective_times_s"],
            "scaled_to_original_objective_agrees": bool(
                domain_result["scaled_to_original_objective_agrees"]),
            "scaler": scaled["scaler"],
            "original_depot_tw_end": scaled["depot_tw_end"],
            "original_coordinate_max": scaled["coordinate_max"],
            "scaled_depot_tw_end": scaled["scaled_depot_tw_end"],
            "scaled_coordinate_max": scaled["scaled_coordinate_max"],
            "input_scaling_protocol": protocol["input_scaling"],
            "input_mapping": mapping,
            "selection": {key: value for key, value in selection.items()
                          if key not in ("canonical_solution", "reported_objective")},
            "constraint_details": compact_constraint_details(
                validation["constraint_details"], problem="CVRPTW"),
            "depot_time_window": list(depot_window),
            "original_depot_time_window": arrays["time_windows"][
                local_index, 0].astype(float).tolist(),
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
