#!/usr/bin/env python3
"""Run the isolated continuous-scaling B audit against existing A paper records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.hashing import sha256_file
from common.objective_agreement import OBJECTIVE_ATOL, OBJECTIVE_RTOL, objective_agrees
from common.paper_results import read_jsonl, validate_record
from common.provenance import (environment_provenance, git_provenance,
                               normalize_git_repository_identity, source_provenance)
from methods.mvmoe.cvrptw.adapter import adapt_batch
from methods.mvmoe.cvrptw.config import SUPPORTED_SIZES, get_size_config
from methods.mvmoe.cvrptw.decode import select_best_candidates
from methods.mvmoe.cvrptw.scaling import assert_continuous_env, scale_instance
from methods.mvmoe.paper_config import (MODEL_CONFIG, UPSTREAM_COMMIT, UPSTREAM_URL,
                                        paper_inference_config)
from methods.mvmoe.paper_runtime import (cuda_device, seed_official_inference,
                                         solve_one)
from problems.cvrptw.validate import validate


PROJECT_BASE_COMMIT = "e64db6016a1cb53de53b2ce6d6039186652538f7"
AUDIT_COUNT = 20
OFFICIAL_ENV_EPSILON = 1e-5


def _load_a_records(chunk_dirs, problem_size, expected):
    records = {}
    sources = []
    for directory in map(Path, chunk_dirs):
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing A metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        identity = metadata.get("resume_identity", {})
        if (metadata.get("state") != "KIT_VALIDATED" or
                identity.get("method") != "MVMoE" or
                identity.get("variant") != "MVMoE/4E" or
                identity.get("problem") != "CVRPTW" or
                identity.get("problem_size") != problem_size or
                identity.get("paper_protocol") != paper_inference_config(
                    problem_size, problem="CVRPTW") or
                identity.get("checkpoint", {}).get("sha256") !=
                expected["checkpoint_sha256"] or
                identity.get("dataset", {}).get("sha256") !=
                expected["dataset_sha256"]):
            raise ValueError(f"A artifact identity/state mismatch: {directory}")
        records_name = metadata.get("validated_records_file")
        if not records_name:
            raise ValueError(f"A artifact has no validated records file: {directory}")
        records_path = directory / records_name
        expected_hash = metadata.get("validated_records_sha256")
        if not records_path.is_file() or sha256_file(records_path) != expected_hash:
            raise ValueError(f"A validated records hash mismatch: {directory}")
        sources.append({
            "directory": str(directory.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "validated_records_path": str(records_path.resolve()),
            "validated_records_sha256": expected_hash,
            "environment": identity.get("environment"),
        })
        for record in read_jsonl(records_path):
            index = validate_record(record, require_kit=True)
            if index >= AUDIT_COUNT:
                continue
            if index in records:
                raise ValueError(f"duplicate A record for index {index}")
            records[index] = record
    expected = set(range(AUDIT_COUNT))
    if set(records) != expected:
        raise ValueError(
            f"A records must cover exactly indices 0..19; missing={sorted(expected-set(records))}")
    return records, sources


def _native(scaled, problem_size, device):
    native, depot_window, mapping = adapt_batch(
        scaled["depot"][None, :], scaled["points"][None, :, :],
        scaled["raw_demands"][None, :],
        np.asarray([scaled["raw_capacity"]], dtype=np.float32),
        scaled["time_windows"][None, :, :],
        scaled["service_times"][None, :],
        problem_size=problem_size, device=device)
    mapping = dict(mapping)
    mapping.update({
        "coordinate_scaling": "audit input divided by per-instance scaler before adapter",
        "time_window_scaling": "audit input divided by the same per-instance scaler",
        "service_time_scaling": "audit input divided by the same per-instance scaler",
        "scaler": scaled["scaler"],
        "loc_scaler": None,
    })
    return native, depot_window, mapping


def _slack_diagnostics(validation, scaler):
    timelines = validation["constraint_details"].get("route_timelines", [])
    customer = [float(event["tw_end"] - event["service_start"])
                for route in timelines for event in route["events"]]
    depot = [float(route["depot_tw_end"] - route["depot_arrival"])
             for route in timelines]
    customer_min = min(customer) if customer else None
    depot_min = min(depot) if depot else None
    all_values = [value for value in (customer_min, depot_min) if value is not None]
    overall = min(all_values) if all_values else None
    scaled = overall / scaler if overall is not None else None
    return {
        "minimum_customer_tw_slack_original": customer_min,
        "minimum_depot_return_slack_original": depot_min,
        "minimum_feasibility_slack_original": overall,
        "minimum_feasibility_slack_scaled": scaled,
        "official_env_epsilon": OFFICIAL_ENV_EPSILON,
        "within_10x_official_epsilon": (
            scaled is not None and scaled <= 10 * OFFICIAL_ENV_EPSILON),
        "uses_positive_official_epsilon_to_pass": (
            scaled is not None and scaled < 0.0),
    }


def _best_index(record, field):
    if field in record:
        return record[field]
    return record.get("selection", {}).get(field)


def summarize(records):
    a_obj = np.asarray([r["comparison_to_A"]["A_original_objective"] for r in records])
    b_obj = np.asarray([r["comparison_to_A"]["B_original_objective"] for r in records])
    a_gap = np.asarray([r["comparison_to_A"]["A_gap"] for r in records])
    b_gap = np.asarray([r["comparison_to_A"]["B_gap"] for r in records])
    ties = np.asarray([objective_agrees(a, b) for a, b in zip(a_obj, b_obj)])
    return {
        "instance_count": len(records),
        "A_mean_objective": float(a_obj.mean()),
        "B_mean_objective": float(b_obj.mean()),
        "A_mean_per_instance_gap_percent": float(a_gap.mean()),
        "B_mean_per_instance_gap_percent": float(b_gap.mean()),
        "mean_B_minus_A_objective": float((b_obj - a_obj).mean()),
        "B_better_count": int(((b_obj < a_obj) & ~ties).sum()),
        "A_better_count": int(((a_obj < b_obj) & ~ties).sum()),
        "tie_count": int(ties.sum()),
        "same_route_count": sum(r["comparison_to_A"]["same_route_as_A"] for r in records),
        "different_route_count": sum(not r["comparison_to_A"]["same_route_as_A"]
                                     for r in records),
        "B_independent_feasible_count": sum(r["validation"]["independent_feasible"]
                                             for r in records),
        "B_kit_feasible_count": sum(r["validation"]["kit_feasible"] for r in records),
        "max_scaled_times_s_original_abs_error": max(
            r["objectives"]["scaled_times_s_original_abs_error"] for r in records),
        "A_mean_runtime_seconds": float(np.mean([
            r["comparison_to_A"]["A_runtime_seconds"] for r in records])),
        "B_mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in records])),
        "epsilon_boundary_observation_count": sum(
            r["epsilon_diagnostics"]["within_10x_official_epsilon"] for r in records),
    }


def _write_artifact(output_dir, metadata, records, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    records_path.write_text("".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records))
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    metadata = dict(metadata)
    metadata["records"] = {"path": "records.jsonl", "sha256": sha256_file(records_path)}
    metadata["summary"] = {"path": "summary.json", "sha256": sha256_file(summary_path)}
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, choices=SUPPORTED_SIZES, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--a-chunk-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-instances", type=int, choices=range(0, 6), default=2)
    args = parser.parse_args()

    expected_output = (ROOT / "artifacts/audit/mvmoe_cvrptw_scaling" /
                       f"cvrptw{args.problem_size}").resolve()
    if args.output_dir.resolve() != expected_output:
        raise ValueError(f"audit output must be isolated at {expected_output}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("audit output directory must not already contain files")

    expected = get_size_config(args.problem_size)
    project = git_provenance(ROOT)
    if project["commit"] != PROJECT_BASE_COMMIT:
        raise ValueError("project HEAD does not match the fixed audit base commit")
    upstream = git_provenance(args.upstream)
    if (upstream["commit"] != UPSTREAM_COMMIT or upstream["dirty"] or
            normalize_git_repository_identity(upstream["url"]) !=
            normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("official MVMoE checkout identity/cleanliness mismatch")
    if sha256_file(args.dataset) != expected["dataset_sha256"]:
        raise ValueError("dataset SHA256 mismatch")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != expected["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 mismatch")
    a_records, a_sources = _load_a_records(
        args.a_chunk_dirs, args.problem_size, expected)

    import ml4co_kit as kit
    wrapper = kit.CVRPTWWrapper()
    wrapper.from_pickle(args.dataset)
    if len(wrapper.task_list) != expected["dataset_count"]:
        raise ValueError("dataset count mismatch")
    tasks = wrapper.task_list[:AUDIT_COUNT]
    if any(type(task) is not kit.CVRPTWTask or task.nodes_num != args.problem_size
           for task in tasks):
        raise ValueError("dataset task type/problem size mismatch")

    import torch
    device = cuda_device(args.device, torch)
    seed_official_inference(torch)
    environment = environment_provenance(device)
    protocol = paper_inference_config(args.problem_size, problem="CVRPTW")
    for source in a_sources:
        a_environment = source.get("environment") or {}
        if "RTX 4090" not in str(a_environment.get("gpu")):
            raise ValueError("A artifact was not produced on the required RTX 4090")
        if (a_environment.get("hostname") and
                a_environment["hostname"] != environment.get("hostname")):
            raise ValueError("A and B must run on the same RTX 4090 host")

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

    for task in tasks[:args.warmup_instances]:
        scaled = scale_instance(task.depots, task.points, task.demands, task.capacity,
                                task.tw, task.service)
        native, depot_window, _ = _native(scaled, args.problem_size, device)
        env.depot_start, env.depot_end = depot_window
        assert_continuous_env(env)
        solve_one(model, env, native, selector=select_best_candidates,
                  problem_size=args.problem_size, device=device, torch=torch, timed=False)

    records = []
    for index, task in enumerate(tasks):
        a_record = a_records[index]
        scaled = scale_instance(task.depots, task.points, task.demands, task.capacity,
                                task.tw, task.service)
        native, depot_window, mapping = _native(scaled, args.problem_size, device)
        env.depot_start, env.depot_end = depot_window
        assert_continuous_env(env)
        selection, runtime = solve_one(
            model, env, native, selector=select_best_candidates,
            problem_size=args.problem_size, device=device, torch=torch, timed=True)
        route = selection["canonical_solution"]
        tolerance = float(task.threshold)
        scaled_validation = validate(
            scaled["depot"], scaled["points"], scaled["raw_demands"],
            scaled["raw_capacity"], scaled["time_windows"], scaled["service_times"],
            route, speed=1.0, start_time=depot_window[0],
            time_tolerance=tolerance / scaled["scaler"],
            capacity_tolerance=tolerance)
        original_validation = validate(
            task.depots, task.points, task.demands, task.capacity, task.tw, task.service,
            route, speed=1.0, start_time=float(task.tw[0, 0]),
            time_tolerance=tolerance, capacity_tolerance=tolerance)
        scaled_route_objective = scaled_validation["independent_objective"]
        original_objective = original_validation["independent_objective"]
        if scaled_route_objective is None or original_objective is None:
            raise RuntimeError(f"independent validation could not score B index {index}")
        scaled_times_s = scaled_route_objective * scaled["scaler"]
        reported_agrees = objective_agrees(
            selection["reported_objective"], scaled_route_objective)
        cross_domain_agrees = objective_agrees(scaled_times_s, original_objective)
        solution = np.asarray(route, dtype=np.int64)
        kit_feasible = bool(task.check_constraints(solution))
        kit_objective = float(task.evaluate(solution))
        kit_agrees = objective_agrees(kit_objective, original_objective)
        reference = float(a_record["reference_objective"])
        if not objective_agrees(float(task.evaluate(task.ref_sol)), reference):
            raise RuntimeError(f"A reference does not match dataset at index {index}")
        b_gap = (original_objective - reference) / reference * 100.0
        epsilon_diagnostics = _slack_diagnostics(
            original_validation, scaled["scaler"])
        gates = (scaled_validation["feasible"], original_validation["feasible"],
                 reported_agrees, cross_domain_agrees, kit_feasible, kit_agrees,
                 not epsilon_diagnostics["uses_positive_official_epsilon_to_pass"])
        if not all(gates):
            raise RuntimeError(f"B correctness gate failed at dataset index {index}")
        record = {
            "dataset_instance_index": index,
            "instance_id": task.name,
            "scaler": scaled["scaler"],
            "original": {
                "coordinate_max": scaled["coordinate_max"],
                "depot_tw_start": scaled["depot_tw_start"],
                "depot_tw_end": scaled["depot_tw_end"],
            },
            "scaled": {
                "coordinate_max": float(max(scaled["depot"].max(),
                                             scaled["points"].max())),
                "depot_tw_start": float(scaled["time_windows"][0, 0]),
                "depot_tw_end": float(scaled["time_windows"][0, 1]),
                "service_min": float(scaled["service_times"][1:].min()),
                "service_max": float(scaled["service_times"][1:].max()),
            },
            "protocol": {
                "pomo_size": args.problem_size, "aug_factor": 8,
                "eval_type": "argmax", "original_batch_size": 1,
                "loc_scaler": None, "speed": 1.0, "seed": 2024,
                "fine_tune_epochs": 0, "training": False, "backward": False,
                "optimizer_created": False,
            },
            "decode": {
                "best_aug_idx": selection["best_aug_idx"],
                "best_pomo_idx": selection["best_pomo_idx"],
                "canonical_solution": route,
                "details": selection["decoding"],
            },
            "objectives": {
                "scaled_model_reward": selection["candidate_reward"],
                "scaled_model_objective": selection["reported_objective"],
                "scaled_route_objective": scaled_route_objective,
                "scaled_objective_times_s": scaled_times_s,
                "independent_original_objective": original_objective,
                "reference_objective": reference,
                "original_domain_gap_percent": b_gap,
                "scaled_times_s_original_abs_error": abs(
                    scaled_times_s - original_objective),
            },
            "validation": {
                "scaled_domain_independent_feasible": bool(
                    scaled_validation["feasible"]),
                "independent_feasible": bool(original_validation["feasible"]),
                "reported_scaled_objective_agrees": reported_agrees,
                "scaled_to_original_objective_agrees": cross_domain_agrees,
                "kit_feasible": kit_feasible,
                "kit_objective": kit_objective,
                "kit_objective_agrees": kit_agrees,
            },
            "comparison_to_A": {
                "same_route_as_A": route == a_record["canonical_solution"],
                "A_original_objective": float(a_record["independent_objective"]),
                "B_original_objective": original_objective,
                "objective_delta_B_minus_A": (
                    original_objective - float(a_record["independent_objective"])),
                "A_gap": float(a_record["gap_percent"]), "B_gap": b_gap,
                "A_runtime_seconds": float(a_record["runtime_seconds"]),
                "A_best_aug_idx": _best_index(a_record, "best_aug_idx"),
                "A_best_pomo_idx": _best_index(a_record, "best_pomo_idx"),
            },
            "runtime_seconds": runtime,
            "epsilon_diagnostics": epsilon_diagnostics,
            "mapping": mapping,
            "status": "PASS",
        }
        records.append(record)
        print(json.dumps({
            "problem_size": args.problem_size, "dataset_instance_index": index,
            "A_objective": record["comparison_to_A"]["A_original_objective"],
            "B_objective": original_objective, "same_route":
            record["comparison_to_A"]["same_route_as_A"], "status": "PASS",
        }, sort_keys=True), flush=True)

    summary = summarize(records)
    metadata = {
        "artifact_type": "MVMoE CVRPTW continuous-scaling A/B audit",
        "status": "SCALING_AUDIT_IMPLEMENTED_AND_RUN",
        "project": project, "upstream": upstream,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": checkpoint_hash,
                       "epoch": 5000, "problem": "Train_ALL", "strict_load": True},
        "dataset": {"path": str(args.dataset.resolve()),
                    "sha256": expected["dataset_sha256"], "indices": list(range(20))},
        "A_artifact_sources": a_sources,
        "B_scaling": {
            "formula": "s=max(max(original coordinates), original depot_tw_end/3.0)",
            "fields_divided": ["coordinates", "time_windows", "service_times"],
            "demand": "raw_demand/raw_capacity exactly once in adapt_batch",
            "loc_scaler": None, "distance_rounding": False,
        },
        "protocol": protocol, "environment": environment,
        "objective_tolerance": {"rtol": OBJECTIVE_RTOL, "atol": OBJECTIVE_ATOL},
        "official_source_provenance": [
            {"path": str((args.upstream / "envs/VRPTWEnv.py").resolve()),
             "sha256": sha256_file(args.upstream / "envs/VRPTWEnv.py")},
            {"path": str((args.upstream / "Tester.py").resolve()),
             "sha256": sha256_file(args.upstream / "Tester.py")},
        ],
        "source_provenance": source_provenance([
            Path(__file__), Path(__file__).with_name("scaling.py"),
            Path(__file__).with_name("adapter.py"), Path(__file__).with_name("decode.py"),
            ROOT / "methods/mvmoe/paper_runtime.py",
            ROOT / "methods/mvmoe/paper_config.py",
            ROOT / "problems/cvrptw/validate.py",
            ROOT / "problems/cvrp/objective.py",
            ROOT / "common/objective_agreement.py",
        ], root=ROOT),
    }
    _write_artifact(args.output_dir, metadata, records, summary)
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
