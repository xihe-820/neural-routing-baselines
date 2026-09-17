#!/usr/bin/env python3
"""UDC Stage S3 ML4CO TSP500/CVRP500 semantic adapter smoke."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from methods.udc.adapter import (adapt_cvrp_task, adapt_tsp_task,
                                 objective_comparison,
                                 validate_cvrp_population,
                                 validate_tsp_population)
from methods.udc.protocol import (ALPHA, DATASET_FILENAMES, OFFICIAL_COMMIT,
                                  SEED, SPECS, discover_datasets,
                                  official_gate, project_gate, s1_gate,
                                  s2_gate)

OFFICIAL_REL = Path("single_objective/UDC-Large-scale-CO-master/UDC")
FAMILY_DIRS = {"tsp": "TSP-AGNN-ICAM", "cvrp": "CVRP-AGNN-ICAM"}
CHECKPOINTS = {
    "tsp": {
        "conquering": ("checkpoint-tsp-460.pt", "724c11c75da5aa042fb36dc0d7594aaec541ae02cf1948996294f1c630f7c79f"),
        "partition": ("checkpoint-partition-460.pt", "83a012cd66a0b23237b4b17d1439bf1dace7cfc19abff62f8d478f55b00e523a"),
    },
    "cvrp": {
        "conquering": ("checkpoint-tsp-230.pt", "9edae6fcf5b0987d91ab39cec608464aa66d34ee6d92260a481d50a4f4831d0a"),
        "partition": ("checkpoint-partition-230.pt", "eadad8388f7e36607aabaa3e82124d8c7be63bb453b1f024eabb4d48309329b8"),
    },
}
MODEL_PARAMS = {
    "tsp": {"embedding_dim": 128, "sqrt_embedding_dim": 128**0.5,
            "encoder_layer_num": 6, "qkv_dim": 16, "head_num": 8,
            "logit_clipping": 10, "ff_hidden_dim": 512, "eval_type": "argmax"},
    "cvrp": {"embedding_dim": 128, "sqrt_embedding_dim": 128**0.5,
             "encoder_layer_num": 6, "qkv_dim": 16, "head_num": 8,
             "logit_clipping": 50, "ff_hidden_dim": 512, "eval_type": "argmax"},
}
TIMING = ("semantic_adapter_smoke_runtime_only: BS=1 wall clock; includes alpha=50 "
          "initial solution generation, all refinement stages and internal CPU/GPU "
          "operations; excludes model/checkpoint load, dataset parsing, adapter "
          "preprocessing, independent/Kit validation and artifact I/O; CUDA synchronized")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_jsonl(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def rng_digest(torch):
    state = np.random.get_state()
    np_bytes = state[1].tobytes() + repr((state[0], *state[2:])).encode()
    return {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            "numpy": hashlib.sha256(np_bytes).hexdigest(),
            "torch_cpu": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
            "torch_cuda": [hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
                           for value in torch.cuda.get_rng_state_all()]}


def torch_load(torch, path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_gate(supplemental_root, family):
    base = supplemental_root / FAMILY_DIRS[family]
    result = {}
    for role, (name, expected) in CHECKPOINTS[family].items():
        path = base / name
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{family} {role} checkpoint SHA256 mismatch")
        result[role] = {"path": str(path.resolve()), "sha256": actual}
    return result


def load_tasks(dataset, family, count):
    import ml4co_kit as kit
    wrapper = kit.TSPWrapper() if family == "tsp" else kit.CVRPWrapper()
    wrapper.from_pickle(dataset)
    if len(wrapper.task_list) < count:
        raise ValueError(f"{family} dataset contains fewer than {count} tasks")
    expected = kit.TSPTask if family == "tsp" else kit.CVRPTask
    tasks = wrapper.task_list[:count]
    if any(type(task) is not expected for task in tasks):
        raise ValueError(f"{family} dataset contains a non-exact ML4CO task class")
    return tasks, kit


def load_models(family, official_root, supplemental_root, torch):
    family_dir = official_root / OFFICIAL_REL / FAMILY_DIRS[family]
    sys.path[:0] = [str(family_dir), str(family_dir.parent), str(family_dir.parent.parent)]
    os.chdir(family_dir)
    if family == "tsp":
        Env = importlib.import_module("TSPEnv").TSPEnv
        Solver = importlib.import_module("TSPModel").TSPModel
        Tester = importlib.import_module("TSPTesterrrc").TSPTester
    else:
        Env = importlib.import_module("CVRPEnv").CVRPEnv
        Solver = importlib.import_module("CVRPModel").CVRPModel
        Tester = importlib.import_module("CVRPTester").CVRPPartitionTrainer
    Partition = importlib.import_module("PartitionModel").PartitionModel
    device = torch.device("cuda:0")
    partition = Partition(64, 2 if family == "tsp" else 3, 100, 2, depth=12).to(device)
    solver = Solver(**MODEL_PARAMS[family]).to(device)
    paths = checkpoint_gate(supplemental_root, family)
    for role, model in (("partition", partition), ("conquering", solver)):
        payload = torch_load(torch, Path(paths[role]["path"]), device)
        incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"{family} {role} strict checkpoint load mismatch")
    partition.eval(); solver.eval()
    env = Env(problem_size_low=500, problem_size_high=1000, sub_size=100,
              pomo_size=SPECS[family]["configured_pomo"],
              sample_size=8 if family == "tsp" else 2,
              optimal={100: 7.7632, 200: 10.7036, 500: 16.5215, 1000: 23.1199})
    harness = Tester.__new__(Tester)
    harness.env, harness.model_p, harness.model_t = env, partition, solver
    return env, harness, paths, device


def solve_tsp(task, env, harness, torch, device, dataset_index):
    points, semantics = adapt_tsp_task(task)
    coordinates = torch.as_tensor(points[None], dtype=torch.float32, device=device)
    harness.tester_params = {"test_episodes": 1, "test_batch_size": 1,
                             "aug_factor": ALPHA}
    harness.node_coords, harness.tours = coordinates, None
    before = rng_digest(torch)
    torch.cuda.synchronize(); started = time.perf_counter()
    with torch.inference_mode():
        solution = torch.stack(harness._load_init_sol(coordinates), dim=0)
        if list(solution.shape) != [1, ALPHA, 500]:
            raise ValueError(f"TSP alpha population shape mismatch: {list(solution.shape)}")
        stages = []
        for k in range(1, SPECS["tsp"]["x"] + 1):
            solution, candidate0, best = harness._test_one_batch(
                solution, batch_size=1, episode=0, k=k)
            stages.append({"stage": k, "candidate0": float(candidate0),
                           "official_best": float(best)})
    torch.cuda.synchronize(); runtime = time.perf_counter() - started
    population = solution[0].detach().cpu().numpy()
    independent = validate_tsp_population(points, population)
    with torch.inference_mode():
        official = env._get_travel_distance2(coordinates, solution)[0].detach().cpu().numpy()
    best = independent["best_alpha"]
    official_best = float(official[best])
    agreement = objective_comparison(official_best, independent["best_objective"])
    returned_agreement = objective_comparison(stages[-1]["official_best"], float(official.min()))
    kit_solution = np.asarray(independent["solution"], dtype=np.int64)
    kit_feasible = bool(task.check_constraints(kit_solution))
    kit_objective = float(task.evaluate(kit_solution))
    kit_agreement = objective_comparison(kit_objective, independent["best_objective"])
    reference = float(task.evaluate(task.ref_sol))
    passed = agreement["pass"] and returned_agreement["pass"] and kit_feasible and kit_agreement["pass"]
    return {
        "instance_id": str(task.name), "dataset_instance_index": dataset_index,
        "problem": "TSP", "size": 500, "seed": SEED, "alpha": ALPHA, "x": 2,
        "configured_pomo": 2, "effective_pomo": 2,
        "rng_before_solve": before, "rng_after_solve": rng_digest(torch),
        "input_semantics": semantics, "completed_stages": stages,
        "runtime_seconds": runtime, "timing_semantics": TIMING,
        "final_solution_population": population.astype(int).tolist(),
        "official_objective_per_alpha": official.astype(float).tolist(),
        "independent_objective_per_alpha": independent["independent_objective_per_alpha"],
        "best_alpha": best, "solution": independent["solution"],
        "udc_internal_objective": official_best,
        "udc_returned_best": stages[-1]["official_best"],
        "official_return_consistency": returned_agreement,
        "independent_feasible": True, "independent_objective": independent["best_objective"],
        "internal_vs_independent": agreement, "kit_feasible": kit_feasible,
        "kit_validation_status": "PASS" if kit_feasible else "FAIL",
        "kit_objective": kit_objective, "independent_vs_kit": kit_agreement,
        "reference_objective": reference,
        "instance_drop_percent": (independent["best_objective"] - reference) / reference * 100,
        "status": "KIT_VALIDATED" if passed else "FAILED"}


def solve_cvrp(task, env, harness, torch, device, dataset_index):
    coordinates_np, demand_np, semantics = adapt_cvrp_task(task)
    coordinates = torch.as_tensor(coordinates_np[None], dtype=torch.float32, device=device)
    demand = torch.as_tensor(demand_np[None], dtype=torch.float32, device=device)
    if env.pomo_size != 10:
        raise ValueError("CVRP configured POMO must start at 10")
    env.pomo_size = 1
    harness.trainer_params = {"validation_aug_factor": ALPHA,
                              "validation_test_episodes": 1,
                              "validation_test_batch_size": 1}
    before = rng_digest(torch)
    torch.cuda.synchronize(); started = time.perf_counter()
    with torch.inference_mode():
        solutions, flags = harness._load_init_sol(coordinates, demand)
        solution, solution_flag = torch.stack(solutions), torch.stack(flags)
        if list(solution.shape) != [1, ALPHA, 500] or solution_flag.shape != solution.shape:
            raise ValueError("CVRP alpha solution/flag population shape mismatch")
        stages = []
        for k in range(1, SPECS["cvrp"]["x"] + 1):
            solution, solution_flag = harness.route_ranking2(
                coordinates, solution, solution_flag)
            solution, solution_flag, candidate0, best = harness._test_one_batch(
                solution, solution_flag, coordinates, demand, 1, 0, k)
            stages.append({"stage": k, "candidate0": float(candidate0),
                           "official_best": float(best)})
    torch.cuda.synchronize(); runtime = time.perf_counter() - started
    population = solution[0].detach().cpu().numpy()
    flag_population = solution_flag[0].detach().cpu().numpy()
    depot = np.asarray(task.depots).reshape(-1, 2)[0]
    independent = validate_cvrp_population(
        depot, task.points, task.demands, task.capacity, population, flag_population)
    with torch.inference_mode():
        official = env.cal_length_total2(coordinates, solution, solution_flag)[0].detach().cpu().numpy()
    best = independent["best_alpha"]
    official_best = float(official[best])
    agreement = objective_comparison(official_best, independent["best_objective"])
    returned_agreement = objective_comparison(stages[-1]["official_best"], float(official.min()))
    kit_solution = np.asarray(independent["canonical_solution"], dtype=np.int64)
    kit_feasible = bool(task.check_constraints(kit_solution))
    kit_objective = float(task.evaluate(kit_solution))
    kit_agreement = objective_comparison(kit_objective, independent["best_objective"])
    reference = float(task.evaluate(task.ref_sol))
    passed = agreement["pass"] and returned_agreement["pass"] and kit_feasible and kit_agreement["pass"]
    return {
        "instance_id": str(task.name), "dataset_instance_index": dataset_index,
        "problem": "CVRP", "size": 500, "seed": SEED, "alpha": ALPHA, "x": 50,
        "configured_pomo": 10, "effective_pomo": 1,
        "rng_before_solve": before, "rng_after_solve": rng_digest(torch),
        "input_semantics": semantics, "completed_stages": stages,
        "runtime_seconds": runtime, "timing_semantics": TIMING,
        "final_solution_population": population.astype(int).tolist(),
        "final_solution_flag_population": flag_population.astype(int).tolist(),
        "official_objective_per_alpha": official.astype(float).tolist(),
        "independent_objective_per_alpha": independent["independent_objective_per_alpha"],
        "best_alpha": best, "solution": independent["solution"],
        "solution_flag": independent["solution_flag"],
        "canonical_solution": independent["canonical_solution"],
        "decoded_routes": independent["decoded_routes"],
        "route_demands": independent["route_demands"],
        "route_count": independent["route_count"],
        "max_route_load": independent["max_route_load"],
        "udc_internal_best": official_best, "independent_feasible": True,
        "udc_returned_best": stages[-1]["official_best"],
        "official_return_consistency": returned_agreement,
        "independent_objective": independent["best_objective"],
        "internal_vs_independent": agreement, "kit_feasible": kit_feasible,
        "kit_validation_status": "PASS" if kit_feasible else "FAIL",
        "kit_objective": kit_objective, "independent_vs_kit": kit_agreement,
        "reference_objective": reference,
        "instance_drop_percent": (independent["best_objective"] - reference) / reference * 100,
        "status": "KIT_VALIDATED" if passed else "FAILED"}


def worker(args):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("S3 requires exactly one visible CUDA GPU")
    if "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("S3 formal semantic smoke requires RTX 4090")
    tasks, kit = load_tasks(args.dataset, args.family, args.count)
    # Adapter audit/preprocessing is deliberately outside every solver timer.
    for task in tasks:
        adapt_tsp_task(task) if args.family == "tsp" else adapt_cvrp_task(task)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED); random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    initial_rng = rng_digest(torch)
    env, harness, checkpoints, device = load_models(
        args.family, args.official_root, args.supplemental_root, torch)
    records = []
    for index, task in enumerate(tasks):
        record = (solve_tsp(task, env, harness, torch, device, index)
                  if args.family == "tsp" else
                  solve_cvrp(task, env, harness, torch, device, index))
        record.update({"dataset_path": str(args.dataset.resolve()),
                       "dataset_sha256": sha256_file(args.dataset)})
        records.append(record)
        if record["status"] != "KIT_VALIDATED":
            raise RuntimeError(f"{args.family} instance {index} failed closed")
    records_path = args.output_dir / f"{args.family}500_records.jsonl"
    summary_path = args.output_dir / f"{args.family}500_summary.json"
    write_jsonl(records_path, records)
    summary = {
        "schema": "udc_s3_summary.v1", "family": args.family, "problem_size": 500,
        "count": args.count, "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset), "dataset_filename": args.dataset.name,
        "instance_order": list(range(args.count)),
        "instance_names": [str(task.name) for task in tasks],
        "reference_source": "ML4CO task.ref_sol evaluated by task.evaluate",
        "kit_module": kit.__file__, "seeded_once": True, "seed": SEED,
        "initial_rng_after_seed_before_model_construction": initial_rng,
        "continuous_rng_across_fixed_instance_order": True,
        "checkpoints": checkpoints, "protocol": {"alpha": ALPHA, **SPECS[args.family]},
        "timing_semantics": TIMING,
        "mean_independent_objective": float(np.mean([r["independent_objective"] for r in records])),
        "mean_instance_drop_percent": float(np.mean([r["instance_drop_percent"] for r in records])),
        "mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in records])),
        "validated_count": sum(r["status"] == "KIT_VALIDATED" for r in records),
        "status": "KIT_VALIDATED" if all(r["status"] == "KIT_VALIDATED" for r in records) else "FAILED",
        "records_path": str(records_path.resolve()), "records_sha256": sha256_file(records_path),
    }
    atomic_json(summary_path, summary)
    return 0


def prior_our2_gate(path, datasets, script_sha):
    metadata_path = path / "metadata.json"
    raw = json.loads(metadata_path.read_text())
    artifact_checks = {}
    for family in ("tsp", "cvrp"):
        summary_path = path / f"{family}500_summary.json"
        records_path = path / f"{family}500_records.jsonl"
        summary = json.loads(summary_path.read_text())
        artifact_checks[family] = {
            "summary_matches_metadata": summary == raw.get("summaries", {}).get(family),
            "summary_status": summary.get("status") == "KIT_VALIDATED",
            "validated_count": summary.get("validated_count") == 2,
            "records_sha256": summary.get("records_sha256") == sha256_file(records_path),
        }
    checks = {
        "state": raw.get("state") == "KIT_VALIDATED",
        "count": raw.get("count") == 2,
        "script": raw.get("script_sha256") == script_sha,
        "tsp_dataset": raw.get("datasets", {}).get("tsp", {}).get("sha256") == sha256_file(datasets["tsp"]),
        "cvrp_dataset": raw.get("datasets", {}).get("cvrp", {}).get("sha256") == sha256_file(datasets["cvrp"]),
        "artifacts": all(all(values.values()) for values in artifact_checks.values()),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"our_5 refused because prior our_2 gate failed: {failed}")
    return {"path": str(path.resolve()), "metadata_sha256": sha256_file(metadata_path),
            "checks": checks, "artifact_checks": artifact_checks, "pass": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--supplemental-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--s1-evidence", type=Path, required=True)
    parser.add_argument("--s2-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=(2, 5), required=True)
    parser.add_argument("--prior-our2", type=Path)
    parser.add_argument("--worker", choices=("tsp", "cvrp"))
    parser.add_argument("--family", choices=("tsp", "cvrp"))
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.official_root = args.official_root.resolve()
    args.supplemental_root = args.supplemental_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.worker:
        args.family = args.worker
        if args.dataset is None:
            parser.error("worker requires --dataset")
        return worker(args)
    if args.count == 5 and args.prior_our2 is None:
        parser.error("our_5 requires --prior-our2")
    if args.count == 2 and args.prior_our2 is not None:
        parser.error("our_2 must not specify --prior-our2")
    targets = [args.output_dir / name for name in
               ("metadata.json", "tsp500_records.jsonl", "tsp500_summary.json",
                "cvrp500_records.jsonl", "cvrp500_summary.json")]
    if any(path.exists() for path in targets):
        parser.error("refusing to overwrite an existing S3 artifact")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve(); script_sha = sha256_file(script)
    metadata = {"schema": "udc_s3_ml4co_adapter.v1", "started_at_utc": utc_now(),
                "count": args.count, "state": "PRECHECK", "script_path": str(script),
                "script_sha256": script_sha, "timing_semantics": TIMING,
                "protocol": {family: {"alpha": ALPHA, **spec} for family, spec in SPECS.items()},
                "blockers": []}
    try:
        datasets = discover_datasets(args.dataset_root.resolve())
        metadata["datasets"] = {family: {"path": str(path), "filename": path.name,
                                                 "sha256": sha256_file(path)}
                                for family, path in datasets.items()}
        metadata["project_pre"] = project_gate(args.project_root)
        metadata["official_pre"] = official_gate(args.official_root)
        metadata["s1"] = s1_gate(args.s1_evidence.resolve())
        metadata["s2"] = s2_gate(args.s2_evidence.resolve())
        metadata["checkpoints"] = {family: checkpoint_gate(args.supplemental_root, family)
                                   for family in ("tsp", "cvrp")}
        if args.count == 5:
            metadata["prior_our2"] = prior_our2_gate(
                args.prior_our2.resolve(), datasets, script_sha)
        preflight = {"project": metadata["project_pre"]["pass"],
                     "official": metadata["official_pre"]["pass"],
                     "s1": metadata["s1"]["pass"], "s2": metadata["s2"]["pass"],
                     "prior_our2": args.count == 2 or metadata["prior_our2"]["pass"]}
        metadata["preflight"] = preflight
        if not all(preflight.values()):
            raise RuntimeError(f"S3 preflight failed closed: {[k for k,v in preflight.items() if not v]}")
        for family in ("tsp", "cvrp"):
            command = [sys.executable, "-B", str(script), "--worker", family,
                       "--project-root", str(args.project_root),
                       "--official-root", str(args.official_root),
                       "--supplemental-root", str(args.supplemental_root),
                       "--dataset-root", str(args.dataset_root),
                       "--s1-evidence", str(args.s1_evidence),
                       "--s2-evidence", str(args.s2_evidence),
                       "--output-dir", str(args.output_dir), "--count", str(args.count),
                       "--dataset", str(datasets[family])]
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                raise RuntimeError(f"{family} S3 worker failed with {completed.returncode}")
        summaries = {family: json.loads(
            (args.output_dir / f"{family}500_summary.json").read_text())
            for family in ("tsp", "cvrp")}
        metadata["summaries"] = summaries
        metadata["project_post"] = project_gate(args.project_root)
        metadata["official_post"] = official_gate(args.official_root)
        final = {"tsp": summaries["tsp"]["status"] == "KIT_VALIDATED"
                        and summaries["tsp"]["validated_count"] == args.count,
                 "cvrp": summaries["cvrp"]["status"] == "KIT_VALIDATED"
                         and summaries["cvrp"]["validated_count"] == args.count,
                 "project_unchanged": metadata["project_post"]["pass"],
                 "official_unchanged": metadata["official_post"]["pass"]}
        metadata["final_gates"] = final
        metadata["state"] = "KIT_VALIDATED" if all(final.values()) else "FAILED"
        metadata["ready_for_stage_s4_scale_preflight"] = (
            "YES" if args.count == 5 and all(final.values()) else "NO")
    except Exception:
        metadata["state"] = "FAILED"
        metadata["fatal_traceback"] = traceback.format_exc()
        metadata["blockers"].append(metadata["fatal_traceback"].splitlines()[-1])
    metadata["finished_at_utc"] = utc_now()
    atomic_json(args.output_dir / "metadata.json", metadata)
    print(f"S3_STATE: {metadata['state']}")
    print("READY_FOR_STAGE_S4_SCALE_PREFLIGHT: " +
          metadata.get("ready_for_stage_s4_scale_preflight", "NO"))
    print(f"S3_METADATA={args.output_dir / 'metadata.json'}")
    return 0 if metadata["state"] == "KIT_VALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
