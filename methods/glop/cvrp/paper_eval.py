#!/usr/bin/env python3
"""Formal BS1 neural GLOP-G evaluation for frozen CVRP protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from common.provenance import environment_provenance, git_provenance, source_provenance
from methods.glop.cvrp.adapter import validate_instance
from methods.glop.cvrp.decode import decode_subtour_coordinates
from methods.glop.paper_protocol import (PARTITIONER_ASSETS, formal_protocol)
from methods.glop.paper_results import (RECORDS_FILE, TIMING_SEMANTICS,
                                        append_record, finalize_chunk,
                                        initialize_chunk, read_jsonl)
from methods.glop.runtime import (activate_upstream, completed_prefix_length,
                                  cuda_device, load_revisers,
                                  official_seeded_setup,
                                  random_insertion_identity,
                                  replay_completed_prefix,
                                  require_cvrp_dataset_prefix,
                                  run_warmup_isolated, verify_file,
                                  verify_upstream)
from problems.cvrp.validate import validate


def _solve_one(depot, points, demands, capacity, *, protocol, partitioner,
               revisers, device, torch, infer, Sampler, trans_tsp, sum_cost,
               reconnect, load_problem, random_insertion_parallel, timed):
    depot, points, demands, capacity = validate_instance(
        depot, points, demands, capacity, problem_size=protocol["problem_size"])
    if timed:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
    coordinates = torch.as_tensor(
        np.concatenate((depot[None], points), axis=0),
        dtype=torch.float32, device=device)
    native_demands = torch.as_tensor(
        np.concatenate((np.zeros(1, dtype=demands.dtype), demands)),
        dtype=torch.float32, device=device)
    heatmap = infer(
        partitioner, coordinates, native_demands, capacity,
        protocol["k_sparse"], False)
    sampler = Sampler(
        native_demands, heatmap, capacity, protocol["n_partition"], "cpu")
    partitions = sampler.gen_subsets(require_prob=False, greedy_mode=True)
    subtsps, n_tsps_per_route = trans_tsp(coordinates.cpu(), partitions)
    if (len(n_tsps_per_route) != protocol["n_partition"] or
            sum(n_tsps_per_route) != subtsps.shape[0]):
        raise ValueError("formal CVRP produced unexpected partition accounting")
    n_subtsps, max_len, _ = subtsps.shape
    if any(length > max_len for length in protocol["revision_lens"]):
        raise ValueError("official reviser is larger than materialized CVRP sub-TSP")
    order = torch.arange(max_len)
    permutations = random_insertion_parallel(subtsps, order)
    permutations = torch.as_tensor(permutations.astype(np.int64))
    seeds = subtsps.gather(
        1, permutations.unsqueeze(-1).repeat(1, 1, 2)).to(device)
    problem = load_problem("tsp")
    opts = SimpleNamespace(
        revision_lens=protocol["revision_lens"],
        revision_iters=protocol["revision_iters"],
        no_aug=not protocol["local_augmentation"],
        no_prune=not protocol["pruning"], eval_batch_size=n_subtsps)
    with torch.no_grad():
        tours, costs = reconnect(
            get_cost_func=lambda data, route: problem.get_costs(
                data, route, return_local=True),
            batch=seeds, opts=opts, revisers=revisers)
    totals = sum_cost(costs, n_tsps_per_route)
    reported, best_partition = totals.min(dim=0)
    best_partition = int(best_partition)
    if best_partition != 0:
        raise ValueError("formal CVRP selected an impossible partition index")
    subtour_start = sum(n_tsps_per_route[:best_partition])
    subtour_count = n_tsps_per_route[best_partition]
    selected = tours[subtour_start:subtour_start + subtour_count]
    preflatten = selected.detach().cpu().numpy()
    canonical, routes = decode_subtour_coordinates(preflatten, depot, points)
    reported = float(reported.detach().cpu())
    if timed:
        torch.cuda.synchronize(device)
        runtime = time.perf_counter() - started
    else:
        runtime = None
    return canonical, routes, reported, runtime, {
        "best_partition_idx": best_partition,
        "subtour_count": int(subtour_count),
        "padded_subtour_length": int(max_len),
        "preflatten_subtour_shape": list(preflatten.shape),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-metadata", type=Path)
    parser.add_argument("--problem-size", type=int, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-instances", type=int, choices=range(0, 6), default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = formal_protocol("CVRP", args.problem_size, args.protocol)
    prepared_path = args.input_metadata or args.input.with_suffix(args.input.suffix + ".json")
    prepared = json.loads(prepared_path.read_text())
    input_hash = sha256_file(args.input)
    if (prepared.get("format") != "glop-paper-cvrp-input-v1" or
            prepared.get("problem_size") != args.problem_size or
            prepared.get("official_protocol_name") != args.protocol or
            prepared.get("expected_dataset_filename") !=
            protocol["expected_dataset_filename"] or
            prepared.get("input_npz_sha256") != input_hash):
        raise ValueError("prepared CVRP input identity/protocol mismatch")
    dataset_path = Path(prepared["dataset_path"])
    if (dataset_path.name != protocol["expected_dataset_filename"] or
            not dataset_path.is_file() or sha256_file(dataset_path) !=
            prepared["dataset_sha256"]):
        raise ValueError("prepared CVRP source dataset is missing or changed")
    upstream = verify_upstream(args.upstream)
    project = git_provenance(ROOT)
    if project["dirty"]:
        raise ValueError("formal GLOP evaluation requires a clean project checkout")

    import torch
    device = cuda_device(args.device, torch)
    with np.load(args.input, allow_pickle=False) as data:
        arrays = {key: data[key] for key in (
            "depots", "points", "raw_demands", "raw_capacities",
            "dataset_indices", "reference_objectives")}
    indices = require_cvrp_dataset_prefix(arrays["dataset_indices"])
    if (prepared.get("dataset_indices") != indices or
            len(arrays["points"]) != len(indices)):
        raise ValueError("prepared CVRP prefix metadata is inconsistent")
    for local in range(len(indices)):
        validate_instance(
            arrays["depots"][local], arrays["points"][local],
            arrays["raw_demands"][local], arrays["raw_capacities"][local],
            problem_size=args.problem_size)

    activate_upstream(args.upstream)
    from heatmap.cvrp.infer import infer, load_partitioner
    from heatmap.cvrp.inst import sum_cost, trans_tsp
    from heatmap.cvrp.sampler import Sampler
    from utils.functions import load_model, load_problem, reconnect
    from utils.insertion import random_insertion_parallel
    insertion = random_insertion_identity()
    partition_spec = PARTITIONER_ASSETS[protocol["partitioner_source_size"]]
    partition_path = args.asset_root / partition_spec["path"]
    partition_hash = verify_file(partition_path, partition_spec)

    def setup_models():
        revisers, reviser_assets = load_revisers(
            args.asset_root, protocol, device=device, torch=torch,
            load_model=load_model)
        partitioner = load_partitioner(
            protocol["partitioner_source_size"], device,
            str(partition_path.resolve()),
            protocol["k_sparse"], protocol["partitioner_depth"])
        payload = torch.load(partition_path, map_location="cpu")
        state = payload.get("model_state_dict", payload)
        strict = partitioner.load_state_dict(state, strict=True)
        if strict.missing_keys or strict.unexpected_keys:
            raise ValueError("partitioner strict load mismatch")
        if (sum(parameter.numel() for parameter in partitioner.parameters()) !=
                partition_spec["parameter_count"]):
            raise ValueError("partitioner parameter count mismatch")
        partitioner.eval()
        return revisers, reviser_assets, partitioner

    revisers, reviser_assets, partitioner = official_seeded_setup(
        torch, protocol["seed"], setup_models)

    environment = environment_provenance(device)
    environment["random_insertion"] = insertion
    sources = source_provenance([
        Path(__file__), Path(__file__).with_name("prepare_instances.py"),
        Path(__file__).with_name("adapter.py"), Path(__file__).with_name("decode.py"),
        ROOT / "methods/glop/paper_protocol.py", ROOT / "methods/glop/runtime.py",
        ROOT / "methods/glop/paper_results.py", ROOT / "problems/cvrp/validate.py",
        ROOT / "problems/cvrp/objective.py", ROOT / "common/objective_agreement.py",
        ROOT / "common/hashing.py", ROOT / "common/provenance.py",
    ], root=ROOT)
    identity = {
        "method": "GLOP", "variant": args.protocol, "problem": "CVRP",
        "problem_size": args.problem_size, "official_protocol_name": args.protocol,
        "paper_protocol": protocol, "project": project, "upstream": upstream,
        "assets": {
            "partitioner": {"path": str(partition_path.resolve()),
                            "partitioner_source_size":
                            protocol["partitioner_source_size"],
                            "sha256": partition_hash,
                            "size_bytes": partition_path.stat().st_size,
                            "strict_load": True,
                            "parameter_count": partition_spec["parameter_count"]},
            "revisers": reviser_assets},
        "dataset": {"path": str(dataset_path.resolve()),
                    "sha256": prepared["dataset_sha256"],
                    "size_bytes": prepared["dataset_size_bytes"],
                    "count": prepared["dataset_count"],
                    "task_class": prepared["dataset_task_class"],
                    "coordinate_shape": prepared["coordinate_shape"],
                    "coordinate_range": [prepared["coordinate_min"],
                                         prepared["coordinate_max"]],
                    "demand_range": [prepared["demand_min"], prepared["demand_max"]],
                    "capacity_values": prepared["capacity_values"],
                    "reference_source": prepared["reference_source"]},
        "prepared_input": {
            "path": str(args.input.resolve()), "sha256": input_hash,
            "metadata_path": str(prepared_path.resolve()),
            "metadata_sha256": sha256_file(prepared_path),
        },
        "chunk": {"offset": indices[0], "count": len(indices),
                  "expected_indices": indices},
        "warmup": {"instances": args.warmup_instances,
                   "policy": "first prepared instances rerun formally; excluded from time"},
        "rng": dict(protocol["rng_semantics"]),
        "environment": environment, "source_provenance": sources,
        "timing_semantics": TIMING_SEMANTICS,
    }
    _, completed = initialize_chunk(args.output_dir, identity)
    records_path = Path(args.output_dir) / RECORDS_FILE
    existing_records = read_jsonl(records_path) if records_path.exists() else []
    prefix_length = completed_prefix_length(existing_records, indices)
    if completed != set(indices[:prefix_length]):
        raise ValueError("CVRP resume metadata and prefix records disagree")
    if completed == set(indices):
        finalize_chunk(args.output_dir)
        return
    solve_kwargs = dict(
        protocol=protocol, partitioner=partitioner, revisers=revisers,
        device=device, torch=torch, infer=infer, Sampler=Sampler,
        trans_tsp=trans_tsp, sum_cost=sum_cost, reconnect=reconnect,
        load_problem=load_problem,
        random_insertion_parallel=random_insertion_parallel)
    def solve(local, *, timed):
        canonical, routes, reported, runtime, selection = _solve_one(
            arrays["depots"][local], arrays["points"][local],
            arrays["raw_demands"][local], arrays["raw_capacities"][local],
            timed=timed, **solve_kwargs)
        checked = validate(
            arrays["depots"][local], arrays["points"][local],
            arrays["raw_demands"][local], arrays["raw_capacities"][local],
            canonical)
        independent = checked["independent_objective"]
        agrees = independent is not None and objective_agrees(reported, independent)
        if not checked["feasible"] or not agrees:
            raise RuntimeError(
                f"CVRP independent gate failed at index {indices[local]}")
        return canonical, routes, reported, runtime, selection, checked, independent

    def warmup():
        for local in range(min(args.warmup_instances, len(indices))):
            solve(local, timed=False)

    run_warmup_isolated(torch, device, warmup)

    def replay(dataset_index):
        local = dataset_index
        canonical, routes, reported, _, selection, _, independent = solve(
            local, timed=False)
        return {
            "canonical_solution": canonical, "canonical_routes": routes,
            "reported_objective": reported,
            "independent_objective": independent, "selection": selection,
        }

    prefix_length = replay_completed_prefix(existing_records, indices, replay)
    for local in range(prefix_length, len(indices)):
        dataset_index = indices[local]
        (canonical, routes, reported, runtime, selection, checked,
         independent) = solve(local, timed=True)
        agrees = True
        reference = float(arrays["reference_objectives"][local])
        record = {
            "dataset_instance_index": dataset_index,
            "instance_id": prepared["instance_names"][local],
            "canonical_solution": canonical, "canonical_routes": routes,
            "reported_objective": reported,
            "independent_objective": independent, "reference_objective": reference,
            "gap_percent": (independent - reference) / reference * 100.0
            if independent is not None else None,
            "runtime_seconds": runtime,
            "independent_feasible": bool(checked["feasible"]),
            "reported_objective_agrees": bool(agrees),
            "selection": selection,
            "constraint_details": checked["constraint_details"],
            "evidence_status": "INDEPENDENT_VERIFIED",
        }
        append_record(args.output_dir, record)
        print(json.dumps({"method": "GLOP", "problem": "CVRP",
                          "size": args.problem_size, "protocol": args.protocol,
                          "dataset_instance_index": dataset_index,
                          "objective": independent, "reference": reference,
                          "gap_percent": record["gap_percent"],
                          "runtime_seconds": runtime, "feasible": True},
                         sort_keys=True, allow_nan=False), flush=True)
    finalize_chunk(args.output_dir)


if __name__ == "__main__":
    main()
