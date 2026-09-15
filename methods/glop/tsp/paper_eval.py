#!/usr/bin/env python3
"""Formal BS1 evaluation for enabled official GLOP TSP protocols."""
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
from methods.glop.paper_protocol import formal_protocol
from methods.glop.paper_results import (TIMING_SEMANTICS, append_record,
                                        finalize_chunk, initialize_chunk)
from methods.glop.runtime import (activate_upstream, cuda_device, load_revisers,
                                  random_insertion_identity, verify_upstream)
from methods.glop.tsp.adapter import adapt_points, validate_initial_permutations
from methods.glop.tsp.decode import decode_coordinate_tour
from problems.tsp.validate import validate


def _solve_one(points, *, protocol, revisers, device, torch, reconnect,
               load_problem, random_insertion_parallel, timed):
    width = protocol["internal_width"]
    torch.manual_seed(protocol["seed"])
    if timed:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
    orders = [torch.randperm(len(points)) for _ in range(width)]
    batched = torch.as_tensor(points[None], dtype=torch.float32)
    permutations = np.asarray(
        [random_insertion_parallel(batched, order) for order in orders],
        dtype=np.int64).reshape(width, 1, len(points))
    validate_initial_permutations(
        permutations, problem_size=len(points), width=width, batch_size=1)
    repeated = batched.repeat(width, 1, 1)
    pi = torch.as_tensor(permutations.reshape(width, len(points)), dtype=torch.long)
    seeds = repeated.gather(1, pi.unsqueeze(-1).repeat(1, 1, 2)).to(device)
    problem = load_problem("tsp")
    opts = SimpleNamespace(
        revision_lens=protocol["revision_lens"],
        revision_iters=protocol["revision_iters"],
        no_aug=not protocol["local_augmentation"],
        no_prune=not protocol["pruning"], eval_batch_size=1)
    with torch.no_grad():
        tours, costs = reconnect(
            get_cost_func=lambda data, route: problem.get_costs(
                data, route, return_local=True),
            batch=seeds, opts=opts, revisers=revisers)
    if tuple(tours.shape) != (1, len(points), 2) or tuple(costs.shape) != (1,):
        raise ValueError("official GLOP TSP output shape mismatch")
    coordinates = tours[0].detach().cpu().numpy()
    reported = float(costs[0].detach().cpu())
    canonical, decoding = decode_coordinate_tour(
        coordinates, points, allowed_transforms=("identity",))
    if timed:
        torch.cuda.synchronize(device)
        runtime = time.perf_counter() - started
    else:
        runtime = None
    return canonical, reported, decoding, runtime


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
    protocol = formal_protocol("TSP", args.problem_size, args.protocol)
    prepared_path = args.input_metadata or args.input.with_suffix(args.input.suffix + ".json")
    prepared = json.loads(prepared_path.read_text())
    input_hash = sha256_file(args.input)
    if (prepared.get("format") != "glop-paper-tsp-input-v1" or
            prepared.get("problem_size") != args.problem_size or
            prepared.get("official_protocol_name") != args.protocol or
            prepared.get("input_npz_sha256") != input_hash):
        raise ValueError("prepared TSP input identity/protocol mismatch")
    dataset_path = Path(prepared["dataset_path"])
    if (not dataset_path.is_file() or sha256_file(dataset_path) !=
            prepared["dataset_sha256"]):
        raise ValueError("prepared TSP source dataset is missing or changed")
    upstream = verify_upstream(args.upstream)
    project = git_provenance(ROOT)
    if project["dirty"]:
        raise ValueError("formal GLOP evaluation requires a clean project checkout")

    import torch
    device = cuda_device(args.device, torch)
    with np.load(args.input, allow_pickle=False) as data:
        points = data["points"]
        indices = [int(value) for value in data["dataset_indices"]]
        references = data["reference_objectives"]
    if (not indices or indices != list(range(indices[0], indices[0] + len(indices))) or
            prepared.get("dataset_indices") != indices or len(points) != len(indices)):
        raise ValueError("prepared TSP chunk must be nonempty, contiguous, and consistent")
    adapt_points(points, problem_size=args.problem_size,
                 top_level_transforms=("identity",), device="cpu")

    activate_upstream(args.upstream)
    from utils.functions import load_model, load_problem, reconnect
    from utils.insertion import random_insertion_parallel
    insertion = random_insertion_identity()
    revisers, assets = load_revisers(
        args.asset_root, protocol, device=device, torch=torch, load_model=load_model)
    environment = environment_provenance(device)
    environment["random_insertion"] = insertion
    sources = source_provenance([
        Path(__file__), Path(__file__).with_name("prepare_instances.py"),
        Path(__file__).with_name("adapter.py"), Path(__file__).with_name("decode.py"),
        ROOT / "methods/glop/paper_protocol.py", ROOT / "methods/glop/runtime.py",
        ROOT / "methods/glop/paper_results.py", ROOT / "problems/tsp/validate.py",
        ROOT / "problems/tsp/objective.py", ROOT / "common/objective_agreement.py",
        ROOT / "common/hashing.py", ROOT / "common/provenance.py",
    ], root=ROOT)
    identity = {
        "method": "GLOP", "variant": args.protocol, "problem": "TSP",
        "problem_size": args.problem_size, "official_protocol_name": args.protocol,
        "paper_protocol": protocol, "project": project, "upstream": upstream,
        "assets": {"revisers": assets},
        "dataset": {"path": str(dataset_path.resolve()),
                    "sha256": prepared["dataset_sha256"],
                    "size_bytes": prepared["dataset_size_bytes"],
                    "count": prepared["dataset_count"],
                    "task_class": prepared["dataset_task_class"],
                    "coordinate_shape": prepared["coordinate_shape"],
                    "coordinate_range": [prepared["coordinate_min"],
                                         prepared["coordinate_max"]],
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
        "environment": environment, "source_provenance": sources,
        "timing_semantics": TIMING_SEMANTICS,
    }
    _, completed = initialize_chunk(args.output_dir, identity)
    if completed == set(indices):
        finalize_chunk(args.output_dir)
        return
    for local in range(min(args.warmup_instances, len(points))):
        _solve_one(points[local], protocol=protocol, revisers=revisers,
                   device=device, torch=torch, reconnect=reconnect,
                   load_problem=load_problem,
                   random_insertion_parallel=random_insertion_parallel, timed=False)
    for local, dataset_index in enumerate(indices):
        if dataset_index in completed:
            continue
        canonical, reported, decoding, runtime = _solve_one(
            points[local], protocol=protocol, revisers=revisers, device=device,
            torch=torch, reconnect=reconnect, load_problem=load_problem,
            random_insertion_parallel=random_insertion_parallel, timed=True)
        checked = validate(points[local], canonical)
        independent = checked["independent_objective"]
        agrees = independent is not None and objective_agrees(reported, independent)
        reference = float(references[local])
        passed = checked["feasible"] and agrees
        record = {
            "dataset_instance_index": dataset_index,
            "instance_id": prepared["instance_names"][local],
            "canonical_solution": canonical, "reported_objective": reported,
            "independent_objective": independent, "reference_objective": reference,
            "gap_percent": (independent - reference) / reference * 100.0
            if independent is not None else None,
            "runtime_seconds": runtime,
            "independent_feasible": bool(checked["feasible"]),
            "reported_objective_agrees": bool(agrees),
            "selection": {"internal_width": protocol["internal_width"],
                          "decoding": decoding},
            "constraint_details": checked["constraint_details"],
            "evidence_status": "INDEPENDENT_VERIFIED" if passed else "FAILED",
        }
        if not passed:
            raise RuntimeError(f"TSP independent gate failed at index {dataset_index}")
        append_record(args.output_dir, record)
        print(json.dumps({"method": "GLOP", "problem": "TSP",
                          "size": args.problem_size, "protocol": args.protocol,
                          "dataset_instance_index": dataset_index,
                          "objective": independent, "reference": reference,
                          "gap_percent": record["gap_percent"],
                          "runtime_seconds": runtime, "feasible": True},
                         sort_keys=True, allow_nan=False), flush=True)
    finalize_chunk(args.output_dir)


if __name__ == "__main__":
    main()
