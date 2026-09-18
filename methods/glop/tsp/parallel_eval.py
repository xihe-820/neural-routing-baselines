#!/usr/bin/env python3
"""Native-batch GLOP TSP evaluator for the appendix Parallel Table."""
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
from methods.glop.paper_protocol import REVISER_ASSETS
from methods.glop.paper_results import fingerprint, utc_now, write_json
from methods.glop.runtime import (activate_upstream, cuda_device, load_revisers,
                                  make_shared_tsp_orders, official_seeded_setup,
                                  random_insertion_identity, run_warmup_isolated,
                                  verify_file, verify_upstream)
from methods.glop.tsp.adapter import adapt_points, validate_initial_permutations
from methods.glop.tsp.decode import decode_coordinate_tour
from methods.glop.tsp.parallel_results import (
    ALLOWED_BATCH_SIZES, BATCH_TIMINGS_FILE, METADATA_FILE, SCHEMA_VERSION,
    SUMMARY_FILE, TIMING_SEMANTICS, VALIDATED_RECORDS_FILE, exact_batch_ranges,
    finalize_parallel, parallel_scope)
from problems.tsp.validate import validate


def _reviser_sha_identities(revisers):
    try:
        return [{
            "reviser_size": int(row["reviser_size"]),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "args_sha256": row["args_sha256"],
        } for row in revisers]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BS1 summary has invalid reviser SHA identity") from exc


def _audit_reviser_sha_identities(asset_root, protocol):
    identities = []
    for size in protocol["revision_lens"]:
        spec = REVISER_ASSETS[size]
        checkpoint = Path(asset_root) / spec["path"]
        args_path = Path(asset_root) / spec["args_path"]
        identities.append({
            "reviser_size": size,
            "checkpoint_sha256": verify_file(checkpoint, spec),
            "args_sha256": verify_file(
                args_path, spec, sha_key="args_sha256",
                size_key="args_size_bytes"),
        })
    return identities


def _verify_bs1_summary(path, *, problem_size, protocol_name, dataset_path,
                        dataset_sha256, dataset_count, upstream_commit,
                        reviser_identities):
    """Bind a parallel cell to its exact frozen BS1 PAPER_READY evidence."""
    summary_path = Path(path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    try:
        summary = json.loads(summary_path.read_text())
        identity = summary["consistency_identity"]
        dataset = identity["dataset"]
        upstream = identity["upstream"]
        assets = identity["assets"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("BS1 summary lacks formal consistency identity") from exc
    expected_size = int(problem_size)
    expected_count = int(dataset_count)
    if summary.get("status") != "PAPER_READY":
        raise ValueError("BS1 summary is not PAPER_READY")
    if (summary.get("method") != "GLOP" or summary.get("problem") != "TSP" or
            identity.get("method") != "GLOP" or identity.get("problem") != "TSP"):
        raise ValueError("BS1 summary method/problem identity mismatch")
    if (summary.get("problem_size") != expected_size or
            identity.get("problem_size") != expected_size):
        raise ValueError("BS1 summary problem size mismatch")
    if (summary.get("variant") != protocol_name or
            identity.get("variant") != protocol_name or
            identity.get("official_protocol_name") != protocol_name):
        raise ValueError("BS1 summary protocol mismatch")
    if (Path(dataset.get("path", "")).name != Path(dataset_path).name or
            dataset.get("sha256") != dataset_sha256 or
            dataset.get("count") != expected_count or
            summary.get("instance_count") != expected_count):
        raise ValueError("BS1 summary dataset identity mismatch")
    if (upstream.get("commit") != upstream_commit or
            upstream.get("dirty") is not False):
        raise ValueError("BS1 summary official GLOP identity mismatch")
    observed_revisers = _reviser_sha_identities(assets.get("revisers"))
    if observed_revisers != reviser_identities:
        raise ValueError("BS1 summary reviser SHA identity mismatch")
    return {
        "path": str(summary_path.resolve()),
        "sha256": sha256_file(summary_path),
        "identity": {
            "status": summary["status"], "method": summary["method"],
            "problem": summary["problem"], "problem_size": expected_size,
            "variant": summary["variant"],
            "official_protocol_name": identity["official_protocol_name"],
            "dataset_filename": Path(dataset["path"]).name,
            "dataset_sha256": dataset["sha256"],
            "dataset_count": dataset["count"],
            "official_glop_commit": upstream["commit"],
            "revisers": observed_revisers,
        },
    }


def _build_candidate_batch(batched, permutations, *, protocol, device, torch):
    """Preserve official candidate-major then original-instance layout."""
    batch_size, problem_size, coordinate_dim = batched.shape
    width = protocol["ri_order_width"]
    values = np.asarray(permutations, dtype=np.int64)
    validate_initial_permutations(
        values, problem_size=problem_size, width=width,
        batch_size=batch_size)
    pi = torch.as_tensor(values, dtype=torch.long)
    repeated = batched.unsqueeze(0).expand(width, -1, -1, -1)
    seeds = repeated.gather(
        2, pi.unsqueeze(-1).expand(width, batch_size, problem_size,
                                   coordinate_dim)).to(device)
    reflected = []
    for transform in protocol["top_level_transforms"]:
        candidate = seeds.clone()
        if transform in ("reflect_x", "reflect_xy"):
            candidate[..., 0] = 1 - candidate[..., 0]
        if transform in ("reflect_y", "reflect_xy"):
            candidate[..., 1] = 1 - candidate[..., 1]
        if transform not in ("identity", "reflect_x", "reflect_y", "reflect_xy"):
            raise ValueError(f"unknown GLOP top-level transform: {transform}")
        reflected.append(candidate)
    candidates = torch.cat(reflected, dim=0)
    expected = protocol["effective_candidate_count"]
    if tuple(candidates.shape) != (expected, batch_size, problem_size, 2):
        raise ValueError("parallel GLOP candidate x batch layout mismatch")
    return candidates.reshape(expected * batch_size, problem_size, 2)


def _solve_batch(points, *, protocol, orders, revisers, device, torch,
                 reconnect, load_problem, random_insertion_parallel, timed):
    batch_size, problem_size, _ = points.shape
    width = protocol["ri_order_width"]
    if len(orders) != width:
        raise ValueError("shared TSP RI order count differs from ri_order_width")
    if timed:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
    batched = torch.as_tensor(points, dtype=torch.float32)
    permutations = np.asarray([
        random_insertion_parallel(batched, order.clone()) for order in orders
    ], dtype=np.int64)
    seeds = _build_candidate_batch(
        batched, permutations, protocol=protocol, device=device, torch=torch)
    problem = load_problem("tsp")
    opts = SimpleNamespace(
        revision_lens=protocol["revision_lens"],
        revision_iters=protocol["revision_iters"],
        no_aug=not protocol["local_reconnect_augmentation"],
        no_prune=not protocol["pruning"], eval_batch_size=batch_size)
    with torch.no_grad():
        tours, costs = reconnect(
            get_cost_func=lambda data, route: problem.get_costs(
                data, route, return_local=True),
            batch=seeds, opts=opts, revisers=revisers)
    if (tuple(tours.shape) != (batch_size, problem_size, 2) or
            tuple(costs.shape) != (batch_size,)):
        raise ValueError("official GLOP parallel TSP output shape mismatch")
    coordinates = tours.detach().cpu().numpy()
    reported = costs.detach().cpu().numpy().astype(np.float64, copy=False)
    canonical, decoding = [], []
    for local in range(batch_size):
        route, details = decode_coordinate_tour(
            coordinates[local], points[local],
            allowed_transforms=protocol["top_level_transforms"])
        canonical.append(route)
        decoding.append(details)
    if timed:
        torch.cuda.synchronize(device)
        runtime = time.perf_counter() - started
    else:
        runtime = None
    return canonical, reported.tolist(), decoding, runtime


def _selection(protocol, decoding):
    return {
        "paper_nominal_width": protocol["paper_nominal_width"],
        "ri_order_width": protocol["ri_order_width"],
        "top_level_reflection_factor": protocol["top_level_reflection_factor"],
        "effective_candidate_count": protocol["effective_candidate_count"],
        "candidate_batch_layout": "candidate-major [candidate,batch,node,xy]",
        "decoding": decoding,
    }


def _load_and_verify_input(args, scope):
    protocol = scope["paper_protocol"]
    metadata_path = (args.input_metadata or
                     args.input.with_suffix(args.input.suffix + ".json"))
    prepared = json.loads(metadata_path.read_text())
    input_hash = sha256_file(args.input)
    if (prepared.get("format") != "glop-paper-tsp-input-v1" or
            prepared.get("problem_size") != args.problem_size or
            prepared.get("official_protocol_name") != args.protocol or
            prepared.get("expected_dataset_filename") !=
            protocol["expected_dataset_filename"] or
            prepared.get("input_npz_sha256") != input_hash or
            prepared.get("dataset_count") != scope["instance_count"]):
        raise ValueError("prepared parallel TSP input identity/protocol mismatch")
    dataset_path = Path(prepared["dataset_path"])
    if (dataset_path.name != protocol["expected_dataset_filename"] or
            not dataset_path.is_file() or
            sha256_file(dataset_path) != prepared["dataset_sha256"]):
        raise ValueError("prepared parallel TSP source dataset is missing or changed")
    with np.load(args.input, allow_pickle=False) as data:
        points = data["points"]
        indices = [int(value) for value in data["dataset_indices"]]
        references = np.asarray(data["reference_objectives"], dtype=np.float64)
    expected_indices = list(range(scope["instance_count"]))
    if (indices != expected_indices or prepared.get("dataset_indices") != indices or
            len(points) != scope["instance_count"] or
            len(references) != scope["instance_count"]):
        raise ValueError("parallel evaluator requires the exact complete ordered dataset")
    adapt_points(points, problem_size=args.problem_size,
                 top_level_transforms=protocol["top_level_transforms"],
                 device="cpu")
    return prepared, metadata_path, dataset_path, input_hash, points, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-metadata", type=Path)
    parser.add_argument("--problem-size", type=int, choices=[100, 500, 1000],
                        required=True)
    parser.add_argument("--protocol", choices=["official_standard", "official_more"],
                        required=True)
    parser.add_argument("--batch-size", type=int, choices=ALLOWED_BATCH_SIZES,
                        required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--bs1-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-batches", type=int, choices=[0, 1], default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            "parallel output directory already exists; refusing overwrite/resume")
    scope = parallel_scope(args.problem_size, args.protocol, args.batch_size)
    protocol = scope["paper_protocol"]
    (prepared, prepared_path, dataset_path, input_hash,
     points, references) = _load_and_verify_input(args, scope)
    project = git_provenance(ROOT)
    upstream = verify_upstream(args.upstream)
    if project["dirty"]:
        raise ValueError("formal GLOP parallel evaluation requires a clean project checkout")
    current_reviser_identities = _audit_reviser_sha_identities(
        args.asset_root, protocol)
    bs1_reference = _verify_bs1_summary(
        args.bs1_summary, problem_size=args.problem_size,
        protocol_name=args.protocol, dataset_path=dataset_path,
        dataset_sha256=prepared["dataset_sha256"],
        dataset_count=prepared["dataset_count"],
        upstream_commit=upstream["commit"],
        reviser_identities=current_reviser_identities)

    import torch
    device = cuda_device(args.device, torch)
    environment = environment_provenance(device)
    if "RTX 4090" not in str(environment["gpu"]):
        raise ValueError("formal GLOP Parallel Table requires NVIDIA RTX 4090")

    import ml4co_kit as kit
    wrapper = kit.TSPWrapper()
    wrapper.from_pickle(dataset_path)
    if len(wrapper.task_list) != scope["instance_count"]:
        raise ValueError("ML4CO-Kit dataset count differs from parallel scope")
    for index, task in enumerate(wrapper.task_list):
        if type(task) is not kit.TSPTask or task.points.shape != (
                args.problem_size, 2):
            raise ValueError("parallel source task is not the exact ML4CO TSPTask")
        if not np.array_equal(np.asarray(task.points, dtype=np.float32), points[index]):
            raise ValueError("prepared coordinates differ from ML4CO source task")
        reference = float(task.evaluate(task.ref_sol))
        if not objective_agrees(reference, float(references[index])):
            raise ValueError("prepared reference differs from ML4CO task reference")

    activate_upstream(args.upstream)
    from utils.functions import load_model, load_problem, reconnect
    from utils.insertion import random_insertion_parallel
    insertion = random_insertion_identity()
    revisers, assets = official_seeded_setup(
        torch, protocol["seed"],
        lambda: load_revisers(
            args.asset_root, protocol, device=device, torch=torch,
            load_model=load_model))
    if _reviser_sha_identities(assets) != current_reviser_identities:
        raise ValueError("loaded reviser identity changed after BS1 provenance gate")
    shared_started = time.perf_counter()
    orders = make_shared_tsp_orders(
        torch, problem_size=args.problem_size,
        width=protocol["ri_order_width"])
    shared_setup_seconds = time.perf_counter() - shared_started
    environment["random_insertion"] = insertion
    sources = source_provenance([
        Path(__file__), Path(__file__).with_name("parallel_results.py"),
        Path(__file__).with_name("prepare_instances.py"),
        Path(__file__).with_name("adapter.py"),
        Path(__file__).with_name("decode.py"),
        ROOT / "methods/glop/paper_protocol.py", ROOT / "methods/glop/runtime.py",
        ROOT / "problems/tsp/validate.py", ROOT / "problems/tsp/objective.py",
        ROOT / "common/objective_agreement.py", ROOT / "common/hashing.py",
        ROOT / "common/provenance.py",
    ], root=ROOT)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "GLOP TSP Parallel Table full-set evaluation",
        "state": "INFERENCE_IN_PROGRESS", "created_at": utc_now(),
        "method": "GLOP", "problem": "TSP", "problem_size": args.problem_size,
        "protocol": args.protocol, "parallel_scope": scope,
        "project": project, "upstream": upstream,
        "bs1_paper_ready_reference": bs1_reference,
        "assets": {"revisers": assets},
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": prepared["dataset_sha256"],
            "size_bytes": prepared["dataset_size_bytes"],
            "count": prepared["dataset_count"],
            "task_class": prepared["dataset_task_class"],
            "reference_source": prepared["reference_source"],
        },
        "prepared_input": {
            "path": str(args.input.resolve()), "sha256": input_hash,
            "metadata_path": str(prepared_path.resolve()),
            "metadata_sha256": sha256_file(prepared_path),
        },
        "rng": {
            **protocol["rng_semantics"],
            "seed": protocol["seed"],
            "parallel_batch_stream": "continuous; no per-batch reseed",
            "shared_ri_orders_fingerprint": fingerprint(
                [order.tolist() for order in orders]),
        },
        "warmup": {
            "batch_count": args.warmup_batches,
            "batch_size": args.batch_size,
            "rng_isolated_and_restored": True,
        },
        "environment": environment, "source_provenance": sources,
        "timing_semantics": TIMING_SEMANTICS,
        "shared_ri_order_generation_seconds": shared_setup_seconds,
        "files": {
            "validated_records": VALIDATED_RECORDS_FILE,
            "batch_timings": BATCH_TIMINGS_FILE,
            "summary": SUMMARY_FILE,
        },
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / METADATA_FILE, metadata)

    records, batch_timings = [], []
    ranges = exact_batch_ranges(scope["instance_count"], args.batch_size)
    try:
        if args.warmup_batches:
            run_warmup_isolated(
                torch, device,
                lambda: _solve_batch(
                    points[:args.batch_size], protocol=protocol, orders=orders,
                    revisers=revisers, device=device, torch=torch,
                    reconnect=reconnect, load_problem=load_problem,
                    random_insertion_parallel=random_insertion_parallel,
                    timed=False))
        for batch_index, (start, stop) in enumerate(ranges):
            canonical, reported, decoding, runtime = _solve_batch(
                points[start:stop], protocol=protocol, orders=orders,
                revisers=revisers, device=device, torch=torch,
                reconnect=reconnect, load_problem=load_problem,
                random_insertion_parallel=random_insertion_parallel,
                timed=True)
            batch_rows = []
            for local, dataset_index in enumerate(range(start, stop)):
                task = wrapper.task_list[dataset_index]
                checked = validate(np.asarray(task.points), canonical[local])
                independent = checked["independent_objective"]
                official_agrees = (
                    independent is not None and
                    objective_agrees(reported[local], independent))
                solution = np.asarray(canonical[local], dtype=np.int64)
                kit_feasible = bool(task.check_constraints(solution))
                kit_objective = float(task.evaluate(solution))
                kit_agrees = (
                    independent is not None and
                    objective_agrees(kit_objective, independent))
                reference = float(references[dataset_index])
                if not (checked["feasible"] and official_agrees and
                        kit_feasible and kit_agrees):
                    raise RuntimeError(
                        f"parallel validation failed at dataset index {dataset_index}")
                batch_rows.append({
                    "dataset_instance_index": dataset_index,
                    "instance_id": prepared["instance_names"][dataset_index],
                    "batch_index": batch_index,
                    "canonical_solution": canonical[local],
                    "reported_objective": reported[local],
                    "independent_objective": independent,
                    "kit_objective": kit_objective,
                    "reference_objective": reference,
                    "gap_percent": (independent - reference) / reference * 100.0,
                    "independent_feasible": True,
                    "reported_objective_agrees": True,
                    "kit_feasible": True, "kit_objective_agrees": True,
                    "selection": _selection(protocol, decoding[local]),
                    "constraint_details": checked["constraint_details"],
                    "evidence_status": "KIT_VALIDATED",
                })
            records.extend(batch_rows)
            batch_timings.append({
                "batch_index": batch_index,
                "dataset_index_start": start,
                "dataset_index_stop_exclusive": stop,
                "original_instance_count": stop - start,
                "solver_runtime_seconds": runtime,
                "shared_ri_order_generation_seconds_charged": (
                    shared_setup_seconds if batch_index == 0 else 0.0),
            })
            print(json.dumps({
                "method": "GLOP", "problem": "TSP",
                "problem_size": args.problem_size, "protocol": args.protocol,
                "batch_size": args.batch_size, "batch_index": batch_index,
                "dataset_index_start": start,
                "dataset_index_stop_exclusive": stop,
                "solver_runtime_seconds": runtime,
            }, sort_keys=True, allow_nan=False), flush=True)
        summary = finalize_parallel(
            args.output_dir, metadata=metadata, records=records,
            batch_timings=batch_timings,
            shared_setup_seconds=shared_setup_seconds)
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
    except Exception as exc:
        metadata.update(
            state=("CUDA_OOM" if isinstance(
                exc, torch.cuda.OutOfMemoryError) else "FAILED"),
            failure_type=type(exc).__name__, failure_message=str(exc),
            failed_at=utc_now(), completed_records=len(records),
            completed_batches=len(batch_timings))
        write_json(args.output_dir / METADATA_FILE, metadata)
        raise


if __name__ == "__main__":
    main()
