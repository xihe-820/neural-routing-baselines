#!/usr/bin/env python3
"""Run pinned official GLOP revisers for formal TSP50/100 first-N evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
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
from common.provenance import (environment_provenance, git_provenance,
                               normalize_git_repository_identity, source_provenance)
from common.result_schema import make_run_metadata, new_result, write_result_bundle
from methods.glop.tsp.adapter import adapt_points
from methods.glop.tsp.config import (REVISER_ASSETS, reviser_paths,
                                     supported_config, validate_reviser_schedule)
from methods.glop.tsp.decode import decode_coordinate_tour, select_best_candidates
from problems.tsp.validate import validate

UPSTREAM_URL = "https://github.com/henry-yeh/GLOP"
UPSTREAM_COMMIT = "e540bc0153a0598e923e35116deeaecaf9c1cfff"


def _device(value, torch):
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return device


def _verify_file(path, *, sha256, size):
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != size:
        raise ValueError(f"unexpected GLOP asset size: {path}")
    with path.open("rb") as stream:
        prefix = stream.read(128)
    if prefix.startswith(b"version https://git-lfs.github.com/spec"):
        raise ValueError(f"GLOP asset is a Git LFS pointer: {path}")
    actual = sha256_file(path)
    if actual != sha256:
        raise ValueError(f"unexpected GLOP asset SHA256: {path}")
    return actual


def _load_revisers(asset_root, config, *, device, torch, load_model):
    revisers, evidence = [], []
    for size in config["required_revisers"]:
        checkpoint, args_path = reviser_paths(asset_root, size)
        identity = REVISER_ASSETS[size]
        checkpoint_hash = _verify_file(
            checkpoint, sha256=identity["checkpoint_sha256"],
            size=identity["checkpoint_size_bytes"])
        args_hash = _verify_file(
            args_path, sha256=identity["args_sha256"], size=identity["args_size_bytes"])
        args_file = json.loads(args_path.read_text())
        if args_file.get("problem") != "local" or args_file.get("graph_size") != size:
            raise ValueError(f"reviser_{size} args.json has unexpected model identity")
        model, loaded_args = load_model(str(checkpoint.resolve()), is_local=True)
        if loaded_args != args_file:
            raise ValueError(f"official loader args disagree with reviser_{size} args.json")
        payload = torch.load(checkpoint, map_location="cpu")
        state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload.state_dict()
        strict = model.load_state_dict(state_dict, strict=True)
        if strict.missing_keys or strict.unexpected_keys:
            raise ValueError(f"reviser_{size} strict load returned incompatible keys")
        model.to(device)
        model.eval()
        model.set_decode_type(config["decode_strategy"])
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != 1301888:
            raise ValueError(f"reviser_{size} parameter count is not the audited architecture")
        revisers.append(model)
        evidence.append({
            "reviser_size": size,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "args_path": str(args_path.resolve()),
            "args_sha256": args_hash,
            "args_size_bytes": args_path.stat().st_size,
            "args_problem": args_file["problem"],
            "args_graph_size": args_file["graph_size"],
            "strict_load": True,
            "missing_keys": list(strict.missing_keys),
            "unexpected_keys": list(strict.unexpected_keys),
            "parameter_count": parameter_count,
            "model_eval": not model.training,
        })
    return revisers, evidence


def _official_imports(upstream):
    # This repository also has a top-level problems package. Its validator is
    # already bound above, so clear that cache before importing pinned GLOP.
    for module_name in list(sys.modules):
        if module_name == "problems" or module_name.startswith("problems."):
            del sys.modules[module_name]
    sys.path.insert(0, str(upstream.resolve()))
    from utils.functions import load_model, load_problem, reconnect
    from utils.insertion import random_insertion_parallel
    forbidden = [name for name in sys.modules if name.startswith(
        ("torch_geometric", "torch_scatter", "torch_sparse", "heatmap"))]
    if forbidden:
        raise RuntimeError(f"TSP-only import unexpectedly loaded CVRP/PyG modules: {forbidden}")
    return load_model, load_problem, reconnect, random_insertion_parallel


def _formal_inference(points, *, config, revisers, device, torch,
                      load_problem, reconnect, random_insertion_parallel,
                      execution_batch_size):
    batch_size, problem_size, _ = points.shape
    width = config["width_after_small_size_branch"]
    torch.manual_seed(config["seed"])
    orders = [torch.randperm(problem_size) for _ in range(width)]
    initial = [random_insertion_parallel(points, order) for order in orders]
    permutations = np.asarray(initial, dtype=np.int64).reshape(width, batch_size, problem_size)
    expected = np.arange(problem_size)
    if not all(np.array_equal(np.sort(row), expected)
               for row in permutations.reshape(-1, problem_size)):
        raise ValueError("official random insertion returned a non-permutation")

    problem = load_problem("tsp")
    get_cost_func = lambda data, pi: problem.get_costs(data, pi, return_local=True)
    selected_tours, selected_costs, selected_indices = [], [], []
    for start in range(0, batch_size, execution_batch_size):
        stop = min(start + execution_batch_size, batch_size)
        current = stop - start
        batch = torch.as_tensor(points[start:stop], dtype=torch.float32)
        batch = batch.repeat(width, 1, 1)
        pi_batch = torch.as_tensor(
            permutations[:, start:stop, :].reshape(-1, problem_size), dtype=torch.long)
        seed = batch.gather(1, pi_batch.unsqueeze(-1).repeat(1, 1, 2)).to(device)
        if config["tsp_aug"]:
            seed_x = torch.cat((1 - seed[:, :, [0]], seed[:, :, [1]]), dim=2)
            seed_y = torch.cat((seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
            seed_xy = torch.cat((1 - seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
            seed = torch.cat((seed, seed_x, seed_y, seed_xy), dim=0)
        candidate_count = width * len(config["top_level_transforms"])
        if seed.shape != (candidate_count * current, problem_size, 2):
            raise ValueError("top-level candidate construction has an unexpected shape")

        # With no_prune=True, official reconnect only groups candidates at its
        # final min. Treat each row as a group there, then apply the identical
        # first-argmin to retain the selected candidate index for provenance.
        reconnect_batch_size = seed.shape[0] if config["no_prune"] else current
        opts = SimpleNamespace(
            revision_lens=list(config["required_revisers"]),
            revision_iters=list(config["revision_iters"]),
            no_aug=config["no_aug"], no_prune=config["no_prune"],
            eval_batch_size=reconnect_batch_size,
        )
        tours, costs = reconnect(
            get_cost_func=get_cost_func, batch=seed, opts=opts, revisers=revisers)
        if config["no_prune"]:
            costs = costs.reshape(candidate_count, current)
            tours = tours.reshape(candidate_count, current, problem_size, 2)
            chosen_tours, chosen_costs, chosen = select_best_candidates(
                costs.detach().cpu().numpy(), tours.detach().cpu().numpy())
            selected_tours.extend(chosen_tours)
            selected_costs.extend(chosen_costs.tolist())
            selected_indices.extend(chosen.tolist())
        else:
            selected_tours.extend(tours.detach().cpu().numpy())
            selected_costs.extend(costs.detach().cpu().numpy().tolist())
            selected_indices.extend([0] * current)
    return (np.asarray(selected_tours, dtype=np.float32),
            np.asarray(selected_costs, dtype=np.float64),
            np.asarray(selected_indices, dtype=np.int64), orders)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, choices=[50, 100], required=True)
    parser.add_argument("--input-metadata", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True,
                        help="Directory containing Reviser-stage2 from the official bundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execution-batch-size", type=int)
    args = parser.parse_args()
    config = supported_config(args.problem_size)
    validate_reviser_schedule(
        args.problem_size, config["required_revisers"], config["revision_iters"])

    metadata_path = args.input_metadata or args.input.with_suffix(args.input.suffix + ".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format") != "glop-tsp-neutral-input-v1":
        raise ValueError("unsupported prepared input metadata")
    if metadata.get("problem_size") != args.problem_size:
        raise ValueError("prepared input problem_size does not match requested problem_size")
    if metadata.get("input_npz_sha256") != sha256_file(args.input):
        raise ValueError("prepared NPZ hash does not match metadata")
    if metadata.get("dataset_sha256") != config["dataset_sha256"]:
        raise ValueError("prepared input is not from the pinned TSP benchmark")
    upstream = git_provenance(args.upstream)
    if (upstream["commit"] != UPSTREAM_COMMIT or upstream["dirty"] or
            normalize_git_repository_identity(upstream["url"]) !=
            normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("official GLOP checkout identity/cleanliness does not match the pin")

    import torch
    device = _device(args.device, torch)
    with np.load(args.input, allow_pickle=False) as data:
        points = data["points"]
        indices = data["dataset_indices"]
        references = data["reference_objectives"]
    _, mapping = adapt_points(points, problem_size=args.problem_size, device="cpu")
    batch_size = len(points)
    execution_batch_size = args.execution_batch_size or batch_size
    if execution_batch_size <= 0 or execution_batch_size > batch_size:
        raise ValueError("execution-batch-size must be within the prepared input batch")

    load_model, load_problem, reconnect, random_insertion_parallel = _official_imports(args.upstream)
    revisers, asset_evidence = _load_revisers(
        args.asset_root, config, device=device, torch=torch, load_model=load_model)
    random_insertion_version = importlib.metadata.version("random-insertion")
    if tuple(int(part) for part in random_insertion_version.split(".post")[0].split(".")) < (0, 3, 0):
        raise RuntimeError("GLOP requires random-insertion >= 0.3.0")

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    with torch.no_grad():
        tours, costs, candidate_indices, orders = _formal_inference(
            points, config=config, revisers=revisers, device=device, torch=torch,
            load_problem=load_problem, reconnect=reconnect,
            random_insertion_parallel=random_insertion_parallel,
            execution_batch_size=execution_batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    finished_at = datetime.now(timezone.utc).isoformat()

    project = git_provenance(ROOT)
    sources = source_provenance([
        Path(__file__), Path(__file__).with_name("adapter.py"),
        Path(__file__).with_name("decode.py"), Path(__file__).with_name("config.py"),
        ROOT / "problems/tsp/validate.py", ROOT / "problems/tsp/objective.py",
        ROOT / "common/objective_agreement.py", ROOT / "common/result_schema.py",
        ROOT / "common/provenance.py",
    ], root=ROOT)
    environment = environment_provenance(device)
    environment["random_insertion_version"] = random_insertion_version
    environment["random_insertion_module"] = sys.modules["random_insertion"].__file__
    width = config["width_after_small_size_branch"]
    rows = []
    for i, (tour, reported, candidate_index) in enumerate(
            zip(tours, costs, candidate_indices)):
        canonical, decode_info = decode_coordinate_tour(
            tour, points[i], allowed_transforms=config["top_level_transforms"])
        transform_index = int(candidate_index) // width
        expected_transform = config["top_level_transforms"][transform_index]
        if decode_info["top_level_transform"] != expected_transform:
            raise ValueError("selected candidate index and exact transform decoding disagree")
        validation = validate(points[i], canonical)
        independent = validation["independent_objective"]
        agrees = objective_agrees(float(reported), independent) if independent is not None else False
        if not validation["feasible"] or not agrees:
            raise RuntimeError(f"GLOP TSP independent gate failed at row {i}")
        abs_error = abs(float(reported) - independent)
        reference = float(references[i])
        primary_asset = asset_evidence[0]
        rows.append(new_result(
            method="GLOP", variant="Reviser-stage2", problem="TSP",
            problem_size=args.problem_size, instance_id=metadata["instance_names"][i],
            project_repo_commit=project["commit"], project_repo_dirty=project["dirty"],
            upstream_url=upstream["url"], upstream_commit=upstream["commit"],
            upstream_dirty=upstream["dirty"],
            checkpoint_path=primary_asset["checkpoint_path"],
            checkpoint_sha256=primary_asset["checkpoint_sha256"],
            dataset_path=metadata["dataset_path"], dataset_sha256=metadata["dataset_sha256"],
            dataset_instance_index=int(indices[i]),
            adapter_provenance={"sources": sources, "mapping": mapping},
            inference_config={
                "problem_type": "tsp", "problem_size": args.problem_size,
                "revision_lens": list(config["required_revisers"]),
                "revision_iters": list(config["revision_iters"]),
                "width_requested": config["width_requested"],
                "width_after_small_size_branch": width,
                "tsp_aug": config["tsp_aug"],
                "top_level_candidate_count": width * len(config["top_level_transforms"]),
                "decode_strategy": config["decode_strategy"],
                "no_aug": config["no_aug"], "no_prune": config["no_prune"],
                "seed": config["seed"], "execution_batch_size": execution_batch_size,
                "configuration_evidence": config["configuration_evidence"],
                "training": False, "backward": False, "optimizer_created": False,
                "reviser_assets": asset_evidence,
            },
            selection_metadata={
                **decode_info,
                "selected_top_level_candidate": int(candidate_index),
                "selected_transform": expected_transform,
                "selected_insertion_order_slot": int(candidate_index) % width,
                "candidate_selection": "numpy.argmin first minimum, matching torch.min",
                "original_node_permutation": canonical[:-1],
            },
            canonical_solution=canonical, reported_objective=float(reported),
            independent_objective=independent, objective_abs_error=abs_error,
            objective_rel_error=abs_error / abs(independent) if independent else None,
            reported_objective_agrees=agrees, reference_objective=reference,
            gap_percent=(independent - reference) / reference * 100,
            independent_feasible=validation["feasible"],
            constraint_details=validation["constraint_details"],
            runtime_seconds=runtime / batch_size,
            runtime_semantics="amortized formal first-N rollout seconds; engineering evidence; NOT PAPER-COMPARABLE",
            environment=environment, evidence_status="LOCAL_VERIFIED_INDEPENDENT",
            error=None,
        ))
    run_metadata = make_run_metadata(
        started_at=started_at, finished_at=finished_at,
        total_runtime_seconds=runtime, batch_size=batch_size,
        problem_size=args.problem_size, execution_batch_size=execution_batch_size,
        formal_candidate_count=width * len(config["top_level_transforms"]),
        initial_order_count=len(orders),
        torch_manual_seed=config["seed"], python_random_seed=None, numpy_random_seed=None,
        random_insertion={
            "version": random_insertion_version,
            "determinism_semantics": "deterministic for identical instances and explicit insertion orders",
            "order_source": "torch.randperm after the sole official torch.manual_seed(seed)",
        },
        asset_evidence=asset_evidence, training=False, backward=False,
        optimizer_created=False, runtime_class="engineering evidence; NOT PAPER-COMPARABLE",
    )
    write_result_bundle(args.output, rows, run_metadata=run_metadata)
    print(args.output)


if __name__ == "__main__":
    main()
