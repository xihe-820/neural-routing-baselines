#!/usr/bin/env python3
"""LEHD ML4CO smoke, preflight, and resumable formal BS1 production."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from common.provenance import (environment_provenance, git_provenance,
                               normalize_git_repository_identity, source_provenance)
from methods.lehd.config import (FORMAL_BATCH_SIZE, FORMAL_PROTOCOLS, PROJECT_SEED,
                                 UPSTREAM_COMMIT, UPSTREAM_URL, WARMUP_POLICY,
                                 resolve_config, validate_checkpoint_location,
                                 validate_dataset_path)
from methods.lehd.paper_results import (TERMINAL_STATES, TIMING_SEMANTICS, append,
                                        capture_rng_state, finalize, fingerprint,
                                        initialize, restore_rng_state)


def comparison(left, right):
    left, right = float(left), float(right)
    absolute = abs(left - right)
    return {
        "left": left, "right": right, "abs_diff": absolute,
        "rel_diff": absolute / max(abs(left), abs(right), 1e-300),
        "pass": objective_agrees(left, right),
    }


def _require_file(value, name):
    if value is None or not Path(value).is_file():
        raise ValueError(f"{name} must be an existing file")
    return Path(value).resolve()


def _require_directory(value, name):
    if value is None or not Path(value).is_dir():
        raise ValueError(f"{name} must be an existing directory")
    return Path(value).resolve()


def _cuda_device(torch, value):
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LEHD server evaluation requires CUDA")
    if device.index is None:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpu = torch.cuda.get_device_name(device)
    if "RTX 4090" not in gpu:
        raise RuntimeError(f"current LEHD correctness workflow requires RTX 4090; observed {gpu!r}")
    return device


def _load_tasks(kit, problem, dataset):
    wrapper = kit.TSPWrapper() if problem == "tsp" else kit.CVRPWrapper()
    wrapper.from_pickle(dataset)
    expected_type = kit.TSPTask if problem == "tsp" else kit.CVRPTask
    if not wrapper.task_list or any(type(task) is not expected_type for task in wrapper.task_list):
        raise ValueError("LEHD dataset contains an unexpected ML4CO task type")
    return wrapper.task_list


def _task_name(task, index):
    value = getattr(task, "name", None)
    return str(value) if value is not None else f"instance-{index}"


def _smoke_count(problem, problem_size, protocol_label):
    if protocol_label == "greedy":
        return 1
    if protocol_label == "fewer":
        return 1 if problem == "cvrp" and problem_size == 2000 else 2
    raise ValueError("our-smoke only supports LEHD greedy or fewer")


def _scope_count(problem, problem_size, protocol_label, scope):
    if scope == "our-smoke":
        return _smoke_count(problem, problem_size, protocol_label)
    if scope == "preflight" and protocol_label == "more":
        return 1
    raise ValueError("LEHD smoke/preflight scope does not match protocol")


def _verify_preflight_evidence(path, *, problem, problem_size, protocol_label,
                               dataset_sha256, checkpoint_sha256,
                               protocol_fingerprint, project_commit, source_files):
    evidence_path = Path(path).resolve()
    if evidence_path.is_dir():
        evidence_path = evidence_path / "summary.json"
    if not evidence_path.is_file():
        raise ValueError("LEHD fullset evidence must be a summary.json or run directory")
    summary = json.loads(evidence_path.read_text())
    expected_scope = "preflight" if protocol_label == "more" else "our-smoke"
    expected = {
        "status": "KIT_VALIDATED", "method": "LEHD", "problem": problem.upper(),
        "problem_size": int(problem_size), "protocol_label": protocol_label,
        "dataset_sha256": dataset_sha256, "checkpoint_sha256": checkpoint_sha256,
        "upstream_commit": UPSTREAM_COMMIT,
        "protocol_fingerprint": protocol_fingerprint,
        "project_commit": project_commit,
        "source_provenance_fingerprint": fingerprint(source_files),
    }
    mismatches = [
        field for field, value in expected.items() if summary.get(field) != value]
    identity = summary.get("identity")
    if not isinstance(identity, dict):
        mismatches.append("identity")
    else:
        embedded_protocol = identity.get("protocol")
        embedded_dataset = identity.get("dataset")
        embedded_checkpoint = identity.get("checkpoint")
        embedded_project = identity.get("project")
        embedded_upstream = identity.get("upstream")
        dictionaries = (
            embedded_protocol, embedded_dataset, embedded_checkpoint,
            embedded_project, embedded_upstream,
        )
        if not all(isinstance(value, dict) for value in dictionaries):
            mismatches.append("identity dictionaries")
        else:
            checks = {
                "identity.method": identity.get("method") == "LEHD",
                "identity.scope": identity.get("scope") == expected_scope,
                "identity.protocol.problem": embedded_protocol.get("problem") == problem.upper(),
                "identity.protocol.problem_size": embedded_protocol.get(
                    "actual_problem_size") == int(problem_size),
                "identity.protocol.label": embedded_protocol.get(
                    "protocol_label") == protocol_label,
                "identity.protocol_fingerprint": identity.get(
                    "protocol_fingerprint") == protocol_fingerprint,
                "identity.protocol_content": fingerprint(
                    embedded_protocol) == protocol_fingerprint,
                "identity.dataset_sha256": embedded_dataset.get(
                    "sha256") == dataset_sha256,
                "identity.checkpoint_sha256": embedded_checkpoint.get(
                    "sha256") == checkpoint_sha256,
                "identity.project_commit": embedded_project.get(
                    "commit") == project_commit,
                "identity.upstream_commit": embedded_upstream.get(
                    "commit") == UPSTREAM_COMMIT,
                "identity.source_files": identity.get("source_files") == source_files,
            }
            mismatches.extend(field for field, passed in checks.items() if not passed)
    count = summary.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        mismatches.append("count")
    else:
        try:
            expected_count = _scope_count(
                problem, int(problem_size), protocol_label, expected_scope)
        except ValueError:
            expected_count = None
        if count != expected_count:
            mismatches.append("count policy")
    if summary.get("validated_count") != count:
        mismatches.append("validated_count")
    if summary.get("failed_count") != 0:
        mismatches.append("failed_count")
    if isinstance(identity, dict) and identity.get("count") != count:
        mismatches.append("identity.count")
    if mismatches:
        raise ValueError(
            "LEHD fullset evidence mismatch: " + ", ".join(sorted(set(mismatches))))
    return {
        "path": str(evidence_path), "sha256": sha256_file(evidence_path),
        "identity": {**expected, "scope": expected_scope, "count": count,
                     "validated_count": count, "failed_count": 0},
    }


def _adapt(problem, task, problem_size):
    if problem == "tsp":
        from methods.lehd.tsp.adapter import adapt_task
    else:
        from methods.lehd.cvrp.adapter import adapt_task
    return adapt_task(task, problem_size=problem_size)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["tsp", "cvrp"], required=True)
    parser.add_argument("--problem-size", type=int, required=True)
    parser.add_argument("--protocol", choices=FORMAL_PROTOCOLS, required=True)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--dump-config", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--preflight-evidence", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scope", choices=["our-smoke", "preflight", "fullset"])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--device", default="cuda:0")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true")
    resume_group.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    protocol = resolve_config(
        args.problem, args.problem_size, args.protocol, batch_size=args.batch_size)
    if args.dump_config:
        print(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.scope is None or args.count is None or args.output_dir is None:
        parser.error("execution requires --scope, --count, and --output-dir")
    if args.offset < 0 or args.count <= 0:
        parser.error("--offset must be nonnegative and --count positive")
    if args.scope in {"our-smoke", "preflight"}:
        try:
            expected_count = _scope_count(
                args.problem, args.problem_size, args.protocol, args.scope)
        except ValueError as exc:
            parser.error(str(exc))
        if args.offset != 0 or args.count != expected_count:
            parser.error("count does not match the frozen LEHD smoke/preflight policy")
    if args.scope == "fullset" and args.preflight_evidence is None:
        parser.error("fullset requires --preflight-evidence")
    if not args.expected_dataset_sha256 or not args.expected_checkpoint_sha256:
        parser.error("execution requires both expected SHA256 arguments")

    dataset = _require_file(args.dataset, "dataset")
    checkpoint = _require_file(args.checkpoint, "checkpoint")
    upstream_path = _require_directory(args.upstream, "upstream")
    validate_checkpoint_location(protocol, checkpoint, upstream_path)
    validate_dataset_path(protocol, dataset)
    dataset_sha = sha256_file(dataset)
    checkpoint_sha = sha256_file(checkpoint)
    if dataset_sha != args.expected_dataset_sha256:
        raise ValueError("LEHD dataset SHA256 mismatch")
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("LEHD checkpoint SHA256 mismatch")

    project = git_provenance(ROOT)
    upstream = git_provenance(upstream_path)
    if project["dirty"]:
        raise ValueError("LEHD server evaluation requires a clean project checkout")
    if (upstream["dirty"] or upstream["commit"] != UPSTREAM_COMMIT
            or normalize_git_repository_identity(upstream["url"])
            != normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("official NCO_code identity/cleanliness mismatch")

    sources = [
        Path(__file__), ROOT / "methods/lehd/config.py",
        ROOT / "methods/lehd/runtime.py", ROOT / "methods/lehd/paper_results.py",
        ROOT / f"methods/lehd/{args.problem}/adapter.py",
        ROOT / f"problems/{args.problem}/validate.py",
    ]
    current_source_files = source_provenance(sources, root=ROOT)
    protocol_fingerprint = fingerprint(protocol)
    evidence = None
    if args.scope == "fullset":
        evidence = _verify_preflight_evidence(
            args.preflight_evidence, problem=args.problem,
            problem_size=args.problem_size, protocol_label=args.protocol,
            dataset_sha256=dataset_sha, checkpoint_sha256=checkpoint_sha,
            protocol_fingerprint=protocol_fingerprint,
            project_commit=project["commit"], source_files=current_source_files)

    import ml4co_kit as kit
    tasks = _load_tasks(kit, args.problem, dataset)
    dataset_count = len(tasks)
    if args.offset + args.count > dataset_count:
        raise ValueError("requested LEHD dataset slice exceeds task count")
    if args.scope == "fullset" and (args.offset != 0 or args.count != dataset_count):
        raise ValueError("LEHD fullset must cover the complete ordered dataset")
    indices = list(range(args.offset, args.offset + args.count))

    import torch
    device = _cuda_device(torch, args.device)
    environment = environment_provenance(device)
    identity = {
        "method": "LEHD", "scope": args.scope, "protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint,
        "offset": args.offset, "count": args.count, "indices": indices,
        "dataset": {
            "path": str(dataset), "expected_filename": protocol["expected_dataset_filename"],
            "actual_filename": dataset.name, "sha256": dataset_sha, "count": dataset_count,
        },
        "checkpoint": {
            "path": str(checkpoint), "filename": checkpoint.name,
            "relative_path": protocol["checkpoint"]["relative_path"],
            "sha256": checkpoint_sha, "epoch": protocol["checkpoint"]["epoch"],
            "trained_on_size": protocol["checkpoint"]["trained_on_size"],
        },
        "project": project, "upstream": upstream, "environment": environment,
        "source_files": current_source_files,
        "timing_semantics": TIMING_SEMANTICS, "warmup": WARMUP_POLICY,
        "preflight_evidence": evidence, "official_source_modified": False,
        "checkpoint_load_semantics": (
            "pinned official Tester torch.load followed by model.load_state_dict "
            "with PyTorch strict=True default"
        ),
        "compatibility_shims": [],
    }
    output_dir = args.output_dir.resolve()
    metadata, records, timings, rng_checkpoint = initialize(
        output_dir, identity, resume=(args.resume or args.skip_existing))
    if metadata["state"] in TERMINAL_STATES:
        print(output_dir / "summary.json")
        return 0

    from methods.lehd.runtime import (build_tester, run_isolated_warmup,
                                      seed_project_rng, solve_one)
    if args.problem == "tsp":
        from methods.lehd.tsp.adapter import (decode_official_solution,
                                              inject_official_data)
        from problems.tsp.validate import validate as independent_validate
    else:
        from methods.lehd.cvrp.adapter import (decode_official_solution,
                                               inject_official_data)
        from problems.cvrp.validate import validate as independent_validate

    def build():
        tester, runtime_protocol = build_tester(
            problem=args.problem, problem_size=args.problem_size,
            protocol_label=args.protocol, upstream=upstream_path,
            checkpoint=checkpoint, device=device, torch=torch)
        if runtime_protocol != protocol:
            raise RuntimeError("runtime LEHD config differs from artifact protocol")
        return tester

    seed_project_rng(torch, PROJECT_SEED)
    tester = None
    if rng_checkpoint is not None:
        tester = build()
        restore_rng_state(rng_checkpoint["rng_state"], torch)

    for index in indices[len(records):]:
        task = tasks[index]
        reference = float(task.evaluate(task.ref_sol))
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError(f"invalid reference objective at dataset index {index}")
        adapted = _adapt(args.problem, task, args.problem_size)
        if args.problem == "tsp":
            source_points, model_points, reference_order, adaptation = adapted
            inject = lambda env: inject_official_data(
                env, model_points[None], reference_order[None], device=device, torch=torch)
        else:
            (depot, points, raw_demands, capacity, coordinates, model_demands,
             reference_native, adaptation) = adapted
            inject = lambda env: inject_official_data(
                env, coordinates[None], model_demands[None], np.asarray([capacity]),
                reference_native[None], device=device, torch=torch)

        if tester is None:
            run_isolated_warmup(
                build,
                lambda warmup_tester: solve_one(
                    warmup_tester, problem=args.problem,
                    problem_size=args.problem_size, inject=inject,
                    device=device, torch=torch, timed=False),
                torch=torch)
            tester = build()

        solved = solve_one(
            tester, problem=args.problem, problem_size=args.problem_size,
            inject=inject, device=device, torch=torch, timed=True)
        decoded = decode_official_solution(solved["solution"], problem_size=args.problem_size)
        if args.problem == "tsp":
            checked = independent_validate(source_points, decoded["canonical_solution"])
        else:
            checked = independent_validate(
                depot, points, raw_demands, capacity, decoded["canonical_solution"],
                capacity_tolerance=1e-5 * capacity)
        if not checked["feasible"]:
            raise RuntimeError(f"independent LEHD validation failed at dataset index {index}")
        independent = float(checked["independent_objective"])
        kit_solution = np.asarray(decoded["canonical_solution"], dtype=np.int64)
        kit_feasible = bool(task.check_constraints(kit_solution))
        kit_objective = float(task.evaluate(kit_solution))
        official_check = comparison(solved["official_objective"], independent)
        kit_check = comparison(independent, kit_objective)
        if not kit_feasible or not official_check["pass"] or not kit_check["pass"]:
            raise RuntimeError(f"LEHD objective/Kit validation failed at dataset index {index}")
        record = {
            "schema": "lehd-validated-record.v1", "method": "LEHD",
            "problem": args.problem.upper(), "problem_size": args.problem_size,
            "trained_on_size": protocol["trained_on_size"],
            "dataset_instance_index": index, "instance_id": _task_name(task, index),
            "protocol": args.protocol, "RRC_budget": protocol["RRC_budget"],
            "budget_mapping_origin": protocol["budget_mapping_origin"],
            "config_origin": protocol["config_origin"],
            "checkpoint_sha256": checkpoint_sha, "dataset_sha256": dataset_sha,
            **decoded, "adaptation": adaptation,
            "official_objective": solved["official_objective"],
            "independent_objective": independent, "reference_objective": reference,
            "gap_percent": (independent - reference) / reference * 100.0,
            "independent_feasible": True,
            "independent_constraint_details": checked["constraint_details"],
            "kit_feasible": kit_feasible, "kit_objective": kit_objective,
            "official_vs_independent": official_check,
            "independent_vs_kit": kit_check,
            "solution_capture": solved["capture"], "evidence_status": "KIT_VALIDATED",
        }
        timing = {
            "schema": "lehd-batch-timing.v1", "dataset_instance_index": index,
            "original_instance_batch_size": 1,
            "runtime_seconds": solved["runtime_seconds"],
            "timing_semantics": TIMING_SEMANTICS,
        }
        append(output_dir, metadata, records, timings, record, timing,
               capture_rng_state(torch))

    summary = finalize(output_dir, metadata, records, timings)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
