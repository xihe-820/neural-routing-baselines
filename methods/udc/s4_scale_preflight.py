#!/usr/bin/env python3
"""One-size UDC formal-scale compatibility, memory and runtime preflight."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from methods.udc.protocol import (ALPHA, FORMAL_SIZES, OFFICIAL_COMMIT, SEED,
                                  SPECS, discover_scale_dataset,
                                  official_gate, s3_gate, s4_project_gate,
                                  scale_dataset_filename)
from methods.udc.s3_eval import (checkpoint_gate, load_models, rng_digest,
                                 SemanticValidationError, solve_cvrp, solve_tsp,
                                 cuda_memory_snapshot)

TIMING = ("preflight_runtime_only: BS=1 wall clock; includes alpha=50 initial "
          "solution generation, all refinement stages and algorithm-internal CPU/GPU "
          "operations; excludes model/checkpoint load, dataset parsing, adapter "
          "preprocessing, independent/Kit validation and artifact I/O; CUDA synchronized")
TERMINAL_STATUSES = {
    "PASS", "CUDA_OOM", "SEMANTIC_FAIL", "RUNTIME_FAIL", "PRECHECK_FAIL",
    "NOT_RUN_AFTER_OOM", "BLOCKED_BY_SEMANTIC_FAILURE",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def bytes_and_gib(value: int) -> dict:
    number = int(value)
    if number < 0:
        raise ValueError("memory bytes must be nonnegative")
    return {"bytes": number, "gib": number / (1024 ** 3)}


def memory_baseline(torch) -> dict:
    allocated = bytes_and_gib(torch.cuda.memory_allocated())
    reserved = bytes_and_gib(torch.cuda.memory_reserved())
    properties = torch.cuda.get_device_properties(0)
    return {"gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory": bytes_and_gib(properties.total_memory),
            "memory_before_solve_allocated": allocated,
            "memory_before_solve_reserved": reserved}


def classify_exception(exc: BaseException, torch) -> str:
    if isinstance(exc, SemanticValidationError):
        return "SEMANTIC_FAIL"
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return "CUDA_OOM"
    text = str(exc).lower()
    if "cuda" in text and "out of memory" in text:
        return "CUDA_OOM"
    return "RUNTIME_FAIL"


def progressive_decision(problem: str, size: int, rows: list[dict]) -> dict:
    global_blockers = [row for row in rows
                       if row.get("status") in ("RUNTIME_FAIL", "PRECHECK_FAIL")]
    if global_blockers:
        blocker = global_blockers[0]
        return {"allowed": False, "status": "PRECHECK_FAIL",
                "reason": ("S4 is globally blocked by "
                           f"{blocker.get('problem')}{blocker.get('size')} "
                           f"{blocker.get('status')}")}
    sizes = FORMAL_SIZES[problem]
    index = sizes.index(size)
    by_size = {row["size"]: row for row in rows if row.get("problem") == problem}
    for preceding in sizes[:index]:
        row = by_size.get(preceding)
        if row is None:
            return {"allowed": False, "status": "PRECHECK_FAIL",
                    "reason": f"required preceding {problem}{preceding} result is missing"}
        status = row.get("status")
        if status in ("CUDA_OOM", "NOT_RUN_AFTER_OOM"):
            return {"allowed": False, "status": "NOT_RUN_AFTER_OOM",
                    "reason": f"preceding {problem}{preceding} encountered CUDA OOM"}
        if status in ("SEMANTIC_FAIL", "BLOCKED_BY_SEMANTIC_FAILURE"):
            return {"allowed": False, "status": "BLOCKED_BY_SEMANTIC_FAILURE",
                    "reason": f"preceding {problem}{preceding} failed semantic validation"}
        if status != "PASS":
            return {"allowed": False, "status": "PRECHECK_FAIL",
                    "reason": f"preceding {problem}{preceding} status is {status}"}
    return {"allowed": True, "status": None, "reason": None}


def summary_row(record: dict) -> dict:
    memory = record.get("solver_memory") or {}
    return {
        "problem": record["problem"].lower(), "size": record["size"],
        "dataset": record.get("dataset_path"),
        "dataset_sha256": record.get("dataset_sha256"),
        "instance_index": record.get("dataset_instance_index", 0),
        "instance_name": record.get("instance_id"), "alpha": record.get("alpha", ALPHA),
        "x": record.get("x"), "configured_pomo": record.get("configured_pomo"),
        "effective_pomo": record.get("effective_pomo"),
        "semantic_status": record.get("semantic_status"),
        "independent_feasible": record.get("independent_feasible"),
        "objective": record.get("independent_objective"),
        "reference": record.get("reference_objective"),
        "drop_percent": record.get("instance_drop_percent"),
        "kit_status": record.get("kit_validation_status"),
        "runtime_seconds": record.get("runtime_seconds"),
        "peak_allocated_gib": memory.get("peak_allocated_gib"),
        "peak_reserved_gib": memory.get("peak_reserved_gib"),
        "status": record["status"],
    }


def placeholder_summary_row(problem: str, size: int, status: str) -> dict:
    row = {"problem": problem, "size": size, "dataset": None,
           "dataset_sha256": None, "instance_index": 0, "instance_name": None,
           "alpha": ALPHA, "x": SPECS[problem]["x"],
           "configured_pomo": SPECS[problem]["configured_pomo"],
           "effective_pomo": SPECS[problem]["effective_pomo"],
           "semantic_status": "NOT_REACHED", "independent_feasible": None,
           "objective": None, "reference": None, "drop_percent": None,
           "kit_status": "NOT_REACHED", "runtime_seconds": None,
           "peak_allocated_gib": None, "peak_reserved_gib": None,
           "status": status}
    return row


def build_summary(output_root: Path) -> dict:
    actual = {}
    for problem, sizes in FORMAL_SIZES.items():
        for size in sizes:
            path = output_root / f"{problem}{size}" / "record.json"
            if path.is_file():
                row = summary_row(json.loads(path.read_text()))
                if row["status"] not in TERMINAL_STATUSES:
                    raise ValueError(f"invalid S4 status in {path}")
                actual[(problem, size)] = row
    rows = []
    for problem, sizes in FORMAL_SIZES.items():
        blocker = None
        for size in sizes:
            row = actual.get((problem, size))
            if row is not None:
                rows.append(row)
                if row["status"] == "CUDA_OOM":
                    blocker = "CUDA_OOM"
                elif row["status"] == "SEMANTIC_FAIL":
                    blocker = "SEMANTIC_FAIL"
                continue
            if blocker:
                rows.append(placeholder_summary_row(
                    problem, size,
                    "NOT_RUN_AFTER_OOM" if blocker == "CUDA_OOM"
                    else "BLOCKED_BY_SEMANTIC_FAILURE"))
    return {"schema": "udc_s4_scale_summary.v1", "updated_at_utc": utc_now(),
            "protocol": {"alpha": ALPHA, "tsp_x": SPECS["tsp"]["x"],
                         "cvrp_x": SPECS["cvrp"]["x"], "sub_size": 100,
                         "batch_size": 1, "timing_semantics": TIMING},
            "rows": rows}


def dataset_evidence(path: Path, problem: str, size: int):
    import ml4co_kit as kit
    wrapper = kit.TSPWrapper() if problem == "tsp" else kit.CVRPWrapper()
    wrapper.from_pickle(path)
    expected = kit.TSPTask if problem == "tsp" else kit.CVRPTask
    if not wrapper.task_list or any(type(task) is not expected for task in wrapper.task_list):
        raise ValueError("dataset contains an empty or non-exact ML4CO task list")
    if any(np.asarray(task.points).shape != (size, 2) for task in wrapper.task_list):
        raise ValueError("dataset task shape does not match requested formal size")
    task = wrapper.task_list[0]
    reference = float(task.evaluate(task.ref_sol))
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("first task reference objective is invalid")
    return task, kit, {
        "path": str(path.resolve()), "filename": path.name,
        "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
        "task_count": len(wrapper.task_list),
        "task_class": f"{type(task).__module__}.{type(task).__name__}",
        "first_instance_name": str(task.name),
        "reference_source": "ML4CO task.ref_sol evaluated by task.evaluate",
        "first_reference_objective": reference,
    }


def precheck_record(problem, size, status, reason):
    return {"schema": "udc_s4_scale_record.v1", "problem": problem.upper(),
            "size": size, "dataset_instance_index": 0, "instance_id": None,
            "alpha": ALPHA, "x": SPECS[problem]["x"],
            "configured_pomo": SPECS[problem]["configured_pomo"],
            "effective_pomo": SPECS[problem]["effective_pomo"],
            "semantic_status": "NOT_REACHED", "status": status,
            "reason": reason}


def run(args) -> int:
    import torch
    problem, size = args.problem, args.size
    output_root = args.output_root.resolve()
    output_dir = output_root / f"{problem}{size}"
    metadata_path, record_path = output_dir / "metadata.json", output_dir / "record.json"
    if metadata_path.exists() or record_path.exists():
        raise ValueError(f"refusing to overwrite existing S4 artifact: {output_dir}")
    if is_within(output_root, args.project_root):
        raise ValueError("S4 evidence must live outside the project repository")
    existing_summary = build_summary(output_root) if output_root.exists() else {"rows": []}
    decision = progressive_decision(problem, size, existing_summary["rows"])
    output_dir.mkdir(parents=True, exist_ok=False)
    record = precheck_record(problem, size, "PRECHECK_FAIL", "preflight not completed")
    metadata = {"schema": "udc_s4_scale_metadata.v1", "started_at_utc": utc_now(),
                "problem": problem, "size": size, "status": "PRECHECK_FAIL",
                "script_path": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "output_dir": str(output_dir), "protocol": {"alpha": ALPHA,
                    "x": SPECS[problem]["x"], "configured_pomo": SPECS[problem]["configured_pomo"],
                    "effective_pomo": SPECS[problem]["effective_pomo"],
                    "sub_size": 100, "batch_size": 1, "seed": SEED,
                    "out_of_training_scale": problem == "tsp" and size == 100,
                    "timing_semantics": TIMING}, "blockers": []}
    last_operation = "output_initialized"
    try:
        if not decision["allowed"]:
            record = precheck_record(problem, size, decision["status"], decision["reason"])
            metadata["status"] = decision["status"]
            metadata["blockers"] = [decision["reason"]]
            return 1
        metadata["project_pre"] = s4_project_gate(args.project_root)
        metadata["official_pre"] = official_gate(args.official_root)
        metadata["s3_authoritative_evidence"] = s3_gate(args.s3_evidence)
        if not metadata["project_pre"]["pass"] or not metadata["official_pre"]["pass"]:
            raise ValueError("project or official repository provenance failed closed")
        last_operation = "repository_and_s3_provenance_verified"
        dataset = discover_scale_dataset(args.dataset_root, problem, size)
        task, kit, metadata["dataset"] = dataset_evidence(dataset, problem, size)
        last_operation = "dataset_index_0_loaded_and_audited"
        metadata["checkpoints"] = checkpoint_gate(args.supplemental_root, problem)
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or "RTX 4090" not in torch.cuda.get_device_name(0)):
            raise ValueError("S4 requires exactly one visible RTX 4090")
        metadata["environment"] = {"python": sys.version, "platform": platform.platform(),
                                   "torch": torch.__version__, "numpy": np.__version__,
                                   "kit_module": kit.__file__,
                                   "gpu_name": torch.cuda.get_device_name(0)}
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED); random.seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(0)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        metadata["rng_after_seed_before_model_construction"] = rng_digest(torch)
        env, harness, loaded_checkpoints, device = load_models(
            problem, args.official_root, args.supplemental_root, torch,
            problem_size=size)
        metadata["loaded_checkpoints"] = loaded_checkpoints
        last_operation = "models_strict_loaded"
        torch.cuda.empty_cache()
        metadata["memory_baseline"] = memory_baseline(torch)
        torch.cuda.reset_peak_memory_stats()
        last_operation = "solver_started"
        solve = solve_tsp if problem == "tsp" else solve_cvrp
        record = solve(task, env, harness, torch, device, 0, problem_size=size,
                       timing_semantics=TIMING, capture_memory=True)
        last_operation = "solver_and_all_validation_completed"
        record.update({"schema": "udc_s4_scale_record.v1",
                       "dataset_path": metadata["dataset"]["path"],
                       "dataset_sha256": metadata["dataset"]["sha256"],
                       "dataset_task_count": metadata["dataset"]["task_count"],
                       "dataset_task_class": metadata["dataset"]["task_class"],
                       "out_of_training_scale": problem == "tsp" and size == 100,
                       "semantic_status": ("PASS" if record["status"] == "KIT_VALIDATED"
                                           else "SEMANTIC_FAIL"),
                       "status": ("PASS" if record["status"] == "KIT_VALIDATED"
                                  else "SEMANTIC_FAIL")})
        metadata["status"] = record["status"]
        metadata["project_post"] = s4_project_gate(args.project_root)
        metadata["official_post"] = official_gate(args.official_root)
        if not metadata["project_post"]["pass"] or not metadata["official_post"]["pass"]:
            raise RuntimeError("repository changed during S4 preflight")
    except Exception as exc:
        status = classify_exception(exc, torch)
        if last_operation not in ("solver_started", "solver_and_all_validation_completed"):
            status = "PRECHECK_FAIL"
        record = precheck_record(problem, size, status, str(exc))
        record.update({"last_successful_operation": last_operation,
                       "exception_type": type(exc).__name__, "exception_text": str(exc)})
        try:
            record["memory_at_failure"] = {
                "current": memory_baseline(torch),
                "peaks": cuda_memory_snapshot(torch),
            }
        except Exception:
            record["memory_at_failure"] = None
        metadata["status"] = status
        metadata["blockers"].append(f"{type(exc).__name__}: {exc}")
        metadata["fatal_traceback"] = traceback.format_exc()
    finally:
        metadata["finished_at_utc"] = utc_now()
        atomic_json(record_path, record)
        metadata["record_path"] = str(record_path)
        metadata["record_sha256"] = sha256_file(record_path)
        atomic_json(metadata_path, metadata)
        atomic_json(output_root / "summary.json", build_summary(output_root))
    print(f"S4_STATUS={record['status']}")
    print(f"S4_RECORD={record_path}")
    return 0 if record["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=tuple(FORMAL_SIZES), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--supplemental-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--s3-evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.size not in FORMAL_SIZES[args.problem]:
        parser.error(f"size must be one of {FORMAL_SIZES[args.problem]}")
    for name in ("project_root", "official_root", "supplemental_root",
                 "dataset_root", "s3_evidence"):
        setattr(args, name, getattr(args, name).resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
