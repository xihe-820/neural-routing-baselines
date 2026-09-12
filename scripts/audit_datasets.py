#!/usr/bin/env python3
"""Audit explicitly supplied trusted ML4CO pickle files using the installed Kit.

Records every task's class/size, first-instance schema, independent reference
validation through problem validators, and secondary Kit checks.
No baseline inputs are exported and no baseline inference is performed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import platform
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.hashing import sha256_file


def audit(path, problem, expected_size):
    result = {"problem": problem, "expected_size": expected_size,
              "realpath": str(path.resolve()), "exists": path.is_file()}
    if not path.is_file():
        return result
    result.update(file_size=path.stat().st_size, sha256=sha256_file(path), format="pickle")
    try:
        import ml4co_kit as kit
        import numpy as np
        result["kit_module"] = kit.__file__
        wrapper = getattr(kit, problem + "Wrapper")()
        wrapper.from_pickle(path)
        tasks = wrapper.task_list
        result["container_type"] = f"{type(tasks).__module__}.{type(tasks).__name__}"
        result["object_count"] = len(tasks)
        result["task_classes"] = dict(Counter(f"{type(t).__module__}.{type(t).__name__}" for t in tasks))
        result["coordinate_shape_counts"] = dict(Counter(str(tuple(t.points.shape)) for t in tasks))
        result["all_expected_size"] = all(t.points.shape == (expected_size, 2) for t in tasks)
        expected_class = getattr(kit, problem + "Task")
        result["all_expected_task_class"] = all(type(t) is expected_class for t in tasks)
        if not tasks:
            raise ValueError("empty task list")
        task = tasks[0]
        if not result["all_expected_size"] or not result["all_expected_task_class"]:
            raise ValueError("dataset content does not match requested problem/size")
        result["task_class"] = f"{type(task).__module__}.{type(task).__name__}"
        source = Path(inspect.getfile(type(task)))
        result["kit_task_source"] = {"path": str(source), "sha256": sha256_file(source)}
        schema = {}
        for key, value in vars(task).items():
            if isinstance(value, np.ndarray):
                schema[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                if value.size and value.dtype.kind in "fiu":
                    schema[key].update(min=float(value.min()), max=float(value.max()))
            elif value is None or isinstance(value, (str, int, float, bool, np.generic)):
                schema[key] = value.item() if isinstance(value, np.generic) else value
            else:
                schema[key] = {"type": type(value).__name__, "repr": str(value)[:200]}
        result["first_instance_schema"] = schema
        if problem != "TSP":
            result["capacity_values"] = sorted({float(t.capacity) for t in tasks})
        if problem == "CVRPTW":
            result["time_semantics"] = {
                "depot_time_window_first": task.tw[0].tolist(),
                "depot_time_windows_all": sorted({tuple(map(float, t.tw[0])) for t in tasks}),
                "depot_service_values_all": sorted({float(t.service[0]) for t in tasks}),
                "customer_service_first_min_max": [float(task.service[1:].min()), float(task.service[1:].max())],
                "customer_tw_start_first_min_max": [float(task.tw[1:, 0].min()), float(task.tw[1:, 0].max())],
                "customer_tw_end_first_min_max": [float(task.tw[1:, 1].min()), float(task.tw[1:, 1].max())],
                "speed": 1.0,
                "speed_basis": "Reviewed Kit CVRPTWTask docstring and CVRPTask._check_route_tw: travel_time = dist_eval.cal_distance",
                "initial_depot_departure": 0.0,
                "window_convention": "service_start=max(arrival,lower); service_start<=upper+threshold",
                "return_depot_convention": "closed routes return arrival <= depot upper+threshold",
                "all_tw_shapes_match": all(t.tw.shape == (expected_size + 1, 2) for t in tasks),
                "all_service_shapes_match": all(t.service.shape == (expected_size + 1,) for t in tasks),
            }
        sol = np.asarray(task.ref_sol)
        if sol.ndim != 1 or sol.dtype.kind not in "iu":
            raise ValueError("reference must be a one-dimensional integer sequence")
        n = expected_size
        # Only continuous Euclidean metrics are implemented here. Do not silently
        # substitute them for other rounding or distance semantics.
        result["metric"] = {"distance_type": str(task.distance_type),
                            "round_type": str(task.dist_eval.round_type)}
        if getattr(task.distance_type, "name", "") != "EUC_2D" or getattr(task.dist_eval.round_type, "name", "") != "NO":
            raise ValueError("independent audit supports only EUC_2D / NO rounding")
        if problem == "TSP":
            cycle = sol[:-1] if len(sol) == n + 1 and sol[-1] == sol[0] else sol
            if not np.array_equal(np.sort(cycle), np.arange(n)):
                raise ValueError("reference is not a node permutation")
            coordinates = np.asarray(task.points, dtype=np.float64)[cycle]
            independent = np.linalg.norm(coordinates - np.roll(coordinates, -1, axis=0), axis=1).sum()
            result["reference_encoding"] = "node permutation; optional repeated start; implicit closing edge"
        else:
            if sol[0] != 0 or sol[-1] != 0 or (sol < 0).any() or (sol > n).any():
                raise ValueError("bad depot encoding or out-of-range reference node")
            if not np.array_equal(np.sort(sol[sol != 0]), np.arange(1, n + 1)):
                raise ValueError("reference customer visit mismatch")
            coordinates = np.concatenate((np.asarray(task.depots, dtype=np.float64).reshape(-1, 2),
                                          np.asarray(task.points, dtype=np.float64)))
            if coordinates.shape != (n + 1, 2):
                raise ValueError("audit supports a single depot")
            independent = np.linalg.norm(coordinates[sol[1:]] - coordinates[sol[:-1]], axis=1).sum()
            result["reference_encoding"] = "0 depot separators, 1..N customers, explicit return to depot"
            result["raw_demands"] = True
            result["normalization_matches_raw_over_capacity"] = bool(np.allclose(task.norm_demands, task.demands / task.capacity))
        kit_value = float(task.evaluate(task.ref_sol))
        if problem == "TSP":
            from problems.tsp.validate import validate
            independent_validation = validate(task.points, task.ref_sol)
        elif problem == "CVRP":
            from problems.cvrp.validate import validate
            independent_validation = validate(task.depots, task.points, task.demands, task.capacity, task.ref_sol)
        else:
            from problems.cvrptw.validate import validate
            independent_validation = validate(task.depots, task.points, task.demands, task.capacity,
                                              task.tw, task.service, task.ref_sol, speed=1.0,
                                              time_tolerance=float(task.threshold))
        result["independent_validation"] = independent_validation
        result["reference"] = {"independent_objective": float(independent), "kit_objective": kit_value,
                               "kit_objective_source": "task.evaluate(task.ref_sol), not a stored cost label",
                               "kit_feasible": bool(task.check_constraints(task.ref_sol)),
                               "absolute_difference": abs(float(independent) - kit_value),
                               "objective_agrees": bool(np.isclose(independent, kit_value, rtol=1e-6, atol=1e-6)),
                               "independent_feasible": independent_validation["feasible"],
                               "validator_objective_agrees": bool(np.isclose(independent_validation["independent_objective"], independent, rtol=1e-12, atol=1e-12)) if independent_validation["independent_objective"] is not None else False,
                               "independent_tw_feasibility": independent_validation["feasible"] if problem == "CVRPTW" else "NOT_APPLICABLE"}
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs=3, action="append", default=[], metavar=("PROBLEM", "SIZE", "PATH"))
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("ML4CO_DATA_ROOT"))
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("BASELINE_ARTIFACT_ROOT", "artifacts")))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", default="local-copy-not-server-verified")
    parser.add_argument("--expected-manifest", type=Path, help="Compare file bytes/SHA against public_datasets.json")
    args = parser.parse_args()
    args.output = args.output or args.artifact_root / "audit/datasets.json"
    candidates = list(args.dataset)
    if args.dataset_root:
        # Names select manageable candidates, never establish task semantics.
        # Every selected file is subsequently checked by actual class and shape.
        for path in sorted(args.dataset_root.rglob("*.pkl")):
            match = re.search(r"(?:^|[^a-z])(cvrptw|cvrp|tsp)[_-]?(50|100)(?![0-9])", str(path).lower())
            if match:
                candidates.append((match[1].upper(), match[2], str(path)))
    if not args.dataset_root and not candidates:
        parser.error("provide --dataset-root / ML4CO_DATA_ROOT or explicit --dataset PROBLEM SIZE PATH")
    result = {"timestamp": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
              "python": sys.version, "executable": sys.executable, "label": args.label, "datasets": [],
              "dataset_root": str(args.dataset_root.resolve()) if args.dataset_root else None,
              "discovery_note": "Filename-filtered candidates, then verified by actual class and all coordinate shapes. Unusual names require explicit --dataset.",
              "root_exists": args.dataset_root.is_dir() if args.dataset_root else None}
    for problem, size, path in dict.fromkeys(tuple(c) for c in candidates):
        if problem not in ("TSP", "CVRP", "CVRPTW") or int(size) not in (50, 100):
            parser.error("supported scope: TSP/CVRP/CVRPTW and size 50/100")
        result["datasets"].append(audit(Path(path), problem, int(size)))
    if args.expected_manifest:
        expected = json.loads(args.expected_manifest.read_text())["files"]
        for row in result["datasets"]:
            matches = [item for item in expected if item["problem"] == row["problem"] and item["target_size"] == row["expected_size"]]
            row["manifest_identity_match"] = any(item["sha256"] == row.get("sha256") and item["bytes"] == row.get("file_size") for item in matches)
    result["coverage"] = [{"problem": p, "size": n,
                           "verified_files": [r["realpath"] for r in result["datasets"] if
                                              r["problem"] == p and r["expected_size"] == n and
                                              r.get("all_expected_size") and r.get("all_expected_task_class") and not r.get("error")],
                           "missing_meaning": "No verified candidate in this audit; not proof the server dataset does not exist"}
                          for p in ("TSP", "CVRP", "CVRPTW") for n in (50, 100)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
