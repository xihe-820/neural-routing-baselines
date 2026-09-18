"""Shared fail-closed contract for the remaining CVRPTW paper baselines."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from common.objective_agreement import objective_agrees
from problems.cvrptw.validate import validate as validate_cvrptw


DATASETS = {
    50: {
        "filename": "cvrptw50_pyvrp-10s_16.038.pkl",
        "sha256": "a16975d9dd242739973191256e1cdbd8166c4405759ef195b985a8c3ec49df40",
        "count": 1000,
        "capacity": 40.0,
    },
    100: {
        "filename": "cvrptw100_pyvrp-20s_25.431.pkl",
        "sha256": "3b74fa520f42f7aa607a5bd00a7d4aaa118e0715ca1672ee854fc850bac67867",
        "count": 1000,
        "capacity": 50.0,
    },
}

TIMING_SEMANTICS = (
    "native original-instance inference-batch wall-clock seconds; batch size is recorded "
    "explicitly and latency is never divided by batch size; input conversion is outside the timed "
    "region; CUDA synchronize precedes perf_counter; the timed region contains official "
    "environment load/reset, the complete Aug8/all-customer-multistart autoregressive "
    "rollout and candidate selection, transfer of the selected reward/action/indices to CPU, "
    "then CUDA synchronize; checkpoint/model/dataset loading, warm-up, hashing, independent "
    "validation, ML4CO-Kit validation, artifact I/O, and aggregation are excluded"
)

DROP_DEFINITION = (
    "mean_i((independent_objective_i-reference_objective_i)"
    "/reference_objective_i*100)"
)


def dataset_config(problem_size: int) -> dict:
    try:
        return dict(DATASETS[int(problem_size)])
    except (KeyError, ValueError) as exc:
        raise ValueError("formal remaining-CVRPTW scope is exactly sizes 50 and 100") from exc


def validate_task(task, problem_size: int) -> None:
    """Validate the exact ML4CO task surface without relying on model packages."""
    cfg = dataset_config(problem_size)
    points = np.asarray(task.points)
    depot = np.asarray(task.depots)
    demand = np.asarray(task.demands)
    tw = np.asarray(task.tw)
    service = np.asarray(task.service)
    if type(task).__name__ != "CVRPTWTask":
        raise ValueError("dataset task must be an exact CVRPTWTask")
    expected = {
        "depot": (2,), "points": (problem_size, 2),
        "demand": (problem_size,), "tw": (problem_size + 1, 2),
        "service": (problem_size + 1,),
    }
    observed = {"depot": depot.shape, "points": points.shape,
                "demand": demand.shape, "tw": tw.shape, "service": service.shape}
    if observed != expected:
        raise ValueError(f"CVRPTW{problem_size} task schema mismatch: {observed}")
    if not all(np.isfinite(a).all() for a in (depot, points, demand, tw, service)):
        raise ValueError("CVRPTW task contains nonfinite values")
    if float(task.capacity) != cfg["capacity"]:
        raise ValueError("CVRPTW task capacity differs from pinned dataset")
    if float(service[0]) != 0.0 or not np.array_equal(tw[0], np.asarray([0.0, 4.6], dtype=tw.dtype)):
        raise ValueError("CVRPTW depot service/time-window semantics differ from benchmark")
    if bool(getattr(task, "cvrp_open", False)):
        raise ValueError("formal CVRPTW tasks must require depot return")


def native_numpy_instance(depot, points, raw_demands, raw_capacity,
                          time_windows, service_times, *, problem_size: int) -> dict:
    """Map original units to the common RF/CaDA/MoSES native MTVRP schema."""
    cfg = dataset_config(problem_size)
    depot = np.asarray(depot, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    demands = np.asarray(raw_demands, dtype=np.float32)
    tw = np.asarray(time_windows, dtype=np.float32)
    service = np.asarray(service_times, dtype=np.float32)
    capacity = float(raw_capacity)
    if (depot.shape != (2,) or points.shape != (problem_size, 2) or
            demands.shape != (problem_size,) or tw.shape != (problem_size + 1, 2) or
            service.shape != (problem_size + 1,)):
        raise ValueError("prepared CVRPTW instance shape mismatch")
    if capacity != cfg["capacity"]:
        raise ValueError("prepared CVRPTW capacity mismatch")
    arrays = (depot, points, demands, tw, service)
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("prepared CVRPTW instance contains nonfinite values")
    if np.any(demands <= 0) or np.any(demands > capacity):
        raise ValueError("raw customer demands are outside (0, capacity]")
    return {
        "locs": np.concatenate((depot[None, :], points), axis=0)[None, ...],
        "demand_linehaul": (demands / capacity)[None, ...],
        "demand_backhaul": np.zeros((1, problem_size), dtype=np.float32),
        "vehicle_capacity": np.ones((1, 1), dtype=np.float32),
        "capacity_original": np.asarray([[capacity]], dtype=np.float32),
        "time_windows": tw[None, ...],
        "service_time": service[None, ...],
        "open_route": np.zeros((1, 1), dtype=bool),
        "distance_limit": np.full((1, 1), np.inf, dtype=np.float32),
        "speed": np.ones((1, 1), dtype=np.float32),
        "backhaul_class": np.ones((1, 1), dtype=np.int32),
    }


def canonicalize_official_actions(actions, problem_size: int) -> list[int]:
    """Represent implicit depot closure; never repair customer/internal-depot order."""
    raw = np.asarray(actions)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError("selected official action must be a one-dimensional integer sequence")
    values = [int(value) for value in raw.tolist()]
    if any(value < 0 or value > problem_size for value in values):
        raise ValueError("selected official action contains an out-of-range node")
    while values and values[-1] == 0:
        values.pop()
    if not values:
        raise ValueError("selected official action contains no customer")
    return [0, *values, 0]


def validate_solution(arrays: dict, local_index: int, canonical_solution,
                      *, reported_objective: float) -> dict:
    """Run the shared independent gate in original ML4CO units."""
    tolerance = float(arrays["time_tolerances"][local_index])
    result = validate_cvrptw(
        arrays["depots"][local_index], arrays["points"][local_index],
        arrays["raw_demands"][local_index], arrays["raw_capacities"][local_index],
        arrays["time_windows"][local_index], arrays["service_times"][local_index],
        canonical_solution, speed=1.0,
        start_time=float(arrays["time_windows"][local_index, 0, 0]),
        time_tolerance=tolerance, capacity_tolerance=tolerance)
    independent = result.get("independent_objective")
    if independent is None or not math.isfinite(float(independent)):
        raise RuntimeError("independent CVRPTW validator did not produce an objective")
    if not result.get("feasible"):
        raise RuntimeError(f"independent CVRPTW feasibility failed: {result.get('error') or result.get('constraint_details')}")
    if not objective_agrees(reported_objective, independent):
        raise RuntimeError("official and independent CVRPTW objectives disagree")
    return result


def kit_validate(task, canonical_solution, independent_objective: float) -> dict:
    solution = np.asarray(canonical_solution, dtype=np.int64)
    feasible = bool(task.check_constraints(solution))
    objective = float(task.evaluate(solution))
    agrees = objective_agrees(objective, independent_objective)
    if not feasible or not agrees:
        raise RuntimeError("ML4CO-Kit CVRPTW validation failed")
    return {"kit_feasible": feasible, "kit_objective": objective,
            "kit_objective_agrees": agrees}


def instance_drop_percent(objective: float, reference: float) -> float:
    if not math.isfinite(objective) or not math.isfinite(reference) or reference <= 0:
        raise ValueError("objective/reference must be finite and reference positive")
    return (objective - reference) / reference * 100.0
