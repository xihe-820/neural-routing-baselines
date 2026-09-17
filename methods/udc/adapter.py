"""ML4CO TSP/CVRP500 boundary and independent UDC population validation."""
from __future__ import annotations

import numpy as np

from common.objective_agreement import objective_agrees
from problems.tsp.validate import validate as validate_tsp
from problems.cvrp.validate import validate as validate_cvrp


def adapt_tsp_task(task, *, size=500):
    points = np.asarray(task.points)
    if points.shape != (size, 2) or points.dtype.kind not in "fiu" or not np.isfinite(points).all():
        raise ValueError("ML4CO TSP task must contain finite [500,2] coordinates")
    return points.astype(np.float32, copy=False), {
        "coordinate_shape": list(points.shape), "coordinate_dtype": str(points.dtype),
        "coordinate_min": float(points.min()), "coordinate_max": float(points.max()),
        "transformation": "none", "node_order": "unchanged",
    }


def adapt_cvrp_task(task, *, size=500):
    depot = np.asarray(task.depots)
    points = np.asarray(task.points)
    demand = np.asarray(task.demands)
    capacity = float(task.capacity)
    if depot.shape == (1, 2):
        depot = depot[0]
    if depot.shape != (2,) or points.shape != (size, 2) or demand.shape != (size,):
        raise ValueError("ML4CO CVRP task field shapes differ from depot/[500,2]/[500]")
    if any(value.dtype.kind not in "fiu" or not np.isfinite(value).all()
           for value in (depot, points, demand)):
        raise ValueError("ML4CO CVRP fields must be finite real arrays")
    if capacity <= 1.0 or (demand < 0).any() or not np.allclose(demand, np.rint(demand)):
        raise ValueError("S3 requires raw nonnegative integer demands and original capacity > 1")
    normalized = demand.astype(np.float64) / capacity
    if (normalized > 1 + 1e-12).any():
        raise ValueError("a customer demand exceeds original capacity")
    coordinates = np.concatenate((depot.reshape(1, 2), points), axis=0).astype(np.float32)
    native_demand = np.concatenate(([0.0], normalized)).astype(np.float32)
    semantics = {
        "depot_shape": [2], "customer_coordinate_shape": list(points.shape),
        "coordinate_dtype": str(points.dtype), "raw_demand_dtype": str(demand.dtype),
        "raw_demand_min": float(demand.min()), "raw_demand_max": float(demand.max()),
        "capacity_original": capacity, "input_demand_representation": "raw_integer",
        "normalization_applied": "raw_demand / capacity exactly once",
        "normalized_demand_min": float(normalized.min()),
        "normalized_demand_max": float(normalized.max()), "node_order": "unchanged",
    }
    return coordinates, native_demand, semantics


def validate_tsp_population(points, population):
    tours = np.asarray(population)
    if tours.shape != (50, len(points)) or tours.dtype.kind not in "iu":
        raise ValueError("TSP final population must be integer [50,N]")
    rows = [validate_tsp(points, tour) for tour in tours]
    feasible = [i for i, row in enumerate(rows) if row["feasible"]]
    if not feasible:
        raise ValueError("TSP alpha population contains no feasible permutation")
    best = min(feasible, key=lambda i: (rows[i]["independent_objective"], i))
    selected = tours[best].astype(int).tolist()
    zero = selected.index(0)
    canonical = selected[zero:] + selected[:zero] + [0]
    return {"candidate_feasible": [row["feasible"] for row in rows],
            "independent_objective_per_alpha": [row["independent_objective"] for row in rows],
            "best_alpha": best, "best_objective": rows[best]["independent_objective"],
            "solution": canonical}


def decode_cyclic_routes(sequence, flags):
    nodes, boundary = np.asarray(sequence), np.asarray(flags)
    if nodes.ndim != 1 or boundary.shape != nodes.shape or nodes.dtype.kind not in "iu":
        raise ValueError("CVRP solution/flag must be equal-length integer vectors")
    if not np.isin(boundary, (0, 1)).all():
        raise ValueError("CVRP flags must be binary")
    ends = np.flatnonzero(boundary == 1)
    if not len(ends):
        raise ValueError("CVRP cyclic encoding contains no route boundary")
    start = (int(ends[-1]) + 1) % len(nodes)
    routes, route = [], []
    for offset in range(len(nodes)):
        index = (start + offset) % len(nodes)
        route.append(int(nodes[index]))
        if int(boundary[index]) == 1:
            routes.append(route); route = []
    if route:
        raise ValueError("CVRP cyclic route did not terminate at a boundary")
    return routes


def canonical_cvrp(routes):
    value = [0]
    for route in routes:
        value.extend(route); value.append(0)
    return value


def validate_cvrp_population(depot, points, raw_demand, capacity, solutions, flags):
    solutions, flags = np.asarray(solutions), np.asarray(flags)
    size = len(points)
    if solutions.shape != (50, size) or flags.shape != solutions.shape:
        raise ValueError("CVRP final solution/flag populations must be [50,N]")
    rows = []
    for solution, boundary in zip(solutions, flags):
        try:
            routes = decode_cyclic_routes(solution, boundary)
            canonical = canonical_cvrp(routes)
            checked = validate_cvrp(depot, points, raw_demand, capacity, canonical,
                                    capacity_tolerance=1e-5 * capacity)
        except (ValueError, TypeError) as exc:
            routes, canonical = None, None
            checked = {"feasible": False, "independent_objective": None,
                       "constraint_details": {"decode_error": str(exc)}}
        rows.append((routes, canonical, checked))
    feasible = [i for i, row in enumerate(rows) if row[2]["feasible"]]
    if not feasible:
        raise ValueError("CVRP alpha population contains no independently feasible candidate")
    best = min(feasible, key=lambda i: (rows[i][2]["independent_objective"], i))
    routes, canonical, checked = rows[best]
    loads = [float(np.asarray(raw_demand)[np.asarray(route)-1].sum()) for route in routes]
    return {"candidate_feasible": [row[2]["feasible"] for row in rows],
            "independent_objective_per_alpha": [row[2]["independent_objective"] for row in rows],
            "best_alpha": best, "best_objective": checked["independent_objective"],
            "solution": solutions[best].astype(int).tolist(),
            "solution_flag": flags[best].astype(int).tolist(),
            "canonical_solution": canonical, "decoded_routes": routes,
            "route_demands": loads, "route_count": len(routes),
            "max_route_load": max(loads), "capacity_original": float(capacity)}


def objective_comparison(left, right):
    absolute = abs(float(left) - float(right))
    return {"abs_diff": absolute,
            "rel_diff": absolute / max(abs(float(left)), abs(float(right)), 1e-300),
            "pass": objective_agrees(float(left), float(right))}
