"""Exact raw-demand ML4CO-to-SIL CVRP boundary and route decoding."""
from __future__ import annotations

import numpy as np


def _fields(task, *, problem_size: int):
    depot = np.asarray(task.depots)
    points = np.asarray(task.points)
    demands = np.asarray(task.demands)
    capacity = float(task.capacity)
    if depot.shape == (1, 2):
        depot = depot[0]
    if depot.shape != (2,) or points.shape != (problem_size, 2):
        raise ValueError("ML4CO CVRP depot/customer coordinate shape mismatch")
    if demands.shape != (problem_size,):
        raise ValueError("ML4CO CVRP raw demand shape mismatch")
    if any(value.dtype.kind not in "fiu" or not np.isfinite(value).all()
           for value in (depot, points, demands)):
        raise ValueError("ML4CO CVRP fields must be finite numeric arrays")
    if not np.isfinite(capacity) or capacity <= 0 or (demands < 0).any():
        raise ValueError("ML4CO CVRP raw demand/capacity is invalid")
    if (not np.allclose(demands, np.rint(demands), rtol=0.0, atol=0.0)
            or not float(capacity).is_integer()):
        raise ValueError(
            "SIL random insertion requires integral raw demands and true capacity")
    if (demands > capacity).any():
        raise ValueError("ML4CO CVRP contains a demand above true capacity")
    return depot, points, demands, capacity


def canonical_to_official(solution, *, problem_size: int) -> tuple[np.ndarray, np.ndarray]:
    route = np.asarray(solution)
    if route.ndim != 1 or route.dtype.kind not in "iu":
        raise ValueError("CVRP canonical reference must be a one-dimensional integer route")
    route = route.astype(np.int64, copy=False)
    if len(route) < 3 or route[0] != 0 or route[-1] != 0:
        raise ValueError("CVRP canonical reference must start and end at depot 0")
    if ((route[:-1] == 0) & (route[1:] == 0)).any():
        raise ValueError("CVRP canonical reference contains an empty route")
    customers = route[route != 0]
    if len(customers) != problem_size or not np.array_equal(
            np.sort(customers), np.arange(1, problem_size + 1)):
        raise ValueError("CVRP reference must visit every customer exactly once")
    flags = []
    previous = 0
    for node in route[1:]:
        if node == 0:
            previous = 0
            continue
        flags.append(1 if previous == 0 else 0)
        previous = int(node)
    return customers.astype(np.int64), np.asarray(flags, dtype=np.int64)


def adapt_task(task, *, problem_size: int):
    depot, points, demands, capacity = _fields(task, problem_size=problem_size)
    nodes, flags = canonical_to_official(task.ref_sol, problem_size=problem_size)
    coordinates = np.concatenate((depot.reshape(1, 2), points), axis=0)
    raw_demands = np.concatenate((np.zeros(1, dtype=demands.dtype), demands))
    reference = np.stack((nodes, flags), axis=1)
    return (
        depot, points, demands, capacity,
        coordinates.astype(np.float32, copy=False),
        raw_demands.astype(np.float32, copy=False),
        reference,
        {
            "source_coordinate_dtype": str(points.dtype),
            "model_coordinate_dtype": "float32",
            "source_demand_dtype": str(demands.dtype),
            "model_demand_dtype": "float32",
            "input_demand_representation": "raw benchmark demand",
            "normalization_before_official_model": False,
            "official_model_normalization": "raw demand / true per-instance capacity exactly once",
            "capacity_source": "float(task.capacity)",
            "capacity": capacity,
            "integral_demand_capacity_gate": True,
            "node_order": "depot then unchanged customer order",
            "synthetic_capacity_helper_used": False,
        },
    )


def inject_official_data(env, coordinates, raw_demands, capacities, references,
                         *, device, torch):
    coordinates = np.asarray(coordinates, dtype=np.float32)
    raw_demands = np.asarray(raw_demands, dtype=np.float32)
    capacities = np.asarray(capacities, dtype=np.float32)
    references = np.asarray(references, dtype=np.int64)
    batch, nodes, _ = coordinates.shape
    if (raw_demands.shape != (batch, nodes) or capacities.shape != (batch,)
            or references.shape != (batch, nodes - 1, 2)):
        raise ValueError("SIL CVRP injected tensors have inconsistent batch/size")
    env.raw_data_nodes = torch.as_tensor(coordinates, dtype=torch.float32, device=device)
    env.raw_data_demand = torch.as_tensor(raw_demands, dtype=torch.float32, device=device)
    env.raw_data_capacity = torch.as_tensor(capacities, dtype=torch.float32, device=device)
    env.raw_data_node_flag = torch.as_tensor(references, dtype=torch.long, device=device)
    env.raw_data_cost = torch.zeros(batch, dtype=torch.float32, device=device)


def decode_official_solution(solution, *, problem_size: int) -> dict:
    values = np.asarray(solution)
    if values.shape != (problem_size, 2) or values.dtype.kind not in "iu":
        raise ValueError("captured SIL CVRP solution must be integer [N,2]")
    nodes = values[:, 0].astype(np.int64)
    flags = values[:, 1].astype(np.int64)
    if not np.array_equal(np.sort(nodes), np.arange(1, problem_size + 1)):
        raise ValueError("captured SIL CVRP node order is not a customer permutation")
    if not np.isin(flags, (0, 1)).all() or not np.any(flags == 1):
        raise ValueError("captured SIL CVRP route-start flags are invalid")
    first = int(np.flatnonzero(flags == 1)[0])
    nodes = np.roll(nodes, -first)
    flags = np.roll(flags, -first)
    routes: list[list[int]] = []
    current: list[int] = []
    for node, starts_route in zip(nodes, flags):
        if starts_route:
            if current:
                routes.append(current)
            current = []
        current.append(int(node))
    if current:
        routes.append(current)
    canonical = [0]
    for route in routes:
        canonical.extend(route)
        canonical.append(0)
    return {
        "raw_solution": values.astype(np.int64).tolist(),
        "customer_visit_order": nodes.astype(int).tolist(),
        "route_start_flags": flags.astype(int).tolist(),
        "routes": routes,
        "canonical_solution": canonical,
    }
