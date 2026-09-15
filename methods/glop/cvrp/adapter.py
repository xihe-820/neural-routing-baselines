"""Exact ML4CO CVRP boundary for the official GLOP synthetic path."""
from __future__ import annotations

import numpy as np

from methods.glop.tsp.adapter import _coordinate_keys


def validate_instance(depot, points, demands, capacity, *, problem_size):
    depot = np.asarray(depot, dtype=np.float32).reshape(-1, 2)
    points = np.asarray(points, dtype=np.float32)
    demands = np.asarray(demands)
    capacity = float(capacity)
    if depot.shape != (1, 2) or points.shape != (problem_size, 2):
        raise ValueError("CVRP depot/customer coordinate shape mismatch")
    if demands.shape != (problem_size,):
        raise ValueError("CVRP demand shape mismatch")
    for name, value in (("depot", depot), ("points", points), ("demands", demands)):
        if not np.isfinite(value).all():
            raise ValueError(f"CVRP {name} contains non-finite values")
    if not np.isfinite(capacity) or capacity <= 0 or (demands < 0).any():
        raise ValueError("CVRP capacity/demand values are invalid")
    coordinates = np.concatenate((depot, points), axis=0)
    if len(set(_coordinate_keys(coordinates))) != problem_size + 1:
        raise ValueError("AMBIGUOUS_COORDINATE_IDENTITY")
    return depot[0], points, demands, capacity


def to_official_tensors(depot, points, demands, capacity, *, problem_size, device, torch):
    depot, points, demands, capacity = validate_instance(
        depot, points, demands, capacity, problem_size=problem_size)
    coordinates = torch.as_tensor(
        np.concatenate((depot[None], points), axis=0),
        dtype=torch.float32, device=device)
    native_demands = torch.as_tensor(
        np.concatenate((np.zeros(1, dtype=demands.dtype), demands)),
        dtype=torch.float32, device=device)
    return coordinates, native_demands, capacity
