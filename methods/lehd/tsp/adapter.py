"""Exact ML4CO-to-LEHD TSP boundary without scaling, padding, or reordering."""
from __future__ import annotations

import numpy as np


def canonical_tour(solution, *, problem_size: int) -> list[int]:
    route = np.asarray(solution)
    if route.ndim != 1 or route.dtype.kind not in "iu":
        raise ValueError("LEHD TSP solution must be a one-dimensional integer route")
    route = route.astype(np.int64, copy=False)
    if len(route) == problem_size + 1 and route[0] == route[-1]:
        route = route[:-1]
    if len(route) != problem_size or not np.array_equal(
            np.sort(route), np.arange(problem_size)):
        raise ValueError("LEHD TSP solution must contain every node exactly once")
    zero = int(np.flatnonzero(route == 0)[0])
    ordered = np.roll(route, -zero).tolist()
    return [int(value) for value in ordered] + [0]


def adapt_task(task, *, problem_size: int):
    source = np.asarray(task.points)
    if (source.shape != (problem_size, 2) or source.dtype.kind not in "fiu"
            or not np.isfinite(source).all()):
        raise ValueError(f"ML4CO TSP task must contain finite [{problem_size},2] coordinates")
    canonical_reference = canonical_tour(task.ref_sol, problem_size=problem_size)
    model_points = source.astype(np.float32, copy=False)
    reference_order = np.asarray(canonical_reference[:-1], dtype=np.int64)
    return source, model_points, reference_order, {
        "source_coordinate_dtype": str(source.dtype),
        "model_input_dtype": str(model_points.dtype),
        "model_input_dtype_cast": source.dtype != np.dtype(np.float32),
        "coordinate_shape": list(source.shape),
        "coordinate_transformation": "none except official float32 tensor boundary",
        "node_order": "unchanged",
        "padding": False,
        "independent_objective_coordinate_source": "original np.asarray(task.points)",
        "reference_usage": "official objective interface only; never model inference",
    }


def inject_official_data(env, model_points, reference_order, *, device, torch):
    points = np.asarray(model_points, dtype=np.float32)
    reference = np.asarray(reference_order, dtype=np.int64)
    if points.ndim != 3 or reference.shape != points.shape[:2]:
        raise ValueError("LEHD TSP injected tensors have inconsistent batch/size")
    env.raw_data_nodes = torch.as_tensor(points, dtype=torch.float32, device=device)
    env.raw_data_tours = torch.as_tensor(reference, dtype=torch.long, device=device)


def decode_official_solution(solution, *, problem_size: int) -> dict:
    values = np.asarray(solution)
    if values.shape != (problem_size,):
        raise ValueError("captured LEHD TSP solution has an unexpected shape")
    canonical = canonical_tour(values, problem_size=problem_size)
    return {
        "raw_solution": values.astype(np.int64).tolist(),
        "canonical_solution": canonical,
    }
