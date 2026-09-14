"""Continuous unit scaling shared by MVMoE CVRPTW audit and formal inference."""
from __future__ import annotations

import numpy as np


def scale_instance(depot, points, raw_demands, raw_capacity, time_windows,
                   service_times):
    """Scale spatial and time units by the official external-path bound."""
    depot = np.asarray(depot)
    points = np.asarray(points)
    demands = np.asarray(raw_demands)
    capacity = np.asarray(raw_capacity)
    tw = np.asarray(time_windows)
    service = np.asarray(service_times)
    if depot.shape == (1, 2):
        depot = depot[0]
    if depot.shape != (2,) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("depot/points must have shapes [2] and [N,2]")
    problem_size = len(points)
    if demands.shape != (problem_size,) or capacity.ndim != 0:
        raise ValueError("raw demands/capacity shapes do not match points")
    if tw.shape != (problem_size + 1, 2):
        raise ValueError("time_windows must have shape [N+1,2]")
    if service.shape != (problem_size + 1,):
        raise ValueError("service_times must have shape [N+1]")
    named = (("depot", depot), ("points", points), ("raw_demands", demands),
             ("raw_capacity", capacity), ("time_windows", tw),
             ("service_times", service))
    if any(value.dtype.kind not in "fiu" or not np.isfinite(value).all()
           for _, value in named):
        raise ValueError("all instance fields must contain finite real values")
    coordinates = np.concatenate((depot[None, :], points), axis=0)
    if (coordinates < 0).any():
        raise ValueError("official coordinate-maximum rule requires nonnegative coordinates")
    coordinate_max = float(coordinates.max())
    depot_start, depot_end = (float(tw[0, 0]), float(tw[0, 1]))
    scaler = max(coordinate_max, depot_end / 3.0)
    if not np.isfinite(scaler) or scaler <= 0:
        raise ValueError("scaler must be finite and positive")

    # Float32 matches the official environment boundary. Demand and node order are
    # copied without scaling; adapt_batch performs the sole capacity normalization.
    result = {
        "depot": (depot.astype(np.float64) / scaler).astype(np.float32),
        "points": (points.astype(np.float64) / scaler).astype(np.float32),
        "raw_demands": demands.copy(),
        "raw_capacity": float(capacity),
        "time_windows": (tw.astype(np.float64) / scaler).astype(np.float32),
        "service_times": (service.astype(np.float64) / scaler).astype(np.float32),
        "scaler": scaler,
        "coordinate_max": coordinate_max,
        "depot_tw_start": depot_start,
        "depot_tw_end": depot_end,
    }
    scaled_coordinates = np.concatenate(
        (result["depot"][None, :], result["points"]), axis=0)
    scaled_values = (
        scaled_coordinates, result["time_windows"], result["service_times"])
    if not all(np.isfinite(value).all() for value in scaled_values):
        raise ValueError("scaled coordinates/time windows/service times must be finite")
    result["scaled_coordinate_max"] = float(scaled_coordinates.max())
    result["scaled_depot_tw_end"] = float(result["time_windows"][0, 1])
    if (result["scaled_coordinate_max"] > 1.0 + 1e-6 or
            result["scaled_depot_tw_end"] <= 0.0 or
            result["scaled_depot_tw_end"] > 3.0 + 1e-6):
        raise ValueError("scaled coordinate/depot-horizon bounds are invalid")
    return result


def scale_prepared_instance(arrays, index):
    """Apply the shared scaler to one original-domain prepared NPZ row."""
    return scale_instance(
        arrays["depots"][index], arrays["points"][index],
        arrays["demands"][index], arrays["capacities"][index],
        arrays["time_windows"][index], arrays["service_times"][index])


def assert_continuous_env(env):
    """Fail closed if official per-edge distance rounding could be active."""
    if not hasattr(env, "loc_scaler") or env.loc_scaler is not None:
        raise RuntimeError("continuous CVRPTW scaling requires env.loc_scaler is None")
