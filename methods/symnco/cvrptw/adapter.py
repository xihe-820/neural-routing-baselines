"""ML4CO CVRPTW to the frozen standalone SymNCO raw TensorDict schema."""
from __future__ import annotations

import hashlib

import numpy as np

from common.cvrptw_formal import dataset_config


def historical_raw_instance_id(depot, points, demands, capacity, time_windows, service):
    """Recreate data.py's ID from ML4CO's float32 TXT serialization exactly."""
    fields = (depot, points, demands, np.asarray([float(capacity)]), time_windows, service)
    digest = hashlib.sha256()
    for field in fields:
        values = np.asarray(field).reshape(-1)
        serialized = np.asarray([float(str(value)) for value in values], dtype="<f8")
        digest.update(serialized.tobytes())
    return digest.hexdigest()


def raw_ids_from_native(native):
    coords = np.asarray(native["coords"])
    demands = np.asarray(native["demand_raw"])
    capacities = np.asarray(native["capacity"])
    tw = np.asarray(native["time_windows"])
    service = np.asarray(native["service_time"])
    return [historical_raw_instance_id(
        coords[index, 0], coords[index, 1:], demands[index, 1:], capacities[index],
        tw[index], service[index]) for index in range(coords.shape[0])]


def adapt_instance(depot, points, raw_demands, raw_capacity, time_windows,
                   service_times, *, problem_size):
    cfg = dataset_config(problem_size)
    depot = np.asarray(depot, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    demand = np.asarray(raw_demands, dtype=np.float32)
    tw = np.asarray(time_windows, dtype=np.float32)
    service = np.asarray(service_times, dtype=np.float32)
    capacity = float(raw_capacity)
    if (depot.shape != (2,) or points.shape != (problem_size, 2) or
            demand.shape != (problem_size,) or tw.shape != (problem_size + 1, 2) or
            service.shape != (problem_size + 1,)):
        raise ValueError("prepared SymNCO CVRPTW instance shape mismatch")
    if capacity != cfg["capacity"]:
        raise ValueError("prepared SymNCO CVRPTW capacity mismatch")
    if not all(np.isfinite(value).all() for value in (depot, points, demand, tw, service)):
        raise ValueError("prepared SymNCO CVRPTW instance contains nonfinite values")
    demand_with_depot = np.concatenate((np.zeros(1, dtype=np.float32), demand))
    native = {
        "coords": np.concatenate((depot[None], points), axis=0)[None],
        "demand_raw": demand_with_depot[None],
        "demand_norm": (demand_with_depot.astype(np.float64) / capacity).astype(np.float32)[None],
        "capacity": np.asarray([capacity], dtype=np.float32),
        "time_windows": tw[None], "service_time": service[None],
        "node_valid": np.ones((1, problem_size + 1), dtype=bool),
    }
    return native, {
        "coordinates": "raw float32, unchanged", "time_windows": "raw float32, unchanged",
        "service_time": "raw float32, unchanged",
        "demand": "raw demand plus depot zero; neural feature raw/capacity exactly once",
        "capacity": capacity,
        "instance_id": historical_raw_instance_id(
            depot, points, demand, capacity, tw, service),
    }
