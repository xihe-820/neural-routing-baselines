"""ML4CO CVRPTW fields -> MVMoE VRPTWEnv.load_problems boundary."""
from __future__ import annotations

import numpy as np

from methods.mvmoe.cvrptw.config import PROBLEM_SIZE, require_problem_size


def adapt_batch(depots, points, raw_demands, raw_capacities, time_windows,
                service_times, *, problem_size, device):
    """Split depot/customer rows and normalize demand exactly once."""
    import torch

    require_problem_size(problem_size)
    depot = np.asarray(depots)
    customers = np.asarray(points)
    demand = np.asarray(raw_demands)
    capacity = np.asarray(raw_capacities)
    tw = np.asarray(time_windows)
    service = np.asarray(service_times)
    if depot.ndim == 2 and depot.shape[1] == 2:
        depot = depot[:, None, :]
    if depot.ndim != 3 or depot.shape[1:] != (1, 2):
        raise ValueError("depots must have shape [B,2] or [B,1,2]")
    if customers.ndim != 3 or customers.shape[1:] != (PROBLEM_SIZE, 2):
        raise ValueError("points must have shape [B,50,2]")
    batch = customers.shape[0]
    if depot.shape[0] != batch or demand.shape != (batch, PROBLEM_SIZE):
        raise ValueError("batch dimensions or demand shape do not match")
    if capacity.shape == (batch, 1):
        capacity = capacity[:, 0]
    if capacity.shape != (batch,):
        raise ValueError("raw_capacities must have shape [B] or [B,1]")
    if tw.shape != (batch, PROBLEM_SIZE + 1, 2):
        raise ValueError("time_windows must have shape [B,51,2], including depot")
    if service.shape != (batch, PROBLEM_SIZE + 1):
        raise ValueError("service_times must have shape [B,51], including depot")
    for name, array in (("depot", depot), ("points", customers),
                        ("raw_demands", demand), ("raw_capacities", capacity),
                        ("time_windows", tw), ("service_times", service)):
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite real values")
    if (demand < 0).any() or (capacity <= 0).any():
        raise ValueError("demands must be nonnegative and capacities positive")
    if (tw < 0).any() or (tw[:, :, 0] > tw[:, :, 1]).any():
        raise ValueError("time windows must be nonnegative with start <= end")
    if (service < 0).any():
        raise ValueError("service times must be nonnegative")
    if not np.all(service[:, 0] == 0):
        raise ValueError("benchmark depot service must be zero")
    depot_windows = tw[:, 0, :]
    if not np.all(depot_windows == depot_windows[0]):
        raise ValueError("batch contains different depot time windows")

    normalized_demand = demand.astype(np.float64) / capacity.astype(np.float64)[:, None]
    native = (
        torch.as_tensor(depot, dtype=torch.float32, device=device),
        torch.as_tensor(customers, dtype=torch.float32, device=device),
        torch.as_tensor(normalized_demand, dtype=torch.float32, device=device),
        torch.as_tensor(service[:, 1:], dtype=torch.float32, device=device),
        torch.as_tensor(tw[:, 1:, 0], dtype=torch.float32, device=device),
        torch.as_tensor(tw[:, 1:, 1], dtype=torch.float32, device=device),
    )
    depot_window = (float(depot_windows[0, 0]), float(depot_windows[0, 1]))
    mapping = {
        "native_depot_id": 0,
        "benchmark_depot_id": 0,
        "customer_id_relation": "identity: native 1..50 == benchmark 1..50",
        "depot_rows_removed_from_customer_service_and_time_windows": True,
        "augmentation_preserves_node_order": True,
        "normalization": "node_demand = raw_demand / raw_capacity exactly once",
        "coordinate_scaling": "none",
        "time_window_scaling": "none",
        "service_time_scaling": "none",
        "problem_size": PROBLEM_SIZE,
        "depot_time_window": list(depot_window),
    }
    return native, depot_window, mapping
