"""ML4CO CVRP fields -> MVMoE CVRPEnv.load_problems tensor boundary."""
from __future__ import annotations

import numpy as np


def adapt_batch(depots, points, raw_demands, raw_capacities, *, device):
    """Normalize demands exactly once; perform no search, repair, or scoring."""
    import torch

    depot = np.asarray(depots)
    customers = np.asarray(points)
    demand = np.asarray(raw_demands)
    capacity = np.asarray(raw_capacities)
    if depot.ndim == 2 and depot.shape[1] == 2:
        depot = depot[:, None, :]
    if depot.ndim != 3 or depot.shape[1:] != (1, 2):
        raise ValueError("depots must have shape [B,2] or [B,1,2]")
    if customers.ndim != 3 or customers.shape[2] != 2:
        raise ValueError("points must have shape [B,N,2]")
    batch, size = customers.shape[:2]
    if size != 50:
        raise ValueError("this integration is intentionally limited to CVRP50")
    if depot.shape[0] != batch or demand.shape != (batch, size):
        raise ValueError("batch dimensions or demand shape do not match")
    if capacity.shape == (batch, 1):
        capacity = capacity[:, 0]
    if capacity.shape != (batch,):
        raise ValueError("raw_capacities must have shape [B] or [B,1]")
    for name, array in (("depot", depot), ("points", customers),
                        ("raw_demands", demand), ("raw_capacities", capacity)):
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite real values")
    if (demand < 0).any() or (capacity <= 0).any():
        raise ValueError("demands must be nonnegative and capacities positive")
    normalized = demand.astype(np.float64) / capacity.astype(np.float64)[:, None]
    native = (
        torch.as_tensor(depot, dtype=torch.float32, device=device),
        torch.as_tensor(customers, dtype=torch.float32, device=device),
        torch.as_tensor(normalized, dtype=torch.float32, device=device),
    )
    mapping = {
        "native_depot_id": 0,
        "benchmark_depot_id": 0,
        "customer_id_relation": "identity: native 1..N == benchmark 1..N",
        "augmentation_preserves_node_order": True,
        "normalization": "node_demand = raw_demand / raw_capacity exactly once",
    }
    return native, mapping
