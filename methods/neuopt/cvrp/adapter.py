"""Map neutral raw CVRP arrays to the official NeuOpt native batch."""
from __future__ import annotations

import numpy as np

from methods.neuopt.cvrp.config import supported_config


def adapt_batch(depots, points, raw_demands, raw_capacities, *, problem_size, device="cpu"):
    import torch

    config = supported_config(problem_size)
    depots = np.asarray(depots, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    demands = np.asarray(raw_demands, dtype=np.float32)
    capacities = np.asarray(raw_capacities, dtype=np.float32)
    batch_size = len(points)
    if depots.shape == (batch_size, 1, 2):
        depots = depots[:, 0]
    if depots.shape != (batch_size, 2) or points.shape != (batch_size, problem_size, 2):
        raise ValueError("depot/customer coordinate shapes do not match the requested problem size")
    if demands.shape != (batch_size, problem_size) or capacities.shape != (batch_size,):
        raise ValueError("demand/capacity shapes do not match the requested problem size")
    if not all(np.isfinite(value).all() for value in (depots, points, demands, capacities)):
        raise ValueError("NeuOpt adapter input contains NaN or Inf")
    if (demands < 0).any() or not np.all(capacities == config["capacity"]):
        raise ValueError(f"raw demands must be nonnegative and capacity exactly {config['capacity']:g}")

    dummy = np.repeat(depots[:, None, :], config["dummy_size"], axis=1)
    coordinates = np.concatenate((dummy, points), axis=1)
    normalized = demands / capacities[:, None]
    native_demands = np.concatenate(
        (np.zeros((batch_size, config["dummy_size"]), dtype=np.float32), normalized), axis=1
    )
    if coordinates.shape[1] != config["sequence_length"]:
        raise AssertionError("constructed sequence length contradicts the pinned size table")
    if not np.array_equal(coordinates[:, :config["dummy_size"]], dummy):
        raise AssertionError("dummy depot construction changed coordinates")
    if not np.array_equal(coordinates[:, config["dummy_size"]:], points):
        raise AssertionError("adapter changed customer order")
    if not np.array_equal(native_demands[:, config["dummy_size"]:], normalized):
        raise AssertionError("adapter demand normalization is not exactly raw/capacity")
    native = {
        "coordinates": torch.as_tensor(coordinates, dtype=torch.float32, device=device),
        "demand": torch.as_tensor(native_demands, dtype=torch.float32, device=device),
    }
    mapping = {
        "problem_size": problem_size,
        "dummy_size": config["dummy_size"],
        "sequence_length": config["sequence_length"],
        "coordinates": "20 exact depot copies followed by benchmark customers in original order; no scaling",
        "normalization": "dummy demands are zero; real raw demands divided by raw capacity exactly once",
        "internal_to_benchmark": "internal ids <20 map to depot 0; internal id i>=20 maps to customer i-19",
    }
    return native, mapping
