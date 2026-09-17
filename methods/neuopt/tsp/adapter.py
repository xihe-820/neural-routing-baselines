"""Adapt the pinned ML4CO TSP100 coordinates to NeuOpt's native batch."""
from __future__ import annotations

import numpy as np

from methods.neuopt.tsp.config import supported_config


def adapt_batch(points, *, problem_size=100, device="cpu"):
    supported_config(problem_size)
    source = np.asarray(points)
    if source.ndim != 3 or source.shape[1:] != (problem_size, 2):
        raise ValueError("NeuOpt TSP input must have shape [batch, 100, 2]")
    if source.shape[0] <= 0:
        raise ValueError("NeuOpt TSP input batch must be nonempty")
    if not np.issubdtype(source.dtype, np.number) or not np.isfinite(source).all():
        raise ValueError("NeuOpt TSP coordinates must be finite numeric values")
    model_points = source.astype(np.float32, copy=False)
    import torch
    native = {
        "coordinates": torch.as_tensor(model_points, dtype=torch.float32, device=device),
    }
    return native, {
        "problem_size": problem_size,
        "source_shape": list(source.shape),
        "source_dtype": str(source.dtype),
        "model_input_dtype": "float32",
        "dtype_cast": source.dtype != np.dtype(np.float32),
        "coordinate_transformation": "none; no normalization, regeneration, or node reordering",
    }
