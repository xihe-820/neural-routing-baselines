"""Neutral-array boundary and permutation provenance for GLOP TSP."""
from __future__ import annotations

import numpy as np

from methods.glop.tsp.config import supported_config


def _coordinate_keys(points):
    values = np.ascontiguousarray(points, dtype=np.float32)
    return [tuple(row.view(np.uint32).tolist()) for row in values]


def apply_top_level_transform(points, transform):
    points = np.asarray(points, dtype=np.float32)
    result = points.copy()
    if transform in ("reflect_x", "reflect_xy"):
        result[..., 0] = np.float32(1.0) - result[..., 0]
    if transform in ("reflect_y", "reflect_xy"):
        result[..., 1] = np.float32(1.0) - result[..., 1]
    if transform not in ("identity", "reflect_x", "reflect_y", "reflect_xy"):
        raise ValueError(f"unknown GLOP top-level transform: {transform}")
    return result


def validate_initial_permutations(permutations, *, problem_size, width, batch_size):
    values = np.asarray(permutations)
    if values.shape != (width, batch_size, problem_size):
        raise ValueError("initial permutation tensor has an unexpected shape")
    expected = np.arange(problem_size)
    if not all(np.array_equal(np.sort(row), expected) for row in values.reshape(-1, problem_size)):
        raise ValueError("initial insertion order has duplicate or missing node ids")
    return True


def apply_top_level_reflections(seeds, *, transforms, expected_candidate_count):
    """Apply the pinned main.py reflection order without changing RI seeds."""
    if seeds.ndim != 3 or seeds.shape[-1] != 2:
        raise ValueError("TSP seed tensor has an unexpected shape")
    reflected = []
    for transform in transforms:
        if transform == "identity":
            candidate = seeds
        elif transform == "reflect_x":
            candidate = seeds.clone()
            candidate[..., 0] = 1 - candidate[..., 0]
        elif transform == "reflect_y":
            candidate = seeds.clone()
            candidate[..., 1] = 1 - candidate[..., 1]
        elif transform == "reflect_xy":
            candidate = 1 - seeds
        else:
            raise ValueError(f"unknown GLOP top-level transform: {transform}")
        reflected.append(candidate)
    if not reflected:
        raise ValueError("at least one top-level transform is required")
    import torch
    candidates = torch.cat(reflected, dim=0)
    if len(candidates) != expected_candidate_count:
        raise ValueError("effective TSP candidate count differs from protocol")
    return candidates


def adapt_points(points, *, problem_size, device="cpu", top_level_transforms=None):
    import torch

    if top_level_transforms is None:
        config = supported_config(problem_size)
        transforms = config["top_level_transforms"]
    else:
        transforms = list(top_level_transforms)
        if not transforms:
            raise ValueError("at least one top-level transform is required")
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (problem_size, 2):
        raise ValueError("TSP points do not match the requested problem size")
    if not np.isfinite(values).all():
        raise ValueError("TSP points contain NaN or Inf")
    for row in values:
        if len(set(_coordinate_keys(row))) != problem_size:
            raise ValueError("exact coordinate identity is ambiguous within a benchmark instance")
        signatures = {
            tuple(sorted(_coordinate_keys(apply_top_level_transform(row, transform))))
            for transform in transforms
        }
        if len(signatures) != len(transforms):
            raise ValueError("top-level augmentation identity is ambiguous for this instance")
    return torch.as_tensor(values, dtype=torch.float32, device=device), {
        "problem_size": problem_size,
        "coordinate_cast": "benchmark float32 preserved exactly",
        "node_identity": "original node id retained by exact float32 coordinate-bit identity",
        "top_level_transforms": transforms,
        "posthoc_tolerance_matching": False,
        "repair": False,
    }
