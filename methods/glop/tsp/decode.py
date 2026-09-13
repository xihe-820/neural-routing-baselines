"""Recover benchmark node ids from unchanged/reflected official GLOP tour coordinates."""
from __future__ import annotations

import numpy as np

from methods.glop.tsp.adapter import _coordinate_keys, apply_top_level_transform


def decode_coordinate_tour(tour_coordinates, original_points, *, allowed_transforms):
    output = np.asarray(tour_coordinates, dtype=np.float32)
    original = np.asarray(original_points, dtype=np.float32)
    if output.shape != original.shape or output.ndim != 2 or output.shape[1] != 2:
        raise ValueError("GLOP coordinate tour and original instance shapes differ")
    matches = []
    for transform in allowed_transforms:
        transformed = apply_top_level_transform(original, transform)
        keys = _coordinate_keys(transformed)
        if len(set(keys)) != len(keys):
            raise ValueError("transformed benchmark coordinates are not exactly unique")
        node_by_key = {key: index for index, key in enumerate(keys)}
        try:
            permutation = [node_by_key[key] for key in _coordinate_keys(output)]
        except KeyError:
            continue
        if sorted(permutation) == list(range(len(original))):
            matches.append((transform, permutation))
    if len(matches) != 1:
        raise ValueError("GLOP output has no unique exact augmentation/node-identity mapping")
    transform, permutation = matches[0]
    zero = permutation.index(0)
    rotated = permutation[zero:] + permutation[:zero]
    return rotated + [0], {
        "top_level_transform": transform,
        "exact_coordinate_bit_match": True,
        "rotation_to_node_zero": zero,
        "repair": False,
    }


def select_best_candidates(costs, coordinate_tours):
    costs = np.asarray(costs)
    tours = np.asarray(coordinate_tours)
    if costs.ndim != 2 or tours.ndim != 4 or tours.shape[:2] != costs.shape:
        raise ValueError("candidate costs/tours have incompatible shapes")
    best = np.argmin(costs, axis=0)
    selected = tours[best, np.arange(costs.shape[1])]
    reported = costs[best, np.arange(costs.shape[1])]
    return selected, reported, best
