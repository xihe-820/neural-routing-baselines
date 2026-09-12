"""Independent unrounded Euclidean TSP objective, accumulated in float64."""
import numpy as np
from common.validation import integer_ids, points2d


def cycle_length(points, tour):
    coordinates = points2d(points)
    route = integer_ids(tour)
    if (route < 0).any() or (route >= len(coordinates)).any():
        raise ValueError("node ID out of range")
    selected = coordinates[route]
    return float(np.linalg.norm(selected - np.roll(selected, -1, axis=0), axis=1).sum(dtype=np.float64))
