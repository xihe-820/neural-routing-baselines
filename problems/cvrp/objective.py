"""Independent depot-return route distance. This function does not prove coverage."""
import numpy as np
from common.validation import integer_ids, points2d


def joined_coordinates(depot, points):
    customers = points2d(points)
    d = np.asarray(depot)
    if d.shape == (2,):
        d = d[None, :]
    depots = points2d(d, "depot")
    if depots.shape != (1, 2):
        raise ValueError("exactly one depot is required")
    return np.concatenate((depots, customers), axis=0)


def route_distance(depot, points, solution):
    coords = joined_coordinates(depot, points)
    route = integer_ids(solution)
    if route[0] != 0 or route[-1] != 0 or (route < 0).any() or (route >= len(coords)).any():
        raise ValueError("invalid depot endpoints or node range")
    return float(np.linalg.norm(coords[route[1:]] - coords[route[:-1]], axis=1).sum(dtype=np.float64))
