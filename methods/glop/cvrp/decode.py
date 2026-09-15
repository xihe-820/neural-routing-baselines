"""Decode official pre-flatten CVRP sub-tour coordinates without repair."""
from __future__ import annotations

import numpy as np

from methods.glop.tsp.adapter import _coordinate_keys


def _cyclic_customer_runs(ids):
    """Split a depot-separated node-ID cycle without reordering customers."""
    customers = np.flatnonzero(ids != 0)
    if not len(customers):
        raise ValueError("official sub-tour contains no customer")
    depots = np.flatnonzero(ids == 0)
    if not len(depots):
        raise ValueError("official CVRP sub-tour contains no depot")
    rotated = np.roll(ids, -int(depots[0]))
    runs = []
    current = []
    for node in rotated:
        if node == 0:
            if current:
                runs.append(current)
                current = []
        else:
            current.append(int(node))
    if current:
        runs.append(current)
    return runs


def decode_subtour_coordinates(subtours, depot, points):
    values = np.asarray(subtours, dtype=np.float32)
    depot = np.asarray(depot, dtype=np.float32).reshape(1, 2)
    points = np.asarray(points, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("official CVRP sub-tours must have shape [routes,L,2]")
    coordinates = np.concatenate((depot, points), axis=0)
    keys = _coordinate_keys(coordinates)
    if len(set(keys)) != len(keys):
        raise ValueError("AMBIGUOUS_COORDINATE_IDENTITY")
    node_by_key = {key: index for index, key in enumerate(keys)}
    routes = []
    for row in values:
        try:
            ids = np.asarray([node_by_key[key] for key in _coordinate_keys(row)],
                             dtype=np.int64)
        except KeyError as exc:
            raise ValueError("official output coordinate has no exact node identity") from exc
        customer_runs = _cyclic_customer_runs(ids)
        routes.extend([0, *customers, 0] for customers in customer_runs)
    canonical = [0]
    for route in routes:
        canonical.extend(route[1:])
    return canonical, routes
