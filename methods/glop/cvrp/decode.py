"""Decode official pre-flatten CVRP sub-tour coordinates without repair."""
from __future__ import annotations

import numpy as np

from methods.glop.tsp.adapter import _coordinate_keys


def _cyclic_zero_run(ids):
    nonzero = np.flatnonzero(ids != 0)
    if not len(nonzero):
        raise ValueError("official sub-tour contains no customer")
    starts = [index for index in nonzero if ids[(index - 1) % len(ids)] == 0]
    if len(starts) != 1:
        raise ValueError("depot padding does not form one cyclic contiguous run")
    start = int(starts[0])
    rotated = np.roll(ids, -start)
    customer_count = int((rotated != 0).sum())
    if (rotated[:customer_count] == 0).any() or (rotated[customer_count:] != 0).any():
        raise ValueError("depot padding does not form one cyclic contiguous run")
    return rotated[:customer_count].astype(int).tolist()


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
        customers = _cyclic_zero_run(ids)
        routes.append([0, *customers, 0])
    canonical = [0]
    for route in routes:
        canonical.extend(route[1:])
    return canonical, routes
