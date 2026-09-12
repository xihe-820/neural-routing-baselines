"""Independent TSP permutation or successor-cycle validation."""
import numpy as np
from common.validation import failure, integer_ids, points2d
from problems.tsp.objective import cycle_length


def decode_successors(successors):
    ids = integer_ids(successors)
    n = len(ids)
    if not np.array_equal(np.sort(ids), np.arange(n)):
        raise ValueError("successor targets must be a permutation of node IDs")
    current, order, visited = 0, [], set()
    for _ in range(n):
        if current in visited:
            raise ValueError("disjoint subtours or premature successor cycle")
        visited.add(current)
        order.append(current)
        current = int(ids[current])
    if current != 0:
        raise ValueError("successor tour does not close")
    return order + [0]


def validate(points, solution, *, representation="tour"):
    try:
        coordinates = points2d(points)
        n = len(coordinates)
        if representation == "successors":
            if len(integer_ids(solution)) != n:
                raise ValueError("wrong successor count")
            route = integer_ids(decode_successors(solution))
        elif representation == "tour":
            route = integer_ids(solution)
        else:
            raise ValueError("representation must be tour or successors")
        closed = len(route) == n + 1 and route[0] == route[-1]
        order = route[:-1] if closed else route
        valid_range = bool(((order >= 0) & (order < n)).all())
        complete = len(order) == n and valid_range and np.array_equal(np.sort(order), np.arange(n))
        details = {"id_range": valid_range, "each_node_once": bool(complete),
                   "single_cycle": bool(complete), "closure": "explicit" if closed else "implicit",
                   "expected_nodes": n, "visited_entries": len(order)}
        return {"feasible": bool(complete),
                "independent_objective": cycle_length(coordinates, order) if complete else None,
                "solution": order.tolist() + [int(order[0])] if complete else None,
                "constraint_details": details}
    except (ValueError, TypeError, OverflowError) as exc:
        return failure(exc)
