"""Independent closed CVRP validator on raw demands and raw capacity."""
import numpy as np
from common.validation import failure, integer_ids, scalar, vector
from problems.cvrp.objective import joined_coordinates, route_distance


def validate(depot, points, demands, capacity, solution, *, capacity_tolerance=0.0):
    try:
        coords = joined_coordinates(depot, points)
        n = len(coords) - 1
        raw = vector(demands, n, "raw demands")
        cap = scalar(capacity, "raw capacity", positive=True)
        tolerance = scalar(capacity_tolerance, "capacity tolerance")
        route = integer_ids(solution)
        valid_range = bool(((route >= 0) & (route <= n)).all())
        endpoints = bool(route[0] == 0 and route[-1] == 0)
        empty = bool(((route[:-1] == 0) & (route[1:] == 0)).any())
        details = {"id_range": valid_range, "depot_endpoints": endpoints, "no_empty_routes": not empty,
                   "capacity": cap, "capacity_tolerance": tolerance}
        if not valid_range or not endpoints:
            return {"feasible": False, "independent_objective": None, "constraint_details": details}
        counts = np.bincount(route.astype(np.int64), minlength=n + 1)[1:]
        details.update(each_customer_once=bool((counts == 1).all()),
                       missing_customers=(np.flatnonzero(counts == 0) + 1).tolist(),
                       duplicate_customers=(np.flatnonzero(counts > 1) + 1).tolist())
        depots = np.flatnonzero(route == 0)
        routes = [route[a:b + 1].tolist() for a, b in zip(depots[:-1], depots[1:])]
        loads = [float(raw[np.asarray(r[1:-1], dtype=np.int64) - 1].sum(dtype=np.float64)) for r in routes]
        details.update(route_loads=loads, capacity_ok=all(load <= cap + tolerance for load in loads))
        feasible = details["each_customer_once"] and details["capacity_ok"] and not empty
        return {"feasible": bool(feasible), "independent_objective": route_distance(depot, points, route),
                "routes": routes, "constraint_details": details}
    except (ValueError, TypeError, OverflowError) as exc:
        return failure(exc)
