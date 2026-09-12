"""Independent CVRPTW simulation with an explicit distance/time speed.

start_time is departure time at the depot, before the first travel arc.
Initial depot service is not added, matching the benchmark Kit evaluator.
Customer service starts within TW; return arrival must meet depot upper TW.
"""
import numpy as np
from common.validation import failure, scalar, vector
from problems.cvrp.objective import joined_coordinates
from problems.cvrp.validate import validate as validate_capacity


def validate(depot, points, demands, capacity, time_windows, service_times, solution,
             *, speed, start_time=0.0, time_tolerance=0.0, capacity_tolerance=0.0):
    result = validate_capacity(depot, points, demands, capacity, solution, capacity_tolerance=capacity_tolerance)
    if "routes" not in result:
        return result
    try:
        coords = joined_coordinates(depot, points)
        velocity = scalar(speed, "speed", positive=True)
        initial_time = scalar(start_time, "start_time")
        tolerance = scalar(time_tolerance, "time tolerance")
        service = vector(service_times, len(coords), "service times (including depot)")
        raw_tw = np.asarray(time_windows)
        if raw_tw.shape != (len(coords), 2) or raw_tw.dtype.kind not in "fiu":
            raise ValueError("time_windows must be [N+1,2], including depot")
        tw = raw_tw.astype(np.float64)
        if not np.isfinite(tw).all() or (tw < 0).any() or (tw[:, 0] > tw[:, 1]).any():
            raise ValueError("time windows must be finite, nonnegative, and start <= end")
        traces, violations = [], []
        if initial_time < tw[0, 0] - tolerance or initial_time > tw[0, 1] + tolerance:
            violations.append({"type": "depot_departure_window", "departure": initial_time})
        for route_id, route in enumerate(result["routes"]):
            clock, previous, events = initial_time, 0, []
            for node in route[1:-1]:
                travel = float(np.linalg.norm(coords[node] - coords[previous])) / velocity
                arrival = clock + travel
                service_start = max(arrival, float(tw[node, 0]))
                departure = service_start + float(service[node])
                late = service_start > tw[node, 1] + tolerance
                events.append({"node": node, "travel_time": travel, "arrival": arrival,
                               "waiting": service_start - arrival, "service_start": service_start,
                               "service_duration": float(service[node]), "departure": departure,
                               "tw_start": float(tw[node, 0]), "tw_end": float(tw[node, 1]),
                               "window_ok": not bool(late)})
                if late:
                    violations.append({"type": "late_service_start", "route": route_id, "node": node,
                                       "lateness": service_start - float(tw[node, 1])})
                clock, previous = departure, node
            return_travel = float(np.linalg.norm(coords[previous] - coords[0])) / velocity
            depot_arrival = clock + return_travel
            return_ok = depot_arrival <= tw[0, 1] + tolerance
            if not return_ok:
                violations.append({"type": "late_depot_return", "route": route_id,
                                   "lateness": depot_arrival - float(tw[0, 1])})
            traces.append({"route": route_id, "depot_departure": initial_time, "events": events,
                           "return_travel_time": return_travel, "depot_arrival": depot_arrival,
                           "depot_tw_end": float(tw[0, 1]), "depot_return_ok": bool(return_ok)})
        result["constraint_details"].update(speed=velocity, start_time=initial_time,
                                            time_tolerance=tolerance, time_windows_ok=not violations,
                                            time_violations=violations, route_timelines=traces)
        result["feasible"] = result["feasible"] and not violations
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        return failure(exc)
