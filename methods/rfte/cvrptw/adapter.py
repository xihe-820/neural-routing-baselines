"""ML4CO CVRPTW to RF-TE native adapter."""
from common.cvrptw_formal import native_numpy_instance


def adapt_instance(*args, problem_size):
    native = native_numpy_instance(*args, problem_size=problem_size)
    return native, {
        "coordinates": "unchanged float32", "time_windows": "unchanged float32",
        "service_time": "unchanged float32", "demand": "raw/capacity exactly once",
        "vehicle_capacity": 1.0, "speed": 1.0,
    }
