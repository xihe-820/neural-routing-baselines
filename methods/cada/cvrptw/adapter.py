"""ML4CO CVRPTW to CaDA native adapter."""
import numpy as np

from common.cvrptw_formal import native_numpy_instance


def adapt_instance(*args, problem_size):
    native = native_numpy_instance(*args, problem_size=problem_size)
    # Unlike the RL4CO RouteFinder/MoSES env, pinned CaDA _reset does not add
    # depot demand rows: its loader/generator already supplies them.
    native["demand_linehaul"] = np.concatenate(
        (np.zeros((1, 1), dtype=np.float32), native["demand_linehaul"]), axis=1)
    native["demand_backhaul"] = np.zeros((1, problem_size + 1), dtype=np.float32)
    native["p_s_tag"] = np.asarray(
        [[1.0, 0.0, 1.0, 0.0, 0.0, problem_size / 2000.0]], dtype="float32")
    return native, {
        "coordinates": "unchanged float32", "time_windows": "unchanged float32",
        "service_time": "unchanged float32", "demand": "raw/capacity exactly once",
        "task_prompt_C_O_TW_L_B": [1, 0, 1, 0, 0], "size_feature": problem_size / 2000.0,
        "depot_demand_row": "explicit, required by pinned CaDA _reset",
    }
