"""Canonical scaled protocol identity for formal MVMoE CVRPTW evaluation."""
from __future__ import annotations

from methods.mvmoe.paper_config import paper_inference_config


SCALED_PROTOCOL_FIELDS = {
    "input_scaling": "continuous_official_style",
    "scaler_rule": "s=max(max(original coordinates), original_depot_tw_end/3.0)",
    "fields_scaled": ["coordinates", "time_windows", "service_times"],
    "demand_normalization": "raw_demand/raw_capacity exactly once",
    "loc_scaler": None,
    "distance_rounding": False,
    "model_inference_domain": "scaled_continuous",
    "final_validation_domain": "original_ml4co",
    "final_objective_domain": "original_ml4co",
    "speed": 1.0,
}


def unscaled_control_inference_config(problem_size):
    """Return the frozen identity of the verified CVRPTW A control."""
    return paper_inference_config(problem_size, problem="CVRPTW")


def scaled_paper_inference_config(problem_size):
    """Return the canonical formal CVRPTW B identity."""
    config = unscaled_control_inference_config(problem_size)
    config.update(SCALED_PROTOCOL_FIELDS)
    return config
