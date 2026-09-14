import unittest
from types import SimpleNamespace

import numpy as np

from common.objective_agreement import objective_agrees
from methods.mvmoe.cvrptw.adapter import adapt_batch
from methods.mvmoe.cvrptw.scaling import assert_continuous_env, scale_instance
from methods.mvmoe.cvrptw.scaling_audit import _slack_diagnostics, summarize
from problems.cvrp.objective import route_distance
from problems.cvrptw.validate import validate


def instance(problem_size=2):
    depot = np.asarray([0.0, 0.0], dtype=np.float32)
    points = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    if problem_size != 2:
        points = np.stack((np.linspace(0.01, 0.99, problem_size),
                           np.linspace(0.99, 0.01, problem_size)), axis=1).astype(np.float32)
    demands = np.full(problem_size, 8.0, dtype=np.float32)
    tw = np.zeros((problem_size + 1, 2), dtype=np.float32)
    tw[0] = [0.0, 4.6]
    tw[1:, 1] = 4.0
    service = np.zeros(problem_size + 1, dtype=np.float32)
    service[1:] = 0.16
    return depot, points, demands, np.float32(40.0), tw, service


class ContinuousScalingTests(unittest.TestCase):
    def test_uniform_scaling_and_depot_horizon(self):
        data = instance()
        scaled = scale_instance(*data)
        expected = np.float32(4.6) / 3.0
        self.assertAlmostEqual(scaled["scaler"], expected)
        np.testing.assert_allclose(scaled["depot"], data[0] / expected)
        np.testing.assert_allclose(scaled["points"], data[1] / expected)
        np.testing.assert_allclose(scaled["time_windows"], data[4] / expected)
        np.testing.assert_allclose(scaled["service_times"], data[5] / expected)
        self.assertAlmostEqual(float(scaled["time_windows"][0, 1]), 3.0, places=6)

    def test_demand_is_not_scaled_and_adapter_normalizes_once(self):
        data = instance(problem_size=50)
        scaled = scale_instance(*data)
        np.testing.assert_array_equal(scaled["raw_demands"], data[2])
        native, _, mapping = adapt_batch(
            scaled["depot"][None], scaled["points"][None],
            scaled["raw_demands"][None], np.asarray([scaled["raw_capacity"]]),
            scaled["time_windows"][None], scaled["service_times"][None],
            problem_size=50, device="cpu")
        np.testing.assert_allclose(native[2].numpy(), np.full((1, 50), 0.2))
        self.assertIn("exactly once", mapping["normalization"])

    def test_loc_scaler_must_be_none(self):
        assert_continuous_env(SimpleNamespace(loc_scaler=None))
        with self.assertRaisesRegex(RuntimeError, "loc_scaler is None"):
            assert_continuous_env(SimpleNamespace(loc_scaler=1.5))
        with self.assertRaisesRegex(RuntimeError, "loc_scaler is None"):
            assert_continuous_env(SimpleNamespace())

    def test_objective_equivalence_and_node_order(self):
        data = instance()
        scaled = scale_instance(*data)
        route = [0, 1, 0, 2, 0]
        original = route_distance(data[0], data[1], route)
        scaled_distance = route_distance(scaled["depot"], scaled["points"], route)
        self.assertTrue(objective_agrees(scaled_distance * scaled["scaler"], original))
        np.testing.assert_allclose(
            scaled["points"] * scaled["scaler"], data[1], rtol=1e-6, atol=1e-6)
        self.assertEqual(route, [0, 1, 0, 2, 0])

    def test_waiting_route_feasibility_is_equivalent(self):
        depot = np.asarray([0.0, 0.0], dtype=np.float32)
        points = np.asarray([[1.0, 0.0]], dtype=np.float32)
        demands = np.asarray([1.0], dtype=np.float32)
        tw = np.asarray([[0.0, 5.0], [2.0, 3.0]], dtype=np.float32)
        service = np.asarray([0.0, 0.5], dtype=np.float32)
        route = [0, 1, 0]
        scaled = scale_instance(depot, points, demands, 10.0, tw, service)
        original_result = validate(
            depot, points, demands, 10.0, tw, service, route,
            speed=1.0, time_tolerance=0.0)
        scaled_result = validate(
            scaled["depot"], scaled["points"], demands, 10.0,
            scaled["time_windows"], scaled["service_times"], route,
            speed=1.0, time_tolerance=0.0)
        self.assertTrue(original_result["feasible"])
        self.assertTrue(scaled_result["feasible"])
        self.assertGreater(
            original_result["constraint_details"]["route_timelines"][0]["events"][0]["waiting"],
            0.0)

    def test_official_coordinate_maximum_dominates_when_larger(self):
        data = list(instance())
        data[1][0, 0] = 6.0
        scaled = scale_instance(*data)
        self.assertEqual(scaled["scaler"], 6.0)
        self.assertAlmostEqual(float(scaled["points"].max()), 1.0)

    def test_summary_is_paired_and_epsilon_diagnostic_is_explicit(self):
        def record(a, b, same, independent, kit, error, runtime, near):
            return {
                "comparison_to_A": {
                    "A_original_objective": a, "B_original_objective": b,
                    "A_gap": a, "B_gap": b, "same_route_as_A": same,
                    "A_runtime_seconds": 0.5,
                },
                "validation": {
                    "independent_feasible": independent, "kit_feasible": kit},
                "objectives": {"scaled_times_s_original_abs_error": error},
                "runtime_seconds": runtime,
                "epsilon_diagnostics": {"within_10x_official_epsilon": near},
            }
        result = summarize([
            record(10.0, 9.0, False, True, True, 2e-6, 0.4, False),
            record(10.0, 10.0, True, True, True, 1e-6, 0.6, True),
        ])
        self.assertEqual(result["B_better_count"], 1)
        self.assertEqual(result["tie_count"], 1)
        self.assertEqual(result["same_route_count"], 1)
        self.assertEqual(result["B_independent_feasible_count"], 2)
        self.assertEqual(result["B_kit_feasible_count"], 2)
        self.assertEqual(result["max_scaled_times_s_original_abs_error"], 2e-6)
        self.assertEqual(result["B_mean_runtime_seconds"], 0.5)
        self.assertEqual(result["epsilon_boundary_observation_count"], 1)

        validation = {"constraint_details": {"route_timelines": [{
            "depot_tw_end": 4.6, "depot_arrival": 4.5,
            "events": [{"tw_end": 2.0, "service_start": 1.99999}],
        }]}}
        diagnostic = _slack_diagnostics(validation, 4.6 / 3.0)
        self.assertFalse(diagnostic["uses_positive_official_epsilon_to_pass"])
        self.assertTrue(diagnostic["within_10x_official_epsilon"])


if __name__ == "__main__":
    unittest.main()
