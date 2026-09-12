import unittest

import numpy as np

from methods.mvmoe.cvrp.adapter import adapt_batch
from methods.mvmoe.cvrp.decode import decode_selected_nodes, select_best_candidates
from problems.cvrp.validate import validate


class MVMoECVRPAdapterTests(unittest.TestCase):
    def test_raw_demands_are_normalized_exactly_once(self):
        depots = np.asarray([[0.1, 0.2]], dtype=np.float32)
        points = np.zeros((1, 50, 2), dtype=np.float32)
        demands = np.arange(1, 51, dtype=np.float32)[None, :]
        native, mapping = adapt_batch(depots, points, demands, [40], device="cpu")
        depot_xy, node_xy, node_demand = native
        self.assertEqual(tuple(depot_xy.shape), (1, 1, 2))
        self.assertEqual(tuple(node_xy.shape), (1, 50, 2))
        self.assertEqual(tuple(node_demand.shape), (1, 50))
        np.testing.assert_allclose(node_demand.numpy(), demands / 40)
        self.assertAlmostEqual(float(node_demand[0, 39]), 1.0)
        self.assertIn("exactly once", mapping["normalization"])

    def test_adapter_rejects_other_size(self):
        with self.assertRaisesRegex(ValueError, "CVRP50"):
            adapt_batch([[0, 0]], np.zeros((1, 49, 2)), np.ones((1, 49)), [40], device="cpu")


class MVMoECVRPDecoderTests(unittest.TestCase):
    def test_single_route(self):
        self.assertEqual(decode_selected_nodes([0, 1, 2, 0], problem_size=2)[0], [0, 1, 2, 0])

    def test_multiple_routes_and_internal_separators(self):
        decoded, info = decode_selected_nodes([0, 1, 0, 2, 0], problem_size=2)
        self.assertEqual(decoded, [0, 1, 0, 2, 0])
        self.assertEqual(info["removed_finish_padding"], 0)

    def test_one_and_multiple_finish_padding_zeros(self):
        one, info_one = decode_selected_nodes([0, 1, 2, 0, 0], problem_size=2)
        many, info_many = decode_selected_nodes([0, 1, 2, 0, 0, 0], problem_size=2)
        self.assertEqual(one, [0, 1, 2, 0])
        self.assertEqual(many, [0, 1, 2, 0])
        self.assertEqual(info_one["removed_finish_padding"], 1)
        self.assertEqual(info_many["removed_finish_padding"], 2)

    def test_duplicate_or_missing_is_not_repaired(self):
        duplicate = decode_selected_nodes([0, 1, 1, 0], problem_size=2)[0]
        missing = decode_selected_nodes([0, 1, 0], problem_size=2)[0]
        self.assertEqual(duplicate, [0, 1, 1, 0])
        self.assertEqual(missing, [0, 1, 0])
        points = [[1, 0], [2, 0]]
        self.assertFalse(validate([0, 0], points, [1, 1], 2, duplicate)["feasible"])
        self.assertFalse(validate([0, 0], points, [1, 1], 2, missing)["feasible"])

    def test_exact_reward_candidate_and_original_ids_after_augmentation(self):
        # A=2, B=1, P=2. Best POMO is selected within each augmentation,
        # then augmentation 1 wins. Customer IDs stay unchanged.
        rewards = np.asarray([[-10.0, -9.0], [-8.0, -7.0]])
        # Equal-length tensor is required by the environment; pad finished POMOs.
        selected = np.asarray([
            [[0, 2, 1, 0, 0], [0, 1, 2, 0, 0]],
            [[0, 2, 1, 0, 0], [0, 1, 0, 2, 0]],
        ], dtype=np.int64)
        result = select_best_candidates(rewards, selected, aug_factor=2, batch_size=1)[0]
        self.assertEqual(result["best_aug_idx"], 1)
        self.assertEqual(result["best_pomo_idx"], 1)
        self.assertEqual(result["reported_objective"], 7.0)
        self.assertEqual(result["canonical_solution"], [0, 1, 0, 2, 0])
        self.assertIn("preserves original node order", result["node_id_mapping"])


if __name__ == "__main__":
    unittest.main()
