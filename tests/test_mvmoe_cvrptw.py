import unittest

import numpy as np

from methods.mvmoe.cvrptw.adapter import adapt_batch
from methods.mvmoe.cvrptw.config import (CAPACITY, CHECKPOINT_SHA256, DATASET_COUNT,
                                        DATASET_SHA256, PROBLEM_SIZE, require_problem_size)
from methods.mvmoe.cvrptw.decode import decode_selected_nodes, select_best_candidates


def valid_batch(batch=2):
    depots = np.zeros((batch, 2), dtype=np.float32)
    points = np.zeros((batch, PROBLEM_SIZE, 2), dtype=np.float32)
    demands = np.tile(np.arange(1, PROBLEM_SIZE + 1, dtype=np.float32), (batch, 1))
    capacities = np.full(batch, CAPACITY, dtype=np.float32)
    tw = np.zeros((batch, PROBLEM_SIZE + 1, 2), dtype=np.float32)
    tw[:, 0, 1] = 4.6
    tw[:, 1:, 1] = 3.0
    service = np.zeros((batch, PROBLEM_SIZE + 1), dtype=np.float32)
    service[:, 1:] = 0.2
    return depots, points, demands, capacities, tw, service


class MVMoECVRPTWAdapterTests(unittest.TestCase):
    def test_demand_normalization_and_depot_customer_split(self):
        data = valid_batch()
        native, depot_window, mapping = adapt_batch(
            *data, problem_size=50, device="cpu")
        depot, points, demand, service, tw_start, tw_end = native
        self.assertEqual(tuple(depot.shape), (2, 1, 2))
        self.assertEqual(tuple(points.shape), (2, 50, 2))
        self.assertEqual(tuple(demand.shape), (2, 50))
        self.assertEqual(tuple(service.shape), (2, 50))
        self.assertEqual(tuple(tw_start.shape), (2, 50))
        self.assertEqual(tuple(tw_end.shape), (2, 50))
        np.testing.assert_allclose(demand.numpy(), data[2] / data[3][:, None])
        np.testing.assert_allclose(service.numpy(), data[5][:, 1:])
        np.testing.assert_allclose(tw_start.numpy(), data[4][:, 1:, 0])
        np.testing.assert_allclose(tw_end.numpy(), data[4][:, 1:, 1])
        self.assertEqual(depot_window, (0.0, np.float32(4.6)))
        self.assertTrue(mapping["depot_rows_removed_from_customer_service_and_time_windows"])
        self.assertIn("exactly once", mapping["normalization"])

    def test_different_depot_windows_in_batch_are_rejected(self):
        data = list(valid_batch())
        data[4][1, 0, 1] = 4.5
        with self.assertRaisesRegex(ValueError, "different depot"):
            adapt_batch(*data, problem_size=50, device="cpu")

    def test_nonzero_depot_service_is_rejected(self):
        data = list(valid_batch())
        data[5][0, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "depot service"):
            adapt_batch(*data, problem_size=50, device="cpu")

    def test_time_window_and_service_shapes_are_strict(self):
        data = list(valid_batch())
        for field, bad in ((4, data[4][:, 1:]), (5, data[5][:, 1:])):
            changed = list(data)
            changed[field] = bad
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "shape"):
                adapt_batch(*changed, problem_size=50, device="cpu")

    def test_exact_size_and_asset_identity(self):
        self.assertEqual(require_problem_size(50), 50)
        with self.assertRaisesRegex(ValueError, "only problem_size 50"):
            require_problem_size(100)
        self.assertEqual(DATASET_COUNT, 1000)
        self.assertEqual(DATASET_SHA256,
                         "a16975d9dd242739973191256e1cdbd8166c4405759ef195b985a8c3ec49df40")
        self.assertEqual(CHECKPOINT_SHA256,
                         "3417f302fbddf232fd19a2a886cd1c7f44290b6d8c7280fcb0ae3777eeed3192")


class MVMoECVRPTWDecoderTests(unittest.TestCase):
    def test_single_route(self):
        self.assertEqual(decode_selected_nodes([0, 1, 2, 0])[0], [0, 1, 2, 0])

    def test_multiple_routes_preserve_internal_depots(self):
        self.assertEqual(decode_selected_nodes([0, 1, 0, 2, 0])[0],
                         [0, 1, 0, 2, 0])

    def test_terminal_finished_pomo_padding_only(self):
        route, info = decode_selected_nodes([0, 1, 2, 0, 0, 0])
        self.assertEqual(route, [0, 1, 2, 0])
        self.assertEqual(info["removed_finish_padding"], 2)

    def test_duplicate_and_missing_customers_are_not_repaired(self):
        self.assertEqual(decode_selected_nodes([0, 1, 1, 0])[0], [0, 1, 1, 0])
        self.assertEqual(decode_selected_nodes([0, 1, 0])[0], [0, 1, 0])

    def test_selected_route_uses_exact_best_reward_indices(self):
        rewards = np.asarray([[-10.0, -9.0], [-8.0, -7.0]])
        selected = np.asarray([
            [[0, 2, 1, 0, 0], [0, 1, 2, 0, 0]],
            [[0, 2, 1, 0, 0], [0, 1, 0, 2, 0]],
        ], dtype=np.int64)
        result = select_best_candidates(
            rewards, selected, aug_factor=2, batch_size=1)[0]
        self.assertEqual((result["best_aug_idx"], result["best_pomo_idx"]), (1, 1))
        self.assertEqual(result["reported_objective"], 7.0)
        self.assertEqual(result["canonical_solution"], [0, 1, 0, 2, 0])


if __name__ == "__main__":
    unittest.main()
