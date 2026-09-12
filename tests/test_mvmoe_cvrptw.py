import unittest

import numpy as np

from methods.mvmoe.cvrptw.adapter import adapt_batch
from methods.mvmoe.cvrptw.config import SIZE_CONFIGS, get_size_config
from methods.mvmoe.cvrptw.decode import decode_selected_nodes, select_best_candidates


def valid_batch(problem_size, batch=2):
    capacity = get_size_config(problem_size)["capacity"]
    depots = np.zeros((batch, 2), dtype=np.float32)
    points = np.zeros((batch, problem_size, 2), dtype=np.float32)
    demands = np.tile(
        np.arange(1, problem_size + 1, dtype=np.float32), (batch, 1))
    capacities = np.full(batch, capacity, dtype=np.float32)
    tw = np.zeros((batch, problem_size + 1, 2), dtype=np.float32)
    tw[:, 0, 1] = 4.6
    tw[:, 1:, 1] = 3.0
    service = np.zeros((batch, problem_size + 1), dtype=np.float32)
    service[:, 1:] = 0.2
    return depots, points, demands, capacities, tw, service


class MVMoECVRPTWAdapterTests(unittest.TestCase):
    def test_both_sizes_shapes_capacity_split_and_normalization(self):
        for problem_size, capacity in ((50, 40.0), (100, 50.0)):
            with self.subTest(problem_size=problem_size):
                data = valid_batch(problem_size)
                native, depot_window, mapping = adapt_batch(
                    *data, problem_size=problem_size, device="cpu")
                depot, points, demand, service, tw_start, tw_end = native
                self.assertEqual(tuple(depot.shape), (2, 1, 2))
                self.assertEqual(tuple(points.shape), (2, problem_size, 2))
                self.assertEqual(tuple(demand.shape), (2, problem_size))
                self.assertEqual(tuple(service.shape), (2, problem_size))
                self.assertEqual(tuple(tw_start.shape), (2, problem_size))
                self.assertEqual(tuple(tw_end.shape), (2, problem_size))
                self.assertTrue(np.all(data[3] == capacity))
                np.testing.assert_allclose(demand.numpy(), data[2] / data[3][:, None])
                np.testing.assert_allclose(service.numpy(), data[5][:, 1:])
                np.testing.assert_allclose(tw_start.numpy(), data[4][:, 1:, 0])
                np.testing.assert_allclose(tw_end.numpy(), data[4][:, 1:, 1])
                self.assertEqual(depot_window, (0.0, np.float32(4.6)))
                self.assertTrue(
                    mapping["depot_rows_removed_from_customer_service_and_time_windows"])
                self.assertIn("exactly once", mapping["normalization"])

    def test_different_depot_windows_in_batch_are_rejected(self):
        data = list(valid_batch(50))
        data[4][1, 0, 1] = 4.5
        with self.assertRaisesRegex(ValueError, "different depot"):
            adapt_batch(*data, problem_size=50, device="cpu")

    def test_nonzero_depot_lower_bound_is_rejected(self):
        data = list(valid_batch(50))
        data[4][:, 0, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "lower bound 0"):
            adapt_batch(*data, problem_size=50, device="cpu")

    def test_nonzero_depot_service_is_rejected(self):
        data = list(valid_batch(50))
        data[5][0, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "depot service"):
            adapt_batch(*data, problem_size=50, device="cpu")

    def test_time_window_and_service_shapes_are_strict_for_both_sizes(self):
        for problem_size in (50, 100):
            data = list(valid_batch(problem_size))
            for field, bad in ((4, data[4][:, 1:]), (5, data[5][:, 1:])):
                changed = list(data)
                changed[field] = bad
                with self.subTest(problem_size=problem_size, field=field):
                    with self.assertRaisesRegex(ValueError, "shape"):
                        adapt_batch(
                            *changed, problem_size=problem_size, device="cpu")

    def test_exact_size_and_asset_identities(self):
        self.assertEqual(tuple(SIZE_CONFIGS), (50, 100))
        self.assertEqual(get_size_config(50)["capacity"], 40.0)
        self.assertEqual(get_size_config(100)["capacity"], 50.0)
        self.assertEqual(
            get_size_config(50)["dataset_sha256"],
            "a16975d9dd242739973191256e1cdbd8166c4405759ef195b985a8c3ec49df40")
        self.assertEqual(
            get_size_config(100)["dataset_sha256"],
            "3b74fa520f42f7aa607a5bd00a7d4aaa118e0715ca1672ee854fc850bac67867")
        self.assertEqual(
            get_size_config(100)["checkpoint_sha256"],
            "554d6daea825e17d62c1b9db40d56869312923848504bdf4970663673c971bdc")
        with self.assertRaisesRegex(ValueError, "50 or 100"):
            get_size_config(75)


class MVMoECVRPTWDecoderTests(unittest.TestCase):
    def test_single_route(self):
        self.assertEqual(
            decode_selected_nodes([0, 1, 2, 0], problem_size=50)[0],
            [0, 1, 2, 0])

    def test_multiple_routes_preserve_internal_depots(self):
        self.assertEqual(
            decode_selected_nodes([0, 1, 0, 2, 0], problem_size=50)[0],
            [0, 1, 0, 2, 0])

    def test_terminal_finished_pomo_padding_only(self):
        route, info = decode_selected_nodes(
            [0, 1, 2, 0, 0, 0], problem_size=50)
        self.assertEqual(route, [0, 1, 2, 0])
        self.assertEqual(info["removed_finish_padding"], 2)

    def test_duplicate_and_missing_customers_are_not_repaired(self):
        self.assertEqual(
            decode_selected_nodes([0, 1, 1, 0], problem_size=50)[0],
            [0, 1, 1, 0])
        self.assertEqual(
            decode_selected_nodes([0, 1, 0], problem_size=50)[0],
            [0, 1, 0])

    def test_size_100_node_ids(self):
        self.assertEqual(
            decode_selected_nodes([0, 100, 0, 0], problem_size=100)[0],
            [0, 100, 0])
        with self.assertRaisesRegex(ValueError, "0..100"):
            decode_selected_nodes([0, 101, 0], problem_size=100)

    def test_selected_route_uses_exact_best_reward_indices(self):
        rewards = np.asarray([[-10.0, -9.0], [-8.0, -7.0]])
        selected = np.asarray([
            [[0, 2, 1, 0, 0], [0, 1, 2, 0, 0]],
            [[0, 2, 1, 0, 0], [0, 1, 0, 2, 0]],
        ], dtype=np.int64)
        result = select_best_candidates(
            rewards, selected, aug_factor=2, batch_size=1, problem_size=2)[0]
        self.assertEqual((result["best_aug_idx"], result["best_pomo_idx"]), (1, 1))
        self.assertEqual(result["reported_objective"], 7.0)
        self.assertEqual(result["canonical_solution"], [0, 1, 0, 2, 0])


if __name__ == "__main__":
    unittest.main()
