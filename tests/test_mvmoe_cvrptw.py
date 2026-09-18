import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from methods.mvmoe.cvrptw.adapter import adapt_batch
from methods.mvmoe.cvrptw.config import SIZE_CONFIGS, get_size_config
from methods.mvmoe.cvrptw.decode import decode_selected_nodes, select_best_candidates
from methods.mvmoe.cvrptw.paper_eval import _slice_native_batch
from methods.mvmoe.cvrptw.batch_artifacts import (
    append_batch_records, initialize_batch_chunk)


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

    def test_bs10_scaling_is_independent_for_ten_different_scalers(self):
        data = list(valid_batch(50, batch=10))
        for index in range(10):
            data[0][index] = [0.1 + index, 0.2]
            data[1][index, 0] = [2.0 + index, 0.5]
            data[4][index, 0, 1] = 4.6 + index
        arrays = {
            "depots": data[0], "points": data[1], "demands": data[2],
            "capacities": data[3], "time_windows": data[4],
            "service_times": data[5],
        }
        scaled, native, windows, mappings = _slice_native_batch(
            arrays, list(range(10)), problem_size=50, device="cpu")
        self.assertEqual(tuple(native[0].shape), (10, 1, 2))
        scalers = [row["scaler"] for row in scaled]
        self.assertEqual(len(set(scalers)), 10)
        for index, scaler in enumerate(scalers):
            np.testing.assert_allclose(
                native[0][index, 0].numpy(), data[0][index] / scaler)
            self.assertAlmostEqual(windows[index, 1], data[4][index, 0, 1] / scaler)
            self.assertEqual(mappings[index]["scaler"], scaler)


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

    def test_bs10_selection_never_mixes_original_instances(self):
        batch, problem, steps = 10, 3, 6
        reward = np.full((8 * batch, problem), -1000.0, dtype=np.float32)
        actions = np.zeros((8 * batch, problem, steps), dtype=np.int64)
        expected = []
        for batch_index in range(batch):
            aug, pomo = batch_index % 8, batch_index % problem
            flat = aug * batch + batch_index
            reward[flat, pomo] = 100 + batch_index
            actions[flat, pomo] = [0, 1, 2, 3, 0, 0]
            expected.append((aug, pomo))
        selected = select_best_candidates(
            reward, actions, aug_factor=8, batch_size=batch,
            problem_size=problem)
        self.assertEqual(len(selected), batch)
        self.assertEqual(
            [(row["best_aug_idx"], row["best_pomo_idx"]) for row in selected],
            expected)


class MVMoEBatchArtifactTests(unittest.TestCase):
    def identity(self):
        return {
            "method": "MVMoE", "variant": "MVMoE/4E", "problem": "CVRPTW",
            "problem_size": 50,
            "paper_protocol": {"original_batch_size": 10},
            "chunk": {"offset": 0, "count": 20,
                      "expected_indices": list(range(20))},
        }

    @staticmethod
    def record(index, batch_index, position, runtime):
        return {
            "dataset_instance_index": index, "instance_id": f"x{index}",
            "canonical_solution": [0, 1, 0], "reported_objective": 2.0,
            "independent_objective": 2.0, "reference_objective": 1.0,
            "gap_percent": 100.0, "runtime_seconds": runtime,
            "independent_feasible": True, "reported_objective_agrees": True,
            "evidence_status": "INDEPENDENT_VERIFIED",
            "batch_index": batch_index, "position_in_batch": position,
        }

    def test_resume_advances_only_after_complete_native_batch(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            identity = self.identity()
            initialize_batch_chunk(path, identity)
            records = [self.record(index, 0, index, 0.4) for index in range(10)]
            append_batch_records(path, records, {
                "batch_index": 0, "dataset_indices": list(range(10)),
                "batch_size": 10, "runtime_seconds": 0.4,
            })
            _, completed, batches = initialize_batch_chunk(path, identity)
            self.assertEqual(completed, set(range(10)))
            self.assertEqual(batches, 1)
            with (path / "inference_records.jsonl").open("a") as stream:
                stream.write(json.dumps(self.record(10, 1, 0, 0.5)) + "\n")
            with self.assertRaisesRegex(ValueError, "partial"):
                initialize_batch_chunk(path, identity)


if __name__ == "__main__":
    unittest.main()
