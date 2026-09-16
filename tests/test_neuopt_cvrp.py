import unittest
from types import ModuleType
from unittest.mock import patch

import numpy as np

from methods.neuopt.cvrp.adapter import adapt_batch
from methods.neuopt.cvrp.compat import ensure_tensorboard_logger
from methods.neuopt.cvrp.config import supported_config
from methods.neuopt.cvrp.decode import (canonicalize_internal_order,
                                        decode_successor, extract_final_best,
                                        extract_final_best_d2a)
from methods.neuopt.cvrp.paper_eval import official_option_args
from methods.neuopt.cvrp.paper_protocol import paper_protocol


def successor_from_order(order):
    successor = np.empty(len(order), dtype=np.int64)
    for current, following in zip(order, order[1:] + order[:1]):
        successor[current] = following
    return successor


class NeuOptTensorboardCompatibilityTests(unittest.TestCase):
    def test_existing_module_uses_real_package_without_shim(self):
        existing = ModuleType("tensorboard_logger")
        with patch.dict("sys.modules", {"tensorboard_logger": existing}):
            result = ensure_tensorboard_logger()
            self.assertTrue(result["tensorboard_logger_available"])
            self.assertFalse(result["tensorboard_logger_import_shim"])
            self.assertIs(__import__("tensorboard_logger"), existing)

    def test_missing_module_registers_importable_guard_shim(self):
        def missing(name):
            raise ModuleNotFoundError("missing tensorboard_logger", name=name)

        with patch.dict("sys.modules", {}, clear=False):
            import sys
            sys.modules.pop("tensorboard_logger", None)
            result = ensure_tensorboard_logger(import_module=missing)
            from tensorboard_logger import Logger
            self.assertFalse(result["tensorboard_logger_available"])
            self.assertTrue(result["tensorboard_logger_import_shim"])
            self.assertFalse(result["official_source_modified"])
            with self.assertRaisesRegex(RuntimeError, "unexpectedly entered TensorBoard path"):
                Logger("unused")
            sys.modules.pop("tensorboard_logger", None)

    def test_unrelated_import_failure_is_not_swallowed(self):
        def transitive_failure(name):
            raise ModuleNotFoundError("missing unrelated dependency", name="unrelated_dependency")

        with self.assertRaises(ModuleNotFoundError) as raised:
            ensure_tensorboard_logger(import_module=transitive_failure)
        self.assertEqual(raised.exception.name, "unrelated_dependency")


class NeuOptCVRPConfigAdapterTests(unittest.TestCase):
    def test_asset_and_size_identities(self):
        expected = {
            50: (40, 0.4, 20, 70,
                 "1cd201ca47888e51068a157389460641c81d71054f064d9c8ea1743312289e3a"),
            100: (50, 0.2, 20, 120,
                  "502a5904182306c1a3f65f7b1a8503a609ff2a044db054abcf93690af36594fb"),
        }
        for size, values in expected.items():
            config = supported_config(size)
            self.assertEqual((config["capacity"], config["dummy_rate"],
                              config["dummy_size"], config["sequence_length"],
                              config["checkpoint_sha256"]), values)
        with self.assertRaisesRegex(ValueError, "50 or 100"):
            supported_config(20)

    def test_adapter_repeats_depot_preserves_customers_and_normalizes_once(self):
        for size, capacity in ((50, 40), (100, 50)):
            with self.subTest(size=size):
                depot = np.asarray([[0.25, 0.75]], dtype=np.float32)
                points = np.arange(size * 2, dtype=np.float32).reshape(1, size, 2)
                demands = np.arange(1, size + 1, dtype=np.float32)[None]
                native, mapping = adapt_batch(depot, points, demands, [capacity],
                                              problem_size=size, device="cpu")
                coordinates = native["coordinates"].numpy()
                native_demand = native["demand"].numpy()
                self.assertEqual(coordinates.shape, (1, size + 20, 2))
                self.assertEqual(native_demand.shape, (1, size + 20))
                np.testing.assert_array_equal(coordinates[0, :20], np.repeat(depot, 20, axis=0))
                np.testing.assert_array_equal(coordinates[0, 20:], points[0])
                np.testing.assert_array_equal(native_demand[0, :20], 0)
                np.testing.assert_allclose(native_demand[0, 20:], demands[0] / capacity)
                self.assertIn("exactly once", mapping["normalization"])

    def test_formal_options_pass_d2a_and_T_to_official_flags(self):
        class Device:
            type = "cuda"

        protocol = paper_protocol(100, T_max=5000)
        args = official_option_args(
            problem_size=100, config=protocol,
            checkpoint="checkpoint.pt", device=Device())
        value = lambda flag: args[args.index(flag) + 1]
        self.assertEqual(value("--val_m"), "5")
        self.assertEqual(value("--T_max"), "5000")
        self.assertEqual(value("--stall_limit"), "10")
        self.assertEqual(value("--k"), "4")
        self.assertEqual(value("--val_batch_size"), "1")


class NeuOptCVRPDecoderTests(unittest.TestCase):
    def test_single_route_and_first_last_customer_mapping_for_both_sizes(self):
        for size in (50, 100):
            order = list(range(size + 20))
            canonical, info = decode_successor(successor_from_order(order), problem_size=size)
            self.assertEqual(canonical, [0] + list(range(1, size + 1)) + [0])
            self.assertEqual(info["internal_order"], order)

    def test_multiple_routes_and_consecutive_dummy_depots(self):
        # Internal dummies are 0..19; real 20,21,22 map to customers 1,2,3.
        order = [0, 20, 1, 2, 21, 3, 22] + list(range(4, 20))
        canonical = canonicalize_internal_order(order, problem_size=3, dummy_size=20)
        self.assertEqual(canonical, [0, 1, 0, 2, 0, 3, 0])

    def test_successor_rejects_duplicate_missing_and_disconnected_cycles(self):
        successor = successor_from_order(list(range(70)))
        successor[0] = successor[1]
        with self.assertRaisesRegex(ValueError, "duplicate/missing"):
            decode_successor(successor, problem_size=50)
        disconnected = np.arange(70, dtype=np.int64)
        disconnected[0], disconnected[1] = 1, 0
        with self.assertRaisesRegex(ValueError, "before visiting"):
            decode_successor(disconnected, problem_size=50)

    def test_internal_order_duplicate_missing_is_not_repaired(self):
        order = list(range(70))
        order[-1] = order[-2]
        with self.assertRaisesRegex(ValueError, "not repaired"):
            canonicalize_internal_order(order, problem_size=50, dummy_size=20)

    def test_official_best_solution_objective_correspondence(self):
        import torch
        solution0 = torch.tensor([[1, 2, 0], [2, 0, 1]])
        solution1 = torch.tensor([[2, 0, 1], [1, 2, 0]])
        obj_history = torch.tensor([
            [[5.0, 5.0, 5.0], [4.0, 4.0, 5.0]],
            [[8.0, 8.0, 8.0], [7.0, 7.0, 8.0]],
        ])
        output = (torch.tensor([4.0, 7.0]), obj_history, torch.empty(2, 1),
                  ([solution0, solution1], [solution0, solution1], [None, None]))
        best, solution = extract_final_best(output, batch_size=2, val_m=1)
        self.assertTrue(torch.equal(best, torch.tensor([4.0, 7.0])))
        self.assertTrue(torch.equal(solution, solution1))
        bad = (torch.tensor([4.0, 6.0]), obj_history, output[2], output[3])
        with self.assertRaisesRegex(ValueError, "does not correspond"):
            extract_final_best(bad, batch_size=2, val_m=1)

    def test_d2a_best_candidate_solution_and_objective_correspondence(self):
        import torch

        class Problem:
            def augment(self, batch, val_m, only_copy=False):
                self.assert_only_copy = only_copy
                return {
                    "coordinates": batch["coordinates"].repeat(val_m, 1, 1),
                    "demand": batch["demand"].repeat(val_m, 1),
                }

            def get_costs(self, batch, solutions, **kwargs):
                self.last_solutions = solutions.clone()
                return torch.tensor([3.0, 1.0, 2.0, 4.0, 5.0])

        candidates = torch.tensor([
            [1, 2, 0], [2, 0, 1], [1, 0, 2], [2, 1, 0], [0, 2, 1]
        ])
        obj_history = torch.tensor([[[3.0, 3.0, 3.0], [1.0, 1.0, 3.0]]])
        output = (
            torch.tensor([1.0]), obj_history, torch.empty(1, 1),
            ([candidates, candidates], [candidates, candidates], [None, None]),
        )
        native = {
            "coordinates": torch.zeros(1, 3, 2), "demand": torch.zeros(1, 3)
        }
        problem = Problem()
        best, solution, selected = extract_final_best_d2a(
            output, problem=problem, native_batch=native, batch_size=1, val_m=5)
        self.assertTrue(problem.assert_only_copy)
        self.assertTrue(torch.equal(best, torch.tensor([1.0])))
        self.assertTrue(torch.equal(solution, candidates[1:2]))
        self.assertTrue(torch.equal(selected, torch.tensor([1])))


if __name__ == "__main__":
    unittest.main()
