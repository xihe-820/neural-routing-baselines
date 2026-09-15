import inspect
import unittest

import numpy as np
import torch

from methods.glop.cvrp.paper_eval import _solve_one as solve_cvrp
from methods.glop.paper_protocol import formal_protocol
from methods.glop.runtime import (completed_prefix_length,
                                  make_shared_tsp_orders,
                                  official_seeded_setup,
                                  replay_completed_prefix,
                                  require_cvrp_dataset_prefix,
                                  run_warmup_isolated)
from methods.glop.tsp.paper_eval import (_solve_one as solve_tsp,
                                         main as tsp_main)


def _fake_stochastic_result(index):
    draw = int(torch.randint(1, 1_000_000, ()).item())
    objective = float(draw)
    return {
        "dataset_instance_index": index,
        "canonical_solution": [0, draw, 0],
        "canonical_routes": [[0, draw, 0]],
        "reported_objective": objective,
        "independent_objective": objective,
        "selection": {"draw": draw},
    }


class GLOPTSPRNGTests(unittest.TestCase):
    def test_solvers_do_not_reseed_per_instance(self):
        self.assertNotIn("manual_seed", inspect.getsource(solve_tsp))
        self.assertNotIn("manual_seed", inspect.getsource(solve_cvrp))

    def test_shared_orders_are_reused_for_different_instances(self):
        orders = (torch.tensor([3, 1, 4, 0, 2]),
                  torch.tensor([2, 4, 1, 3, 0]))
        seen = []

        def insertion(batch, order):
            seen.append(order.tolist())
            return np.tile(order.numpy(), (len(batch), 1))

        def reconnect(*, get_cost_func, batch, opts, revisers):
            tour = batch[:1]
            cost = ((tour[:, 1:] - tour[:, :-1]).norm(dim=2).sum(1) +
                    (tour[:, 0] - tour[:, -1]).norm(dim=1))
            return tour, cost

        class Problem:
            @staticmethod
            def get_costs(data, route, return_local=True):
                return None

        protocol = {
            "internal_width": 2, "revision_lens": [], "revision_iters": [],
            "local_augmentation": True, "pruning": True,
        }
        first = np.asarray(
            [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float32)
        second = first + np.asarray([0, 1], dtype=np.float32)
        outputs = []
        for points in (first, second):
            outputs.append(solve_tsp(
                points, protocol=protocol, orders=orders, revisers=[],
                device=torch.device("cpu"), torch=torch, reconnect=reconnect,
                load_problem=lambda _: Problem,
                random_insertion_parallel=insertion, timed=False)[0])
        self.assertEqual(seen, [orders[0].tolist(), orders[1].tolist()] * 2)
        self.assertEqual(outputs[0], outputs[1])

    def test_seed_precedes_model_setup_and_shared_order_generation(self):
        torch.manual_seed(1)
        torch.rand(17)  # official model construction consumes CPU RNG
        expected = make_shared_tsp_orders(torch, problem_size=19, width=3)

        torch.manual_seed(999)
        setup_value = official_seeded_setup(
            torch, 1, lambda: torch.rand(17))
        actual = make_shared_tsp_orders(torch, problem_size=19, width=3)
        self.assertEqual(tuple(setup_value.shape), (17,))
        for left, right in zip(expected, actual):
            self.assertTrue(torch.equal(left, right))

    def test_runner_generates_one_shared_order_set_after_model_setup(self):
        source = inspect.getsource(tsp_main)
        setup = source.index("official_seeded_setup(")
        timing_start = source.index("shared_order_started = time.perf_counter()")
        generation = source.index("orders = make_shared_tsp_orders(")
        timing_stop = source.index(
            "time.perf_counter() - shared_order_started")
        warmup = source.index("run_warmup_isolated(")
        self.assertLess(setup, timing_start)
        self.assertLess(timing_start, generation)
        self.assertLess(generation, timing_stop)
        self.assertLess(timing_stop, warmup)
        self.assertEqual(source.count("orders = make_shared_tsp_orders("), 1)

    def test_tsp_warmup_does_not_change_formal_output(self):
        def formal_output():
            return int(torch.randint(0, 1_000_000, ()).item())

        torch.manual_seed(1)
        without_warmup = formal_output()
        torch.manual_seed(1)
        run_warmup_isolated(
            torch, torch.device("cpu"),
            lambda: [formal_output() for _ in range(2)])
        with_warmup = formal_output()
        self.assertEqual(without_warmup, with_warmup)


class GLOPCVRPRNGTests(unittest.TestCase):
    def test_sampling_stream_advances_across_instances(self):
        torch.manual_seed(1)
        first = _fake_stochastic_result(0)
        second = _fake_stochastic_result(1)
        self.assertNotEqual(first["selection"], second["selection"])

        torch.manual_seed(1)
        self.assertEqual(first, _fake_stochastic_result(0))
        self.assertEqual(second, _fake_stochastic_result(1))

    def test_warmup_restoration_preserves_formal_sequence(self):
        torch.manual_seed(1)
        expected = [_fake_stochastic_result(i) for i in range(4)]
        torch.manual_seed(1)
        run_warmup_isolated(
            torch, torch.device("cpu"),
            lambda: [_fake_stochastic_result(i) for i in range(2)])
        actual = [_fake_stochastic_result(i) for i in range(4)]
        self.assertEqual(expected, actual)

    def test_prefix_replay_matches_uninterrupted_stream(self):
        torch.manual_seed(1)
        uninterrupted = [_fake_stochastic_result(i) for i in range(4)]
        prefix = uninterrupted[:2]

        torch.manual_seed(1)
        replay_completed_prefix(
            prefix, [0, 1, 2, 3], _fake_stochastic_result)
        resumed = prefix + [_fake_stochastic_result(i) for i in range(2, 4)]
        self.assertEqual(uninterrupted, resumed)

    def test_non_prefix_resume_records_fail_closed(self):
        records = [{"dataset_instance_index": 0},
                   {"dataset_instance_index": 2}]
        with self.assertRaisesRegex(ValueError, "contiguous ordered prefix"):
            completed_prefix_length(records, [0, 1, 2, 3])

    def test_nonzero_prepared_offset_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "starting at index 0"):
            require_cvrp_dataset_prefix([2, 3])
        self.assertEqual(require_cvrp_dataset_prefix([0, 1]), [0, 1])

    def test_seed_and_rng_identity_are_frozen(self):
        tsp = formal_protocol("TSP", 500, "official_standard")
        cvrp = formal_protocol("CVRP", 1000, "official_single")
        self.assertEqual((tsp["seed"], cvrp["seed"]), (1, 1))
        self.assertFalse(tsp["rng_semantics"]["per_instance_reseed"])
        self.assertFalse(cvrp["rng_semantics"]["per_instance_reseed"])
        self.assertEqual(cvrp["rng_semantics"]["resume_rng_policy"],
                         "completed-prefix replay")


if __name__ == "__main__":
    unittest.main()
