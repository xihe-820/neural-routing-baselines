import inspect
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from methods.glop.paper_protocol import formal_protocol
from methods.glop.paper_results import fingerprint
from methods.glop.tsp.parallel_eval import (_build_candidate_batch,
                                             _solve_batch,
                                             _verify_bs1_summary,
                                             main as parallel_main)
from methods.glop.tsp.parallel_results import (TIMING_SEMANTICS,
                                                exact_batch_ranges,
                                                parallel_scope,
                                                summarize_parallel)


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "e540bc0153a0598e923e35116deeaecaf9c1cfff"
REVISERS = [{
    "reviser_size": 100,
    "checkpoint_sha256": "1" * 64,
    "args_sha256": "2" * 64,
}]
CURRENT_PROTOCOL = formal_protocol("TSP", 500, "official_more")


def _bs1_summary(**changes):
    result = {
        "status": "PAPER_READY", "method": "GLOP", "variant": "official_more",
        "problem": "TSP", "problem_size": 500, "instance_count": 128,
        "consistency_identity": {
            "method": "GLOP", "variant": "official_more", "problem": "TSP",
            "problem_size": 500,
            "official_protocol_name": "official_more",
            "paper_protocol": deepcopy(CURRENT_PROTOCOL),
            "dataset": {"path": "/data/tsp500_concorde_16.546.pkl",
                        "sha256": "a" * 64, "count": 128},
            "upstream": {"commit": UPSTREAM_COMMIT, "dirty": False},
            "assets": {"revisers": REVISERS},
        },
    }
    for dotted, value in changes.items():
        target = result
        parts = dotted.split("__")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return result


def _check_bs1(summary):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "summary.json"
        path.write_text(json.dumps(summary) + "\n")
        return _verify_bs1_summary(
            path, problem_size=500, protocol_name="official_more",
            dataset_path=Path("/current/tsp500_concorde_16.546.pkl"),
            dataset_sha256="a" * 64, dataset_count=128,
            upstream_commit=UPSTREAM_COMMIT, reviser_identities=REVISERS,
            current_protocol=CURRENT_PROTOCOL)


def _records(count, batch_size=16):
    return [{
        "dataset_instance_index": index,
        "batch_index": index // batch_size,
        "canonical_solution": [0, 1, 0],
        "reported_objective": float(index + 2),
        "independent_objective": float(index + 2),
        "kit_objective": float(index + 2),
        "reference_objective": 1.0,
        "gap_percent": float(index + 1) * 100.0,
        "independent_feasible": True,
        "reported_objective_agrees": True,
        "kit_feasible": True,
        "kit_objective_agrees": True,
        "evidence_status": "KIT_VALIDATED",
    } for index in range(count)]


def _timings(count, batch_size, runtime=2.0):
    return [{
        "batch_index": batch_index,
        "dataset_index_start": start,
        "dataset_index_stop_exclusive": stop,
        "original_instance_count": batch_size,
        "solver_runtime_seconds": runtime,
        "shared_ri_order_generation_seconds_charged": 0.5 if batch_index == 0 else 0.0,
    } for batch_index, (start, stop) in enumerate(
        exact_batch_ranges(count, batch_size))]


class GLOPParallelProtocolTests(unittest.TestCase):
    def test_bs1_same_name_with_tampered_revision_iters_fails(self):
        tampered = deepcopy(CURRENT_PROTOCOL)
        tampered["revision_iters"] = [20, 24, 5]
        with self.assertRaisesRegex(ValueError, "frozen paper protocol"):
            _check_bs1(_bs1_summary(
                consistency_identity__paper_protocol=tampered))

    def test_bs1_same_name_with_tampered_width_and_candidates_fails(self):
        tampered = deepcopy(CURRENT_PROTOCOL)
        tampered["paper_nominal_width"] = 9
        tampered["ri_order_width"] = 9
        tampered["effective_candidate_count"] = 9
        with self.assertRaisesRegex(ValueError, "frozen paper protocol"):
            _check_bs1(_bs1_summary(
                consistency_identity__paper_protocol=tampered))

    def test_bs1_exact_complete_paper_protocol_passes(self):
        result = _check_bs1(_bs1_summary())
        identity = result["identity"]
        self.assertEqual(identity["paper_protocol"], CURRENT_PROTOCOL)
        self.assertEqual(identity["paper_protocol_fingerprint"],
                         fingerprint(CURRENT_PROTOCOL))

    def test_bs1_dataset_sha_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "dataset identity"):
            _check_bs1(_bs1_summary(
                consistency_identity__dataset__sha256="f" * 64))

    def test_bs1_protocol_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            _check_bs1(_bs1_summary(variant="official_standard"))

    def test_bs1_size_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "problem size"):
            _check_bs1(_bs1_summary(problem_size=100))

    def test_bs1_non_paper_ready_fails(self):
        with self.assertRaisesRegex(ValueError, "not PAPER_READY"):
            _check_bs1(_bs1_summary(status="KIT_VALIDATED"))

    def test_matching_bs1_identity_passes_and_records_provenance(self):
        result = _check_bs1(_bs1_summary())
        self.assertTrue(Path(result["path"]).is_absolute())
        self.assertEqual(len(result["sha256"]), 64)
        self.assertEqual(result["identity"]["dataset_sha256"], "a" * 64)
        self.assertEqual(result["identity"]["official_glop_commit"],
                         UPSTREAM_COMMIT)
        self.assertEqual(result["identity"]["revisers"], REVISERS)

    def test_only_native_batch_sizes_16_and_128(self):
        for batch_size in (1, 15, 17, 64, 256):
            with self.subTest(batch_size=batch_size):
                with self.assertRaisesRegex(ValueError, "16 or 128"):
                    parallel_scope(100, "official_standard", batch_size)
        for batch_size in (16, 128):
            self.assertEqual(
                parallel_scope(100, "official_standard", batch_size)
                ["parallel_original_instance_batch_size"], batch_size)

    def test_exact_batch_counts_and_no_drop_last(self):
        expected = {
            (1280, 16): 80, (1280, 128): 10,
            (128, 16): 8, (128, 128): 1,
        }
        for (count, batch_size), batch_count in expected.items():
            ranges = exact_batch_ranges(count, batch_size)
            self.assertEqual(len(ranges), batch_count)
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], count)
            self.assertTrue(all(stop - start == batch_size
                                for start, stop in ranges))
        with self.assertRaisesRegex(ValueError, "exact native batches"):
            exact_batch_ranges(129, 128)

    def test_parallel_scope_preserves_every_frozen_search_parameter(self):
        expected = {
            (100, "official_standard"): (35, [100, 50, 20, 10], [20, 10, 10, 5]),
            (100, "official_more"): (140, [100, 50, 20, 10], [20, 10, 10, 5]),
            (500, "official_standard"): (1, [100, 50, 20], [20, 25, 5]),
            (500, "official_more"): (10, [100, 50, 20], [20, 25, 5]),
            (1000, "official_standard"): (1, [100, 50, 20], [20, 25, 5]),
            (1000, "official_more"): (10, [100, 50, 20], [20, 25, 5]),
        }
        for (size, name), (width, lens, iters) in expected.items():
            with self.subTest(size=size, name=name):
                frozen = formal_protocol("TSP", size, name)
                scope = parallel_scope(size, name, 16)
                self.assertEqual(scope["paper_protocol"], frozen)
                self.assertEqual(frozen["paper_nominal_width"], width)
                self.assertEqual(frozen["revision_lens"], lens)
                self.assertEqual(frozen["revision_iters"], iters)
                self.assertEqual(frozen["original_batch_size"], 1)

    def test_candidate_major_layout_keeps_each_instance_column(self):
        for batch_size in (16, 128):
            with self.subTest(batch_size=batch_size):
                width, nodes = 10, 10
                points = torch.zeros((batch_size, nodes, 2), dtype=torch.float32)
                for instance in range(batch_size):
                    points[instance, :, 0] = instance * 100 + torch.arange(nodes)
                permutations = np.stack([
                    np.tile(np.roll(np.arange(nodes), -candidate),
                            (batch_size, 1))
                    for candidate in range(width)
                ])
                protocol = {
                    "ri_order_width": width,
                    "top_level_transforms": ["identity"],
                    "effective_candidate_count": width,
                }
                flat = _build_candidate_batch(
                    points, permutations, protocol=protocol,
                    device=torch.device("cpu"), torch=torch)
                self.assertEqual(tuple(flat.shape),
                                 (width * batch_size, nodes, 2))
                shaped = flat.reshape(width, batch_size, nodes, 2)
                for candidate in range(width):
                    for instance in range(batch_size):
                        expected_first = instance * 100 + candidate
                        self.assertEqual(
                            float(shaped[candidate, instance, 0, 0]),
                            expected_first)
                costs = torch.arange(width * batch_size).reshape(width, batch_size)
                self.assertEqual(tuple(costs.min(0).values.shape), (batch_size,))

    def test_solver_calls_official_reconnect_once_for_whole_native_batch(self):
        for batch_size in (16, 128):
            with self.subTest(batch_size=batch_size):
                nodes, width = 10, 10
                points = np.zeros((batch_size, nodes, 2), dtype=np.float32)
                points[:, :, 0] = np.arange(nodes, dtype=np.float32) / nodes
                points[:, :, 1] = (
                    np.arange(batch_size, dtype=np.float32)[:, None] /
                    batch_size)
                orders = tuple(torch.arange(nodes) for _ in range(width))
                calls = []

                def insertion(batch, order):
                    return np.tile(order.numpy(), (len(batch), 1))

                def reconnect(*, get_cost_func, batch, opts, revisers):
                    calls.append((tuple(batch.shape), opts.eval_batch_size))
                    return batch[:opts.eval_batch_size], torch.zeros(opts.eval_batch_size)

                class Problem:
                    @staticmethod
                    def get_costs(data, route, return_local=True):
                        return None

                protocol = {
                    "ri_order_width": width,
                    "effective_candidate_count": width,
                    "top_level_transforms": ["identity"],
                    "revision_lens": [10], "revision_iters": [1],
                    "local_reconnect_augmentation": True, "pruning": True,
                }
                canonical, _, _, runtime = _solve_batch(
                    points, protocol=protocol, orders=orders, revisers=[],
                    device=torch.device("cpu"), torch=torch,
                    reconnect=reconnect, load_problem=lambda _: Problem,
                    random_insertion_parallel=insertion, timed=False)
                self.assertEqual(calls, [((width * batch_size, nodes, 2),
                                          batch_size)])
                self.assertEqual(len(canonical), batch_size)
                self.assertIsNone(runtime)

    def test_total_and_mean_batch_time_charge_shared_setup_once(self):
        scope = {
            "paper_protocol": formal_protocol("TSP", 500, "official_more"),
            "parallel_original_instance_batch_size": 16,
            "instance_count": 128,
            "batch_count": 8,
        }
        summary = summarize_parallel(
            _records(128), _timings(128, 16), scope=scope,
            shared_setup_seconds=0.5)
        self.assertEqual(summary["batch_count"], 8)
        self.assertEqual(summary["total_runtime_seconds"], 16.5)
        self.assertEqual(summary["time_mean_batch_seconds"], 16.5 / 8)
        self.assertEqual(summary["drop_mean_per_instance_gap_percent"], 6450.0)
        self.assertEqual(summary["timing_semantics"], TIMING_SEMANTICS)

    def test_coverage_and_partial_batch_timing_fail_closed(self):
        scope = {
            "paper_protocol": formal_protocol("TSP", 500, "official_standard"),
            "parallel_original_instance_batch_size": 128,
            "instance_count": 128,
            "batch_count": 1,
        }
        with self.assertRaisesRegex(ValueError, "fullset"):
            summarize_parallel(
                _records(127), _timings(128, 128), scope=scope,
                shared_setup_seconds=0.0)
        bad = _timings(128, 128)
        bad[0]["original_instance_count"] = 127
        with self.assertRaisesRegex(ValueError, "exact native batch"):
            summarize_parallel(
                _records(128, 128), bad, scope=scope,
                shared_setup_seconds=0.0)

    def test_timer_wraps_solver_and_excludes_validation_and_io(self):
        solve = inspect.getsource(_solve_batch)
        started = solve.index("started = time.perf_counter()")
        self.assertLess(solve.index("torch.cuda.synchronize(device)"),
                        started)
        self.assertLess(started,
                        solve.index("random_insertion_parallel(", started))
        self.assertGreater(solve.rindex("torch.cuda.synchronize(device)"),
                           solve.index("decode_coordinate_tour"))
        self.assertNotIn("validate(", solve)
        self.assertNotIn("check_constraints", solve)
        self.assertNotIn("write_json", solve)
        self.assertNotIn("write_jsonl", solve)

    def test_cuda_oom_is_recorded_and_reraised_without_batch_fallback(self):
        source = inspect.getsource(parallel_main)
        self.assertIn('"CUDA_OOM"', source)
        self.assertIn("torch.cuda.OutOfMemoryError", source)
        self.assertIn("raise\n", source)
        self.assertNotIn("batch_size //", source)
        self.assertNotIn("smaller_batch", source)

    def test_frozen_bs1_runner_and_official_source_are_unmodified(self):
        project_diff = subprocess.run(
            ["git", "diff", "--", "methods/glop/tsp/paper_eval.py",
             "methods/glop/paper_protocol.py"], cwd=ROOT,
            capture_output=True, text=True, check=True).stdout
        upstream_dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT / "external/GLOP",
            capture_output=True, text=True, check=True).stdout
        self.assertEqual(project_diff, "")
        self.assertEqual(upstream_dirty, "")


if __name__ == "__main__":
    unittest.main()
