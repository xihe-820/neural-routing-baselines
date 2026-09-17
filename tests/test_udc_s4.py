import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from methods.udc.adapter import (adapt_cvrp_task, adapt_tsp_task,
                                 validate_cvrp_population,
                                 validate_tsp_population)
from methods.udc.protocol import (FORMAL_SIZES, OFFICIAL_COMMIT,
                                  S3_IMPLEMENTATION_COMMIT, S3_SCRIPT_SHA256,
                                  SCALE_DATASET_FILENAMES,
                                  discover_scale_dataset, s3_gate)
from methods.udc.s4_scale_preflight import (build_summary, bytes_and_gib,
                                            classify_exception,
                                            memory_baseline,
                                            progressive_decision)
from methods.udc.s3_eval import SemanticValidationError, cuda_memory_snapshot


class Task:
    pass


class GenericAdapterTests(unittest.TestCase):
    def test_generic_tsp_shapes_100_500_1000(self):
        for size in (100, 500, 1000):
            with self.subTest(size=size):
                task = Task()
                task.points = np.arange(size * 2, dtype=np.float64).reshape(size, 2)
                source, model, evidence = adapt_tsp_task(task, size=size)
                self.assertEqual(source.shape, (size, 2))
                self.assertEqual(model.shape, (size, 2))
                self.assertEqual(source.dtype, np.float64)
                self.assertEqual(model.dtype, np.float32)
                self.assertEqual(evidence["node_order"], "unchanged")

    def test_generic_cvrp_shapes_and_single_normalization_200_500_1000(self):
        for size in (200, 500, 1000):
            with self.subTest(size=size):
                task = Task()
                task.depots = np.array([0.25, 0.75])
                task.points = np.zeros((size, 2), dtype=np.float32)
                task.demands = np.arange(size) % 9 + 1
                task.capacity = 50
                coordinates, demand, evidence = adapt_cvrp_task(task, size=size)
                self.assertEqual(coordinates.shape, (size + 1, 2))
                self.assertEqual(demand.shape, (size + 1,))
                np.testing.assert_allclose(demand[1:], task.demands / 50)
                self.assertEqual(evidence["normalization_applied"],
                                 "raw_demand / capacity exactly once")

    def test_tsp_best_alpha_is_independent_argmin(self):
        points = np.column_stack((np.arange(6), np.zeros(6)))
        population = np.tile(np.arange(6), (50, 1))
        population[3] = [0, 2, 4, 5, 3, 1]
        result = validate_tsp_population(points, population)
        objectives = result["independent_objective_per_alpha"]
        self.assertEqual(result["best_alpha"], min(range(50), key=lambda i: (objectives[i], i)))

    def test_cvrp_best_alpha_keeps_matching_flag(self):
        points = np.column_stack((np.arange(1, 5), np.zeros(4)))
        solutions = np.tile(np.arange(1, 5), (50, 1))
        flags = np.ones((50, 4), dtype=np.int64)
        flags[1] = [0, 0, 0, 1]
        result = validate_cvrp_population(
            np.array([0, 0]), points, np.ones(4), 4, solutions, flags)
        best = result["best_alpha"]
        self.assertEqual(result["solution_flag"], flags[best].tolist())
        self.assertEqual(result["solution"], solutions[best].tolist())


class ProtocolAndProvenanceTests(unittest.TestCase):
    def test_exact_formal_size_and_dataset_mapping(self):
        self.assertEqual(FORMAL_SIZES["tsp"], (100, 500, 1000, 2000, 5000, 10000))
        self.assertEqual(FORMAL_SIZES["cvrp"], (200, 500, 1000, 2000))
        self.assertEqual(SCALE_DATASET_FILENAMES, {
            ("tsp", 100): "tsp100_concorde_7.756.pkl",
            ("tsp", 500): "tsp500_concorde_16.546.pkl",
            ("tsp", 1000): "tsp1000_concorde_23.118.pkl",
            ("tsp", 2000): "tsp2000_lkh_500_32.436.pkl",
            ("tsp", 5000): "tsp5000_lkh_500_50.968.pkl",
            ("tsp", 10000): "tsp10000_lkh_500_71.782.pkl",
            ("cvrp", 200): "cvrp200_hgs-60s_19.630.pkl",
            ("cvrp", 500): "cvrp500_hgs-300s_37.154.pkl",
            ("cvrp", 1000): "cvrp1000_hgs-360s_41.171.pkl",
            ("cvrp", 2000): "cvrp2000_hgs-360s_57.181.pkl",
        })

    def test_scale_discovery_requires_one_exact_match(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            filename = SCALE_DATASET_FILENAMES[("tsp", 100)]
            (root / filename).touch()
            self.assertEqual(discover_scale_dataset(root, "tsp", 100).name, filename)
            duplicate = root / "copy" / filename
            duplicate.parent.mkdir(); duplicate.touch()
            with self.assertRaises(ValueError):
                discover_scale_dataset(root, "tsp", 100)

    def _s3_payload(self):
        return {"state": "KIT_VALIDATED", "count": 5,
                "ready_for_stage_s4_scale_preflight": "YES",
                "script_sha256": S3_SCRIPT_SHA256,
                "project_post": {"head": S3_IMPLEMENTATION_COMMIT, "pass": True},
                "official_post": {"head": OFFICIAL_COMMIT, "pass": True}}

    def test_s3_authoritative_gate(self):
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "df082675" / "s3_ml4co_adapter" / "our_5"
            path.mkdir(parents=True)
            (path / "metadata.json").write_text(json.dumps(self._s3_payload()))
            result = s3_gate(path)
            self.assertTrue(result["pass"])
            self.assertEqual(result["script_sha256"], S3_SCRIPT_SHA256)

    def test_s3_gate_fails_on_nonready_or_wrong_script(self):
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "df082675" / "s3_ml4co_adapter" / "our_5"
            path.mkdir(parents=True)
            payload = self._s3_payload(); payload["script_sha256"] = "0" * 64
            (path / "metadata.json").write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                s3_gate(path)


class ProgressiveAndMemoryTests(unittest.TestCase):
    def test_cuda_oom_classification(self):
        class OOM(Exception):
            pass

        class Cuda:
            OutOfMemoryError = OOM

        class Torch:
            cuda = Cuda()

        self.assertEqual(classify_exception(OOM("boom"), Torch), "CUDA_OOM")
        self.assertEqual(classify_exception(RuntimeError("CUDA out of memory"), Torch),
                         "CUDA_OOM")
        self.assertEqual(classify_exception(RuntimeError("shape mismatch"), Torch),
                         "RUNTIME_FAIL")
        self.assertEqual(classify_exception(
            SemanticValidationError("no feasible candidate"), Torch), "SEMANTIC_FAIL")

    def test_progressive_stop_after_oom_and_semantic_failure(self):
        rows = [{"problem": "tsp", "size": 100, "status": "PASS"},
                {"problem": "tsp", "size": 500, "status": "CUDA_OOM"}]
        decision = progressive_decision("tsp", 1000, rows)
        self.assertFalse(decision["allowed"])
        self.assertEqual(decision["status"], "NOT_RUN_AFTER_OOM")
        rows = [{"problem": "cvrp", "size": 200, "status": "SEMANTIC_FAIL"}]
        decision = progressive_decision("cvrp", 500, rows)
        self.assertEqual(decision["status"], "BLOCKED_BY_SEMANTIC_FAILURE")

    def test_progressive_requires_every_preceding_pass(self):
        self.assertEqual(progressive_decision("tsp", 500, [])["status"],
                         "PRECHECK_FAIL")
        rows = [{"problem": "tsp", "size": 100, "status": "PASS"}]
        self.assertTrue(progressive_decision("tsp", 500, rows)["allowed"])

    def test_runtime_failure_blocks_both_families(self):
        rows = [{"problem": "tsp", "size": 100, "status": "RUNTIME_FAIL"}]
        decision = progressive_decision("cvrp", 200, rows)
        self.assertFalse(decision["allowed"])
        self.assertEqual(decision["status"], "PRECHECK_FAIL")

    def test_summary_marks_all_larger_sizes_not_run_after_oom(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            path = root / "tsp100"; path.mkdir()
            record = {"problem": "TSP", "size": 100, "status": "CUDA_OOM"}
            (path / "record.json").write_text(json.dumps(record))
            rows = build_summary(root)["rows"]
            statuses = {(row["problem"], row["size"]): row["status"] for row in rows}
            self.assertEqual(statuses[("tsp", 100)], "CUDA_OOM")
            self.assertEqual(statuses[("tsp", 10000)], "NOT_RUN_AFTER_OOM")

    def test_memory_stat_bookkeeping_bytes_and_gib(self):
        class Properties:
            total_memory = 24 * 1024 ** 3

        class Cuda:
            memory_allocated = staticmethod(lambda: 2 * 1024 ** 3)
            memory_reserved = staticmethod(lambda: 3 * 1024 ** 3)
            max_memory_allocated = staticmethod(lambda: 4 * 1024 ** 3)
            max_memory_reserved = staticmethod(lambda: 5 * 1024 ** 3)
            get_device_properties = staticmethod(lambda _index: Properties())
            get_device_name = staticmethod(lambda _index: "NVIDIA GeForce RTX 4090")

        class Torch:
            cuda = Cuda()

        result = memory_baseline(Torch)
        self.assertEqual(result["memory_before_solve_allocated"],
                         {"bytes": 2 * 1024 ** 3, "gib": 2.0})
        self.assertEqual(result["memory_before_solve_reserved"]["gib"], 3.0)
        self.assertEqual(result["gpu_total_memory"]["gib"], 24.0)
        self.assertEqual(bytes_and_gib(1024 ** 3), {"bytes": 1024 ** 3, "gib": 1.0})
        peaks = cuda_memory_snapshot(Torch)
        self.assertEqual(peaks["peak_allocated_bytes"], 4 * 1024 ** 3)
        self.assertEqual(peaks["peak_allocated_gib"], 4.0)
        self.assertEqual(peaks["peak_reserved_bytes"], 5 * 1024 ** 3)
        self.assertEqual(peaks["peak_reserved_gib"], 5.0)


if __name__ == "__main__":
    unittest.main()
