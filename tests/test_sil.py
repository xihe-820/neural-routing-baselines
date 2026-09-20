import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from methods.sil.config import (CHECKPOINTS, SIZE_REGISTRY, resolve_config,
                                validate_checkpoint_path)
from methods.sil.cvrp.adapter import (adapt_task as adapt_cvrp,
                                      canonical_to_official,
                                      decode_official_solution as decode_cvrp)
from methods.sil.paper_eval import main as paper_main
from methods.sil.paper_results import append, finalize, initialize
from methods.sil.runtime import capture_official_solution
from methods.sil.tsp.adapter import (adapt_task as adapt_tsp,
                                     decode_official_solution as decode_tsp)
from problems.cvrp.validate import validate as validate_cvrp
from problems.tsp.validate import validate as validate_tsp


class Task:
    pass


class SILConfigTests(unittest.TestCase):
    def test_all_size_checkpoint_mappings(self):
        expected = {
            ("tsp", 500): ("tsp1k", 1000, "senior_approved_adaptation"),
            ("tsp", 1000): ("tsp1k", 1000, "official_native"),
            ("tsp", 2000): ("tsp1k", 1000, "senior_approved_adaptation"),
            ("tsp", 5000): ("tsp5k", 5000, "official_native"),
            ("tsp", 10000): ("tsp10k", 10000, "official_native"),
            ("cvrp", 500): ("cvrp1k", 1000, "senior_approved_adaptation"),
            ("cvrp", 1000): ("cvrp1k", 1000, "official_native"),
            ("cvrp", 2000): ("cvrp1k", 1000, "senior_approved_adaptation"),
        }
        self.assertEqual(SIZE_REGISTRY, expected)
        for (problem, size), (checkpoint, adapted, origin) in expected.items():
            config = resolve_config(problem, size, "fewer")
            self.assertEqual(config["checkpoint_key"], checkpoint)
            self.assertEqual(config["adapted_from_size"], adapted)
            self.assertEqual(config["config_origin"], origin)

    def test_fewer_more_are_same_pipeline_different_budget(self):
        fewer = resolve_config("tsp", 1000, "fewer")
        more = resolve_config("tsp", 1000, "more")
        self.assertEqual(fewer["budget"], 50)
        self.assertEqual(more["budget"], 500)
        for field in ("random_insertion", "PRC", "repair_max_sub_length",
                      "pomo_size", "decode_method", "seed"):
            self.assertEqual(fewer[field], more[field])
        self.assertFalse(fewer["protocol_pending"])

    def test_knn_strict_threshold(self):
        self.assertFalse(resolve_config("tsp", 500, "fewer")["initial_knn_path_active"])
        self.assertFalse(resolve_config("tsp", 1000, "fewer")["initial_knn_path_active"])
        self.assertTrue(resolve_config("tsp", 2000, "fewer")["initial_knn_path_active"])

    def test_formal_batch_is_one(self):
        with self.assertRaises(ValueError):
            resolve_config("tsp", 1000, "fewer", batch_size=2)

    def test_checkpoint_filename_gate(self):
        config = resolve_config("cvrp", 500, "more")
        validate_checkpoint_path(config, Path("/tmp/checkpoint-cvrp1k.pt"))
        with self.assertRaises(ValueError):
            validate_checkpoint_path(config, Path("/tmp/wrong.pt"))

    def test_registry_has_no_claimed_local_sha(self):
        self.assertTrue(all(row["actual_sha256"] is None for row in CHECKPOINTS.values()))

    def test_dump_config_does_not_import_official_runtime(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = paper_main(["--problem", "tsp", "--problem-size", "500",
                                 "--budget", "fewer", "--dump-config"])
        payload = json.loads(stream.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["checkpoint_key"], "tsp1k")


class SILAdapterAndValidationTests(unittest.TestCase):
    def test_tsp_adapter_and_decoder_preserve_real_size(self):
        task = Task()
        task.points = np.arange(20, dtype=np.float64).reshape(10, 2)
        task.ref_sol = np.arange(10, dtype=np.int64)
        source, model, reference, semantics = adapt_tsp(task, problem_size=10)
        self.assertEqual(source.shape, (10, 2))
        self.assertEqual(model.shape, (10, 2))
        self.assertEqual(model.dtype, np.float32)
        self.assertEqual(reference.tolist(), list(range(10)))
        self.assertFalse(semantics["padding"])
        decoded = decode_tsp(np.array([4, 5, 6, 7, 8, 9, 0, 1, 2, 3]), problem_size=10)
        self.assertEqual(decoded["canonical_solution"], list(range(10)) + [0])

    def test_tsp_independent_validator_malformed_routes(self):
        points = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        self.assertTrue(validate_tsp(points, [0, 1, 2])["feasible"])
        self.assertFalse(validate_tsp(points, [0, 1, 1])["feasible"])
        self.assertFalse(validate_tsp(points, [0, 1])["feasible"])
        self.assertFalse(validate_tsp(points, [0, 1, 3])["feasible"])

    def _cvrp_task(self):
        task = Task()
        task.depots = np.array([[0.0, 0.0]], dtype=np.float64)
        task.points = np.array([[1., 0.], [2., 0.], [0., 1.], [0., 2.]], dtype=np.float64)
        task.demands = np.array([4, 6, 3, 7], dtype=np.int64)
        task.capacity = 10
        task.ref_sol = np.array([0, 1, 2, 0, 3, 4, 0], dtype=np.int64)
        return task

    def test_cvrp_adapter_passes_raw_demand_and_true_capacity(self):
        task = self._cvrp_task()
        (_, _, raw, capacity, coordinates, native_demand,
         reference, semantics) = adapt_cvrp(task, problem_size=4)
        self.assertEqual(raw.tolist(), [4, 6, 3, 7])
        self.assertEqual(capacity, 10)
        self.assertEqual(native_demand.tolist(), [0, 4, 6, 3, 7])
        self.assertEqual(coordinates.shape, (5, 2))
        self.assertEqual(reference.tolist(), [[1, 1], [2, 0], [3, 1], [4, 0]])
        self.assertFalse(semantics["normalization_before_official_model"])
        self.assertFalse(semantics["synthetic_capacity_helper_used"])

    def test_cvrp_route_start_flag_decode(self):
        official = np.array([[3, 1], [4, 0], [1, 1], [2, 0]], dtype=np.int64)
        decoded = decode_cvrp(official, problem_size=4)
        self.assertEqual(decoded["routes"], [[3, 4], [1, 2]])
        self.assertEqual(decoded["canonical_solution"], [0, 3, 4, 0, 1, 2, 0])

    def test_cvrp_independent_validator_failures(self):
        task = self._cvrp_task()
        self.assertTrue(validate_cvrp(task.depots, task.points, task.demands,
                                      task.capacity, task.ref_sol)["feasible"])
        self.assertFalse(validate_cvrp(task.depots, task.points, task.demands,
                                       task.capacity, [0, 1, 1, 3, 4, 0])["feasible"])
        self.assertFalse(validate_cvrp(task.depots, task.points, task.demands,
                                       task.capacity, [0, 1, 2, 3, 0])["feasible"])
        self.assertFalse(validate_cvrp(task.depots, task.points, task.demands,
                                       9, [0, 1, 2, 0, 3, 4, 0])["feasible"])

    def test_bad_cvrp_reference_fails_closed(self):
        with self.assertRaises(ValueError):
            canonical_to_official([0, 1, 1, 0], problem_size=2)

    def test_cvrp_fractional_raw_units_fail_before_official_integer_cast(self):
        task = self._cvrp_task()
        task.demands = task.demands.astype(float)
        task.demands[0] = 4.5
        with self.assertRaisesRegex(ValueError, "integral raw demands"):
            adapt_cvrp(task, problem_size=4)
        task = self._cvrp_task()
        task.capacity = 10.5
        with self.assertRaisesRegex(ValueError, "integral raw demands"):
            adapt_cvrp(task, problem_size=4)


class SILCaptureAndArtifactTests(unittest.TestCase):
    def test_capture_forwards_calls_and_restores_method(self):
        class Env:
            def __init__(self):
                self.calls = 0

            def _get_travel_distance_2(self, problems, solution, **kwargs):
                self.calls += 1
                return np.asarray([float(np.asarray(solution).sum())])

        env = Env()
        original = env._get_travel_distance_2
        solution = np.array([[0, 1, 2]])
        with capture_official_solution(env, problem="tsp", problem_size=3) as captured:
            result = env._get_travel_distance_2(None, solution)
        self.assertEqual(env.calls, 1)
        self.assertEqual(result.tolist(), [3.0])
        self.assertIs(captured["solution"], solution)
        self.assertEqual(captured["eligible_calls"], 1)
        self.assertEqual(env._get_travel_distance_2.__func__, original.__func__)

    def test_artifact_aggregates_mean_instance_gap_and_resume(self):
        identity = {
            "scope": "fullset", "offset": 0, "indices": [0, 1],
            "dataset": {"count": 2},
            "protocol": {"problem": "TSP", "actual_problem_size": 3,
                         "budget_label": "fewer", "budget": 50},
        }
        records = []
        for index, objective, reference in ((0, 2.0, 1.0), (1, 3.0, 2.0)):
            records.append({
                "dataset_instance_index": index, "independent_objective": objective,
                "reference_objective": reference,
                "gap_percent": (objective-reference)/reference*100,
                "evidence_status": "KIT_VALIDATED", "independent_feasible": True,
                "kit_feasible": True, "official_vs_independent": {"pass": True},
                "independent_vs_kit": {"pass": True},
            })
        timings = [{"dataset_instance_index": i, "runtime_seconds": i + 1.0}
                   for i in range(2)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            metadata, empty_records, empty_timings, checkpoint = initialize(
                root, identity, resume=False)
            self.assertEqual((empty_records, empty_timings, checkpoint), ([], [], None))
            live_records, live_timings = [], []
            for record, timing in zip(records, timings):
                append(root, metadata, live_records, live_timings, record, timing,
                       {"test_rng_state": record["dataset_instance_index"]})
            summary = finalize(root, metadata, live_records, live_timings)
            self.assertEqual(summary["status"], "PAPER_READY")
            self.assertAlmostEqual(summary["mean_instance_gap_percent"], 75.0)
            self.assertAlmostEqual(summary["total_runtime_seconds"], 3.0)
            loaded = json.loads((root / "summary.json").read_text())
            self.assertEqual(loaded["status"], "PAPER_READY")
            with self.assertRaises(ValueError):
                initialize(root, identity, resume=False)
            resumed = initialize(root, identity, resume=True)
            self.assertEqual(len(resumed[1]), 2)

    def test_resume_refuses_hash_tamper_and_unowned_empty_directory(self):
        identity = {"indices": [], "scope": "preflight", "dataset": {"count": 1},
                    "protocol": {"problem": "TSP", "actual_problem_size": 3,
                                 "budget_label": "fewer", "budget": 50},
                    "offset": 0}
        with tempfile.TemporaryDirectory() as temp:
            empty = Path(temp) / "empty"
            empty.mkdir()
            with self.assertRaises(ValueError):
                initialize(empty, identity, resume=True)
        identity["indices"] = [0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            metadata, records, timings, _ = initialize(root, identity, resume=False)
            record = {"dataset_instance_index": 0}
            timing = {"dataset_instance_index": 0, "runtime_seconds": 1.0}
            append(root, metadata, records, timings, record, timing, {"state": 1})
            with (root / "validated_records.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "records hash mismatch"):
                initialize(root, identity, resume=True)


if __name__ == "__main__":
    unittest.main()
