import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from common.hashing import sha256_file
from methods.udc.adapter import (adapt_cvrp_task, adapt_tsp_task,
                                 decode_cyclic_routes,
                                 validate_cvrp_population,
                                 validate_tsp_population)
from methods.udc.protocol import (DATASET_FILENAMES, OFFICIAL_COMMIT,
                                  S2_SCRIPT_SHA256, discover_datasets,
                                  s2_gate)
from methods.udc.s3_eval import prior_our2_gate
from methods.udc import s3_eval


class Task:
    pass


class UdcS3AdapterTests(unittest.TestCase):
    def test_tsp_adapter_preserves_order_and_dtype_audit(self):
        task = Task()
        task.points = np.arange(1000, dtype=np.float64).reshape(500, 2) / 1000
        source, native, evidence = adapt_tsp_task(task)
        np.testing.assert_array_equal(source, task.points)
        np.testing.assert_array_equal(native, task.points.astype(np.float32))
        self.assertEqual(source.dtype, np.float64)
        self.assertEqual(native.dtype, np.float32)
        self.assertEqual(evidence["source_coordinate_dtype"], "float64")
        self.assertEqual(evidence["model_input_dtype"], "float32")
        self.assertTrue(evidence["model_input_dtype_cast"])
        self.assertEqual(evidence["node_order"], "unchanged")

    def test_cvrp_raw_demand_is_normalized_exactly_once(self):
        task = Task()
        task.depots = np.array([[0.5, 0.5]])
        task.points = np.zeros((500, 2), dtype=np.float32)
        task.demands = np.arange(500) % 10 + 1
        task.capacity = 50
        coordinates, demand, evidence = adapt_cvrp_task(task)
        self.assertEqual(coordinates.shape, (501, 2))
        np.testing.assert_allclose(demand[1:], task.demands / 50)
        self.assertEqual(evidence["input_demand_representation"], "raw_integer")
        self.assertEqual(evidence["normalization_applied"],
                         "raw_demand / capacity exactly once")

    def test_normalized_cvrp_input_fails_closed(self):
        task = Task()
        task.depots = np.zeros((1, 2)); task.points = np.zeros((500, 2))
        task.demands = np.full(500, 0.1); task.capacity = 1
        with self.assertRaises(ValueError):
            adapt_cvrp_task(task)

    def test_cyclic_boundary_semantics(self):
        self.assertEqual(decode_cyclic_routes([1, 2, 3], [1, 0, 1]),
                         [[1], [2, 3]])

    def test_tsp_best_feasible_alpha(self):
        points = np.zeros((500, 2))
        population = np.tile(np.arange(500), (50, 1))
        population[0, 0] = 1
        result = validate_tsp_population(points, population)
        self.assertEqual(result["best_alpha"], 1)
        self.assertFalse(result["candidate_feasible"][0])
        self.assertEqual(result["solution"][0], 0)
        self.assertEqual(result["solution"][-1], 0)
        self.assertEqual(len(result["solution"]), 501)

    def test_cvrp_population_selects_feasible(self):
        points = np.column_stack((np.arange(1, 501), np.zeros(500)))
        solutions = np.tile(np.arange(1, 501), (50, 1))
        flags = np.ones((50, 500), dtype=int)
        solutions[0, 0] = 2
        result = validate_cvrp_population(
            np.array([0, 0]), points, np.ones(500), 1, solutions, flags)
        self.assertEqual(result["best_alpha"], 1)
        self.assertEqual(result["route_count"], 500)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    def __getitem__(self, key):
        return FakeTensor(self.value[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeTorch:
    float32 = "float32"

    class cuda:
        @staticmethod
        def synchronize():
            pass

    @staticmethod
    def as_tensor(value, **_kwargs):
        return FakeTensor(value)

    @staticmethod
    def stack(values, dim=0):
        return FakeTensor(np.stack([value.value for value in values], axis=dim))

    @staticmethod
    def inference_mode():
        return FakeInferenceMode()


class UdcS3SolveStateTests(unittest.TestCase):
    def _task(self):
        task = Task()
        task.name = "fake"
        task.depots = np.zeros(2)
        task.points = np.zeros((500, 2))
        task.demands = np.ones(500)
        task.capacity = 100
        task.ref_sol = np.array([0, 1, 0])
        task.check_constraints = lambda _solution: True
        task.evaluate = lambda _solution: 1.0
        return task

    def test_two_cvrp_solves_restore_configured_pomo_between_instances(self):
        class Env:
            pomo_size = 10

            def cal_length_total2(self, *_args):
                return FakeTensor(np.ones((1, 50)))

        class Harness:
            def __init__(self, env):
                self.env = env
                self.pomo_seen_at_load = []

            def _load_init_sol(self, *_args):
                self.pomo_seen_at_load.append(self.env.pomo_size)
                solutions = FakeTensor(np.tile(np.arange(1, 501), (50, 1)))
                flags = FakeTensor(np.ones((50, 500), dtype=np.int64))
                return [solutions], [flags]

            def route_ranking2(self, _coordinates, solution, flags):
                return solution, flags

            def _test_one_batch(self, solution, flags, *_args):
                return solution, flags, 1.0, 1.0

        independent = {
            "best_alpha": 0, "best_objective": 1.0,
            "solution": list(range(1, 501)), "solution_flag": [1] * 500,
            "canonical_solution": [0, *range(1, 501), 0],
            "decoded_routes": [list(range(1, 501))],
            "route_demands": [500.0], "route_count": 1,
            "max_route_load": 500.0,
            "independent_objective_per_alpha": [1.0] * 50,
        }
        env = Env()
        harness = Harness(env)
        adapted = (np.zeros((501, 2)), np.zeros(501), {})
        with mock.patch.object(s3_eval, "adapt_cvrp_task", return_value=adapted), \
                mock.patch.object(s3_eval, "validate_cvrp_population",
                                  return_value=independent), \
                mock.patch.object(s3_eval, "rng_digest", return_value={}):
            s3_eval.solve_cvrp(self._task(), env, harness, FakeTorch, "cpu", 0)
            self.assertEqual(env.pomo_size, 10)
            s3_eval.solve_cvrp(self._task(), env, harness, FakeTorch, "cpu", 1)
        self.assertEqual(harness.pomo_seen_at_load, [1, 1])
        self.assertEqual(env.pomo_size, 10)

    def test_cvrp_exception_also_restores_configured_pomo(self):
        class Env:
            pomo_size = 10

        class Harness:
            def _load_init_sol(self, *_args):
                raise RuntimeError("synthetic inference failure")

        env = Env()
        adapted = (np.zeros((501, 2)), np.zeros(501), {})
        with mock.patch.object(s3_eval, "adapt_cvrp_task", return_value=adapted), \
                mock.patch.object(s3_eval, "rng_digest", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
                s3_eval.solve_cvrp(self._task(), env, Harness(), FakeTorch, "cpu", 0)
        self.assertEqual(env.pomo_size, 10)

    def test_tsp_independent_objective_receives_original_coordinates(self):
        source = np.arange(1000, dtype=np.float64).reshape(500, 2) / 997.0
        model = source.astype(np.float32)
        task = Task()
        task.name = "tsp-fake"; task.points = source
        task.ref_sol = np.r_[np.arange(500), 0]
        task.check_constraints = lambda _solution: True
        task.evaluate = lambda _solution: 1.0

        class Env:
            def _get_travel_distance2(self, *_args):
                return FakeTensor(np.ones((1, 50)))

        class Harness:
            def _load_init_sol(self, *_args):
                return [FakeTensor(np.tile(np.arange(500), (50, 1)))]

            def _test_one_batch(self, solution, *_args, **_kwargs):
                return solution, 1.0, 1.0

        observed = []

        def validate(points, _population):
            observed.append(points)
            return {"best_alpha": 0, "best_objective": 1.0,
                    "solution": [*range(500), 0],
                    "independent_objective_per_alpha": [1.0] * 50}

        with mock.patch.object(s3_eval, "adapt_tsp_task",
                               return_value=(source, model, {})), \
                mock.patch.object(s3_eval, "validate_tsp_population",
                                  side_effect=validate), \
                mock.patch.object(s3_eval, "rng_digest", return_value={}):
            s3_eval.solve_tsp(task, Env(), Harness(), FakeTorch, "cpu", 0)
        self.assertIs(observed[0], source)
        self.assertEqual(observed[0].dtype, np.float64)


class UdcS3ProvenanceTests(unittest.TestCase):
    def test_discovery_requires_exactly_one_file_per_family(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            for filename in DATASET_FILENAMES.values():
                (root / filename).touch()
            found = discover_datasets(root)
            self.assertEqual(set(found), {"tsp", "cvrp"})
            duplicate = root / "copy" / DATASET_FILENAMES["tsp"]
            duplicate.parent.mkdir(); duplicate.touch()
            with self.assertRaises(ValueError):
                discover_datasets(root)

    def _s2(self):
        return {
            "schema": "udc_s2_official_smoke.v1",
            "ready_for_stage_s3_ml4co_adapter": "YES",
            "script_sha256": S2_SCRIPT_SHA256,
            "project_pre": {"head": "15d02f5b95ecb418c19f12dacbe6a2acc7516fed", "pass": True},
            "project_post": {"head": "15d02f5b95ecb418c19f12dacbe6a2acc7516fed", "pass": True},
            "official_pre": {"head": OFFICIAL_COMMIT, "pass": True},
            "official_post": {"head": OFFICIAL_COMMIT, "pass": True},
            "tsp": {"final_pass": True}, "cvrp": {"final_pass": True},
        }

    def test_s2_authoritative_exact_gate(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value) / "s2_official_smoke_v3"; root.mkdir()
            (root / "metadata.json").write_text(json.dumps(self._s2()))
            self.assertTrue(s2_gate(root)["pass"])

    def test_s2_wrong_status_fails_closed(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value) / "s2_official_smoke_v3"; root.mkdir()
            payload = self._s2(); payload["cvrp"]["final_pass"] = False
            (root / "metadata.json").write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                s2_gate(root)

    def test_our5_requires_untampered_our2_artifacts(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            datasets = {}
            for family in ("tsp", "cvrp"):
                dataset = root / DATASET_FILENAMES[family]
                dataset.write_text(family)
                datasets[family] = dataset
            our2 = root / "our_2"; our2.mkdir()
            summaries = {}
            for family in ("tsp", "cvrp"):
                records = our2 / f"{family}500_records.jsonl"
                records.write_text("{}\n")
                summary = {"status": "KIT_VALIDATED", "validated_count": 2,
                           "records_sha256": sha256_file(records)}
                (our2 / f"{family}500_summary.json").write_text(json.dumps(summary))
                summaries[family] = summary
            metadata = {"state": "KIT_VALIDATED", "count": 2,
                        "script_sha256": "abc", "summaries": summaries,
                        "datasets": {family: {"sha256": sha256_file(path)}
                                     for family, path in datasets.items()}}
            (our2 / "metadata.json").write_text(json.dumps(metadata))
            self.assertTrue(prior_our2_gate(our2, datasets, "abc")["pass"])
            (our2 / "tsp500_records.jsonl").write_text("tampered\n")
            with self.assertRaises(ValueError):
                prior_our2_gate(our2, datasets, "abc")


if __name__ == "__main__":
    unittest.main()
