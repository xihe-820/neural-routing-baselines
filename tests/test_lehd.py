import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import types
import unittest

import numpy as np

from methods.lehd.config import (AUTHOR_BATCH_REGISTRY, CHECKPOINTS,
                                 DATASET_FILENAMES, FORMAL_GPU, FORMAL_PROTOCOLS,
                                 FORMAL_SIZES, MODEL_PARAMS, RRC_BUDGETS,
                                 SIZE_ORIGINS, TIMING_PROBE_COUNTS,
                                 resolve_author_batch_config, resolve_config,
                                 validate_checkpoint_location, validate_dataset_path)
from methods.lehd.author_batch_eval import (batch_slices as author_batch_slices,
                                            _cuda_device as author_cuda_device,
                                            prepare_batch as prepare_author_batch,
                                            validate_batch as validate_author_batch,
                                            _verify_runtime_algorithm_config)
from methods.lehd.cvrp.adapter import (adapt_task as adapt_cvrp,
                                       canonical_to_official,
                                       decode_official_solution as decode_cvrp)
from methods.lehd.paper_eval import (_scope_count, _verify_preflight_evidence,
                                     main as paper_main)
from methods.lehd.paper_results import (append, capture_rng_state, finalize,
                                        fingerprint, initialize)
from methods.lehd.runtime import (capture_official_solution,
                                  run_isolated_warmup,
                                  solve_batch as solve_official_batch)
from methods.lehd.timing_probe import timing_count
from methods.lehd.tsp.adapter import (adapt_task as adapt_tsp,
                                      decode_official_solution as decode_tsp)
from problems.cvrp.validate import validate as validate_cvrp
from problems.tsp.validate import validate as validate_tsp


class LEHDConfigTests(unittest.TestCase):
    def test_exact_formal_scope_and_cell_count(self):
        self.assertEqual(FORMAL_SIZES, {
            "tsp": (100, 500, 1000),
            "cvrp": (50, 100, 200, 500, 1000, 2000),
        })
        self.assertEqual(FORMAL_PROTOCOLS, ("greedy", "fewer", "more"))
        self.assertEqual(sum(map(len, FORMAL_SIZES.values())) * len(FORMAL_PROTOCOLS), 27)
        for problem, size in (("tsp", 200), ("tsp", 2000), ("cvrp", 20), ("cvrp", 5000)):
            with self.subTest(problem=problem, size=size), self.assertRaises(ValueError):
                resolve_config(problem, size, "greedy")

    def test_checkpoint_and_train_size_mapping(self):
        for problem, sizes in FORMAL_SIZES.items():
            for size in sizes:
                config = resolve_config(problem, size, "fewer")
                self.assertEqual(config["trained_on_size"], 100)
                self.assertEqual(config["checkpoint"]["trained_on_size"], 100)
                self.assertEqual(config["checkpoint"]["filename"],
                                 "checkpoint-150.pt" if problem == "tsp" else "checkpoint-40.pt")
                self.assertIsNone(config["checkpoint"]["actual_sha256"])

    def test_size_provenance_is_exact(self):
        expected = {
            ("tsp", 100): "official_native_training_size",
            ("tsp", 500): "official_paper_generalization_size",
            ("tsp", 1000): "official_paper_generalization_size",
            ("cvrp", 50): "senior_approved_scale_adaptation",
            ("cvrp", 100): "official_native_training_size",
            ("cvrp", 200): "official_paper_generalization_size",
            ("cvrp", 500): "official_paper_generalization_size",
            ("cvrp", 1000): "official_paper_generalization_size",
            ("cvrp", 2000): "senior_approved_scale_adaptation",
        }
        self.assertEqual(SIZE_ORIGINS, expected)
        for key, origin in expected.items():
            self.assertEqual(resolve_config(*key, "greedy")["config_origin"], origin)

    def test_protocols_are_exact_rrc_budgets(self):
        self.assertEqual(RRC_BUDGETS, {"greedy": 0, "fewer": 50, "more": 500})
        for problem, sizes in FORMAL_SIZES.items():
            for size in sizes:
                for label, budget in RRC_BUDGETS.items():
                    config = resolve_config(problem, size, label)
                    self.assertEqual(config["RRC_budget"], budget)
                    self.assertEqual(config["original_instance_batch_size"], 1)
                    self.assertEqual(config["formal_gpu"], FORMAL_GPU)
                    self.assertNotIn("hardware_" + "protocol_pending", config)
                    self.assertNotIn("manuscript_" + "hardware_statement", config)
                    self.assertEqual(config["budget_mapping_origin"],
                                     "project_protocol_mapping_of_author_reported_budgets")
                    for forbidden in ("PRC", "random_insertion", "repair_max_sub_length",
                                      "pomo_size", "beam_width", "use_k_nearest"):
                        self.assertNotIn(forbidden, config)

    def test_architecture_matches_official_test_files(self):
        self.assertEqual(MODEL_PARAMS, {
            "mode": "test", "embedding_dim": 128,
            "sqrt_embedding_dim": 128 ** 0.5, "decoder_layer_num": 6,
            "qkv_dim": 16, "head_num": 8, "ff_hidden_dim": 512,
        })

    def test_dataset_registry_and_path_gate(self):
        self.assertEqual(len(DATASET_FILENAMES), 9)
        for (problem, size), filename in DATASET_FILENAMES.items():
            config = resolve_config(problem, size, "greedy")
            self.assertEqual(config["expected_dataset_filename"], filename)
            validate_dataset_path(config, Path("/data") / filename)
            with self.assertRaises(ValueError):
                validate_dataset_path(config, Path("/data/wrong.pkl"))

    def test_checkpoint_must_be_repository_owned_asset(self):
        config = resolve_config("tsp", 500, "greedy")
        with tempfile.TemporaryDirectory() as temp:
            upstream = Path(temp)
            exact = upstream / config["checkpoint"]["relative_path"]
            validate_checkpoint_location(config, exact, upstream)
            with self.assertRaises(ValueError):
                validate_checkpoint_location(
                    config, upstream / "copy/checkpoint-150.pt", upstream)

    def test_smoke_and_preflight_counts(self):
        for problem, sizes in FORMAL_SIZES.items():
            for size in sizes:
                self.assertEqual(_scope_count(problem, size, "greedy", "our-smoke"), 1)
                expected = 1 if problem == "cvrp" and size == 2000 else 2
                self.assertEqual(_scope_count(problem, size, "fewer", "our-smoke"), expected)
                self.assertEqual(_scope_count(problem, size, "more", "preflight"), 1)
        with self.assertRaises(ValueError):
            _scope_count("tsp", 100, "more", "our-smoke")

    def test_dump_config_is_dependency_light(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = paper_main([
                "--problem", "cvrp", "--problem-size", "2000",
                "--protocol", "more", "--dump-config",
            ])
        payload = json.loads(stream.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["RRC_budget"], 500)
        self.assertEqual(payload["trained_on_size"], 100)


class LEHDAdapterTests(unittest.TestCase):
    @staticmethod
    def _tsp_task(dtype=np.float64):
        return types.SimpleNamespace(
            points=np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=dtype),
            ref_sol=np.asarray([2, 3, 0, 1, 2], dtype=np.int64),
        )

    @staticmethod
    def _cvrp_task():
        return types.SimpleNamespace(
            depots=np.asarray([[0.0, 0.0]]),
            points=np.asarray([[1, 0], [2, 0], [0, 1], [0, 2]], dtype=float),
            demands=np.asarray([4, 6, 3, 7], dtype=np.int64), capacity=10,
            ref_sol=np.asarray([0, 1, 2, 0, 3, 4, 0], dtype=np.int64),
        )

    def test_tsp_adapter_preserves_source_and_decode(self):
        task = self._tsp_task()
        source, model, reference, audit = adapt_tsp(task, problem_size=4)
        self.assertIs(source, task.points)
        self.assertEqual(model.dtype, np.float32)
        self.assertEqual(reference.tolist(), [0, 1, 2, 3])
        self.assertTrue(audit["model_input_dtype_cast"])
        decoded = decode_tsp(np.asarray([3, 2, 1, 0]), problem_size=4)
        self.assertEqual(decoded["canonical_solution"], [0, 3, 2, 1, 0])
        self.assertTrue(validate_tsp(task.points, decoded["canonical_solution"])["feasible"])
        with self.assertRaises(ValueError):
            decode_tsp(np.asarray([0, 1, 1, 3]), problem_size=4)

    def test_cvrp_raw_units_and_bidirectional_route_representation(self):
        task = self._cvrp_task()
        adapted = adapt_cvrp(task, problem_size=4)
        depot, points, demands, capacity, coords, model_demands, reference, audit = adapted
        self.assertEqual(demands.tolist(), [4, 6, 3, 7])
        self.assertEqual(capacity, 10)
        self.assertEqual(model_demands.tolist(), [0, 4, 6, 3, 7])
        self.assertEqual(reference.tolist(), [[1, 1], [2, 0], [3, 1], [4, 0]])
        self.assertFalse(audit["normalization_before_official_model"])
        decoded = decode_cvrp(reference, problem_size=4)
        self.assertEqual(decoded["canonical_solution"], [0, 1, 2, 0, 3, 4, 0])
        self.assertTrue(validate_cvrp(
            depot, points, demands, capacity, decoded["canonical_solution"])["feasible"])
        nodes, flags = canonical_to_official(task.ref_sol, problem_size=4)
        self.assertEqual(nodes.tolist(), [1, 2, 3, 4])
        self.assertEqual(flags.tolist(), [1, 0, 1, 0])

    def test_fractional_demand_or_capacity_fails_closed(self):
        task = self._cvrp_task()
        task.demands = task.demands.astype(float)
        task.demands[0] = 4.5
        with self.assertRaisesRegex(ValueError, "integral raw demands"):
            adapt_cvrp(task, problem_size=4)
        task = self._cvrp_task()
        task.capacity = 10.5
        with self.assertRaisesRegex(ValueError, "integral raw demands"):
            adapt_cvrp(task, problem_size=4)

    def test_cvrp_decoder_rejects_duplicate_missing_and_bad_flags(self):
        with self.assertRaises(ValueError):
            decode_cvrp(np.asarray([[1, 1], [1, 0], [3, 1], [4, 0]]), problem_size=4)
        with self.assertRaises(ValueError):
            decode_cvrp(np.asarray([[1, 2], [2, 0], [3, 1], [4, 0]]), problem_size=4)


class LEHDRuntimeTests(unittest.TestCase):
    def test_capture_retains_last_eligible_call_and_restores_method(self):
        class Env:
            def __init__(self):
                self.calls = []

            def _get_travel_distance_2(self, problems, solution):
                self.calls.append(solution)
                return np.asarray([float(np.asarray(solution).sum())])

        env = Env()
        original = env._get_travel_distance_2
        first = np.asarray([[0, 1, 2]])
        last = np.asarray([[2, 1, 0]])
        with capture_official_solution(env, problem="tsp", problem_size=3) as observed:
            env._get_travel_distance_2(None, first)
            env._get_travel_distance_2(None, np.asarray([[0, 1]]))
            env._get_travel_distance_2(None, last)
        self.assertEqual(len(env.calls), 3)
        self.assertIs(observed["solution"], last)
        self.assertEqual(observed["eligible_calls"], 2)
        self.assertEqual(env._get_travel_distance_2.__func__, original.__func__)

    def test_isolated_warmup_restores_rng_and_discards_state(self):
        import torch
        random.seed(7)
        np.random.seed(8)
        torch.manual_seed(9)
        before = capture_rng_state(torch)
        objects = []

        def build():
            value = types.SimpleNamespace(mutated=False)
            objects.append(value)
            torch.rand(2)
            return value

        def solve(value):
            value.mutated = True
            random.random()
            np.random.random()
            torch.rand(3)

        run_isolated_warmup(build, solve, torch=torch)
        self.assertEqual(capture_rng_state(torch), before)
        formal = build()
        self.assertTrue(objects[0].mutated)
        self.assertFalse(formal.mutated)
        self.assertIsNot(objects[0], formal)

    def test_official_rng_and_strict_load_source_semantics(self):
        root = Path("external/NCO_code/single_objective/LEHD")
        tsp = (root / "TSP/TSPTester.py").read_text()
        cvrp = (root / "CVRP/VRPTester.py").read_text()
        self.assertNotIn("manual_seed", tsp)
        self.assertIn("torch.manual_seed(random_seed)", cvrp)
        self.assertIn("random_seed = 12", cvrp)
        self.assertIn("self.model.load_state_dict(checkpoint['model_state_dict'])", tsp)
        self.assertIn("self.model.load_state_dict(checkpoint['model_state_dict'])", cvrp)
        self.assertIn("for bbbb in range(budget)", tsp)
        self.assertIn("for bbbb in range(budget)", cvrp)


class LEHDAuthorBatchProtocolTests(unittest.TestCase):
    class _ArrayTorch:
        float32 = np.float32
        long = np.int64

        @staticmethod
        def as_tensor(value, *, dtype, device):
            return np.asarray(value, dtype=dtype)

        @staticmethod
        def zeros(*shape, dtype, device):
            return np.zeros(shape, dtype=dtype)

    def test_cuda_gate_accepts_rtx_4090_runtime_aliases_only(self):
        class Device:
            type, index = "cuda", 0

        class Cuda:
            def __init__(self, name):
                self.name, self.selected = name, None

            @staticmethod
            def is_available():
                return True

            def set_device(self, device):
                self.selected = device

            def get_device_name(self, device):
                return self.name

        class Torch:
            def __init__(self, name):
                self.cuda = Cuda(name)

            @staticmethod
            def device(value):
                return Device()

        self.assertEqual(FORMAL_GPU, "NVIDIA RTX 4090")
        for name in ("NVIDIA GeForce RTX 4090", "NVIDIA RTX 4090"):
            with self.subTest(name=name):
                torch = Torch(name)
                self.assertIsInstance(author_cuda_device(torch, "cuda:0"), Device)
                self.assertIsNotNone(torch.cuda.selected)
        with self.assertRaisesRegex(RuntimeError, "RTX 4090"):
            author_cuda_device(Torch("NVIDIA A100-SXM4-80GB"), "cuda:0")

    @staticmethod
    def _tsp_task(offset=0.0):
        points = np.asarray([[0, 0], [1 + offset, 0], [0, 1 + offset]], dtype=np.float64)
        task = types.SimpleNamespace(points=points, ref_sol=np.asarray([0, 1, 2], dtype=np.int64))
        task.check_constraints = lambda route: validate_tsp(points, route)["feasible"]
        task.evaluate = lambda route: validate_tsp(points, route)["independent_objective"]
        return task

    @staticmethod
    def _cvrp_task(offset=0.0, capacity=5):
        points = np.asarray([[1 + offset, 0], [2 + offset, 0], [0, 1 + offset],
                             [0, 2 + offset]], dtype=np.float64)
        demands = np.asarray([2, 2, 2, 2], dtype=np.int64)
        task = types.SimpleNamespace(
            depots=np.asarray([[0.0, 0.0]]), points=points, demands=demands,
            capacity=capacity, ref_sol=np.asarray([0, 1, 2, 0, 3, 4, 0], dtype=np.int64))
        task.check_constraints = lambda route: validate_cvrp(
            task.depots, points, demands, capacity, route)["feasible"]
        task.evaluate = lambda route: validate_cvrp(
            task.depots, points, demands, capacity, route)["independent_objective"]
        return task

    def test_author_batch_registry_and_adaptation_provenance(self):
        expected = {
            ("tsp", 100): (1280, 1280), ("tsp", 500): (128, 128),
            ("tsp", 1000): (128, 128), ("cvrp", 50): (10000, 10000),
            ("cvrp", 100): (10000, 10000), ("cvrp", 200): (100, 100),
            ("cvrp", 500): (100, 100), ("cvrp", 1000): (100, 100),
            ("cvrp", 2000): (100, 100),
        }
        self.assertEqual({key: (value["dataset_count"], value["batch_size"])
                          for key, value in AUTHOR_BATCH_REGISTRY.items()}, expected)
        for key in (("cvrp", 50), ("cvrp", 2000)):
            protocol = resolve_author_batch_config(*key, "fewer")
            self.assertEqual(protocol["config_origin"], "senior_approved_project_adaptation")
            self.assertTrue(protocol["batch_protocol_origin"].startswith("adapted_from_"))
        protocol = resolve_author_batch_config("tsp", 100, "greedy")
        self.assertEqual(protocol["artifact_class"], "baseline_result_reproduction")
        self.assertEqual(protocol["original_instance_batch_size"], 1280)
        self.assertEqual(protocol["batch_size_requested"], 1280)
        with self.assertRaises(ValueError):
            resolve_author_batch_config("tsp", 100, "greedy", batch_size=64)
        overridden = resolve_author_batch_config(
            "tsp", 100, "greedy", batch_size=64, batch_override_reason="documented OOM")
        self.assertEqual(overridden["batch_size_requested"], 64)
        _verify_runtime_algorithm_config(resolve_config("tsp", 100, "greedy"), protocol)
        bad_algorithm = dict(protocol)
        bad_algorithm["RRC_budget"] = 50
        with self.assertRaises(RuntimeError):
            _verify_runtime_algorithm_config(resolve_config("tsp", 100, "greedy"), bad_algorithm)

    def test_real_batch_stack_injection_and_raw_capacity_gate(self):
        torch = self._ArrayTorch()
        tsp = prepare_author_batch("tsp", [self._tsp_task(), self._tsp_task(2)], 3,
                                   device="cuda:0", torch=torch)
        tsp_env = types.SimpleNamespace()
        tsp["inject"](tsp_env)
        self.assertEqual(tsp_env.raw_data_nodes.shape, (2, 3, 2))
        self.assertEqual(tsp_env.raw_data_tours.shape, (2, 3))
        cvrp_tasks = [self._cvrp_task(), self._cvrp_task(2)]
        cvrp = prepare_author_batch("cvrp", cvrp_tasks, 4, device="cuda:0", torch=torch)
        cvrp_env = types.SimpleNamespace()
        cvrp["inject"](cvrp_env)
        self.assertEqual(cvrp_env.raw_data_nodes.shape, (2, 5, 2))
        self.assertEqual(cvrp_env.raw_data_demand.shape, (2, 5))
        self.assertEqual(cvrp_env.raw_data_capacity.shape, (2,))
        self.assertEqual(cvrp_env.raw_data_node_flag.shape, (2, 4, 2))
        with self.assertRaisesRegex(ValueError, "identical true capacities"):
            prepare_author_batch("cvrp", [self._cvrp_task(), self._cvrp_task(capacity=6)],
                                 4, device="cuda:0", torch=torch)

    def test_per_instance_validation_and_kit_aggregation(self):
        torch = self._ArrayTorch()
        tsp_tasks = [self._tsp_task(), self._tsp_task(2)]
        tsp = prepare_author_batch("tsp", tsp_tasks, 3, device="cuda:0", torch=torch)
        tsp_solutions = np.asarray([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
        tsp_records = validate_author_batch(
            "tsp", tsp_tasks, tsp, tsp_solutions,
            np.asarray([task.evaluate([*solution, solution[0]])
                        for task, solution in zip(tsp_tasks, tsp_solutions)]),
            3, dataset_offset=7)
        self.assertEqual([row["dataset_instance_index"] for row in tsp_records], [7, 8])
        self.assertTrue(all(row["independent_feasible"] and row["kit_feasible"]
                            for row in tsp_records))
        cvrp_tasks = [self._cvrp_task(), self._cvrp_task(2)]
        cvrp = prepare_author_batch("cvrp", cvrp_tasks, 4, device="cuda:0", torch=torch)
        cvrp_solutions = np.asarray([[[1, 1], [2, 0], [3, 1], [4, 0]],
                                     [[3, 1], [4, 0], [1, 1], [2, 0]]], dtype=np.int64)
        cvrp_records = validate_author_batch(
            "cvrp", cvrp_tasks, cvrp, cvrp_solutions,
            np.asarray([task.evaluate([0, 1, 2, 0, 3, 4, 0]) for task in cvrp_tasks]),
            4, dataset_offset=10)
        self.assertEqual([row["dataset_instance_index"] for row in cvrp_records], [10, 11])
        self.assertTrue(all(row["official_vs_independent"]["pass"] and
                            row["independent_vs_kit"]["pass"] for row in cvrp_records))

    def test_batch_capture_shapes_objective_vector_and_timing_separation(self):
        class Env:
            def _get_travel_distance_2(self, problems, solution):
                return np.asarray([3.0, 4.0])

        env = Env()
        tsp = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
        with capture_official_solution(env, problem="tsp", problem_size=3, batch_size=2) as captured:
            env._get_travel_distance_2(None, tsp)
        self.assertIs(captured["solution"], tsp)
        self.assertEqual(captured["objective"].shape, (2,))
        cvrp = np.asarray([[[1, 1], [2, 0]], [[2, 1], [1, 0]]], dtype=np.int64)
        with capture_official_solution(env, problem="cvrp", problem_size=2, batch_size=2) as captured:
            env._get_travel_distance_2(None, cvrp)
        self.assertIs(captured["solution"], cvrp)
        self.assertEqual(captured["objective"].shape, (2,))
        self.assertEqual(author_batch_slices(10, 4), [(0, 4), (4, 8), (8, 10)])
        self.assertEqual({key: timing_count(key, None) for key in FORMAL_PROTOCOLS},
                         TIMING_PROBE_COUNTS)
        with self.assertRaises(ValueError):
            timing_count("greedy", 0)
        author_source = Path("methods/lehd/author_batch_eval.py").read_text()
        timing_source = Path("methods/lehd/timing_probe.py").read_text()
        self.assertIn("np.stack", author_source)
        self.assertIn("solve_batch(", author_source)
        self.assertNotIn("solve_one(", author_source)
        self.assertNotIn("mean_batch_runtime_seconds", author_source)
        self.assertIn("solve_one(", timing_source)
        self.assertNotIn('add_argument("--batch-size"', timing_source)
        self.assertIn('"original_instance_batch_size": 1', timing_source)

    def test_solve_batch_returns_one_solution_and_objective_per_original_instance(self):
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            @property
            def shape(self):
                return self.value.shape

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        solution, objective = Tensor([[0, 1, 2], [2, 1, 0]]), Tensor([3.0, 4.0])

        class Env:
            def _get_travel_distance_2(self, problems, candidate):
                self.seen = candidate
                return objective

        class Tester:
            def __init__(self):
                self.env = Env()
                self.time_estimator_2 = types.SimpleNamespace(reset=lambda: None)

            def _test_one_batch(self, episode, batch_size, **kwargs):
                self.env._get_travel_distance_2(None, solution)
                return (None, 3.5)

        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda device: None))
        solved = solve_official_batch(
            Tester(), problem="tsp", problem_size=3, batch_size=2,
            inject=lambda env: None, device="cuda:0", torch=torch)
        self.assertEqual(solved["solutions"].shape, (2, 3))
        self.assertEqual(solved["official_objectives"].tolist(), [3.0, 4.0])
        self.assertIsNone(solved["runtime_seconds"])


class LEHDArtifactTests(unittest.TestCase):
    @staticmethod
    def _identity():
        protocol = resolve_config("tsp", 100, "fewer")
        return {
            "method": "LEHD", "scope": "fullset", "offset": 0,
            "count": 2, "indices": [0, 1],
            "dataset": {"count": 2, "sha256": "dataset"},
            "checkpoint": {"sha256": "checkpoint"},
            "protocol": protocol, "protocol_fingerprint": fingerprint(protocol),
            "upstream": {"commit": "upstream"}, "project": {"commit": "project"},
            "source_files": [],
        }

    def test_mean_instance_gap_resume_and_paper_ready(self):
        identity = self._identity()
        records = []
        for index, objective, reference in ((0, 2.0, 1.0), (1, 3.0, 2.0)):
            records.append({
                "dataset_instance_index": index, "independent_objective": objective,
                "reference_objective": reference,
                "gap_percent": (objective - reference) / reference * 100,
                "evidence_status": "KIT_VALIDATED", "independent_feasible": True,
                "kit_feasible": True, "official_vs_independent": {"pass": True},
                "independent_vs_kit": {"pass": True},
            })
        timings = [{"dataset_instance_index": index, "runtime_seconds": index + 1.0}
                   for index in range(2)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            metadata, live_records, live_timings, _ = initialize(
                root, identity, resume=False)
            for record, timing in zip(records, timings):
                append(root, metadata, live_records, live_timings, record, timing,
                       {"state": record["dataset_instance_index"]})
            summary = finalize(root, metadata, live_records, live_timings)
            self.assertEqual(summary["status"], "PAPER_READY")
            self.assertAlmostEqual(summary["mean_instance_gap_percent"], 75.0)
            self.assertAlmostEqual(summary["total_runtime_seconds"], 3.0)
            self.assertTrue(summary["paper_ready"])
            resumed = initialize(root, identity, resume=True)
            self.assertEqual(len(resumed[1]), 2)

    def test_resume_refuses_hash_tamper(self):
        identity = self._identity()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            metadata, records, timings, _ = initialize(root, identity, resume=False)
            append(root, metadata, records, timings,
                   {"dataset_instance_index": 0},
                   {"dataset_instance_index": 0, "runtime_seconds": 1.0},
                   {"state": 1})
            with (root / "validated_records.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "records hash mismatch"):
                initialize(root, identity, resume=True)


class LEHDEvidenceTests(unittest.TestCase):
    PROJECT_COMMIT = "project"
    SOURCE_FILES = [{"path": "methods/lehd/config.py", "sha256": "source"}]

    def _summary(self, protocol_label="fewer", size=100, **changes):
        protocol = resolve_config("tsp", size, protocol_label)
        protocol_hash = fingerprint(protocol)
        scope = "preflight" if protocol_label == "more" else "our-smoke"
        count = _scope_count("tsp", size, protocol_label, scope)
        summary = {
            "status": "KIT_VALIDATED", "method": "LEHD", "problem": "TSP",
            "problem_size": size, "protocol_label": protocol_label,
            "dataset_sha256": "dataset", "checkpoint_sha256": "checkpoint",
            "upstream_commit": "274df3c4975384592b60fe7f79fbb2441ce11c15",
            "project_commit": self.PROJECT_COMMIT,
            "source_provenance_fingerprint": fingerprint(self.SOURCE_FILES),
            "protocol_fingerprint": protocol_hash,
            "count": count, "validated_count": count, "failed_count": 0,
            "identity": {
                "method": "LEHD", "scope": scope, "count": count,
                "protocol": protocol, "protocol_fingerprint": protocol_hash,
                "dataset": {"sha256": "dataset"},
                "checkpoint": {"sha256": "checkpoint"},
                "project": {"commit": self.PROJECT_COMMIT},
                "upstream": {
                    "commit": "274df3c4975384592b60fe7f79fbb2441ce11c15"},
                "source_files": self.SOURCE_FILES,
            },
        }
        summary.update(changes)
        return summary, protocol

    def _verify(self, summary, requested_label=None, requested_size=None):
        label = requested_label or summary["protocol_label"]
        size = requested_size or summary["problem_size"]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "summary.json"
            path.write_text(json.dumps(summary))
            return _verify_preflight_evidence(
                path, problem="tsp", problem_size=size, protocol_label=label,
                dataset_sha256="dataset", checkpoint_sha256="checkpoint",
                protocol_fingerprint=fingerprint(resolve_config("tsp", size, label)),
                project_commit=self.PROJECT_COMMIT, source_files=self.SOURCE_FILES)

    def test_exact_greedy_fewer_more_evidence_passes(self):
        for label in FORMAL_PROTOCOLS:
            summary, _ = self._summary(label)
            self.assertEqual(self._verify(summary)["identity"]["status"], "KIT_VALIDATED")

    def test_wrong_size_protocol_sha_or_source_fails_closed(self):
        cases = (
            {"problem_size": 500}, {"protocol_label": "more"},
            {"dataset_sha256": "wrong"}, {"checkpoint_sha256": "wrong"},
            {"source_provenance_fingerprint": "wrong"},
            {"project_commit": "wrong"}, {"status": "PAPER_READY"},
        )
        for changes in cases:
            summary, _ = self._summary(**changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self._verify(summary, requested_label="fewer", requested_size=100)

    def test_embedded_identity_tamper_fails_closed(self):
        for field in ("protocol", "source_files", "scope"):
            summary, protocol = self._summary()
            if field == "protocol":
                summary["identity"][field] = json.loads(json.dumps(protocol))
                summary["identity"][field]["RRC_budget"] = 500
            elif field == "source_files":
                summary["identity"][field] = []
            else:
                summary["identity"][field] = "preflight"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self._verify(summary)

    def test_fullset_cli_requires_evidence(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            paper_main([
                "--problem", "tsp", "--problem-size", "100", "--protocol", "fewer",
                "--scope", "fullset", "--count", "1", "--output-dir", "/tmp/unused",
            ])


class LEHDManifestTests(unittest.TestCase):
    def test_checkpoint_manifest_has_only_two_unclaimed_hashes(self):
        payload = json.loads(Path("manifests/lehd_checkpoints.json").read_text())
        self.assertEqual(payload["method"], "LEHD")
        self.assertEqual(len(payload["assets"]), 2)
        self.assertTrue(all(asset["sha256"] is None for asset in payload["assets"]))
        self.assertEqual({asset["trained_on_size"] for asset in payload["assets"]}, {100})


if __name__ == "__main__":
    unittest.main()
