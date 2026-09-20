import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from methods.sil.config import (CHECKPOINTS, FORMAL_PROTOCOLS, FORMAL_SIZES,
                                SIZE_REGISTRY, WARMUP_POLICY,
                                effective_repair_max, resolve_config,
                                validate_checkpoint_path, validate_dataset_path)
from methods.sil.cvrp.adapter import (adapt_task as adapt_cvrp,
                                      canonical_to_official,
                                      decode_official_solution as decode_cvrp)
from methods.sil.paper_eval import (_our_smoke_count, _verify_preflight_evidence,
                                    _warmup_required, main as paper_main)
from methods.sil.paper_results import (append, capture_rng_state, finalize,
                                       fingerprint, initialize)
from methods.sil.runtime import (build_env_params, build_tester,
                                 capture_official_solution,
                                 run_rng_preserving_warmup)
from methods.sil.tsp.adapter import (adapt_task as adapt_tsp,
                                     decode_official_solution as decode_tsp)
from problems.cvrp.validate import validate as validate_cvrp
from problems.tsp.validate import validate as validate_tsp


class Task:
    pass


class SILConfigTests(unittest.TestCase):
    OFFICIAL_MODEL_PARAM_KEYS = {
        "tsp": {
            "mode", "embedding_dim", "sqrt_embedding_dim", "encoder_layer_num",
            "qkv_dim", "head_num", "logit_clipping", "ff_hidden_dim",
            "eval_type", "use_k_nearest", "k_nearest_num",
        },
        "cvrp": {
            "mode", "embedding_dim", "sqrt_embedding_dim", "decoder_layer_num",
            "qkv_dim", "head_num", "logit_clipping", "ff_hidden_dim",
            "eval_type", "use_k_nearest", "k_nearest_num",
        },
    }

    def test_model_config_has_complete_pinned_official_parameter_sets(self):
        for problem in ("tsp", "cvrp"):
            with self.subTest(problem=problem):
                model = resolve_config(problem, 1000, "greedy")["model"]
                self.assertEqual(set(model), self.OFFICIAL_MODEL_PARAM_KEYS[problem])
                self.assertEqual(model["mode"], "test")
                self.assertEqual(model["sqrt_embedding_dim"], 128 ** 0.5)

    def test_all_size_checkpoint_mappings(self):
        expected = {
            ("tsp", 1000): ("tsp1k", 1000, None, "official_native"),
            ("tsp", 2000): ("tsp1k", 1000, 1000, "senior_approved_adaptation"),
            ("tsp", 5000): ("tsp5k", 5000, None, "official_native"),
            ("tsp", 10000): ("tsp10k", 10000, None, "official_native"),
            ("cvrp", 1000): ("cvrp1k", 1000, None, "official_native"),
            ("cvrp", 2000): ("cvrp1k", 1000, 1000, "senior_approved_adaptation"),
        }
        self.assertEqual(set(SIZE_REGISTRY), set(expected))
        for (problem, size), (checkpoint, setting, adapted, origin) in expected.items():
            config = resolve_config(problem, size, "fewer")
            self.assertEqual(config["checkpoint_key"], checkpoint)
            self.assertEqual(config["setting_size"], setting)
            self.assertEqual(config["adapted_from_size"], adapted)
            self.assertEqual(config["config_origin"], origin)

    def test_fewer_more_are_same_pipeline_different_budget(self):
        fewer = resolve_config("tsp", 1000, "fewer")
        more = resolve_config("tsp", 1000, "more")
        self.assertEqual(fewer["budget"], 50)
        self.assertEqual(more["budget"], 500)
        for field in ("random_insertion", "PRC", "repair_max_sub_length_nominal",
                      "repair_max_sub_length_effective", "repair_max_rule",
                      "pomo_size", "decode_method", "seed"):
            self.assertEqual(fewer[field], more[field])
        self.assertFalse(fewer["protocol_pending"])

    def test_three_formal_protocols_are_exact(self):
        self.assertEqual(FORMAL_PROTOCOLS, ("greedy", "fewer", "more"))
        expected = {
            "greedy": (0, False, False),
            "fewer": (50, True, True),
            "more": (500, True, True),
        }
        for label, (budget, insertion, knn) in expected.items():
            config = resolve_config("tsp", 1000, label)
            self.assertEqual(config["budget"], budget)
            self.assertEqual(config["random_insertion"], insertion)
            self.assertEqual(config["PRC"], True)
            self.assertEqual(config["use_k_nearest"], knn)
            self.assertTrue(config["paper_result_eligible"])
            self.assertEqual(config["artifact_class"], "formal_paper_protocol")
            self.assertTrue(config["hardware_protocol_pending"])

    def test_diagnostic_requires_explicit_nonformal_api(self):
        with self.assertRaises(ValueError):
            resolve_config("tsp", 1000, "greedy_diagnostic")
        config = resolve_config(
            "tsp", 1000, "greedy_diagnostic", allow_diagnostic=True)
        self.assertFalse(config["paper_result_eligible"])
        self.assertEqual(config["artifact_class"], "diagnostic")
        self.assertEqual(config["budget"], 0)
        self.assertFalse(config["random_insertion"])
        self.assertFalse(config["use_k_nearest"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            paper_main(["--problem", "tsp", "--problem-size", "1000",
                        "--budget", "greedy_diagnostic", "--dump-config"])

    def test_senior_approved_formal_scope_and_cell_count(self):
        self.assertEqual(FORMAL_SIZES, {
            "tsp": (1000, 2000, 5000, 10000), "cvrp": (1000, 2000)})
        self.assertEqual(len(SIZE_REGISTRY) * len(FORMAL_PROTOCOLS), 18)
        for problem, size in (("tsp", 100), ("tsp", 500), ("cvrp", 500)):
            with self.assertRaises(ValueError):
                resolve_config(problem, size, "greedy")

    def test_knn_strict_threshold(self):
        self.assertFalse(resolve_config("tsp", 1000, "fewer")["initial_knn_path_active"])
        self.assertTrue(resolve_config("tsp", 2000, "fewer")["initial_knn_path_active"])

    def test_formal_batch_is_one(self):
        with self.assertRaises(ValueError):
            resolve_config("tsp", 1000, "fewer", batch_size=2)

    def test_checkpoint_filename_gate(self):
        config = resolve_config("cvrp", 2000, "more")
        validate_checkpoint_path(config, Path("/tmp/checkpoint-cvrp1k.pt"))
        with self.assertRaises(ValueError):
            validate_checkpoint_path(config, Path("/tmp/wrong.pt"))

    def test_registry_has_no_claimed_local_sha(self):
        self.assertTrue(all(row["actual_sha256"] is None for row in CHECKPOINTS.values()))

    def test_repair_max_is_clamped_to_actual_problem_size(self):
        for problem, size, effective in (
                ("tsp", 1000, 1000), ("tsp", 2000, 1000),
                ("cvrp", 1000, 1000),
                ("cvrp", 2000, 1000)):
            config = resolve_config(problem, size, "fewer")
            self.assertEqual(config["repair_max_sub_length_nominal"], 1000)
            self.assertEqual(config["repair_max_sub_length_effective"], effective)
            self.assertEqual(build_env_params(config)["repair_max_sub_length"], effective)
        self.assertEqual(effective_repair_max(500), 500)
        self.assertEqual(effective_repair_max(500, 1000), 500)
        with self.assertRaises(ValueError):
            resolve_config("cvrp", 500, "fewer")

    def test_build_tester_passes_effective_repair_max(self):
        captured = {}

        class Estimator:
            def reset(self):
                pass

        class Tester:
            def __init__(self, *, env_params, model_params, tester_params):
                captured["env_params"] = env_params
                self.time_estimator_2 = Estimator()

        modules = {
            "CVRP": types.ModuleType("CVRP"),
            "CVRP.Test_All": types.ModuleType("CVRP.Test_All"),
            "CVRP.Test_All.Tester": types.ModuleType("CVRP.Test_All.Tester"),
        }
        modules["CVRP.Test_All.Tester"].VRPTester = Tester
        device = types.SimpleNamespace(index=0)
        with mock.patch.dict(sys.modules, modules), mock.patch(
                "methods.sil.runtime._clear_official_namespaces"):
            _, config = build_tester(
                problem="cvrp", problem_size=1000, budget_label="fewer",
                upstream=Path("/tmp"), checkpoint=Path("/tmp/checkpoint-cvrp1k.pt"),
                device=device, torch=types.SimpleNamespace())
        self.assertEqual(config["repair_max_sub_length_effective"], 1000)
        self.assertEqual(captured["env_params"]["repair_max_sub_length"], 1000)

    def test_formal_dataset_registry_and_filename_gate(self):
        expected = {
            ("tsp", 1000): "tsp1000_concorde_23.118.pkl",
            ("tsp", 2000): "tsp2000_lkh_500_32.436.pkl",
            ("tsp", 5000): "tsp5000_lkh_500_50.968.pkl",
            ("tsp", 10000): "tsp10000_lkh_500_71.782.pkl",
            ("cvrp", 1000): "cvrp1000_hgs-360s_41.171.pkl",
            ("cvrp", 2000): "cvrp2000_hgs-360s_57.181.pkl",
        }
        for (problem, size), filename in expected.items():
            config = resolve_config(problem, size, "fewer")
            self.assertEqual(config["expected_dataset_filename"], filename)
            validate_dataset_path(config, Path("/data") / filename)
            with self.assertRaises(ValueError):
                validate_dataset_path(config, Path("/data/wrong.pkl"))
        for problem, size, filename in (
                ("tsp", 500, "tsp500_concorde_16.546.pkl"),
                ("cvrp", 500, "cvrp500_hgs-300s_37.154.pkl")):
            with self.assertRaises(ValueError):
                config = resolve_config(problem, size, "fewer")
                validate_dataset_path(config, Path("/data") / filename)

    def test_warmup_policy_is_frozen_in_protocol_identity(self):
        self.assertEqual(WARMUP_POLICY["batches"], 1)
        self.assertTrue(WARMUP_POLICY["excluded_from_timing"])
        self.assertTrue(WARMUP_POLICY["excluded_from_records"])
        self.assertTrue(WARMUP_POLICY["rng_state_restored"])
        self.assertEqual(resolve_config("tsp", 1000, "fewer")["warmup"], WARMUP_POLICY)

    def test_large_tsp_smoke_is_one_instance_and_other_sizes_are_two(self):
        for problem, size in SIZE_REGISTRY:
            self.assertEqual(_our_smoke_count(problem, size, "greedy"), 1)
        self.assertEqual(_our_smoke_count("tsp", 5000, "fewer"), 1)
        self.assertEqual(_our_smoke_count("tsp", 10000, "fewer"), 1)
        self.assertEqual(_our_smoke_count("tsp", 2000, "fewer"), 2)
        self.assertEqual(_our_smoke_count("cvrp", 2000, "fewer"), 2)

    def test_dump_config_does_not_import_official_runtime(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = paper_main(["--problem", "tsp", "--problem-size", "1000",
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
        protocol = {"problem": "TSP", "actual_problem_size": 3,
                    "budget_label": "fewer", "budget": 50,
                    "hardware_protocol_pending": True}
        identity = {
            "scope": "fullset", "offset": 0, "indices": [0, 1],
            "dataset": {"count": 2, "sha256": "dataset"},
            "protocol": protocol, "protocol_fingerprint": fingerprint(protocol),
            "checkpoint": {"sha256": "checkpoint"},
            "upstream": {"commit": "upstream"},
            "project": {"commit": "project"}, "source_files": [],
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
            self.assertEqual(summary["status"], "HARDWARE_PROTOCOL_PENDING")
            self.assertFalse(summary["paper_ready"])
            self.assertTrue(summary["full_dataset_complete"])
            self.assertAlmostEqual(summary["mean_instance_gap_percent"], 75.0)
            self.assertAlmostEqual(summary["total_runtime_seconds"], 3.0)
            loaded = json.loads((root / "summary.json").read_text())
            self.assertEqual(loaded["status"], "HARDWARE_PROTOCOL_PENDING")
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

    def test_warmup_restores_rng_and_does_not_append_artifacts(self):
        import torch
        random.seed(13)
        np.random.seed(14)
        torch.manual_seed(15)
        before = capture_rng_state(torch)
        records, timings = [], []

        def warmup_solve():
            random.random()
            np.random.random()
            torch.rand(4)

        run_rng_preserving_warmup(warmup_solve, torch=torch)
        self.assertEqual(capture_rng_state(torch), before)
        self.assertEqual(records, [])
        self.assertEqual(timings, [])
        self.assertTrue(_warmup_required(0))
        self.assertFalse(_warmup_required(1))

    def test_warmup_and_formal_capture_are_isolated(self):
        class Env:
            def _get_travel_distance_2(self, problems, solution, **kwargs):
                return np.asarray([float(np.asarray(solution).sum())])

        env = Env()
        with capture_official_solution(env, problem="tsp", problem_size=3) as warmup:
            env._get_travel_distance_2(None, np.array([[0, 1, 2]]))
        with capture_official_solution(env, problem="tsp", problem_size=3) as formal:
            env._get_travel_distance_2(None, np.array([[2, 1, 0]]))
        self.assertEqual(warmup["eligible_calls"], 1)
        self.assertEqual(formal["eligible_calls"], 1)
        self.assertIsNot(warmup, formal)


class SILPreflightGateTests(unittest.TestCase):
    PROJECT_COMMIT = "project-commit"
    SOURCE_FILES = [{"path": "methods/sil/config.py", "sha256": "source"}]

    def _summary(self, protocol_label="fewer", **changes):
        protocol = resolve_config("tsp", 1000, protocol_label)
        protocol_hash = fingerprint(protocol)
        scope = "preflight" if protocol_label == "more" else "our-smoke"
        summary = {
            "status": "KIT_VALIDATED", "method": "SIL", "problem": "TSP",
            "problem_size": 1000, "budget_label": protocol_label,
            "dataset_sha256": "dataset", "checkpoint_sha256": "checkpoint",
            "upstream_commit": "9ec783e90a1631f7b95f84eb20f8f9751cb45c10",
            "protocol_fingerprint": protocol_hash,
            "project_commit": self.PROJECT_COMMIT,
            "source_provenance_fingerprint": fingerprint(self.SOURCE_FILES),
            "count": 2, "validated_count": 2, "failed_count": 0,
            "identity": {
                "method": "SIL", "scope": scope, "count": 2,
                "protocol": protocol,
                "protocol_fingerprint": protocol_hash,
                "dataset": {"sha256": "dataset"},
                "checkpoint": {"sha256": "checkpoint"},
                "project": {"commit": self.PROJECT_COMMIT},
                "source_files": self.SOURCE_FILES,
                "upstream": {
                    "commit": "9ec783e90a1631f7b95f84eb20f8f9751cb45c10"},
            },
        }
        summary.update(changes)
        return summary, protocol

    def _verify(self, summary, protocol, budget_label=None):
        budget_label = protocol["budget_label"] if budget_label is None else budget_label
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "summary.json"
            path.write_text(json.dumps(summary))
            return _verify_preflight_evidence(
                path, problem="tsp", problem_size=1000, budget_label=budget_label,
                dataset_sha256="dataset", checkpoint_sha256="checkpoint",
                protocol_fingerprint=fingerprint(resolve_config(
                    "tsp", 1000, budget_label)), project_commit=self.PROJECT_COMMIT,
                source_files=self.SOURCE_FILES)

    def test_matching_greedy_fewer_more_evidence_passes(self):
        for label in FORMAL_PROTOCOLS:
            summary, protocol = self._summary(label)
            evidence = self._verify(summary, protocol)
            self.assertEqual(evidence["identity"]["status"], "KIT_VALIDATED")
            self.assertEqual(evidence["identity"]["count"], 2)

    def test_mismatched_evidence_fails_closed(self):
        for field, value in (
                ("problem", "CVRP"), ("problem_size", 500),
                ("budget_label", "more"), ("dataset_sha256", "wrong"),
                ("checkpoint_sha256", "wrong"), ("status", "PAPER_READY"),
                ("upstream_commit", "wrong"), ("protocol_fingerprint", "wrong"),
                ("project_commit", "wrong"),
                ("source_provenance_fingerprint", "wrong")):
            summary, protocol = self._summary(**{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self._verify(summary, protocol)

    def test_invalid_validation_counts_fail_closed(self):
        for changes in ({"count": 0, "validated_count": 0},
                        {"validated_count": 1}, {"failed_count": 1}):
            summary, protocol = self._summary(**changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self._verify(summary, protocol)

    def test_embedded_protocol_tamper_fails_closed(self):
        summary, protocol = self._summary()
        summary["identity"]["protocol"] = json.loads(json.dumps(protocol))
        summary["identity"]["protocol"]["budget"] = 500
        with self.assertRaisesRegex(ValueError, "identity.protocol_content"):
            self._verify(summary, protocol)

    def test_embedded_project_source_and_scope_tamper_fail_closed(self):
        for field in ("project", "source_files", "scope"):
            summary, protocol = self._summary()
            if field == "project":
                summary["identity"][field] = {"commit": "wrong"}
            elif field == "source_files":
                summary["identity"][field] = []
            else:
                summary["identity"][field] = "preflight"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self._verify(summary, protocol)

    def test_protocol_evidence_cannot_cross_greedy_fewer_more(self):
        for evidence_label, requested_label in (
                ("greedy", "fewer"), ("fewer", "more"), ("more", "greedy")):
            summary, protocol = self._summary(evidence_label)
            with self.subTest(evidence=evidence_label, requested=requested_label), \
                    self.assertRaises(ValueError):
                self._verify(summary, protocol, budget_label=requested_label)

    def test_fullset_requires_preflight_cli_argument(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            paper_main(["--problem", "tsp", "--problem-size", "1000",
                        "--budget", "fewer", "--scope", "fullset",
                        "--count", "1", "--output-dir", "/tmp/unused"])


if __name__ == "__main__":
    unittest.main()
