"""CPU regression: configuration equivalence, native call boundaries and artifact gates."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

import numpy as np

from common.hashing import sha256_file
from methods.glop.paper_protocol import formal_protocol
from methods.glop.paper_results import fingerprint, write_json
from methods.udc.paper_protocol import REGISTRY_PATH, REGISTRY_SHA256, budget_config
from rebuttal.tsp2k.config import (COUNT, DATASET, GLOP_SWEEP, UDC_SWEEP, SIL_SWEEP,
                                  UPSTREAMS, protocol, solver_config)
from rebuttal.tsp2k.artifacts import Writer, read_artifact, summarize, indices_for
from rebuttal.tsp2k.plot import checked_report
from rebuttal.tsp2k.run_calibration import commands

HAS_SIL = importlib.util.find_spec("methods.sil") is not None


def identity(method="GLOP", budget=1, phase="calibration", count=2):
    config = protocol(method, budget)
    return {**config, "phase": phase, "requested_count": count,
        "dataset_indices": list(range(count)),
        "timing_indices": list(range(min(3, count))) if method == "SIL" else [],
        "actual_quality_batch_size": count if method == "SIL" else 1,
        "dataset": {"path": "/dataset/" + DATASET, "sha256": "d"*64, "count": COUNT,
                    "reference_objectives_sha256": "e"*64},
        "project": {"commit": "a"*40, "dirty": False}, "project_commit": "a"*40,
        "upstream": {"commit": UPSTREAMS[method][1], "dirty": False},
        "upstream_commit": UPSTREAMS[method][1], "gpu": "NVIDIA GeForce RTX 4090",
        "environment": {"gpu": "NVIDIA GeForce RTX 4090", "device": "cuda:0",
                        "python": "CPU-test-fixture", "torch": "CPU-test-fixture",
                        "numpy": "CPU-test-fixture"},
        "checkpoint_sha256": {"model": "b"*64},
        "source_provenance": [{"path": "solver.py", "sha256": "c"*64}],
        "evidence_kind": "SYNTHETIC_TEST"}


def row(index, source="quality", runtime=1.0):
    obj, ref = (10.0, 5.0) if index == 0 else (12.0, 10.0)
    return {"instance_index": index, "source": source,
        "canonical_solution": list(range(2000)), "objective": obj,
        "official_objective": obj, "kit_objective": obj, "reference_objective": ref,
        "gap_percent": (obj-ref)/ref*100, "runtime_seconds": runtime,
        "feasible": True, "kit_feasible": True, "official_objective_agrees": True,
        "kit_objective_agrees": True,
        "runtime_components": {"per_instance_solve_seconds": runtime,
                              "shared_ri_order_generation_seconds": 0}}


class BudgetTests(unittest.TestCase):
    def test_glop_six_vectors_and_1x_exact_formal_protocol(self):
        standard = formal_protocol("TSP", 2000, "official_standard")
        self.assertEqual(solver_config("GLOP", 1), standard)
        for budget, expected in GLOP_SWEEP.items():
            config = solver_config("GLOP", budget)
            self.assertEqual(config["revision_iters"], expected)
            unchanged = deepcopy(config)
            unchanged["revision_iters"] = standard["revision_iters"]
            self.assertEqual(unchanged, standard)
            self.assertEqual(config["ri_order_width"], 1)
            self.assertEqual(config["effective_candidate_count"], 1)
            self.assertEqual(config["decode_strategy"], "greedy")

    def test_udc_exact_x_only_and_unchanged_registry(self):
        standard = budget_config("tsp", "fewer")
        for x in UDC_SWEEP:
            config = solver_config("UDC", x)
            self.assertEqual(config["x"], x)
            config["x"] = 2
            self.assertEqual(config, standard)
        more = solver_config("UDC", 50)
        formal_more = budget_config("tsp", "more")
        for key in ("alpha", "x", "sub_size", "configured_pomo", "effective_pomo",
                    "seed", "batch_size", "official_source_commit"):
            self.assertEqual(more[key], formal_more[key])
        self.assertEqual(sha256_file(REGISTRY_PATH), REGISTRY_SHA256)

    @unittest.skipUnless(HAS_SIL, "SIL runs in its independent feat/lehd-derived worktree")
    def test_sil_integer_budgets_only_metadata_and_budget_change(self):
        from methods.sil.config import resolve_config
        standard = resolve_config("tsp", 2000, "fewer")
        for budget in SIL_SWEEP:
            config = solver_config("SIL", budget)
            self.assertEqual(config["budget"], budget)
            self.assertEqual(config["budget_label"], f"prc{budget}")
            self.assertTrue(config["random_insertion"])
            self.assertTrue(config["PRC"])
            for key in ("budget", "budget_label", "artifact_class", "paper_result_eligible",
                        "evaluation_mapping", "budget_mapping_origin"):
                config[key] = standard[key]
            self.assertEqual(config, standard)
        self.assertEqual(standard["budget"], 10)
        self.assertEqual(standard["model"]["mode"], "test")
        self.assertEqual(standard["model"]["sqrt_embedding_dim"], 128**0.5)

    def test_invalid_budgets_and_scope_fail_closed(self):
        for method in ("GLOP", "UDC", "SIL"):
            for budget in (0, -1, True, 1.0, 999):
                with self.subTest(method=method, budget=budget), self.assertRaises(ValueError):
                    solver_config(method, budget)
        with self.assertRaises(ValueError):
            indices_for("fullset", 2)
        self.assertEqual(len(indices_for("fullset")), 64)


class ArtifactTests(unittest.TestCase):
    def test_native_decoders_explicit_closed_tours_pass_without_repair(self):
        from methods.glop.tsp.decode import decode_coordinate_tour
        from methods.udc.adapter import validate_tsp_population
        from problems.tsp.validate import validate
        from rebuttal.tsp2k.runner import validated
        from rebuttal.tsp2k.artifacts import check_record
        theta = np.linspace(0, 2*np.pi, 2000, endpoint=False)
        points = np.stack((0.5+0.4*np.cos(theta), 0.5+0.4*np.sin(theta)), axis=1).astype(np.float32)
        permutation = np.arange(2000)
        objective = validate(points, permutation)["independent_objective"]
        task = types.SimpleNamespace(points=points, ref_sol=permutation,
            evaluate=lambda tour: validate(points, tour)["independent_objective"],
            check_constraints=lambda tour: validate(points, tour)["feasible"])
        glop_tour, _ = decode_coordinate_tour(points, points, allowed_transforms=["identity"])
        population = np.tile(permutation, (50, 1))
        udc_tour = validate_tsp_population(points, population)["solution"]
        tours = [glop_tour, udc_tour]
        if HAS_SIL:
            from methods.sil.tsp.adapter import decode_official_solution
            tours.append(decode_official_solution(permutation, problem_size=2000)["canonical_solution"])
        for tour in tours:
            self.assertEqual(len(tour), 2001)
            result = validated(task, 0, tour, objective)
            check_record(result)
            self.assertEqual(result["canonical_solution"], tour)
            malformed = dict(result, canonical_solution=tour[:-1]+[1])
            with self.assertRaises(ValueError):
                check_record(malformed)

    def test_lock_rejects_changed_dataset_or_checkpoint_before_solver_import(self):
        from rebuttal.tsp2k.audit import verify_lock, file_identity, unique_bytes
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, checkpoint = root / DATASET, root / "model.pt"
            dataset.write_bytes(b"dataset")
            checkpoint.write_bytes(b"model")
            lock = {"schema": "tsp2k-asset-lock.v1",
                    "projects": {"GLOP": {"commit": "reviewed"}},
                    "dataset": dict(file_identity(dataset), count=64),
                    "methods": {"GLOP": {"upstream_path": "/official",
                        "upstream": {"commit": "pinned"},
                        "assets": [file_identity(checkpoint)]}}}
            with patch("rebuttal.tsp2k.audit.project", return_value={"commit": "reviewed"}), \
                 patch("rebuttal.tsp2k.audit.checkout", return_value={"commit": "pinned"}):
                self.assertEqual(verify_lock(lock, "GLOP", root), lock["methods"]["GLOP"])
                for asset in (dataset, checkpoint):
                    original = asset.read_bytes()
                    asset.write_bytes(b"changed")
                    with self.assertRaises(ValueError):
                        verify_lock(lock, "GLOP", root)
                    asset.write_bytes(original)
            with self.assertRaises(ValueError):
                unique_bytes([dataset, checkpoint], "conflicting assets")
            checkpoint.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaises(ValueError):
                file_identity(checkpoint)

    def test_real_gap_mean_and_timing_and_atomic_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "point"
            writer = Writer(out, identity())
            writer.append(row(0))
            writer.append(row(1))
            summary = writer.finish()
            self.assertEqual(summary["mean_drop_percent"], 60)
            self.assertEqual(summary["quality_instance_count"], 2)
            self.assertEqual(summary["status"], "MEASURED_TIMING_SUBSET")
            self.assertEqual(read_artifact(out)[2], summary)
            with self.assertRaises(FileExistsError):
                Writer(out, identity())

    def test_missing_duplicate_nan_mixed_budget_and_provenance_fail(self):
        cases = [
            (identity(), [row(0)]),
            (identity(), [row(0), row(0)]),
            (identity(), [row(0), dict(row(1), objective=float("nan"))]),
            (dict(identity(), budget_value=2), [row(0), row(1)]),
            (dict(identity(), dataset_indices=[1, 2]), [row(0), row(1)]),
            (dict(identity(), upstream_commit="wrong"), [row(0), row(1)]),
            (dict(identity(), checkpoint_sha256={}), [row(0), row(1)]),
            (dict(identity(), gpu="NVIDIA H100"), [row(0), row(1)]),
        ]
        for ident, rows in cases:
            with self.subTest(identity=ident["budget_value"]), self.assertRaises(ValueError):
                summarize(ident, rows)

    def test_glop_shared_cost_only_once_and_component_sum(self):
        rows = [row(0), row(1)]
        rows[0]["runtime_components"]["shared_ri_order_generation_seconds"] = 0.2
        rows[0]["runtime_seconds"] = 1.2
        self.assertAlmostEqual(summarize(identity(), rows)["mean_time_seconds"], 1.1)
        rows[1]["runtime_components"]["shared_ri_order_generation_seconds"] = 0.2
        with self.assertRaises(ValueError):
            summarize(identity(), rows)

    def test_record_tamper_and_failed_run_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = Writer(Path(temp) / "p", identity())
            writer.append(row(0)); writer.append(row(1)); writer.finish()
            path = writer.directory / "inference_records.jsonl"
            path.write_text(path.read_text() + "\n")
            with self.assertRaises(ValueError):
                read_artifact(writer.directory)
            writer.fail("simulated failure")
            with self.assertRaises(ValueError):
                read_artifact(writer.directory)

    def test_synthetic_data_never_reaches_official_plot(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.json"
            write_json(path, {"evidence_kind": "SYNTHETIC_TEST", "status": "MEASURED_FULLSET"})
            with self.assertRaises(ValueError):
                checked_report(path)

    def test_plot_renderer_exports_three_formats_to_temporary_directory(self):
        from rebuttal.tsp2k.plot import main
        # Test rendering only; the input validator has a separate synthetic rejection test.
        report = {"points": [{"method": method, "mean_time_seconds": budget,
                              "mean_drop_percent": 5.0 / budget}
                            for method in ("GLOP", "UDC", "SIL") for budget in (1, 2)]}
        with tempfile.TemporaryDirectory() as temp:
            prefix = Path(temp) / "synthetic-render-test"
            with patch("rebuttal.tsp2k.plot.checked_report", return_value=report):
                main(["--report", "/unused-test-only.json", "--output-prefix", str(prefix)])
            for ext in ("png", "pdf", "svg"):
                self.assertGreater(prefix.with_suffix("." + ext).stat().st_size, 1000)

    @unittest.skipUnless(HAS_SIL, "independent SIL worktree")
    def test_sil_quality64_and_bs1_timing_sources_are_distinct(self):
        ident = identity("SIL", 10, "fullset", 64)
        rows = [row(i, runtime=None) for i in range(64)]
        rows += [row(i, "timing", 2.0) for i in range(3)]
        summary = summarize(ident, rows)
        self.assertEqual(summary["quality_instance_count"], 64)
        self.assertEqual(summary["quality_batch_size"], 64)
        self.assertEqual(summary["timing_batch_size"], 1)
        self.assertEqual(summary["timing_sample_count"], 3)
        self.assertEqual(summary["mean_time_seconds"], 2.0)
        rows[0]["runtime_seconds"] = 0.01
        with self.assertRaises(ValueError):
            summarize(ident, rows)


class DelegationTests(unittest.TestCase):
    def test_glop_delegates_to_original_solve_one_with_shared_orders(self):
        from rebuttal.tsp2k.runner import glop_execute
        import methods.glop.runtime as runtime
        import methods.glop.tsp.paper_eval as native
        import sys
        writer = types.SimpleNamespace(metadata={}, save=Mock(), append=Mock())
        tasks = [types.SimpleNamespace(points=np.zeros((2000, 2))) for _ in range(2)]
        order = types.SimpleNamespace(tolist=lambda: list(range(2000)))
        official = {"utils": types.ModuleType("utils"),
                    "utils.functions": types.SimpleNamespace(load_model=Mock(),
                        load_problem=Mock(), reconnect=Mock()),
                    "utils.insertion": types.SimpleNamespace(random_insertion_parallel=Mock())}
        with patch.object(runtime, "activate_upstream"), \
             patch.object(runtime, "load_revisers", return_value=([], [])), \
             patch.object(runtime, "random_insertion_identity", return_value={}), \
             patch.object(runtime, "make_shared_tsp_orders", return_value=(order,)) as orders, \
             patch.object(runtime, "run_warmup_isolated") as warmup, \
             patch.object(native, "_solve_one", return_value=(list(range(2000)), 1, {}, 2)) as solve, \
             patch("rebuttal.tsp2k.runner.validated", return_value=row(0)), \
             patch.dict(sys.modules, official):
            glop_execute(tasks, solver_config("GLOP", 1),
                {"upstream_path": "/GLOP", "asset_root": "/assets"},
                Mock(), Mock(), [0, 1], writer)
        self.assertEqual(orders.call_count, 1)
        self.assertEqual(warmup.call_count, 1)
        self.assertEqual(solve.call_count, 2)
        self.assertTrue(all(c.kwargs["timed"] for c in solve.call_args_list))
        self.assertTrue(all(c.kwargs["orders"] == (order,) for c in solve.call_args_list))
        self.assertEqual(solve.call_args.kwargs["protocol"],
                         formal_protocol("TSP", 2000, "official_standard"))

    @unittest.skipUnless(HAS_SIL, "independent SIL worktree")
    def test_sil_constructor_receives_integer_budget_and_full_model_keys(self):
        import sys
        import methods.sil.runtime as runtime
        factory = Mock(side_effect=lambda **kwargs: types.SimpleNamespace(
            time_estimator_2=types.SimpleNamespace(reset=Mock())))
        module = types.SimpleNamespace(TSPTester=factory)
        with patch.dict(sys.modules, {"TSP.Test_All.TSPTester": module}), \
             patch.object(runtime, "_clear_official_namespaces"), \
             patch.object(sys, "path", list(sys.path)):
            for value in SIL_SWEEP:
                tester, config = runtime.build_tester(
                    problem="tsp", problem_size=2000, budget_label="fewer",
                    upstream=Path("/SIL"), checkpoint=Path("/checkpoint-tsp1k.pt"),
                    device=types.SimpleNamespace(index=0), torch=Mock(),
                    rebuttal_prc_budget=value)
                self.assertEqual(factory.call_args.kwargs["env_params"]["budget"], value)
                self.assertEqual(factory.call_args.kwargs["model_params"], config["model"])
                self.assertEqual(config["model"]["mode"], "test")

    def test_udc_reuses_native_solver_with_requested_x_and_continuous_seed(self):
        from rebuttal.tsp2k.runner import udc_execute
        import methods.udc.paper_production as production
        import methods.udc.s3_eval as native
        for x in (2, 50):
            writer = types.SimpleNamespace(metadata={}, save=Mock(), append=Mock())
            solved = {"status": "KIT_VALIDATED", "completed_stages": [None]*x,
                      "solution": list(range(2000)), "udc_internal_objective": 10,
                      "runtime_seconds": 1}
            with patch.object(production, "cuda_allocator_gate"), \
                 patch.object(production, "seed_once") as seed, \
                 patch.object(production, "prepare_solver_memory"), \
                 patch.object(production, "verify_loaded_environment"), \
                 patch.object(native, "load_models", return_value=(None, None, {}, "cuda")), \
                 patch.object(native, "solve_tsp", return_value=solved) as solve, \
                 patch("rebuttal.tsp2k.runner.validated", return_value=row(0)):
                udc_execute([None, None], solver_config("UDC", x),
                    {"upstream_path": "/official", "supplemental_root": "/assets"},
                    Mock(), "cuda", [0, 1], writer)
                self.assertEqual(seed.call_count, 1)
                self.assertEqual(solve.call_count, 2)
                self.assertEqual([c.kwargs["x_stages"] for c in solve.call_args_list], [x, x])
                self.assertTrue(all(c.kwargs["capture_memory"] is False for c in solve.call_args_list))

    @unittest.skipUnless(HAS_SIL, "independent SIL worktree")
    def test_sil_native64_batch_and_singleton_timing_calls(self):
        from rebuttal.tsp2k.runner import sil_execute
        import methods.sil.runtime as native
        import methods.sil.author_batch_eval as helpers
        writer = types.SimpleNamespace(identity={"timing_indices": [0, 1, 2]}, append=Mock())
        config = solver_config("SIL", 5)
        solved_batch = {"solutions": np.zeros((64, 2000)), "official_objectives": np.ones(64), "capture": {}}
        solved_one = {"solution": np.zeros(2000), "official_objective": 1, "capture": {}, "runtime_seconds": 2}
        with patch.object(native, "build_tester", return_value=(Mock(), config)) as build, \
             patch.object(native, "run_rng_preserving_warmup"), \
             patch.object(native, "solve_batch", return_value=solved_batch) as batched, \
             patch.object(native, "solve_one", return_value=solved_one) as single, \
             patch.object(helpers, "prepare_batch", return_value={"inject": Mock()}), \
             patch.object(helpers, "validate_batch", side_effect=lambda problem, tasks, *a, **k:
                 [{"canonical_solution": list(range(2000)), "official_objective": 1} for t in tasks]), \
             patch("rebuttal.tsp2k.runner.validated", return_value=row(0)):
            sil_execute([None]*64, config, {"upstream_path": "/SIL", "checkpoint": "/model"},
                        Mock(), Mock(), list(range(64)), writer)
        self.assertEqual(batched.call_count, 1)
        self.assertEqual(batched.call_args.kwargs["batch_size"], 64)
        self.assertFalse(batched.call_args.kwargs["timed"])
        self.assertEqual(single.call_count, 3)
        self.assertTrue(all(c.kwargs["timed"] for c in single.call_args_list))
        self.assertTrue(all(c.kwargs["rebuttal_prc_budget"] == 5 for c in build.call_args_list))

    def test_subprocess_commands_are_separate_and_budget_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "lock.json"
            write_json(lock, {"projects": {m: {"path": f"/source/{m}"} for m in SWEEPS}})
            jobs = commands(lock, Path(temp), "calibration")
            self.assertEqual(len(jobs), 16)
            for command in jobs:
                self.assertIn("--budget", command)
                self.assertEqual(command[command.index("--count")+1], "2")


from rebuttal.tsp2k.config import SWEEPS


@unittest.skipUnless(HAS_SIL, "aggregation executes in the independent SIL worktree")
class AggregationTests(unittest.TestCase):
    def make_points(self, root):
        paths = []
        for method, budgets in SWEEPS.items():
            for budget in budgets:
                ident = identity(method, budget, "fullset", COUNT)
                writer = Writer(root / method / str(budget), ident)
                for i in range(COUNT):
                    writer.append(row(i, runtime=None if method == "SIL" else 1.0))
                if method == "SIL":
                    for i in range(3):
                        writer.append(row(i, "timing", 2.0))
                writer.finish()
                paths.append(writer.directory)
        return paths

    def measured_read(self, path):
        # CPU fixtures remain marked SYNTHETIC_TEST on disk. This mock exercises
        # the aggregate's measured-data branch; no mock results leave this test.
        ident, records, summary = read_artifact(path)
        ident["evidence_kind"] = "SERVER_MEASUREMENT"
        return ident, records, summary

    def test_full_curve_aggregation_and_missing_duplicate_dataset_checkpoint_gates(self):
        from rebuttal.tsp2k.aggregate import aggregate
        with tempfile.TemporaryDirectory() as temp:
            paths = self.make_points(Path(temp))
            with self.assertRaises(ValueError):
                aggregate(paths, require_fullset=True)  # Synthetic rejected.
            with patch("rebuttal.tsp2k.aggregate.read_artifact", side_effect=self.measured_read):
                report = aggregate(paths, require_fullset=True)
                self.assertEqual(len(report["points"]), 16)
                self.assertEqual(report["status"], "MEASURED_FULLSET")
                for selected in (paths[:-1], paths + [paths[0]]):
                    with self.assertRaises(ValueError):
                        aggregate(selected, require_fullset=True)
            for field in ("dataset", "checkpoint_sha256", "project_commit"):
                counter = [0]
                def altered(path):
                    ident, records, summary = self.measured_read(path)
                    counter[0] += 1
                    if counter[0] == 2:
                        if field == "dataset":
                            ident["dataset"]["sha256"] = "f"*64
                        elif field == "checkpoint_sha256":
                            ident[field] = {"model": "e"*64}
                        else:
                            ident[field] = "f"*40
                    return ident, records, summary
                with patch("rebuttal.tsp2k.aggregate.read_artifact", side_effect=altered):
                    with self.assertRaises(ValueError):
                        aggregate(paths, require_fullset=True)

if __name__ == "__main__":
    unittest.main()
