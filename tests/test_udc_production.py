import ast
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from common.hashing import sha256_file
from methods.udc.build_paper_table import build_table, load_complete_run
from methods.udc.paper_production import (_pilot_summary, freeze_decision_gate,
                                          load_formal_dataset, solve_record)
from methods.udc.paper_protocol import (EXPECTED_BUDGETS, REGISTRY_PATH,
                                        REGISTRY_SHA256, formal_cells,
                                        load_budget_registry, s4_gate)
from methods.udc.paper_results import (METADATA_FILE, RECORDS_FILE, SUMMARY_FILE,
                                       atomic_json, atomic_jsonl,
                                       capture_rng_state, fingerprint,
                                       initialize_or_resume,
                                       initialize_rng_checkpoint,
                                       restore_rng_state, summarize_records,
                                       validate_production_record)
from methods.udc.protocol import FORMAL_SIZES, SCALE_DATASET_FILENAMES


def record(index=0, *, problem="tsp", size=100, alpha=50, x=2,
           objective=2.0, reference=1.0, runtime=0.5, label="fewer"):
    row = {
        "dataset_instance_index": index, "instance_id": f"item-{index}",
        "problem": problem, "size": size, "budget_label": label,
        "alpha": alpha, "x": x, "best_alpha": 0, "completed_x_stages": x,
        "selected_solution": [0, 1], "canonical_solution": [0, 1, 0],
        "official_objective": objective, "independent_objective": objective,
        "reference_objective": reference,
        "drop_percent": (objective - reference) / reference * 100,
        "runtime_seconds": runtime, "timing_semantics": "paper production timer",
        "independent_feasible": True, "kit_feasible": True,
        "internal_vs_independent": {"pass": True},
        "independent_vs_kit": {"pass": True}, "evidence_status": "KIT_VALIDATED",
    }
    if problem == "cvrp":
        row.update(selected_solution=[1, 2], canonical_solution=[0, 1, 2, 0],
                   solution_flag=[0, 1])
    return row


def identity(dataset_path, dataset_sha, *, problem="tsp", size=100,
             label="fewer", count=1):
    budget = dict(load_budget_registry()["entries"][(problem, label)])
    return {
        "problem": problem, "size": size, "budget": budget,
        "dataset": {"path": str(dataset_path),
                    "filename": SCALE_DATASET_FILENAMES[(problem, size)],
                    "sha256": dataset_sha, "count": count,
                    "instance_names": [f"item-{index}" for index in range(count)],
                    "instance_names_sha256": fingerprint(
                        [f"item-{index}" for index in range(count)])},
        "environment": {"gpu_name": "NVIDIA GeForce RTX 4090"},
        "project": {"head": "abc"}, "registry": {"sha256": REGISTRY_SHA256},
        "checkpoints": {},
        "batch_size": 1, "seed_once": 1234, "timing_semantics": "paper production",
    }


class BudgetRegistryTests(unittest.TestCase):
    def test_exact_four_frozen_budgets(self):
        registry = load_budget_registry()
        observed = {key: {field: entry[field] for field in expected}
                    for key, expected in EXPECTED_BUDGETS.items()
                    for entry in (registry["entries"][key],)}
        self.assertEqual(observed, EXPECTED_BUDGETS)
        self.assertEqual(registry["sha256"], REGISTRY_SHA256)
        self.assertTrue(all(entry["selection_origin"] ==
                            "PROJECT-SELECTED PAPER BUDGETS"
                            for entry in registry["entries"].values()))

    def test_registry_mutation_fails_hash_gate(self):
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "registry.json"
            raw = json.loads(REGISTRY_PATH.read_text())
            raw["entries"][0]["x"] = 3
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_budget_registry(path)

    def test_freeze_decision_recomputes_pilot_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            runs = {}
            checks = {}
            decisions = {}
            for problem in ("tsp", "cvrp"):
                summaries = {}
                for label in ("fewer", "more"):
                    budget = dict(load_budget_registry()["entries"][(problem, label)])
                    objective = 2.0 if label == "fewer" else 1.0
                    runtime = 1.0 if label == "fewer" else 2.0
                    rows = [record(index, problem=problem, size=500,
                                   x=budget["x"], objective=objective, label=label,
                                   reference=1.0, runtime=runtime)
                            for index in range(3)]
                    directory = (root / "budget_freeze" / "pilot_runs" /
                                 problem / label)
                    directory.mkdir(parents=True)
                    atomic_json(directory / "records.json", rows)
                    atomic_json(directory / METADATA_FILE, {
                        "state": "KIT_VALIDATED",
                        "gates": {
                            "project": {"head": "production-head", "pass": True},
                            "official": {
                                "head": "274df3c4975384592b60fe7f79fbb2441ce11c15",
                                "pass": True}}})
                    summary = _pilot_summary(rows, problem, label, budget)
                    summaries[label] = summary
                    runs[f"{problem}_{label}"] = {
                        "artifact_path": str(directory),
                        "metadata_sha256": sha256_file(directory / METADATA_FILE),
                        "records_sha256": sha256_file(directory / "records.json"),
                        "summary": summary, "records": rows}
                fewer, more = summaries["fewer"], summaries["more"]
                checks[problem] = {
                    "all_semantics_pass": True,
                    "exact_fewer_stage_count": True,
                    "exact_more_stage_count": True,
                    "more_has_larger_fixed_x": True,
                    "more_mean_runtime_exceeds_fewer": True,
                    "more_mean_objective_nonworsening": True}
                decisions[problem] = {
                    label: {"alpha": summaries[label]["alpha"],
                            "x": summaries[label]["x"]}
                    for label in ("fewer", "more")}
            freeze = root / "budget_freeze"
            pilot = {"registry_sha256": REGISTRY_SHA256,
                     "project_head": "production-head", "runs": runs}
            atomic_json(freeze / "pilot.json", pilot)
            decision = {"status": "FROZEN", "registry_sha256": REGISTRY_SHA256,
                        "project_head": "production-head",
                        "pilot_sha256": sha256_file(freeze / "pilot.json"),
                        "checks": checks, "budgets": decisions}
            atomic_json(freeze / "decision.json", decision)
            self.assertTrue(freeze_decision_gate(root, REGISTRY_PATH)["pass"])
            rows = json.loads((Path(runs["tsp_fewer"]["artifact_path"]) /
                               "records.json").read_text())
            rows[0]["runtime_seconds"] = 99
            atomic_json(Path(runs["tsp_fewer"]["artifact_path"]) / "records.json", rows)
            with self.assertRaisesRegex(ValueError, "hash integrity"):
                freeze_decision_gate(root, REGISTRY_PATH)

    def test_s4_gate_requires_all_ten_pass_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value) / "99b85d29" / "s4_scale_preflight"
            rows = []
            for problem, sizes in FORMAL_SIZES.items():
                for size in sizes:
                    directory = root / f"{problem}{size}"; directory.mkdir(parents=True)
                    record_path = directory / "record.json"
                    dataset_sha = f"{size:064x}"[-64:]
                    atomic_json(record_path, {"status": "PASS",
                                              "dataset_sha256": dataset_sha})
                    gate = {"head": "99b85d299987d7275c45f75d08dad038759d10ab",
                            "pass": True}
                    official = {"head": "274df3c4975384592b60fe7f79fbb2441ce11c15",
                                "pass": True}
                    atomic_json(directory / "metadata.json", {
                        "status": "PASS", "record_sha256": sha256_file(record_path),
                        "dataset": {"sha256": dataset_sha},
                        "project_pre": gate, "project_post": gate,
                        "official_pre": official, "official_post": official})
                    rows.append({"problem": problem, "size": size, "status": "PASS",
                                 "dataset_sha256": dataset_sha,
                                 "dataset": f"/data/{SCALE_DATASET_FILENAMES[(problem, size)]}"})
            atomic_json(root / "summary.json",
                        {"schema": "udc_s4_scale_summary.v1", "rows": rows})
            self.assertTrue(s4_gate(root)["pass"])
            rows[0]["status"] = "CUDA_OOM"
            atomic_json(root / "summary.json",
                        {"schema": "udc_s4_scale_summary.v1", "rows": rows})
            with self.assertRaisesRegex(ValueError, "missing, duplicated, or not PASS"):
                s4_gate(root)


class DatasetAndRngTests(unittest.TestCase):
    def test_dataset_count_is_loaded_from_complete_wrapper(self):
        class TSPTask:
            def __init__(self, index):
                self.name = f"task-{index}"
                self.points = np.zeros((100, 2), dtype=np.float32)

        class Wrapper:
            def from_pickle(self, _path):
                self.task_list = [TSPTask(index) for index in range(7)]

        fake = types.SimpleNamespace(TSPWrapper=Wrapper, CVRPWrapper=None,
                                     TSPTask=TSPTask, CVRPTask=object)
        with tempfile.TemporaryDirectory() as value, \
                mock.patch.dict(sys.modules, {"ml4co_kit": fake}):
            path = Path(value) / "data.pkl"; path.write_bytes(b"dataset")
            tasks, _kit, evidence = load_formal_dataset(path, "tsp", 100)
        self.assertEqual(len(tasks), 7)
        self.assertEqual(evidence["count"], 7)
        self.assertEqual(evidence["indices"],
                         {"first": 0, "last": 6, "order": "0..N-1 without shuffle"})

    def test_rng_serialization_restores_continuous_stream(self):
        import torch
        random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
        state = capture_rng_state(torch)
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        random.random(); np.random.rand(); torch.rand(1)
        restore_rng_state(state, torch)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertEqual(actual, expected)

    def test_resume_rolls_back_one_record_not_committed_with_rng_state(self):
        import torch
        with tempfile.TemporaryDirectory() as value:
            output = Path(value) / "run"
            ident = identity("unused", "0" * 64)
            _metadata, rows, checkpoint = initialize_or_resume(output, ident)
            self.assertEqual(rows, [])
            self.assertIsNone(checkpoint)
            initialize_rng_checkpoint(output, ident, capture_rng_state(torch))
            atomic_jsonl(output / RECORDS_FILE, [record()])
            _metadata, rows, checkpoint = initialize_or_resume(output, ident)
            self.assertEqual(rows, [])
            self.assertEqual(checkpoint["next_instance_index"], 0)

    def test_production_loop_has_no_per_instance_reseed(self):
        source = Path("methods/udc/paper_production.py").read_text()
        tree = ast.parse(source)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_production")
        seed_calls = [node for node in ast.walk(function)
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "seed_once"]
        self.assertEqual(len(seed_calls), 1)
        loop = next(node for node in ast.walk(function) if isinstance(node, ast.For))
        self.assertFalse(any(call in set(ast.walk(loop)) for call in seed_calls))

    def test_production_passes_registry_x_to_solver_and_requires_exact_completion(self):
        budget = dict(load_budget_registry()["entries"][("tsp", "more")])
        raw = {"best_alpha": 0,
               "final_solution_population": [[0, 1]] * 50,
               "solution": [0, 1, 0], "udc_internal_objective": 1.0,
               "instance_id": "task", "completed_stages": [{}] * 50,
               "independent_objective": 1.0, "reference_objective": 1.0,
               "instance_drop_percent": 0.0, "runtime_seconds": 2.0,
               "independent_feasible": True, "kit_feasible": True,
               "internal_vs_independent": {"pass": True},
               "independent_vs_kit": {"pass": True}, "kit_objective": 1.0,
               "input_semantics": {}, "rng_before_solve": {}, "rng_after_solve": {},
               "status": "KIT_VALIDATED"}
        with mock.patch("methods.udc.paper_production.solve_tsp", return_value=raw) as solve:
            result = solve_record(object(), "tsp", 500, budget, object(), object(),
                                  object(), "cuda:0", 0, "pilot timer")
        self.assertEqual(solve.call_args.kwargs["x_stages"], 50)
        self.assertEqual(result["completed_x_stages"], 50)


class RecordAndAggregationTests(unittest.TestCase):
    def test_mean_drop_is_mean_of_instance_drops_not_gap_of_means(self):
        rows = [record(0, objective=2, reference=1),
                record(1, objective=100, reference=100)]
        ident = identity("unused", "0" * 64, count=2)
        summary = summarize_records(rows, ident)
        gap_of_means = ((summary["mean_objective"] - summary["mean_reference_objective"])
                        / summary["mean_reference_objective"] * 100)
        self.assertEqual(summary["mean_instance_drop_percent"], 50.0)
        self.assertFalse(math.isclose(summary["mean_instance_drop_percent"], gap_of_means))

    def test_timing_and_selected_solution_are_required(self):
        for missing in ("runtime_seconds", "timing_semantics", "selected_solution"):
            with self.subTest(missing=missing):
                row = record(); del row[missing]
                with self.assertRaisesRegex(ValueError, "missing"):
                    validate_production_record(row, problem="tsp", size=100,
                                               alpha=50, x_stages=2)

    def test_cvrp_requires_matching_solution_and_flag(self):
        row = record(problem="cvrp", size=200, x=50)
        row["solution_flag"] = [1]
        with self.assertRaisesRegex(ValueError, "matching solution_flag"):
            validate_production_record(row, problem="cvrp", size=200,
                                       alpha=50, x_stages=50)

    def test_aggregation_refuses_failed_instance(self):
        row = record(); row["kit_feasible"] = False
        with self.assertRaisesRegex(ValueError, "failed independent/Kit"):
            summarize_records([row], identity("unused", "0" * 64))

    def _complete_artifact(self, root):
        directory = root / "tsp100" / "fewer"; directory.mkdir(parents=True)
        dataset = root / SCALE_DATASET_FILENAMES[("tsp", 100)]
        dataset.write_bytes(b"formal-dataset")
        ident = identity(dataset, sha256_file(dataset))
        rows = [record()]
        atomic_jsonl(directory / RECORDS_FILE, rows)
        summary = summarize_records(rows, ident)
        summary["validated_records_sha256"] = sha256_file(directory / RECORDS_FILE)
        atomic_json(directory / SUMMARY_FILE, summary)
        metadata = {"state": "KIT_VALIDATED", "resume_identity": ident,
                    "resume_fingerprint": fingerprint(ident),
                    "validated_records_sha256": sha256_file(directory / RECORDS_FILE),
                    "summary_sha256": sha256_file(directory / SUMMARY_FILE),
                    "project_post": {"head": "abc", "pass": True},
                    "official_post": {
                        "head": "274df3c4975384592b60fe7f79fbb2441ce11c15",
                        "pass": True}}
        atomic_json(directory / METADATA_FILE, metadata)
        return directory, dataset

    def test_artifact_hash_integrity_and_failed_run_rejection(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            directory, dataset = self._complete_artifact(root)
            self.assertEqual(load_complete_run(root, "tsp", 100, "fewer")["summary"]["count"], 1)
            dataset.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash integrity"):
                load_complete_run(root, "tsp", 100, "fewer")
            metadata = json.loads((directory / METADATA_FILE).read_text())
            metadata["state"] = "FAILED"
            atomic_json(directory / METADATA_FILE, metadata)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_complete_run(root, "tsp", 100, "fewer")

    def test_aggregation_requires_exact_20_cell_scope_without_x8_or_cvrp50_100(self):
        cells = formal_cells()
        self.assertEqual(len(cells), 20)
        self.assertFalse(any(problem == "cvrp" and size in (50, 100)
                             for problem, size, _label in cells))

        def fake(_root, problem, size, label):
            ident = {"project": {"head": "production-head"}}
            return {"directory": f"/{problem}{size}/{label}",
                    "metadata_sha256": "a" * 64, "records_sha256": "b" * 64,
                    "summary_sha256": "c" * 64,
                    "summary": {"resume_identity": ident, "mean_objective": 1.0,
                                "mean_instance_drop_percent": 2.0,
                                "mean_runtime_seconds": 3.0, "count": 4,
                                "dataset_sha256": "d" * 64}}

        with mock.patch("methods.udc.build_paper_table.load_complete_run", side_effect=fake):
            table = build_table(Path("/unused"))
        self.assertEqual(len(table["cells"]), 20)
        self.assertEqual(table["scope"]["augmentation"], "none")
        self.assertTrue(all(cell["main_table_key"] == cell["appendix_table_key"]
                            for cell in table["cells"]))


if __name__ == "__main__":
    unittest.main()
