import json
from pathlib import Path
import tempfile
import unittest

from common.hashing import sha256_file
from common.paper_results import (METADATA_FILE, RECORDS_FILE, SCHEMA_VERSION,
                                  TIMING_SEMANTICS, VALIDATED_RECORDS_FILE, append_record,
                                  finalize_chunk, initialize_chunk,
                                  json_fingerprint, summarize_chunks, write_json)
from methods.mvmoe.cvrp.config import supported_config
from methods.mvmoe.cvrp.run import MODEL_CONFIG as CVRP_INTEGRATION_MODEL_CONFIG
from methods.mvmoe.cvrptw.config import get_size_config
from methods.mvmoe.cvrptw.paper_protocol import (
    scaled_paper_inference_config, unscaled_control_inference_config)
from methods.mvmoe.cvrptw.batch_artifacts import BATCH_TIMINGS_FILE
from methods.mvmoe.cvrptw.run import MODEL_CONFIG as CVRPTW_INTEGRATION_MODEL_CONFIG
from methods.mvmoe.paper_config import MODEL_CONFIG, paper_inference_config
from scripts.summarize_paper_results import summarize_for_protocol


def record(index, independent, reference, *, feasible=True, runtime=0.1,
           scaled=False):
    value = {
        "dataset_instance_index": index,
        "instance_id": f"instance-{index}",
        "canonical_solution": [0, 1, 0],
        "reported_objective": independent,
        "independent_objective": independent,
        "reference_objective": reference,
        "gap_percent": (independent - reference) / reference * 100.0,
        "runtime_seconds": runtime,
        "independent_feasible": feasible,
        "reported_objective_agrees": True,
        "evidence_status": "INDEPENDENT_VERIFIED" if feasible else "FAILED",
        "kit_feasible": feasible,
        "kit_objective": independent,
        "kit_objective_agrees": True,
    }
    if scaled:
        value.update(
            input_scaling_protocol="continuous_official_style",
            scaler=2.0,
            original_depot_tw_end=6.0,
            original_coordinate_max=1.0,
            scaled_depot_tw_end=3.0,
            scaled_coordinate_max=0.5,
            scaled_reported_objective=independent / 2.0,
            scaled_route_objective=independent / 2.0,
            scaled_objective_times_s=independent,
            scaled_reported_objective_agrees=True,
            scaled_to_original_objective_agrees=True,
        )
    return value


def identity(indices, *, dataset_count=10000, problem="CVRP", problem_size=50,
             batch_size=1):
    protocol = (paper_inference_config(problem_size, problem="CVRP")
                if problem == "CVRP" else scaled_paper_inference_config(
                    problem_size, batch_size))
    return {
        "method": "MVMoE", "variant": "MVMoE/4E", "problem": problem,
        "problem_size": problem_size,
        "paper_protocol": protocol,
        "project": {"commit": "a" * 40, "dirty": False, "url": "project"},
        "upstream": {"commit": "b" * 40, "dirty": False, "url": "upstream"},
        "checkpoint": {"path": "/checkpoint", "sha256": "c" * 64},
        "dataset": {"path": "/dataset", "sha256": "d" * 64,
                    "count": dataset_count},
        "prepared_input": {"path": "/input", "sha256": "e" * 64,
                           "metadata_path": "/input.json"},
        "chunk": {"offset": min(indices), "count": len(indices),
                  "expected_indices": list(indices)},
        "warmup": {"instances": 2, "policy": "fixed"},
        "environment": {"gpu": "NVIDIA GeForce RTX 4090", "device": "cuda:0"},
        "source_provenance": [{"path": "paper_eval.py", "sha256": "f" * 64}],
        "timing_semantics": TIMING_SEMANTICS,
    }


def write_chunk(root, name, records, *, mutate_identity=None, expected_indices=None,
                dataset_count=10000, problem="CVRP", problem_size=50,
                batch_size=1, batch_timings=None):
    directory = root / name
    directory.mkdir()
    indices = expected_indices if expected_indices is not None else [
        item["dataset_instance_index"] for item in records]
    value = identity(indices, dataset_count=dataset_count, problem=problem,
                     problem_size=problem_size, batch_size=batch_size)
    if mutate_identity:
        mutate_identity(value)
    records_path = directory / VALIDATED_RECORDS_FILE
    with records_path.open("w") as stream:
        for item in records:
            stream.write(json.dumps(item, allow_nan=True) + "\n")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "state": "KIT_VALIDATED",
        "resume_identity": value,
        "resume_fingerprint": json_fingerprint(value),
        "validated_records_file": VALIDATED_RECORDS_FILE,
        "validated_records_sha256": sha256_file(records_path),
    }
    if batch_timings is not None:
        timings_path = directory / BATCH_TIMINGS_FILE
        with timings_path.open("w") as stream:
            for timing in batch_timings:
                stream.write(json.dumps(timing) + "\n")
        metadata.update(
            batch_timings_file=BATCH_TIMINGS_FILE,
            batch_timings_sha256=sha256_file(timings_path),
        )
    (directory / METADATA_FILE).write_text(json.dumps(metadata) + "\n")
    return directory


class MVMoEPaperProtocolTests(unittest.TestCase):
    def test_cvrp_protocol_fingerprints_are_unchanged(self):
        self.assertEqual(
            json_fingerprint(paper_inference_config(50, problem="CVRP")),
            "41850bd8205727b523e1ba6ffc0a19dc15ad1a87a5daff16e7216e266ac4524e")
        self.assertEqual(
            json_fingerprint(paper_inference_config(100, problem="CVRP")),
            "844d315d9ccb2bdce5f9373aa7745e0dd0d9fca0719958f6ae80f7a86f977688")

    def test_exact_formal_configuration_for_both_problems_and_sizes(self):
        for problem in ("CVRP", "CVRPTW"):
            for size in (50, 100):
                with self.subTest(problem=problem, size=size):
                    config = (paper_inference_config(size, problem="CVRP")
                              if problem == "CVRP"
                              else scaled_paper_inference_config(size))
                    self.assertEqual(config["variant"], "MVMoE/4E")
                    self.assertEqual(config["model_type"], "MOE")
                    self.assertEqual(config["num_experts"], 4)
                    self.assertEqual(config["routing_level"], "node")
                    self.assertEqual(config["routing_method"], "input_choice")
                    self.assertEqual(config["original_batch_size"], 1)
                    self.assertEqual(config["pomo_size"], size)
                    self.assertEqual(config["aug_factor"], 8)
                    self.assertEqual(config["eval_type"], "argmax")
                    self.assertEqual(config["seed"], 2024)
                    self.assertEqual(config["fine_tune_epochs"], 0)
                    self.assertFalse(config["training"])
                    self.assertFalse(config["backward"])
                    self.assertFalse(config["optimizer_created"])
                    self.assertEqual(config["model"], MODEL_CONFIG)

    def test_cvrptw_scaled_protocol_is_explicit_for_both_sizes(self):
        expected = {
            "input_scaling": "continuous_official_style",
            "scaler_rule": "s=max(max(original coordinates), original_depot_tw_end/3.0)",
            "fields_scaled": ["coordinates", "time_windows", "service_times"],
            "demand_normalization": "raw_demand/raw_capacity exactly once",
            "loc_scaler": None,
            "distance_rounding": False,
            "model_inference_domain": "scaled_continuous",
            "final_validation_domain": "original_ml4co",
            "final_objective_domain": "original_ml4co",
            "speed": 1.0,
        }
        for size in (50, 100):
            with self.subTest(size=size):
                config = scaled_paper_inference_config(size)
                for key, value in expected.items():
                    self.assertEqual(config[key], value)
                old = unscaled_control_inference_config(size)
                self.assertNotIn("input_scaling", old)
                self.assertNotEqual(config, old)

    def test_full_dataset_counts(self):
        self.assertEqual(supported_config(50)["dataset_count"], 10000)
        self.assertEqual(supported_config(100)["dataset_count"], 10000)
        self.assertEqual(get_size_config(50)["dataset_count"], 1000)
        self.assertEqual(get_size_config(100)["dataset_count"], 1000)

    def test_model_config_matches_both_server_verified_runners(self):
        self.assertEqual(MODEL_CONFIG, CVRP_INTEGRATION_MODEL_CONFIG)
        self.assertEqual(MODEL_CONFIG, CVRPTW_INTEGRATION_MODEL_CONFIG)

    def test_timing_semantics_are_single_instance_and_paper_comparable(self):
        self.assertIn("single-original-instance", TIMING_SEMANTICS)
        self.assertIn("best-candidate selection", TIMING_SEMANTICS)
        self.assertNotIn("amortized batch runtime", TIMING_SEMANTICS)
        self.assertNotIn("not paper-comparable", TIMING_SEMANTICS)


class PaperAggregationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_means_objective_per_instance_gap_and_runtime(self):
        # Gaps are 100% and 0%; gap-of-means would be about 0.99%, not 50%.
        records = [record(index, 2.0, 1.0, runtime=0.2)
                   for index in range(5000)]
        records.extend(record(index, 100.0, 100.0, runtime=0.4)
                       for index in range(5000, 10000))
        chunk = write_chunk(self.root, "chunk", records)
        summary = summarize_chunks([chunk])
        dispatched = summarize_for_protocol([chunk])
        self.assertEqual(summary["status"], "PAPER_READY")
        self.assertAlmostEqual(summary["obj_mean_independent_objective"], 51.0)
        self.assertAlmostEqual(summary["drop_mean_per_instance_gap_percent"], 50.0)
        self.assertAlmostEqual(summary["time_mean_single_instance_seconds"], 0.3)
        self.assertEqual(
            {key: value for key, value in dispatched.items() if key != "created_at"},
            {key: value for key, value in summary.items() if key != "created_at"})

    def test_resume_accepts_exact_identity_and_rejects_changed_identity(self):
        directory = self.root / "resume"
        value = identity([0], dataset_count=1)
        _, completed = initialize_chunk(directory, value)
        self.assertEqual(completed, set())
        item = record(0, 2.0, 1.0)
        item.pop("kit_feasible")
        item.pop("kit_objective")
        item.pop("kit_objective_agrees")
        append_record(directory, item)
        finalize_chunk(directory)
        _, completed = initialize_chunk(directory, value)
        self.assertEqual(completed, {0})
        changed = json.loads(json.dumps(value))
        changed["paper_protocol"]["aug_factor"] = 1
        with self.assertRaisesRegex(ValueError, "resume refused"):
            initialize_chunk(directory, changed)

    def test_unscaled_control_cannot_resume_as_scaled_cvrptw(self):
        directory = self.root / "cvrptw-resume"
        old = identity([0], dataset_count=1000, problem="CVRPTW")
        old["paper_protocol"] = unscaled_control_inference_config(50)
        initialize_chunk(directory, old)
        scaled = identity([0], dataset_count=1000, problem="CVRPTW")
        with self.assertRaisesRegex(ValueError, "resume refused"):
            initialize_chunk(directory, scaled)

    def test_kit_validated_resume_rejects_valid_post_finalization_mutation(self):
        directory = self.root / "kit-validated-resume"
        value = identity([0], dataset_count=1)
        initialize_chunk(directory, value)
        item = record(0, 2.0, 1.0)
        item.pop("kit_feasible")
        item.pop("kit_objective")
        item.pop("kit_objective_agrees")
        append_record(directory, item)
        metadata = finalize_chunk(directory)

        validated = dict(item, kit_feasible=True, kit_objective=2.0,
                         kit_objective_agrees=True)
        validated_path = directory / VALIDATED_RECORDS_FILE
        validated_path.write_text(json.dumps(validated) + "\n")
        metadata.update(
            state="KIT_VALIDATED",
            validated_records_sha256=sha256_file(validated_path),
        )
        write_json(directory / METADATA_FILE, metadata)
        validated_hash = sha256_file(validated_path)

        _, completed = initialize_chunk(directory, value)
        self.assertEqual(completed, {0})
        mutated = dict(item, runtime_seconds=0.2)
        (directory / RECORDS_FILE).write_text(json.dumps(mutated) + "\n")
        with self.assertRaisesRegex(
                ValueError, "inference records changed after finalization"):
            initialize_chunk(directory, value)
        self.assertEqual(sha256_file(validated_path), validated_hash)

    def test_missing_index_is_rejected(self):
        chunk = write_chunk(self.root, "chunk", [record(0, 2.0, 1.0)],
                            expected_indices=[0, 1])
        with self.assertRaisesRegex(ValueError, "chunk.*missing"):
            summarize_chunks([chunk])

    def test_duplicate_index_across_chunks_is_rejected(self):
        first = write_chunk(self.root, "a", [record(0, 2.0, 1.0)])
        second = write_chunk(self.root, "b", [record(0, 2.0, 1.0)])
        with self.assertRaisesRegex(ValueError, "duplicate dataset index"):
            summarize_chunks([first, second])

    def test_mixed_checkpoint_dataset_and_inference_config_are_rejected(self):
        changes = {
            "checkpoint": lambda value: value["checkpoint"].update(sha256="0" * 64),
            "dataset": lambda value: value["dataset"].update(sha256="1" * 64),
            "config": lambda value: value["paper_protocol"].update(aug_factor=1),
        }
        for name, mutation in changes.items():
            with self.subTest(name=name):
                case = self.root / name
                case.mkdir()
                first = write_chunk(case, "a", [record(0, 2.0, 1.0)])
                second = write_chunk(case, "b", [record(1, 2.0, 1.0)],
                                     mutate_identity=mutation)
                with self.assertRaisesRegex(ValueError, "mixed paper"):
                    summarize_chunks([first, second])

    def test_mixed_unscaled_and_scaled_cvrptw_chunks_are_rejected(self):
        def make_unscaled(value):
            value["paper_protocol"] = unscaled_control_inference_config(50)

        first = write_chunk(
            self.root, "unscaled", [record(0, 2.0, 1.0)],
            mutate_identity=make_unscaled, dataset_count=1000, problem="CVRPTW")
        second = write_chunk(
            self.root, "scaled", [record(1, 2.0, 1.0, scaled=True)],
            dataset_count=1000, problem="CVRPTW")
        with self.assertRaisesRegex(ValueError, "mixed paper"):
            summarize_for_protocol([second, first])

    def test_unscaled_cvrptw_is_recognized_as_control_not_paper_result(self):
        def make_unscaled(value):
            value["paper_protocol"] = unscaled_control_inference_config(50)

        chunk = write_chunk(
            self.root, "unscaled-only", [record(0, 2.0, 1.0)],
            mutate_identity=make_unscaled, dataset_count=1000, problem="CVRPTW")
        with self.assertRaisesRegex(ValueError, "unscaled CVRPTW control"):
            summarize_for_protocol([chunk])

    def test_synthetic_scaled_cvrptw_full_set_is_paper_ready(self):
        records = [record(index, 2.0, 1.0, scaled=True)
                   for index in range(1000)]
        chunk = write_chunk(
            self.root, "scaled-full", records,
            dataset_count=1000, problem="CVRPTW")
        summary = summarize_for_protocol([chunk])
        self.assertEqual(summary["status"], "PAPER_READY")
        self.assertEqual(
            summary["consistency_identity"]["paper_protocol"]["input_scaling"],
            "continuous_official_style")

    def test_scaled_bs10_summary_uses_native_batch_latency_and_total(self):
        records = []
        timings = []
        for batch_index in range(100):
            runtime = 0.2 + batch_index / 1000.0
            indices = list(range(batch_index * 10, (batch_index + 1) * 10))
            timings.append({
                "batch_index": batch_index, "dataset_indices": indices,
                "batch_size": 10, "runtime_seconds": runtime,
            })
            for position, index in enumerate(indices):
                item = record(index, 2.0, 1.0, runtime=runtime, scaled=True)
                item.update(batch_index=batch_index, position_in_batch=position)
                records.append(item)
        chunk = write_chunk(
            self.root, "scaled-bs10", records, dataset_count=1000,
            problem="CVRPTW", batch_size=10, batch_timings=timings)
        summary = summarize_for_protocol([chunk])
        self.assertEqual(summary["batch_size"], 10)
        self.assertEqual(summary["num_batches"], 100)
        self.assertAlmostEqual(
            summary["time_mean_batch_seconds"],
            sum(row["runtime_seconds"] for row in timings) / 100)
        self.assertAlmostEqual(
            summary["time_total_seconds"],
            sum(row["runtime_seconds"] for row in timings))
        self.assertIsNone(summary["time_mean_single_instance_seconds"])

    def test_scaled_record_conversion_or_identity_mismatch_is_rejected(self):
        for name, mutation in (
                ("conversion", lambda item: item.update(
                    scaled_objective_times_s=3.0)),
                ("identity", lambda item: item.pop("input_scaling_protocol"))):
            with self.subTest(name=name):
                case = self.root / name
                case.mkdir()
                items = [record(index, 2.0, 1.0, scaled=True)
                         for index in range(1000)]
                mutation(items[0])
                chunk = write_chunk(
                    case, "chunk", items,
                    dataset_count=1000, problem="CVRPTW")
                with self.assertRaisesRegex(ValueError, "scaled|scaling"):
                    summarize_for_protocol([chunk])

    def test_infeasible_record_cannot_be_paper_ready(self):
        chunk = write_chunk(self.root, "chunk", [record(0, 2.0, 1.0, feasible=False),
                                                  record(1, 2.0, 1.0)])
        with self.assertRaisesRegex(ValueError, "failed independent feasibility"):
            summarize_chunks([chunk])

    def test_nan_and_infinity_are_rejected(self):
        for name, field, value in (("nan", "independent_objective", float("nan")),
                                   ("inf", "runtime_seconds", float("inf"))):
            with self.subTest(name=name):
                item = record(0, 2.0, 1.0)
                item[field] = value
                case = self.root / name
                case.mkdir()
                chunk = write_chunk(case, "chunk", [item, record(1, 2.0, 1.0)])
                with self.assertRaisesRegex(ValueError, "finite"):
                    summarize_chunks([chunk])


if __name__ == "__main__":
    unittest.main()
