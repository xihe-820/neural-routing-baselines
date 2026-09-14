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
from methods.mvmoe.cvrptw.run import MODEL_CONFIG as CVRPTW_INTEGRATION_MODEL_CONFIG
from methods.mvmoe.paper_config import MODEL_CONFIG, paper_inference_config


def record(index, independent, reference, *, feasible=True, runtime=0.1):
    return {
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


def identity(indices, *, dataset_count=10000):
    return {
        "method": "MVMoE", "variant": "MVMoE/4E", "problem": "CVRP",
        "problem_size": 50,
        "paper_protocol": paper_inference_config(50, problem="CVRP"),
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
                dataset_count=10000):
    directory = root / name
    directory.mkdir()
    indices = expected_indices if expected_indices is not None else [
        item["dataset_instance_index"] for item in records]
    value = identity(indices, dataset_count=dataset_count)
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
    (directory / METADATA_FILE).write_text(json.dumps(metadata) + "\n")
    return directory


class MVMoEPaperProtocolTests(unittest.TestCase):
    def test_exact_formal_configuration_for_both_problems_and_sizes(self):
        for problem in ("CVRP", "CVRPTW"):
            for size in (50, 100):
                with self.subTest(problem=problem, size=size):
                    config = paper_inference_config(size, problem=problem)
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
        self.assertEqual(summary["status"], "PAPER_READY")
        self.assertAlmostEqual(summary["obj_mean_independent_objective"], 51.0)
        self.assertAlmostEqual(summary["drop_mean_per_instance_gap_percent"], 50.0)
        self.assertAlmostEqual(summary["time_mean_single_instance_seconds"], 0.3)

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
