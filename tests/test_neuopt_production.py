import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from methods.neuopt.cvrp.compat import configure_production_decoder
from methods.neuopt.cvrp.config import supported_config
from methods.neuopt.cvrp.paper_protocol import (
    FORMAL_BATCH_SIZES, FORMAL_T_VALUES, UPSTREAM_COMMIT,
    legacy_calibration_protocol, paper_protocol, protocol_fingerprint,
)
from methods.neuopt.cvrp.production_results import (
    HYBRID_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION, TIMING_SEMANTICS,
    aggregate_chunks, append_batch, build_hybrid_summary, finalize_chunk,
    initialize_or_resume, validate_identity, validate_record,
    _verify_aggregate_summary,
)


ROOT = Path(__file__).resolve().parents[1]
PINNED_DECODER = ROOT / "external/NeuOpt/nets/graph_layers.py"


def identity(*, size=50, T_max=20, batch_size=1, offset=0, count=1):
    protocol = paper_protocol(size, T_max=T_max, batch_size=batch_size)
    config = supported_config(size)
    return {
        "method": "NeuOpt", "variant": "NeuOpt-GIRE", "problem": "CVRP",
        "problem_size": size, "paper_protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "project": {"commit": "a" * 40, "dirty": False, "url": "project"},
        "upstream": {"commit": UPSTREAM_COMMIT, "dirty": False, "url": "upstream"},
        "checkpoint": {"path": "/checkpoint", "sha256": config["checkpoint_sha256"]},
        "dataset": {"path": "/dataset", "sha256": config["dataset_sha256"],
                    "count": config["dataset_count"]},
        "prepared_input": {"path": "/input", "sha256": "b" * 64,
                           "metadata_path": "/input.json"},
        "chunk": {"offset": offset, "count": count,
                  "expected_indices": list(range(offset, offset + count))},
        "warmup": {"batches": 1, "rng_state_restored": True},
        "rng_policy": "sequential continuous RNG with exact replay",
        "environment": {"device": "cuda:0", "gpu": "NVIDIA GeForce RTX 4090"},
        "tensorboard_compatibility": {"official_source_modified": False},
        "decoder_compatibility": {
            "bs1_shape_shim": batch_size == 1,
            "original_batch_size": batch_size, "D2A": 1, "val_m": 1,
            "internal_decoder_batch_size": batch_size,
            "official_source_modified": False,
            "action_reward_logits_rng_budget_changed": False,
        },
        "source_provenance": [], "timing_semantics": TIMING_SEMANTICS,
    }


def record(index, *, batch_index=0, position=0, objective=10.0, reference=10.0):
    gap = (objective - reference) / reference * 100.0
    return {
        "dataset_instance_index": index, "instance_id": str(index),
        "batch_index": batch_index, "position_in_batch": position,
        "canonical_solution": [0, 1, 0], "successor": [1, 0],
        "reported_objective": objective, "timed_official_objective": objective,
        "replay_official_objective": objective,
        "official_recomputed_objective": objective,
        "independent_objective": objective, "reference_objective": reference,
        "kit_reference_objective": reference, "gap_percent": gap,
        "independent_feasible": True, "reported_objective_agrees": True,
        "kit_feasible": True, "kit_objective": objective,
        "kit_objective_agrees": True, "constraint_details": {},
        "evidence_status": "KIT_VALIDATED",
    }


def timing(batch_index, indices, seconds=1.0):
    return {
        "batch_index": batch_index, "dataset_indices": list(indices),
        "batch_size": len(indices), "runtime_seconds": seconds,
        "timed_replay": {
            "timed_rollout_record": False, "evidence_replay_record": True,
            "rng_state_restored": True,
            "timed_replay_official_objective_exact": True,
            "timed_replay_rng_after_exact": True,
            "cuda_synchronized_before_timing": True,
            "cuda_synchronized_after_timing": True,
            "evidence_replay_excluded_from_runtime": True,
        },
    }


def write_chunk(root, name, current_identity, records, timings):
    directory = Path(root) / name
    initialize_or_resume(directory, current_identity)
    batch_size = current_identity["paper_protocol"]["original_batch_size"]
    for batch_index, current_timing in enumerate(timings):
        start = batch_index * batch_size
        append_batch(directory, records[start:start + batch_size], current_timing)
    finalize_chunk(directory)
    return directory


class NeuOptFinalProtocolTests(unittest.TestCase):
    def test_exact_eight_frozen_protocols(self):
        observed = set()
        for size in (50, 100):
            for T_max in FORMAL_T_VALUES:
                for batch_size in FORMAL_BATCH_SIZES:
                    protocol = paper_protocol(size, T_max=T_max, batch_size=batch_size)
                    observed.add(protocol_fingerprint(protocol))
                    self.assertEqual(protocol["D2A"], 1)
                    self.assertEqual(protocol["val_m"], 1)
                    self.assertEqual(protocol["stall_limit"], 10)
                    self.assertEqual(protocol["k"], 4)
                    self.assertEqual(protocol["seed"], 6666)
                    self.assertEqual(protocol["original_batch_size"], batch_size)
        self.assertEqual(len(observed), 8)

    def test_nonfinal_d2a_T_and_batch_are_rejected(self):
        for kwargs in (
                {"T_max": 20, "batch_size": 1, "d2a": 5},
                {"T_max": 1000, "batch_size": 1},
                {"T_max": 20, "batch_size": 10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                paper_protocol(50, **kwargs)

    def test_legacy_d2a5_identity_is_rejected_by_production_gate(self):
        current = identity()
        current["paper_protocol"] = legacy_calibration_protocol(50, T_max=1000)
        current["protocol_fingerprint"] = protocol_fingerprint(current["paper_protocol"])
        with self.assertRaisesRegex(ValueError, "D2A=1"):
            validate_identity(current)


class NeuOptProductionDecoderTests(unittest.TestCase):
    SOURCE = """\
def forward(self):
    stopped = stopped | (action == next_of_last_action).squeeze()
    stopped = (action == next_of_last_action).squeeze()
"""

    def test_bs1_uses_guarded_in_memory_shape_shim_without_disk_edit(self):
        class Decoder:
            def forward(self):
                return None

        before = hashlib.sha256(PINNED_DECODER.read_bytes()).hexdigest()
        with patch("methods.neuopt.cvrp.compat.inspect.getsource", return_value=self.SOURCE):
            provenance = configure_production_decoder(
                Decoder, original_batch_size=1, val_m=1)
        after = hashlib.sha256(PINNED_DECODER.read_bytes()).hexdigest()
        self.assertTrue(provenance["bs1_shape_shim"])
        self.assertEqual(provenance["internal_decoder_batch_size"], 1)
        self.assertFalse(provenance["official_source_modified"])
        self.assertEqual(before, after)

    def test_bs100_requires_unmodified_pinned_shape_source(self):
        class Decoder:
            def forward(self):
                return None

        with patch("methods.neuopt.cvrp.compat.inspect.getsource", return_value=self.SOURCE):
            provenance = configure_production_decoder(
                Decoder, original_batch_size=100, val_m=1)
        self.assertFalse(provenance["bs1_shape_shim"])
        self.assertEqual(provenance["internal_decoder_batch_size"], 100)


class NeuOptProductionArtifactTests(unittest.TestCase):
    def test_metrics_use_independent_objectives_per_instance_gaps_and_batch_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_chunk(
                tmp, "first", identity(offset=0, count=1),
                [record(0, objective=12.0, reference=10.0)], [timing(0, [0], 2.0)])
            second = write_chunk(
                tmp, "second", identity(offset=1, count=1),
                [record(1, objective=21.0, reference=20.0)], [timing(0, [1], 4.0)])
            summary = aggregate_chunks(
                [second, first], expected_offset=0, expected_count=2,
                scope="timing_subset")
        self.assertEqual(summary["obj_mean_independent_objective"], 16.5)
        self.assertEqual(summary["drop_mean_per_instance_gap_percent"], 12.5)
        self.assertEqual(summary["total_inference_seconds"], 6.0)
        self.assertEqual(summary["mean_batch_latency_seconds"], 3.0)
        self.assertEqual(summary["table_time_seconds"], 3.0)
        self.assertTrue(summary["estimated"])
        self.assertEqual(summary["sample_indices"], [0, 1])

    def test_one_infeasible_record_fails_closed(self):
        changed = record(0)
        changed["kit_feasible"] = False
        with self.assertRaisesRegex(ValueError, "correctness gate"):
            validate_record(changed)

    def test_bs100_batch_runtime_is_stored_once_and_never_divided_for_table_time(self):
        records = [record(index, position=index) for index in range(100)]
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_chunk(
                tmp, "bs100", identity(batch_size=100, count=100), records,
                [timing(0, range(100), 5.0)])
            summary = __import__("json").loads(
                (directory / "summary.json").read_text())
        self.assertEqual(summary["num_batches"], 1)
        self.assertEqual(summary["total_inference_seconds"], 5.0)
        self.assertEqual(summary["mean_batch_latency_seconds"], 5.0)
        self.assertEqual(summary["table_time_seconds"], 5.0)
        self.assertEqual(summary["mean_instance_latency_seconds"], 0.05)

    def test_resume_rolls_back_one_uncommitted_record_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "interrupted"
            current_identity = identity()
            initialize_or_resume(directory, current_identity)
            (directory / "validated_records.jsonl").write_text(
                json.dumps(record(0)) + "\n")
            metadata, records, timings = initialize_or_resume(
                directory, current_identity)
        self.assertEqual(metadata["completed_records"], 0)
        self.assertEqual(records, [])
        self.assertEqual(timings, [])

    def test_missing_and_duplicate_dataset_indices_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_chunk(tmp, "first", identity(offset=0, count=1),
                                [record(0)], [timing(0, [0])])
            with self.assertRaisesRegex(ValueError, "missing or extra"):
                aggregate_chunks([first], expected_offset=0, expected_count=2,
                                 scope="timing_subset")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                aggregate_chunks([first, first], expected_offset=0, expected_count=1,
                                 scope="timing_subset")

    def test_mixed_T_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            low = write_chunk(tmp, "low", identity(T_max=20, offset=0, count=1),
                              [record(0)], [timing(0, [0])])
            high = write_chunk(tmp, "high", identity(T_max=50, offset=1, count=1),
                               [record(1)], [timing(0, [1])])
            with self.assertRaisesRegex(ValueError, "mixed T"):
                aggregate_chunks([low, high], expected_offset=0, expected_count=2,
                                 scope="timing_subset")

    def test_hybrid_source_summary_is_recomputed_from_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_chunk(tmp, "one", identity(),
                                    [record(0)], [timing(0, [0])])
            summary = aggregate_chunks(
                [directory], expected_offset=0, expected_count=1,
                scope="timing_subset")
            _verify_aggregate_summary(summary)
            summary["table_time_seconds"] = 99.0
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                _verify_aggregate_summary(summary)

    def test_aggregator_rejects_mixed_batch_checkpoint_and_dataset(self):
        base = identity()
        cases = {}
        changed_batch = identity(batch_size=100, count=100)
        cases["batch size"] = changed_batch
        changed_checkpoint = identity()
        changed_checkpoint["checkpoint"] = dict(
            changed_checkpoint["checkpoint"], sha256="c" * 64)
        cases["checkpoint"] = changed_checkpoint
        changed_dataset = identity()
        changed_dataset["dataset"] = dict(
            changed_dataset["dataset"], sha256="d" * 64)
        cases["dataset"] = changed_dataset
        for label, changed in cases.items():
            first_metadata = {
                "resume_identity": base,
                "validated_records_sha256": "1" * 64,
                "batch_timings_sha256": "2" * 64,
            }
            second_metadata = {
                "resume_identity": changed,
                "validated_records_sha256": "3" * 64,
                "batch_timings_sha256": "4" * 64,
            }
            with self.subTest(label=label), patch(
                    "methods.neuopt.cvrp.production_results.load_chunk",
                    side_effect=[(first_metadata, [], []),
                                 (second_metadata, [], [])]):
                with self.assertRaisesRegex(ValueError, "mixed T"):
                    aggregate_chunks(["one", "two"], expected_offset=0,
                                     expected_count=1, scope="timing_subset")


class NeuOptHybridTests(unittest.TestCase):
    def test_hybrid_keeps_fullset_quality_and_estimated_bs1_time_distinct(self):
        quality_identity = identity(batch_size=100, count=100)
        timing_identity = identity(batch_size=1, count=2)
        quality = {
            "schema_version": SUMMARY_SCHEMA_VERSION, "fullset": True,
            "scope": "fullset", "instance_count": 10000,
            "obj_mean_independent_objective": 10.5,
            "drop_mean_per_instance_gap_percent": 1.2,
            "consistency_identity": quality_identity,
            "source_chunks": [{"path": "/quality"}],
        }
        timed = {
            "schema_version": SUMMARY_SCHEMA_VERSION, "fullset": False,
            "scope": "timing_subset", "estimated": True, "instance_count": 2,
            "sample_indices": [0, 1], "mean_batch_latency_seconds": 0.3,
            "consistency_identity": timing_identity,
            "source_chunks": [{"path": "/timing"}],
        }
        quality_records = {0: record(0, objective=10.0), 1: record(1, objective=12.0)}
        timing_records = {0: record(0, objective=10.1), 1: record(1, objective=12.1)}
        with (patch(
                "methods.neuopt.cvrp.production_results._verify_aggregate_summary"),
              patch(
                "methods.neuopt.cvrp.production_results._records_from_summary",
                side_effect=[quality_records, timing_records])):
            hybrid = build_hybrid_summary(quality, timed)
        self.assertEqual(hybrid["schema_version"], HYBRID_SCHEMA_VERSION)
        self.assertFalse(hybrid["bs1_fullset_completed"])
        self.assertTrue(hybrid["time_source"]["estimated"])
        self.assertEqual(hybrid["table10_candidate"]["Obj"], 10.5)
        self.assertEqual(hybrid["table10_candidate"]["Time_seconds"], 0.3)
        self.assertTrue(
            hybrid["table10_candidate"]["quality_and_time_from_different_experiments"])
        self.assertEqual(
            hybrid["same_subset_comparison"]["warning_status"],
            "PENDING_USER_REVIEW_NO_AUTOMATIC_THRESHOLD")


if __name__ == "__main__":
    unittest.main()
