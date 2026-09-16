import tempfile
import unittest
from pathlib import Path

from methods.neuopt.cvrp.compat import dimension_preserving_decoder_source
from methods.neuopt.cvrp.config import supported_config
from methods.neuopt.cvrp.paper_protocol import (
    CALIBRATION_TARGETS, MANUSCRIPT_CANDIDATE_T, paper_protocol,
    pending_final_t, protocol_fingerprint,
)
from methods.neuopt.cvrp.paper_results import (
    TIMING_SEMANTICS, build_calibration_report, json_fingerprint, load_artifact,
    summarize_same_protocol, write_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def identity(size=50, T_max=1000):
    protocol = paper_protocol(size, T_max=T_max)
    config = supported_config(size)
    return {
        "method": "NeuOpt", "variant": "NeuOpt-GIRE", "problem": "CVRP",
        "problem_size": size, "paper_protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "original_batch_size": 1,
        "project": {"commit": "a" * 40, "dirty": False, "url": "project"},
        "upstream": {
            "commit": "ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b",
            "dirty": False, "url": "https://github.com/yining043/NeuOpt",
        },
        "checkpoint": {"path": "/checkpoint", "sha256": config["checkpoint_sha256"]},
        "dataset": {"path": "/dataset", "sha256": config["dataset_sha256"],
                    "count": config["dataset_count"]},
        "prepared_input": {"path": "/input", "sha256": "c" * 64,
                           "metadata_path": "/input.json"},
        "subset": {"offset": 0, "count": 1, "dataset_indices": [0]},
        "warmup": {"instances": 1, "rng_state_restored": True},
        "environment": {"device": "cuda:0", "gpu": "NVIDIA GeForce RTX 4090"},
        "tensorboard_compatibility": {"official_source_modified": False},
        "bs1_compatibility": {
            "bs1_shape_shim": True, "official_source_modified": False,
            "action_reward_logits_rng_budget_changed": False,
        },
        "source_provenance": [], "timing_semantics": TIMING_SEMANTICS,
    }


def record(runtime=0.2, objective=10.0):
    return {
        "dataset_instance_index": 0, "instance_id": "0",
        "canonical_solution": [0, 1, 0], "successor": [1, 0],
        "selected_d2a_candidate": 0,
        "reported_objective": objective,
        "official_recomputed_objective": objective,
        "independent_objective": objective, "reference_objective": objective,
        "kit_reference_objective": objective,
        "gap_percent": 0.0, "runtime_seconds": runtime,
        "runtime_semantics": TIMING_SEMANTICS,
        "independent_feasible": True, "reported_objective_agrees": True,
        "kit_feasible": True, "kit_objective": objective,
        "kit_objective_agrees": True, "constraint_details": {},
        "evidence_status": "KIT_VALIDATED",
    }


class NeuOptPaperProtocolTests(unittest.TestCase):
    def test_frozen_size_identities_and_runtime_targets(self):
        expected = {
            50: ("cvrp50_hgs-1s_10.366.pkl", 40.0, 0.4,
                 "eea12fbefe9c1bcc008d56ecfc1c50dadd64ac774f3547774c9fade8a7baa6c2",
                 "1cd201ca47888e51068a157389460641c81d71054f064d9c8ea1743312289e3a",
                 0.197, 0.804),
            100: ("cvrp100_hgs-20s_15.563.pkl", 50.0, 0.2,
                  "bb47d5a113848e5a404edefc562d5d2ef6b0ade1aafc287828bdf60364e23532",
                  "502a5904182306c1a3f65f7b1a8503a609ff2a044db054abcf93690af36594fb",
                  0.307, 1.193),
        }
        for size, values in expected.items():
            config = supported_config(size)
            target = CALIBRATION_TARGETS[size]
            self.assertEqual(
                (config["dataset_filename"], config["capacity"], config["dummy_rate"],
                 config["dataset_sha256"], config["checkpoint_sha256"],
                 target["fewer_seconds"], target["more_seconds"]), values)
            self.assertEqual(config["dataset_count"], 10000)

    def test_d2a_five_is_val_m_five_and_original_batch_is_one(self):
        protocol = paper_protocol(50, T_max=1000, d2a=5)
        self.assertEqual(protocol["D2A"], 5)
        self.assertEqual(protocol["val_m"], 5)
        self.assertEqual(protocol["original_batch_size"], 1)
        with self.assertRaisesRegex(ValueError, "D2A=5"):
            paper_protocol(50, T_max=1000, d2a=1)

    def test_T_max_changes_protocol_fingerprint(self):
        low = paper_protocol(50, T_max=1000)
        high = paper_protocol(50, T_max=5000)
        self.assertNotEqual(protocol_fingerprint(low), protocol_fingerprint(high))
        self.assertNotEqual(json_fingerprint(identity(T_max=1000)),
                            json_fingerprint(identity(T_max=5000)))
        self.assertEqual((low["budget_status"], high["budget_status"]),
                         ("MANUSCRIPT_CANDIDATE", "MANUSCRIPT_CANDIDATE"))

    def test_problem_sizes_can_retain_different_final_T(self):
        selected = pending_final_t(
            cvrp50_fewer=100, cvrp50_more=400,
            cvrp100_fewer=150, cvrp100_more=600)
        self.assertEqual(selected["50"], {"fewer": 100, "more": 400})
        self.assertEqual(selected["100"], {"fewer": 150, "more": 600})


class NeuOptBS1SourceCompatibilityTests(unittest.TestCase):
    def test_official_stopped_squeeze_is_patched_exactly(self):
        source_path = ROOT / "external/NeuOpt/nets/graph_layers.py"
        source = source_path.read_text()
        patched = dimension_preserving_decoder_source(source)
        self.assertEqual(source.count("(action == next_of_last_action).squeeze()"), 2)
        self.assertEqual(patched.count("(action == next_of_last_action).squeeze()"), 0)
        self.assertEqual(patched.count("(action == next_of_last_action).squeeze(-1)"), 2)
        before = source.replace(
            "(action == next_of_last_action).squeeze()",
            "(action == next_of_last_action).squeeze(-1)")
        self.assertEqual(patched, before)

    def test_source_drift_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "expected BS1 squeeze"):
            dimension_preserving_decoder_source("def forward(): pass")


class NeuOptPaperArtifactTests(unittest.TestCase):
    def write(self, root, name, current_identity, current_record=None):
        path = Path(root) / name
        write_artifact(path, current_identity, [current_record or record()])
        return path

    def test_timing_metadata_is_single_original_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, "one", identity())
            metadata, records = load_artifact(path)
            self.assertEqual(metadata["resume_identity"]["original_batch_size"], 1)
            self.assertEqual(metadata["resume_identity"]["timing_semantics"],
                             TIMING_SEMANTICS)
            self.assertNotIn("amortized", TIMING_SEMANTICS)
            self.assertEqual(len(records), 1)

    def test_objective_claims_fail_closed(self):
        changed = record()
        changed["reported_objective"] = 11.0
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "disagrees"):
                self.write(tmp, "bad_objective", identity(), changed)

    def test_same_protocol_aggregator_rejects_mixed_T(self):
        with tempfile.TemporaryDirectory() as tmp:
            low = self.write(tmp, "low", identity(T_max=1000))
            high = self.write(tmp, "high", identity(T_max=5000))
            with self.assertRaisesRegex(ValueError, "mixed T"):
                summarize_same_protocol([low, high])

    def test_mixed_D2A_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            changed = identity()
            changed["paper_protocol"] = dict(changed["paper_protocol"], D2A=1, val_m=1)
            changed["protocol_fingerprint"] = protocol_fingerprint(changed["paper_protocol"])
            path = self.write(tmp, "bad_d2a", changed)
            with self.assertRaisesRegex(ValueError, "D2A=5"):
                load_artifact(path)

    def test_calibration_rejects_mixed_checkpoint_and_dataset(self):
        for field in ("checkpoint", "dataset"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                low_identity = identity(T_max=1000)
                high_identity = identity(T_max=5000)
                high_identity[field]["sha256"] = "d" * 64
                low = self.write(tmp, "low", low_identity)
                high = self.write(tmp, "high", high_identity)
                with self.assertRaisesRegex(ValueError, "mix"):
                    build_calibration_report([low, high], problem_size=50)

    def test_report_retains_candidates_without_freezing_T(self):
        with tempfile.TemporaryDirectory() as tmp:
            low = self.write(tmp, "t1000", identity(T_max=1000), record(runtime=0.21))
            high = self.write(tmp, "t5000", identity(T_max=5000), record(runtime=0.81))
            report = build_calibration_report([high, low], problem_size=50)
            self.assertEqual([row["T_max"] for row in report["candidates"]],
                             list(MANUSCRIPT_CANDIDATE_T))
            self.assertEqual(report["targets"]["fewer_seconds"], 0.197)
            self.assertEqual(report["targets"]["more_seconds"], 0.804)
            self.assertIsNone(report["final_T_fewer"])
            self.assertIsNone(report["final_T_more"])
            self.assertFalse(report["paper_ready"])


if __name__ == "__main__":
    unittest.main()
