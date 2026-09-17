import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common.objective_agreement import OBJECTIVE_ATOL, OBJECTIVE_RTOL
from methods.neuopt.cvrp.compat import configure_production_decoder
from methods.neuopt.tsp.adapter import adapt_batch
from methods.neuopt.tsp.config import supported_config
from methods.neuopt.tsp.compat import (PINNED_TSP_STEP_SHA256,
                                       replay_record_compatibility)
from methods.neuopt.tsp.decode import decode_successor
from methods.neuopt.tsp.paper_eval import (
    official_option_args, timed_rollout_with_evidence_replay,
)
from methods.neuopt.tsp.paper_protocol import (
    CALIBRATION_INDICES, CALIBRATION_TARGETS, FIRST_ROUND_T_VALUES,
    UPSTREAM_COMMIT, UPSTREAM_URL, calibration_protocol, protocol_fingerprint,
)
from methods.neuopt.tsp.runtime_calibration import (
    RNG_POLICY, TIMING_SEMANTICS, WARMUP_POLICY, build_calibration_report,
    validate_identity, validate_record, write_calibration_report, write_candidate,
)
from problems.tsp.objective import cycle_length
from problems.tsp.validate import validate


ROOT = Path(__file__).resolve().parents[1]


def successor_from_order(order):
    successor = np.empty(len(order), dtype=np.int64)
    for current, following in zip(order, order[1:] + order[:1]):
        successor[current] = following
    return successor


def fake_identity(T_max=1):
    protocol = calibration_protocol(100, T_max=T_max)
    config = supported_config(100)
    return {
        "method": "NeuOpt", "variant": "NeuOpt-GIRE", "problem": "TSP",
        "problem_size": 100, "paper_protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "project": {"commit": "1" * 40, "dirty": False,
                    "url": "https://github.com/xihe-820/neural-routing-baselines"},
        "upstream": {"commit": UPSTREAM_COMMIT, "dirty": False, "url": UPSTREAM_URL},
        "checkpoint": {"path": "/assets/tsp100.pt",
                       "sha256": config["checkpoint_sha256"]},
        "dataset": {"path": "/data/" + config["dataset_filename"],
                    "filename": config["dataset_filename"],
                    "sha256": config["dataset_sha256"],
                    "count": config["dataset_count"]},
        "sample_indices": list(CALIBRATION_INDICES),
        "adapter_mapping": {"coordinate_transformation": "none"},
        "environment": {"gpu": "NVIDIA GeForce RTX 4090", "device": "cuda:0",
                        "torch": "2.5.0+cu124", "torch_cuda_build": "12.4"},
        "tensorboard_compatibility": {"official_source_modified": False},
        "decoder_compatibility": {
            "bs1_shape_shim": True, "original_batch_size": 1,
            "D2A": 1, "val_m": 1, "internal_decoder_batch_size": 1,
            "official_source_modified": False,
            "action_reward_logits_rng_budget_changed": False,
        },
        "record_compatibility": {
            "record_step_shim": True,
            "pinned_step_sha256": PINNED_TSP_STEP_SHA256,
            "timed_rollout_shim_active": False,
            "evidence_replay_shim_active": True,
            "official_source_modified": False,
            "action_reward_logits_rng_budget_changed": False,
        },
        "timing_semantics": TIMING_SEMANTICS,
        "warmup_policy": dict(WARMUP_POLICY), "rng_policy": RNG_POLICY,
        "source_provenance": [],
    }


def fake_record(index, *, runtime=0.01):
    objective, reference = 10.0, 9.0
    return {
        "index": index, "instance_id": f"tsp-{index}",
        "runtime_seconds": runtime,
        "successor": successor_from_order(list(range(100))).tolist(),
        "canonical_solution": list(range(100)) + [0],
        "official_objective": objective,
        "replay_official_objective": objective,
        "official_recomputed_objective": objective,
        "independent_objective": objective,
        "reference_objective": reference,
        "gap_percent": (objective - reference) / reference * 100.0,
        "feasible": True, "kit_feasible": True,
        "kit_objective": objective, "kit_reference_objective": reference,
        "objective_agreement": {
            "timed_replay_exact": True,
            "official_independent": True,
            "official_recomputed_independent": True,
            "kit_independent": True,
            "rtol": OBJECTIVE_RTOL, "atol": OBJECTIVE_ATOL,
        },
        "constraint_details": {"each_node_once": True, "single_cycle": True},
        "timed_replay": {
            "timed_rollout_record": False,
            "evidence_replay_record": True,
            "rng_state_restored": True,
            "timed_replay_official_objective_exact": True,
            "timed_replay_rng_after_exact": True,
            "cuda_synchronized_before_timing": True,
            "cuda_synchronized_after_timing": True,
            "evidence_replay_excluded_from_runtime": True,
        },
        "evidence_status": "KIT_VALIDATED",
    }


def fake_records(runtime=0.01):
    return [fake_record(index, runtime=runtime + index / 100000) for index in CALIBRATION_INDICES]


class NeuOptTSPProtocolTests(unittest.TestCase):
    def test_only_tsp100_is_allowed(self):
        self.assertEqual(calibration_protocol(100, T_max=1)["problem_size"], 100)
        for size in (50, 500, 1000):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "only TSP100"):
                calibration_protocol(size, T_max=1)

    def test_positive_T_and_first_round_candidates(self):
        for value in FIRST_ROUND_T_VALUES + (30, 50):
            self.assertEqual(calibration_protocol(100, T_max=value)["T_max"], value)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                calibration_protocol(100, T_max=value)

    def test_fixed_bs1_d2a_stall_and_k(self):
        protocol = calibration_protocol(100, T_max=5)
        self.assertEqual((protocol["original_batch_size"], protocol["D2A"],
                          protocol["stall_limit"], protocol["k"]), (1, 1, 10, 4))
        for kwargs in ({"batch_size": 16}, {"d2a": 5},
                       {"stall_limit": 9}, {"k": 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                calibration_protocol(100, T_max=5, **kwargs)

    def test_official_options_are_eval_only_bs1(self):
        class Device:
            type = "cuda"
        args = official_option_args(
            config=calibration_protocol(100, T_max=20),
            checkpoint="tsp100.pt", device=Device())
        value = lambda flag: args[args.index(flag) + 1]
        self.assertEqual(value("--val_m"), "1")
        self.assertEqual(value("--T_max"), "20")
        self.assertEqual(value("--stall_limit"), "10")
        self.assertEqual(value("--k"), "4")
        self.assertEqual(value("--val_batch_size"), "1")
        self.assertIn("--eval_only", args)
        self.assertNotIn("--record", args)


class NeuOptTSPAdapterDecoderTests(unittest.TestCase):
    def test_adapter_shape_and_exact_coordinates(self):
        points = np.arange(400, dtype=np.float32).reshape(2, 100, 2) / 401
        native, mapping = adapt_batch(points, device="cpu")
        self.assertEqual(tuple(native["coordinates"].shape), (2, 100, 2))
        np.testing.assert_array_equal(native["coordinates"].numpy(), points)
        self.assertFalse(mapping["dtype_cast"])
        self.assertIn("no normalization", mapping["coordinate_transformation"])

    def test_successor_decodes_one_hamiltonian_cycle(self):
        order = [0] + list(range(99, 0, -1))
        tour, metadata = decode_successor(successor_from_order(order))
        self.assertEqual(tour, order + [0])
        self.assertFalse(metadata["repair"])

    def test_duplicate_and_missing_node_reject(self):
        successor = successor_from_order(list(range(100)))
        successor[0] = successor[1]
        with self.assertRaisesRegex(ValueError, "duplicate/missing"):
            decode_successor(successor)

    def test_subtour_rejects(self):
        successor = np.arange(100, dtype=np.int64)
        successor[0], successor[1] = 1, 0
        successor[2:] = np.roll(np.arange(2, 100), -1)
        with self.assertRaisesRegex(ValueError, "subtour"):
            decode_successor(successor)

    def test_objective_recomputation(self):
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        points = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        tour = list(range(100)) + [0]
        result = validate(points, tour)
        self.assertTrue(result["feasible"])
        self.assertAlmostEqual(result["independent_objective"],
                               cycle_length(points, list(range(100))), places=12)

    def test_tsp_bs1_uses_guarded_decoder_shim(self):
        class Decoder:
            def forward(self):
                return None
        source = """\
def forward(self):
    stopped = stopped | (action == next_of_last_action).squeeze()
    stopped = (action == next_of_last_action).squeeze()
"""
        with patch("methods.neuopt.cvrp.compat.inspect.getsource", return_value=source):
            provenance = configure_production_decoder(
                Decoder, original_batch_size=1, val_m=1)
        self.assertTrue(provenance["bs1_shape_shim"])
        self.assertEqual(provenance["internal_decoder_batch_size"], 1)
        self.assertFalse(provenance["official_source_modified"])
        self.assertFalse(provenance["action_reward_logits_rng_budget_changed"])

    def test_record_shim_is_replay_only_and_restores_step(self):
        class Problem:
            def step(self, batch, rec, action, obj, feasible_history, t, weights=0):
                return ("state", "reward", "objective", None, None, None, None)
        problem = Problem()
        original = problem.step
        with patch("methods.neuopt.tsp.compat.inspect.getsource", return_value="pinned"):
            digest = __import__("hashlib").sha256(b"pinned").hexdigest()
            with patch("methods.neuopt.tsp.compat.PINNED_TSP_STEP_SHA256", digest):
                with replay_record_compatibility(problem) as provenance:
                    history = object()
                    self.assertIs(problem.step(None, None, None, None, history, 0)[3], history)
                    self.assertFalse(provenance["timed_rollout_shim_active"])
        self.assertEqual(problem.step, original)


class NeuOptTSPTimedReplayTests(unittest.TestCase):
    @staticmethod
    def native(torch):
        return {"coordinates": torch.zeros(1, 100, 2)}

    @staticmethod
    def run_pair(agent):
        import torch
        with (patch.object(torch.cuda, "synchronize"),
              patch.object(torch.cuda, "get_rng_state_all", return_value=[]),
              patch.object(torch.cuda, "set_rng_state_all"),
              patch("methods.neuopt.tsp.paper_eval.time.perf_counter",
                    side_effect=[10.0, 10.25]),
              patch("methods.neuopt.tsp.paper_eval.replay_record_compatibility") as compatibility):
            compatibility.return_value.__enter__.return_value = {
                "record_step_shim": True,
                "timed_rollout_shim_active": False,
            }
            return timed_rollout_with_evidence_replay(
                agent, object(), NeuOptTSPTimedReplayTests.native(torch),
                config=calibration_protocol(100, T_max=1),
                device="cuda:0", torch=torch)

    def test_timed_false_replay_true(self):
        import torch
        class Agent:
            def __init__(self): self.records = []
            def rollout(self, problem, **kwargs):
                self.records.append(kwargs["record"])
                torch.rand(1)
                return (torch.tensor([7.0]), None, None,
                        ([], [], []) if kwargs["record"] else None)
        agent = Agent()
        timed, replay, elapsed, provenance, record_compatibility = self.run_pair(agent)
        self.assertEqual(agent.records, [False, True])
        self.assertTrue(torch.equal(timed, replay[0]))
        self.assertEqual(elapsed, 0.25)
        self.assertTrue(provenance["timed_replay_rng_after_exact"])
        self.assertTrue(record_compatibility["record_step_shim"])

    def test_objective_mismatch_rejects(self):
        import torch
        class Agent:
            def rollout(self, problem, **kwargs):
                torch.rand(1)
                return (torch.tensor([2.0 if kwargs["record"] else 1.0]), None, None, None)
        with self.assertRaisesRegex(RuntimeError, "objective differs"):
            self.run_pair(Agent())

    def test_rng_mismatch_rejects(self):
        import torch
        class Agent:
            def rollout(self, problem, **kwargs):
                torch.rand(2 if kwargs["record"] else 1)
                return (torch.tensor([1.0]), None, None, None)
        with self.assertRaisesRegex(RuntimeError, "different RNG"):
            self.run_pair(Agent())


class NeuOptTSPArtifactTests(unittest.TestCase):
    def test_checkpoint_dataset_upstream_and_dirty_fail_closed(self):
        cases = []
        bad = fake_identity(); bad["checkpoint"]["sha256"] = "0" * 64
        cases.append((bad, "checkpoint"))
        bad = fake_identity(); bad["dataset"]["sha256"] = "0" * 64
        cases.append((bad, "dataset"))
        bad = fake_identity(); bad["upstream"]["commit"] = "0" * 40
        cases.append((bad, "provenance"))
        bad = fake_identity(); bad["project"]["dirty"] = True
        cases.append((bad, "dirty"))
        bad = fake_identity(); bad["upstream"]["dirty"] = True
        cases.append((bad, "dirty"))
        for identity, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_identity(identity)

    def test_timed_replay_objective_record_mismatch_rejects(self):
        record = fake_record(0)
        record["replay_official_objective"] = 11.0
        with self.assertRaisesRegex(ValueError, "replay objective differs"):
            validate_record(record)

    def test_output_overwrite_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                write_candidate(existing, fake_identity(), fake_records())

    def test_report_requires_all_candidates_and_never_freezes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = []
            for T_max in FIRST_ROUND_T_VALUES:
                path = root / f"t{T_max}"
                write_candidate(path, fake_identity(T_max),
                                fake_records(runtime=0.04 if T_max == 1 else 0.01 * T_max))
                candidates.append(path)
            extra = root / "t30"
            write_candidate(extra, fake_identity(30), fake_records(runtime=0.30))
            candidates.append(extra)
            report = build_calibration_report(candidates[::-1])
            self.assertEqual([row["T_max"] for row in report["candidates"]],
                             list(FIRST_ROUND_T_VALUES) + [30])
            self.assertEqual(report["targets"], CALIBRATION_TARGETS)
            self.assertEqual(report["minimum_valid_T"], 1)
            self.assertTrue(report["t1_runtime_exceeds_fewer_target"])
            self.assertIsNone(report["final_T_fewer"])
            self.assertIsNone(report["final_T_more"])
            self.assertFalse(report["automatic_freeze"])
            self.assertFalse(report["paper_ready"])
            output = root / "report.json"
            write_calibration_report(output, report)
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                write_calibration_report(output, report)

    def test_report_rejects_missing_first_round_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t1"
            write_candidate(path, fake_identity(1), fake_records())
            with self.assertRaisesRegex(ValueError, "missing first-round"):
                build_calibration_report([path])


if __name__ == "__main__":
    unittest.main()
