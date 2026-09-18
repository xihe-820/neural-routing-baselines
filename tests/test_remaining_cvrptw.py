import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from common.cvrptw_artifacts import (append, finalize, initialize,
                                     require_our2_gate, require_our5_gate,
                                     require_our5_prefix_matches_our2)
from common.cvrptw_formal import (DATASETS, TIMING_SEMANTICS,
                                  canonicalize_official_actions,
                                  instance_drop_percent, native_numpy_instance)
from common.cvrptw_runtime import select_cada_output, select_rl4co_output
from methods.cada.cvrptw.adapter import adapt_instance as cada_adapt
from methods.cada.cvrptw.decode import ActionCapture
from methods.cada.cvrptw.official_runtime import checkpoint_metadata as cada_metadata
from methods.cada.cvrptw.config import protocol as cada_protocol
from methods.moses_cada.cvrptw.adapter import adapt_instance as moses_adapt
from methods.moses_cada.cvrptw.config import CHECKPOINTS as MOSES_CHECKPOINTS
from methods.moses_cada.cvrptw.config import protocol as moses_protocol
from methods.moses_cada.cvrptw.official_runtime import checkpoint_metadata as moses_metadata
from methods.rfte.cvrptw.adapter import adapt_instance as rfte_adapt
from methods.rfte.cvrptw.config import protocol as rfte_protocol
from methods.rfte.cvrptw.official_runtime import checkpoint_metadata as rfte_metadata
import methods.rfte.cvrptw.official_runtime as rfte_runtime


def instance(size):
    capacity = DATASETS[size]["capacity"]
    depot = np.asarray([0.1, 0.2], dtype=np.float64)
    points = np.linspace(0.01, 0.99, size * 2, dtype=np.float64).reshape(size, 2)
    demand = np.resize(np.arange(1, 10, dtype=np.float64), size)
    tw = np.zeros((size + 1, 2), dtype=np.float64)
    tw[:, 1] = 4.6
    service = np.zeros(size + 1, dtype=np.float64)
    service[1:] = 0.15
    return depot, points, demand, capacity, tw, service


class AdapterDecoderProtocolTests(unittest.TestCase):
    def test_all_adapters_share_original_units_and_one_demand_normalization(self):
        for size in (50, 100):
            source = instance(size)
            for adapter in (rfte_adapt, cada_adapt, moses_adapt):
                with self.subTest(size=size, adapter=adapter.__module__):
                    native, mapping = adapter(*source, problem_size=size)
                    np.testing.assert_allclose(native["locs"][0, 0], source[0].astype(np.float32))
                    np.testing.assert_allclose(native["locs"][0, 1:], source[1].astype(np.float32))
                    observed_demand = native["demand_linehaul"][0]
                    if adapter is cada_adapt:
                        self.assertEqual(observed_demand[0], 0.0)
                        observed_demand = observed_demand[1:]
                    np.testing.assert_allclose(observed_demand, source[2] / source[3])
                    np.testing.assert_allclose(native["time_windows"][0], source[4])
                    np.testing.assert_allclose(native["service_time"][0], source[5])
                    self.assertIn("exactly once", mapping["demand"])

    def test_adapter_rejects_wrong_size_capacity_and_shapes(self):
        values = list(instance(50))
        values[3] = 50
        with self.assertRaisesRegex(ValueError, "capacity"):
            native_numpy_instance(*values, problem_size=50)
        values = list(instance(50))
        values[1] = values[1][:-1]
        with self.assertRaisesRegex(ValueError, "shape"):
            native_numpy_instance(*values, problem_size=50)

    def test_decoder_only_removes_terminal_depot_padding(self):
        self.assertEqual(canonicalize_official_actions([1, 0, 2, 0, 0], 2),
                         [0, 1, 0, 2, 0])
        self.assertEqual(canonicalize_official_actions([1, 1, 0], 2), [0, 1, 1, 0])
        with self.assertRaisesRegex(ValueError, "range"):
            canonicalize_official_actions([1, 51], 50)

    def test_protocols_are_exact_single_row_scope(self):
        for size in (50, 100):
            self.assertEqual(rfte_protocol(size)["num_starts"], size)
            self.assertEqual(cada_protocol(size)["task_prompt"], [1, 0, 1, 0, 0])
            moses = moses_protocol(size)
            self.assertEqual(moses["backbone"], "CaDA")
            self.assertEqual(moses["lora_activation"], "sigmoid")
            self.assertEqual(moses["num_augmentations"], 8)
        for protocol in (rfte_protocol, cada_protocol, moses_protocol):
            with self.assertRaises(ValueError):
                protocol(200)

    def test_action_capture_restores_even_on_error(self):
        class Tensor:
            def detach(self): return self
        class Env:
            def get_reward(self, td, actions): return "reward"
        env = Env()
        original = env.get_reward
        tensor = Tensor()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with ActionCapture(env) as capture:
                self.assertEqual(env.get_reward(None, tensor), "reward")
                self.assertIs(capture.actions[0], tensor)
                raise RuntimeError("boom")
        self.assertEqual(env.get_reward(None, None), original(None, None))
        self.assertNotIn("get_reward", vars(env))


class CheckpointAndTimingTests(unittest.TestCase):
    def test_checkpoint_metadata_parsers_fail_closed(self):
        self.assertEqual(rfte_metadata({"state_dict": {"x": 1}})["payload_keys"], ["state_dict"])
        self.assertEqual(cada_metadata({"epoch": 300, "model_state_dict": {"x": 1}})["epoch"], 300)
        moses = {"epoch": 299, "global_step": 117300,
                 "pytorch-lightning_version": "2.5.0.post0",
                 "hyper_parameters": {"test_decode_type": "multistart_greedy",
                                      "lora_temperature": 1.0},
                 "state_dict": {"policy.x": 1}}
        self.assertEqual(moses_metadata(moses)["global_step"], 117300)
        for bad in ({}, {"epoch": 299, "model_state_dict": {}},
                    {**moses, "state_dict": {"wrong.x": 1}}):
            with self.assertRaises(ValueError):
                if "model_state_dict" in bad:
                    cada_metadata(bad)
                else:
                    moses_metadata(bad)

    def test_moses_checkpoint_hashes_are_frozen(self):
        self.assertEqual(MOSES_CHECKPOINTS[50]["sha256"],
                         "1aa499f3fce5d3412c2544c9632bbb9709309a7299b4298b535fa7e9014ef803")
        self.assertEqual(MOSES_CHECKPOINTS[100]["sha256"],
                         "2eac9b038ae4655581aa73e4dbe8ad529aefd1963368c9a92d254b6269f8aabf")

    def test_timing_boundary_places_reset_inside_timed_callable(self):
        source = inspect.getsource(rfte_runtime.Runtime.solve)
        self.assertLess(source.index("def call"), source.index("self.env.reset"))
        self.assertLess(source.index("select_rl4co_output"), source.index("timed_call(call"))
        self.assertLess(source.index("self.env.reset"), source.index("timed_call(call"))
        self.assertIn("official environment load/reset", TIMING_SEMANTICS)

    def test_exact_candidate_indices_for_rl4co_and_cada_layouts(self):
        import torch
        rewards = torch.arange(16, dtype=torch.float32).neg()
        rewards[7] = 1.0  # aug=3, start=1 for N=2
        actions = torch.arange(1 * 8 * 2 * 3).reshape(1, 8, 2, 3)
        selected_action = actions[0, 3, 1]
        rl_out = {
            "reward": rewards,
            "actions": actions,
            "max_aug_reward": torch.tensor([1.0]),
            "best_aug_actions": selected_action[None, :],
        }
        selected = select_rl4co_output(rl_out, problem_size=2, torch=torch)
        self.assertEqual(selected["selected_candidate"]["flat_index"], 7)
        self.assertEqual(selected["selected_candidate"]["official_flat_index"], 7)
        cada_rewards = torch.zeros(2, 8, 1)
        cada_rewards[1, 3, 0] = 2.0
        cada_actions = torch.arange(2 * 8 * 1 * 3).reshape(2, 8, 1, 3)
        selected = select_cada_output(cada_rewards.reshape(-1), cada_actions.reshape(16, 3),
                                      problem_size=2, torch=torch)
        self.assertEqual(selected["selected_candidate"]["flat_index"], 7)
        self.assertEqual(selected["selected_candidate"]["official_flat_index"], 11)


class ArtifactTests(unittest.TestCase):
    def record(self, index):
        raw = list(range(1, 51))
        return {
            "dataset_instance_index": index, "instance_id": f"x{index}",
            "raw_official_action": raw, "canonical_solution": [0, *raw, 0],
            "selected_candidate": {"augmentation_index": 0, "start_index": 0, "flat_index": 0},
            "official_reward": -2.0, "reported_objective": 2.0,
            "independent_objective": 2.0, "kit_objective": 2.0,
            "reference_objective": 1.0, "instance_drop_percent": 100.0,
            "runtime_seconds": 0.1, "independent_feasible": True,
            "kit_feasible": True, "reported_objective_agrees": True,
            "kit_objective_agrees": True, "route_count": 1,
            "route_loads": [1.0], "route_timelines": [], "status": "KIT_VALIDATED",
        }

    def identity(self, scope="our_5", count=5):
        return {"method": "RF-TE", "variant": "RouteFinder Transformer",
                "problem": "CVRPTW", "problem_size": 50, "scope": scope,
                "dataset": {"sha256": DATASETS[50]["sha256"], "count": 1000},
                "checkpoint": {"sha256": "a" * 64},
                "chunk": {"expected_indices": list(range(count))},
                "project": {"commit": "p", "dirty": False},
                "upstream": {"commit": "u", "dirty": False},
                "protocol": {"fixed": True}, "environment": {"gpu": "RTX 4090"},
                "source_provenance": [], "timing_semantics": TIMING_SEMANTICS}

    def test_drop_is_per_instance(self):
        self.assertEqual(instance_drop_percent(3.0, 2.0), 50.0)
        drops = [instance_drop_percent(2.0, 1.0), instance_drop_percent(3.0, 3.0)]
        self.assertEqual(sum(drops) / 2, 50.0)

    def test_resume_finalize_and_our5_provenance_gate(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            initialize(path, self.identity())
            for index in range(5):
                append(path, self.record(index))
            summary = finalize(path)
            self.assertEqual(summary["state"], "KIT_VALIDATED")
            self.assertEqual(summary["mean_independent_objective"], 2.0)
            self.assertEqual(summary["mean_instance_drop_percent"], 100.0)
            self.assertEqual(summary["mean_runtime_seconds"], 0.1)
            require_our5_gate(path, method="RF-TE", problem_size=50,
                              dataset_sha256=DATASETS[50]["sha256"],
                              checkpoint_sha256="a" * 64)
            with (path / "validated_records.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaises(ValueError):
                require_our5_gate(path, method="RF-TE", problem_size=50,
                                  dataset_sha256=DATASETS[50]["sha256"],
                                  checkpoint_sha256="a" * 64)

    def test_our5_first_two_must_match_our2_semantics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            our2, our5 = root / "our2", root / "our5"
            initialize(our2, self.identity("our_2", 2))
            initialize(our5, self.identity("our_5", 5))
            for index in range(2):
                append(our2, self.record(index))
            for index in range(5):
                append(our5, self.record(index))
            finalize(our2)
            metadata2, records2 = require_our2_gate(
                our2, method="RF-TE", problem_size=50,
                dataset_sha256=DATASETS[50]["sha256"], checkpoint_sha256="a" * 64)
            self.assertTrue(require_our5_prefix_matches_our2(our5, metadata2, records2))
            rows = (our5 / "validated_records.jsonl").read_text().splitlines()
            changed = json.loads(rows[0])
            changed["raw_official_action"] = list(reversed(changed["raw_official_action"]))
            rows[0] = json.dumps(changed)
            (our5 / "validated_records.jsonl").write_text("\n".join(rows) + "\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                require_our5_prefix_matches_our2(our5, metadata2, records2)


if __name__ == "__main__":
    unittest.main()
