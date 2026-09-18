import inspect
import importlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest

import numpy as np

from common.cvrptw_artifacts import (append, append_batch, finalize, initialize,
                                     require_small_gate, require_validation_gate,
                                     require_our2_gate, require_our5_gate,
                                     require_our5_prefix_matches_our2,
                                     scope_instance_count)
from common.cvrptw_formal import (DATASETS, TIMING_SEMANTICS,
                                  canonicalize_official_actions,
                                  instance_drop_percent, native_numpy_instance)
from common.cvrptw_runtime import (select_cada_output,
                                   select_rl4co_batch_output,
                                   select_rl4co_output,
                                   rl4co_unbatchify_tensor, stack_native,
                                   to_tensordict)
from methods.cada.cvrptw.adapter import adapt_instance as cada_adapt
from methods.cada.cvrptw.decode import ActionCapture
from methods.cada.cvrptw.official_runtime import checkpoint_metadata as cada_metadata
from methods.cada.cvrptw.config import protocol as cada_protocol
from methods.moses_cada.cvrptw.adapter import adapt_instance as moses_adapt
from methods.moses_cada.cvrptw.config import CHECKPOINTS as MOSES_CHECKPOINTS
from methods.moses_cada.cvrptw.config import protocol as moses_protocol
from methods.moses_cada.cvrptw.official_runtime import (
    POLICY_KWARGS as MOSES_POLICY_KWARGS,
    _construct_policy as construct_moses_policy,
    _official_module_context as moses_official_module_context,
    checkpoint_metadata as moses_metadata,
    effective_runtime_protocol as moses_effective_protocol,
)
import methods.moses_cada.cvrptw.official_runtime as moses_runtime
from methods.rfte.cvrptw.adapter import adapt_instance as rfte_adapt
from methods.rfte.cvrptw.config import protocol as rfte_protocol
from methods.rfte.cvrptw.official_runtime import checkpoint_metadata as rfte_metadata
import methods.rfte.cvrptw.official_runtime as rfte_runtime


def official_rl4co_output(rewards, actions, torch):
    """Construct the exact tensors returned by pinned RF-TE/MoSES test.py."""
    max_reward, max_indices = rewards.max(dim=-1)
    batch_size, augmentations = rewards.shape[:2]
    batch_indices = torch.arange(batch_size)[:, None]
    augmentation_indices = torch.arange(augmentations)[None, :]
    best_multistart_actions = actions[
        batch_indices, augmentation_indices, max_indices]
    max_aug_reward, max_aug_indices = max_reward.max(dim=1)
    best_aug_actions = best_multistart_actions[
        torch.arange(batch_size), max_aug_indices]
    # Inverse of RL4CO unbatchify: flat storage is [start,augmentation,batch].
    flat_reward = rewards.permute(2, 1, 0).reshape(-1)
    return {
        "reward": flat_reward,
        "actions": actions,
        "max_reward": max_reward,
        "best_multistart_actions": best_multistart_actions,
        "max_aug_reward": max_aug_reward,
        "best_aug_actions": best_aug_actions,
    }


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
            self.assertEqual(moses["augmentation"], "dihedral8")
            self.assertEqual(moses["policy_test_decode_type"], "greedy")
            self.assertEqual(moses["decode_type"], "greedy")
            self.assertIs(moses["multistart"], True)
            self.assertEqual(moses["num_starts"], size)
            self.assertEqual(moses["start_selector"], "all customers 1..N")
        for protocol in (rfte_protocol, cada_protocol, moses_protocol):
            with self.assertRaises(ValueError):
                protocol(200)

    def test_batch_protocols_accept_only_one_or_ten(self):
        for protocol in (rfte_protocol, moses_protocol):
            for batch_size in (1, 10):
                self.assertEqual(
                    protocol(50, batch_size)["original_instance_batch_size"],
                    batch_size)
            with self.assertRaises(ValueError):
                protocol(50, 2)

    def test_stack_native_and_tensordict_preserve_original_batch_axis(self):
        class FakeTensorDict(dict):
            def __init__(self, values, batch_size, device):
                super().__init__(values)
                self.batch_size, self.device = batch_size, device

        fake_module = ModuleType("tensordict")
        fake_module.TensorDict = FakeTensorDict
        import torch
        prior = sys.modules.get("tensordict")
        sys.modules["tensordict"] = fake_module
        try:
            for batch_size in (1, 10):
                rows = []
                for index in range(batch_size):
                    native, _ = rfte_adapt(*instance(50), problem_size=50)
                    native["locs"] = native["locs"] + index
                    rows.append(native)
                batch = stack_native(rows)
                self.assertEqual(batch["locs"].shape[0], batch_size)
                td = to_tensordict(
                    batch, torch=torch, device="cpu",
                    expected_batch_size=batch_size)
                self.assertEqual(td.batch_size, [batch_size])
        finally:
            if prior is None:
                sys.modules.pop("tensordict", None)
            else:
                sys.modules["tensordict"] = prior

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
                 "hyper_parameters": {"lora_temperature": 1.0},
                 "state_dict": {"policy.x": 1}}
        parsed = moses_metadata(moses)
        self.assertEqual(parsed["global_step"], 117300)
        self.assertEqual(parsed["checkpoint_test_decode_type_candidates"], [])
        with_greedy = {**moses, "hyper_parameters": {
            **moses["hyper_parameters"], "test_decode_type": "greedy"}}
        candidates = moses_metadata(with_greedy)[
            "checkpoint_test_decode_type_candidates"]
        self.assertEqual(candidates, [{
            "path": "$.hyper_parameters.test_decode_type", "value": "greedy"}])
        embedded = type("EmbeddedPolicy", (), {})()
        embedded.test_decode_type = "greedy"
        with_object = {**moses, "hyper_parameters": {"policy": embedded}}
        self.assertEqual(
            moses_metadata(with_object)["checkpoint_test_decode_type_candidates"],
            [{"path": "$.hyper_parameters.policy.test_decode_type",
              "value": "greedy"}])
        for bad in ({}, {"epoch": 299, "model_state_dict": {}},
                    {**moses, "state_dict": {"wrong.x": 1}},
                    {**moses, "epoch": 298}, {**moses, "global_step": 117299}):
            with self.assertRaises(ValueError):
                if "model_state_dict" in bad:
                    cada_metadata(bad)
                else:
                    moses_metadata(bad)

    def test_moses_constructor_matches_official_cada_multilora_arguments(self):
        class CapturePolicy:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        CapturePolicy.__signature__ = inspect.Signature([
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name in MOSES_POLICY_KWARGS
        ])

        policy = construct_moses_policy(CapturePolicy)
        self.assertEqual(policy.kwargs, MOSES_POLICY_KWARGS)
        self.assertNotIn("lora_use_gate", policy.kwargs)
        del CapturePolicy.__signature__
        with self.assertRaisesRegex(RuntimeError, "unused kwargs"):
            construct_moses_policy(CapturePolicy)

    def test_moses_effective_runtime_protocol_checks_constructed_modules(self):
        class Block:
            normalization = "rms"
            use_prenorm = False
            parallel_gated_kwargs = {"mlp_activation": "silu"}
            attn_sparse_ratio = 0.5
            sparse_applied_to_score = True

        class Encoder:
            global_layers = [Block()]
            sparse_layers = [Block()]
            post_layers_norm = None

        ParallelGatedMLP = type("ParallelGatedMLP", (), {})
        mlp = ParallelGatedMLP()
        mlp.act_type = "silu"
        GatedMultiLoRALayer = type("GatedMultiLoRALayer", (), {})
        gated = GatedMultiLoRALayer()
        gated.act_func = "sigmoid"
        gated.n_experts = 4
        gated.top_k = 4
        gated.temperature = 1.0
        gated.use_trainable_layer = True
        gated.use_dynamic_topK = False
        gated.use_basis_variants = False
        gated.use_basis_variants_as_input = False
        gated.lora_layers = []
        for _ in range(5):
            layer = type("LoRALayer", (), {})()
            layer.rank, layer.alpha, layer.use_linear = 32, 1.0, False
            gated.lora_layers.append(layer)

        class Policy:
            test_decode_type = "greedy"
            temperature = 1.0
            encoder = type("Wrapper", (), {"encoder": Encoder()})()
            def modules(self):
                return [self, mlp, gated]

        effective = moses_effective_protocol(Policy(), problem_size=50)
        self.assertEqual(effective["policy_test_decode_type"], "greedy")
        self.assertIs(effective["multistart"], True)
        self.assertEqual(effective["lora_rank"], [32] * 5)
        wrong = Policy()
        wrong.test_decode_type = "sampling"
        with self.assertRaisesRegex(RuntimeError, "test_decode_type"):
            moses_effective_protocol(wrong, problem_size=50)

    def test_moses_official_module_context_restores_colliding_modules_and_path(self):
        with TemporaryDirectory() as tmp:
            upstream = Path(tmp)
            (upstream / "envs").mkdir()
            (upstream / "envs" / "__init__.py").write_text("ORIGIN = 'pinned-moses'\n")
            prior = sys.modules.get("envs")
            collision = ModuleType("envs")
            collision.ORIGIN = "other-upstream"
            sys.modules["envs"] = collision
            old_path = list(sys.path)
            try:
                with moses_official_module_context(upstream):
                    loaded = importlib.import_module("envs")
                    self.assertEqual(loaded.ORIGIN, "pinned-moses")
                    self.assertNotEqual(loaded, collision)
                self.assertIs(sys.modules["envs"], collision)
                self.assertEqual(sys.path, old_path)
            finally:
                if prior is None:
                    sys.modules.pop("envs", None)
                else:
                    sys.modules["envs"] = prior

    def test_moses_checkpoint_hashes_are_frozen(self):
        self.assertEqual(MOSES_CHECKPOINTS[50]["sha256"],
                         "1aa499f3fce5d3412c2544c9632bbb9709309a7299b4298b535fa7e9014ef803")
        self.assertEqual(MOSES_CHECKPOINTS[100]["sha256"],
                         "2eac9b038ae4655581aa73e4dbe8ad529aefd1963368c9a92d254b6269f8aabf")

    def test_timing_boundary_places_reset_inside_timed_callable(self):
        for runtime in (rfte_runtime, moses_runtime):
            source = inspect.getsource(runtime.Runtime.solve_batch)
            self.assertLess(source.index("def call"), source.index("self.env.reset"))
            self.assertLess(source.index("select_rl4co_batch_output"),
                            source.index("timed_call(call"))
            self.assertLess(source.index("self.env.reset"), source.index("timed_call(call"))
        self.assertIn("official environment load/reset", TIMING_SEMANTICS)

    def test_rl4co_batch_selection_is_independent_per_original_instance(self):
        import torch
        steps = 5
        for batch_size in (1, 10):
            for problem_size in (50, 100):
                rewards = torch.full((batch_size, 8, problem_size), -1000.0)
                actions = torch.arange(
                    batch_size * 8 * problem_size * steps).reshape(
                        batch_size, 8, problem_size, steps)
                expected = []
                for batch_index in range(batch_size):
                    aug = batch_index % 8
                    start = (batch_index * 7 + 3) % problem_size
                    rewards[batch_index, aug, start] = float(100 + batch_index)
                    expected.append((aug, start))
                out = official_rl4co_output(rewards, actions, torch)
                selected = select_rl4co_batch_output(
                    out, problem_size=problem_size, torch=torch)
                self.assertEqual(len(selected), batch_size)
                for batch_index, row in enumerate(selected):
                    aug, start = expected[batch_index]
                    self.assertEqual(
                        (row["selected_candidate"]["augmentation_index"],
                         row["selected_candidate"]["start_index"]), (aug, start))
                    self.assertEqual(
                        row["selected_candidate"]["flat_index"],
                        aug * problem_size + start)
                    self.assertEqual(
                        row["raw_action"],
                        actions[batch_index, aug, start].tolist())

    def test_rl4co_unbatchify_is_not_plain_reshape(self):
        import torch
        flat = torch.arange(6)
        actual = rl4co_unbatchify_tensor(flat, (2, 3))
        plain = flat.reshape(1, 2, 3)
        expected = torch.tensor([[[0, 2, 4], [1, 3, 5]]])
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, plain))

    def test_rl4co_selector_audits_every_official_intermediate(self):
        import torch
        rewards = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
        actions = torch.arange(1 * 8 * 2 * 4).reshape(1, 8, 2, 4)
        valid = official_rl4co_output(rewards, actions, torch)
        select_rl4co_batch_output(valid, problem_size=2, torch=torch)
        for field in ("max_reward", "best_multistart_actions",
                      "max_aug_reward", "best_aug_actions"):
            tampered = dict(valid)
            tampered[field] = valid[field].clone()
            tampered[field].reshape(-1)[0] += 1
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, field):
                    select_rl4co_batch_output(
                        tampered, problem_size=2, torch=torch)

    def test_exact_candidate_indices_for_rl4co_and_cada_layouts(self):
        import torch
        rewards = torch.arange(16, dtype=torch.float32).neg().reshape(1, 8, 2)
        rewards[0, 3, 1] = 1.0
        actions = torch.arange(1 * 8 * 2 * 3).reshape(1, 8, 2, 3)
        rl_out = official_rl4co_output(rewards, actions, torch)
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

    def identity(self, scope="our_5", count=5, batch_size=1):
        return {"method": "RF-TE", "variant": "RouteFinder Transformer",
                "problem": "CVRPTW", "problem_size": 50, "scope": scope,
                "dataset": {"sha256": DATASETS[50]["sha256"], "count": 1000},
                "checkpoint": {"sha256": "a" * 64},
                "chunk": {"expected_indices": list(range(count))},
                "project": {"commit": "p", "dirty": False},
                "upstream": {"commit": "u", "dirty": False},
                "protocol": {"fixed": True,
                             "original_instance_batch_size": batch_size},
                "environment": {"gpu": "RTX 4090"},
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

    def test_bs10_complete_batch_resume_summary_and_gate_identity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            small = root / "small"
            identity = self.identity("small_gate", 20, batch_size=10)
            initialize(small, identity)
            for batch_index in range(2):
                indices = list(range(batch_index * 10, (batch_index + 1) * 10))
                runtime = 0.2 + batch_index
                records = [self.record(index) for index in indices]
                for record in records:
                    record["runtime_seconds"] = runtime
                append_batch(small, records, {
                    "batch_index": batch_index, "dataset_indices": indices,
                    "batch_size": 10, "runtime_seconds": runtime,
                })
            _, completed = initialize(small, identity)
            self.assertEqual(completed, set(range(20)))
            summary = finalize(small)
            self.assertEqual(summary["num_instances"], 20)
            self.assertEqual(summary["num_batches"], 2)
            self.assertEqual(summary["batch_size"], 10)
            self.assertAlmostEqual(summary["mean_batch_runtime_seconds"], 0.7)
            self.assertAlmostEqual(summary["total_runtime_seconds"], 1.4)
            require_small_gate(
                small, method="RF-TE", problem_size=50,
                dataset_sha256=DATASETS[50]["sha256"],
                checkpoint_sha256="a" * 64, batch_size=10)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                require_small_gate(
                    small, method="RF-TE", problem_size=50,
                    dataset_sha256=DATASETS[50]["sha256"],
                    checkpoint_sha256="a" * 64, batch_size=1)

    def test_bs10_partial_batch_resume_fails_closed(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            identity = self.identity("small_gate", 20, batch_size=10)
            initialize(path, identity)
            (path / "validated_records.jsonl").write_text(
                json.dumps(self.record(0)) + "\n")
            with self.assertRaisesRegex(ValueError, "partial/corrupt"):
                initialize(path, identity)

    def test_batch_scope_counts_are_real_native_batches(self):
        self.assertEqual(scope_instance_count("preflight", 10, 1000), 10)
        self.assertEqual(scope_instance_count("small_gate", 10, 1000), 20)
        self.assertEqual(scope_instance_count("validation_gate", 10, 1000), 50)
        self.assertEqual(scope_instance_count("production", 10, 1000), 1000)
        self.assertEqual(1000 // 10, 100)
        with self.assertRaises(ValueError):
            scope_instance_count("our_5", 10, 1000)


if __name__ == "__main__":
    unittest.main()
