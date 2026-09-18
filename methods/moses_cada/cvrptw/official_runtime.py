"""Pinned MoSES(CaDA) construction, audited load, and official evaluation call."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from common.cvrptw_runtime import select_rl4co_output, timed_call, to_tensordict


def _nested_values(value, key):
    found = []
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                found.append(child)
            found.extend(_nested_values(child, key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_nested_values(child, key))
    return found


def checkpoint_metadata(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("MoSES checkpoint must contain a state_dict")
    if payload.get("epoch") != 299 or payload.get("global_step") != 117300:
        raise ValueError("MoSES checkpoint epoch/global_step mismatch")
    if not all(name.startswith("policy.") for name in payload["state_dict"]):
        raise ValueError("MoSES checkpoint contains a non-policy state key")
    decode = _nested_values(payload, "test_decode_type")
    temperature = _nested_values(payload, "lora_temperature")
    if not decode or not all(str(value).lower() == "multistart_greedy" for value in decode):
        raise ValueError("MoSES checkpoint does not freeze multistart_greedy test decode")
    if not temperature or not all(float(value) == 1.0 for value in temperature):
        raise ValueError("MoSES checkpoint does not freeze LoRA temperature=1")
    return {"epoch": payload["epoch"], "global_step": payload["global_step"],
            "lightning_version": payload.get("pytorch-lightning_version"),
            "test_decode_type": [str(value) for value in decode],
            "lora_temperature": [float(value) for value in temperature]}


class Runtime:
    def __init__(self, upstream, checkpoint, problem_size, device, torch):
        self.upstream, self.problem_size, self.device, self.torch = (
            Path(upstream), problem_size, device, torch)
        sys.path.insert(0, str(self.upstream))
        from envs import MTVRPEnv
        from models import CadaMultiLoRAPolicy
        module_spec = importlib.util.spec_from_file_location("pinned_moses_test", self.upstream / "test.py")
        official_test = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(official_test)
        self.official_test = official_test.test
        self.policy = CadaMultiLoRAPolicy(
            normalization="rms", encoder_use_prenorm=False,
            encoder_use_post_layers_norm=False,
            parallel_gated_kwargs={"mlp_activation": "silu"},
            attn_sparse_ratio=0.5, sparse_applied_to_score=True,
            lora_rank=[32] * 5, lora_alpha=1.0, lora_use_gate=True,
            lora_act_func="sigmoid", lora_n_experts=4, lora_top_k=4,
            lora_temperature=1.0, lora_use_trainable_layer=True,
            lora_use_dynamic_topK=False, lora_use_basis_variants=False,
            lora_use_basis_variants_as_input=False, lora_use_linear=False)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        parsed = checkpoint_metadata(payload)
        state = payload.get("state_dict")
        policy_state = {name.removeprefix("policy."): value for name, value in state.items()}
        incompatible = self.policy.load_state_dict(policy_state, strict=False)
        self.checkpoint_state = {
            "official_load_strict": False, **parsed,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        if self.checkpoint_state["missing_keys"] or self.checkpoint_state["unexpected_keys"]:
            raise RuntimeError("MoSES checkpoint has missing/unexpected policy keys")
        self.policy = self.policy.to(device).eval()
        self.env = MTVRPEnv()
        self.smoke_env = MTVRPEnv(
            generator_params={"num_loc": problem_size, "variant_preset": "vrptw"},
            check_solution=True)

    def official_format_smoke(self):
        def call():
            reset = self.smoke_env.reset(batch_size=[1]).to(self.device)
            out = self.official_test(
                self.policy, reset, self.smoke_env, num_augment=8,
                augment_fn="dihedral8", num_starts=self.problem_size, device=self.device)
            return select_rl4co_output(out, problem_size=self.problem_size,
                                       torch=self.torch)
        selection, _ = timed_call(call, torch=self.torch, device=self.device, timed=False)
        return selection

    def solve(self, native, *, timed=True):
        td = to_tensordict(native, torch=self.torch, device=self.device)

        def call():
            reset = self.env.reset(td)
            out = self.official_test(self.policy, reset, self.env, num_augment=8,
                                     augment_fn="dihedral8", num_starts=self.problem_size,
                                     device=self.device)
            return select_rl4co_output(out, problem_size=self.problem_size,
                                       torch=self.torch)
        selection, elapsed = timed_call(call, torch=self.torch, device=self.device,
                                        timed=timed)
        return selection, elapsed
