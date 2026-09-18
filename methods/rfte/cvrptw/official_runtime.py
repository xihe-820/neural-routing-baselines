"""Pinned RF-TE construction, audited load, and official evaluation call."""
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
        raise ValueError("RF-TE checkpoint must contain a state_dict")
    hyper_parameters = payload.get("hyper_parameters")
    return {"payload_keys": sorted(payload), "hyper_parameters": hyper_parameters,
            "test_decode_type_candidates": _nested_values(hyper_parameters, "test_decode_type"),
            "temperature_candidates": _nested_values(hyper_parameters, "temperature")}


class Runtime:
    def __init__(self, upstream, checkpoint, problem_size, device, torch):
        self.upstream, self.problem_size, self.device, self.torch = (
            Path(upstream), problem_size, device, torch)
        sys.path.insert(0, str(self.upstream))
        from routefinder.envs import MTVRPEnv
        from routefinder.models import RouteFinderBase
        module_spec = importlib.util.spec_from_file_location("pinned_rfte_test", self.upstream / "test.py")
        official_test = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(official_test)
        self.official_test = official_test.test
        self.model = RouteFinderBase.load_from_checkpoint(
            checkpoint, map_location="cpu", strict=False)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        parsed = checkpoint_metadata(payload)
        incompatible = self.model.load_state_dict(payload["state_dict"], strict=False)
        self.checkpoint_state = {
            "official_load_strict": False,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "payload_keys": parsed["payload_keys"],
        }
        if self.checkpoint_state["missing_keys"] or self.checkpoint_state["unexpected_keys"]:
            raise RuntimeError("RF-TE checkpoint has missing/unexpected model keys")
        if type(self.model).__name__ != "RouteFinderBase":
            raise RuntimeError("checkpoint did not construct the RF-TE RouteFinderBase")
        self.policy = self.model.policy.to(device).eval()
        decode_candidates = list(parsed["test_decode_type_candidates"])
        if getattr(self.policy, "test_decode_type", None) is not None:
            decode_candidates.append(self.policy.test_decode_type)
        temperature_candidates = list(parsed["temperature_candidates"])
        if getattr(self.policy, "temperature", None) is not None:
            temperature_candidates.append(self.policy.temperature)
        decoder = getattr(self.policy, "decoder", None)
        if getattr(decoder, "temperature", None) is not None:
            temperature_candidates.append(decoder.temperature)
        if (not decode_candidates or
                not all(str(value).lower() in ("greedy", "multistart_greedy")
                        for value in decode_candidates)):
            raise RuntimeError(
                f"RF-TE preflight could not confirm greedy test decode: {decode_candidates}")
        if (not temperature_candidates or
                not all(float(value) == 1.0 for value in temperature_candidates)):
            raise RuntimeError(
                f"RF-TE preflight could not confirm temperature=1: {temperature_candidates}")
        self.checkpoint_state.update(
            effective_test_decode_type=[str(value) for value in decode_candidates],
            effective_temperature=[float(value) for value in temperature_candidates])
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
