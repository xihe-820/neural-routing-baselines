"""Pinned size-specific CaDA construction and non-invasive action capture."""
from __future__ import annotations

from pathlib import Path
import random
from types import SimpleNamespace
import sys

from common.cvrptw_runtime import select_cada_output, timed_call, to_tensordict
from methods.cada.cvrptw.decode import ActionCapture


def checkpoint_metadata(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("CaDA checkpoint must contain model_state_dict")
    if payload.get("epoch") != 300:
        raise ValueError("CaDA checkpoint epoch must be 300")
    return {"epoch": 300, "payload_keys": sorted(payload)}


def _model_params(upstream_size_dir):
    import yaml
    config = yaml.safe_load((Path(upstream_size_dir) / "config.yaml").read_text())
    params = dict(config["model_params"])
    params["sqrt_embedding_dim"] = params["embedding_dim"] ** 0.5
    return params


class Runtime:
    def __init__(self, upstream, checkpoint, problem_size, device, torch):
        self.upstream = Path(upstream) / str(problem_size)
        self.problem_size, self.device, self.torch = problem_size, device, torch
        sys.path.insert(0, str(self.upstream))
        # Exact official run.py seed surface: Python and Torch CPU/CUDA only.
        random.seed(7)
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        from envs.env import MTVRPEnv
        from envs.transformer import StateAugmentation
        from model import VRPModel
        args = SimpleNamespace(model_params=_model_params(self.upstream),
                               log=lambda *unused: None)
        self.model = VRPModel(args).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        parsed = checkpoint_metadata(payload)
        incompatible = self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.checkpoint_state = {
            "official_load_strict": True, **parsed,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        self.model.eval()
        self.env = MTVRPEnv(generator_params={"num_loc": problem_size,
                                              "variant_preset": "all"},
                            check_solution=False, seed=7, device=str(device))
        self.augmentation = StateAugmentation()

    def official_format_smoke(self):
        prior = self.env.generator.variant_preset
        try:
            self.env.generator.reset_variant_preset("vrptw")
            td = self.env.generator(1)

            def call():
                reset = self.env.reset(td=td)
                augmented = self.augmentation(reset)
                prior_check = self.env.check_solution
                self.env.check_solution = True
                try:
                    with self.torch.inference_mode(), ActionCapture(self.env) as capture:
                        out = self.model(augmented, self.env)
                finally:
                    self.env.check_solution = prior_check
                if len(capture.actions) != 1:
                    raise RuntimeError("CaDA official smoke did not capture one action tensor")
                return select_cada_output(out["reward"], capture.actions[0],
                                          problem_size=self.problem_size, torch=self.torch)
            selection, _ = timed_call(call, torch=self.torch, device=self.device, timed=False)
            return selection
        finally:
            self.env.generator.reset_variant_preset(prior)

    def solve(self, native, *, timed=True):
        td = to_tensordict(native, torch=self.torch, device=self.device)

        def call():
            reset = self.env.reset(td=td)
            augmented = self.augmentation(reset)
            with self.torch.inference_mode(), ActionCapture(self.env) as capture:
                out = self.model(augmented, self.env)
            if len(capture.actions) != 1:
                raise RuntimeError("CaDA action capture expected exactly one reward call")
            return select_cada_output(out["reward"], capture.actions[0],
                                      problem_size=self.problem_size, torch=self.torch)
        selection, elapsed = timed_call(call, torch=self.torch, device=self.device,
                                        timed=timed)
        return selection, elapsed
