"""Pinned MoSES(CaDA) construction, audited load, and official evaluation call."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import inspect
import importlib.util
from pathlib import Path
import sys

from common.cvrptw_runtime import (select_rl4co_batch_output,
                                   select_rl4co_output, timed_call,
                                   to_tensordict)


CHECKPOINT_DIAGNOSTIC_FIELDS = (
    "test_decode_type", "train_decode_type", "val_decode_type", "temperature",
    "lora_temperature", "lora_act_func", "lora_n_experts", "lora_top_k",
    "lora_rank", "lora_use_trainable_layer", "lora_use_dynamic_topK",
    "lora_use_basis_variants", "lora_use_basis_variants_as_input",
    "lora_use_linear",
)

POLICY_KWARGS = {
    "normalization": "rms",
    "encoder_use_prenorm": False,
    "encoder_use_post_layers_norm": False,
    "parallel_gated_kwargs": {"mlp_activation": "silu"},
    "attn_sparse_ratio": 0.5,
    "sparse_applied_to_score": True,
    "lora_rank": [32] * 5,
    "lora_alpha": 1.0,
    "lora_act_func": "sigmoid",
    "lora_n_experts": 4,
    "lora_top_k": 4,
    "lora_temperature": 1.0,
    "lora_use_trainable_layer": True,
    "lora_use_dynamic_topK": False,
    "lora_use_basis_variants": False,
    "lora_use_basis_variants_as_input": False,
    "lora_use_linear": False,
}
OFFICIAL_TEST_MODULE = "_neural_routing_baselines_pinned_moses_test"


class CheckpointCompatibilityError(RuntimeError):
    def __init__(self, checkpoint_state):
        self.checkpoint_state = checkpoint_state
        super().__init__(
            "MoSES checkpoint has missing/unexpected policy keys: "
            f"missing={checkpoint_state['missing_keys']!r}, "
            f"unexpected={checkpoint_state['unexpected_keys']!r}")


def _json_diagnostic(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_diagnostic(child) for child in value]
    if isinstance(value, Mapping):
        return {str(key): _json_diagnostic(child) for key, child in value.items()}
    return repr(value)


def _nested_candidates(value, key, path="$", seen=None):
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return []
    identity = id(value)
    if identity in seen:
        return []
    seen.add(identity)
    found = []
    if isinstance(value, Mapping):
        for name, child in value.items():
            child_path = f"{path}.{name}"
            if name == key:
                found.append({"path": child_path, "value": _json_diagnostic(child)})
            if name != "state_dict":
                found.extend(_nested_candidates(child, key, child_path, seen))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_nested_candidates(child, key, f"{path}[{index}]", seen))
    else:
        try:
            attributes = vars(value)
        except TypeError:
            attributes = {}
        ignored = {
            "_parameters", "_buffers", "_non_persistent_buffers_set",
            "_backward_hooks", "_backward_pre_hooks", "_forward_hooks",
            "_forward_pre_hooks", "_state_dict_hooks", "_load_state_dict_pre_hooks",
            "_load_state_dict_post_hooks",
        }
        for name, child in attributes.items():
            child_path = f"{path}.{name}"
            if name == key:
                found.append({"path": child_path, "value": _json_diagnostic(child)})
            if name not in ignored:
                found.extend(_nested_candidates(child, key, child_path, seen))
    return found


def checkpoint_metadata(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("MoSES checkpoint must contain a state_dict")
    if payload.get("epoch") != 299 or payload.get("global_step") != 117300:
        raise ValueError("MoSES checkpoint epoch/global_step mismatch")
    if not all(name.startswith("policy.") for name in payload["state_dict"]):
        raise ValueError("MoSES checkpoint contains a non-policy state key")
    result = {
        "payload_type": f"{type(payload).__module__}.{type(payload).__qualname__}",
        "top_level_keys": sorted(str(key) for key in payload),
        "epoch": payload["epoch"], "global_step": payload["global_step"],
        "pytorch_lightning_version": payload.get("pytorch-lightning_version"),
        "state_dict_count": len(payload["state_dict"]),
        "all_state_dict_keys_policy_prefixed": True,
    }
    for field in CHECKPOINT_DIAGNOSTIC_FIELDS:
        result[f"checkpoint_{field}_candidates"] = _nested_candidates(payload, field)
    return result


def _module_is_below(module, root):
    filename = getattr(module, "__file__", None)
    if filename is None:
        return False
    try:
        Path(filename).resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


def _require_official_module(module, upstream, label):
    if not _module_is_below(module, Path(upstream).resolve()):
        raise ImportError(
            f"pinned MoSES {label} resolved outside the official checkout: "
            f"{getattr(module, '__file__', None)!r}")


@contextmanager
def _official_module_context(upstream):
    """Temporarily bind MoSES' top-level packages without leaking/colliding."""
    upstream = Path(upstream).resolve()
    package_roots = ("envs", "models", "utils")
    original_path = list(sys.path)
    displaced = {
        name: module for name, module in list(sys.modules.items())
        if name.split(".", 1)[0] in package_roots or name == OFFICIAL_TEST_MODULE
    }
    for name in displaced:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(upstream))
    try:
        yield
    finally:
        for name, module in list(sys.modules.items()):
            if (name.split(".", 1)[0] in package_roots or
                    name == OFFICIAL_TEST_MODULE or
                    _module_is_below(module, upstream)):
                sys.modules.pop(name, None)
        sys.modules.update(displaced)
        sys.path[:] = original_path


def _load_official_test(upstream):
    spec = importlib.util.spec_from_file_location(
        OFFICIAL_TEST_MODULE, Path(upstream) / "test.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load pinned MoSES test.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[OFFICIAL_TEST_MODULE] = module
    spec.loader.exec_module(module)
    return module.test


def _construct_policy(policy_class):
    parameters = inspect.signature(policy_class).parameters
    explicit = {
        name for name, parameter in parameters.items()
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    }
    unused = sorted(set(POLICY_KWARGS) - explicit)
    if unused:
        raise RuntimeError(
            f"MoSES constructor arguments would become unused kwargs: {unused}")
    return policy_class(**POLICY_KWARGS)


def _require_equal(name, observed, expected):
    if observed != expected:
        raise RuntimeError(
            f"MoSES effective runtime protocol mismatch for {name}: "
            f"observed {observed!r}, expected {expected!r}")


def effective_runtime_protocol(policy, *, problem_size):
    """Inspect constructed objects; checkpoint hparams are not protocol gates."""
    _require_equal("policy.test_decode_type", policy.test_decode_type, "greedy")
    _require_equal("policy.temperature", float(policy.temperature), 1.0)

    encoder = policy.encoder.encoder
    global_blocks = list(encoder.global_layers)
    sparse_blocks = list(encoder.sparse_layers)
    if not global_blocks or len(global_blocks) != len(sparse_blocks):
        raise RuntimeError("MoSES effective encoder layers are incomplete")
    for index, block in enumerate([*global_blocks, *sparse_blocks]):
        _require_equal(f"encoder block {index} normalization", block.normalization, "rms")
        _require_equal(f"encoder block {index} prenorm", block.use_prenorm, False)
        _require_equal(
            f"encoder block {index} parallel_gated_kwargs",
            block.parallel_gated_kwargs, {"mlp_activation": "silu"})
    _require_equal("encoder post-layer normalization", encoder.post_layers_norm, None)
    for index, block in enumerate(sparse_blocks):
        _require_equal(f"sparse block {index} ratio", float(block.attn_sparse_ratio), 0.5)
        _require_equal(f"sparse block {index} score application",
                       block.sparse_applied_to_score, True)

    modules = list(policy.modules())
    mlps = [module for module in modules if type(module).__name__ == "ParallelGatedMLP"]
    if not mlps:
        raise RuntimeError("MoSES policy contains no ParallelGatedMLP")
    for index, module in enumerate(mlps):
        _require_equal(f"ParallelGatedMLP {index} activation", module.act_type, "silu")

    gated = [module for module in modules if type(module).__name__ == "GatedMultiLoRALayer"]
    if not gated:
        raise RuntimeError("MoSES policy contains no GatedMultiLoRALayer")
    rank_bearing = 0
    for index, module in enumerate(gated):
        expected = {
            "act_func": "sigmoid", "n_experts": 4, "top_k": 4,
            "temperature": 1.0, "use_trainable_layer": True,
            "use_dynamic_topK": False, "use_basis_variants": False,
            "use_basis_variants_as_input": False,
        }
        for field, value in expected.items():
            observed = getattr(module, field)
            if isinstance(value, float):
                observed = float(observed)
            _require_equal(f"GatedMultiLoRALayer {index}.{field}", observed, value)
        lora_layers = list(module.lora_layers)
        _require_equal(f"GatedMultiLoRALayer {index} layer count", len(lora_layers), 5)
        for layer_index, layer in enumerate(lora_layers):
            _require_equal(f"LoRA layer {index}.{layer_index} alpha",
                           float(layer.alpha), 1.0)
            if hasattr(layer, "rank"):
                rank_bearing += 1
                _require_equal(f"LoRA layer {index}.{layer_index} rank", layer.rank, 32)
            if hasattr(layer, "use_linear"):
                _require_equal(f"LoRA layer {index}.{layer_index} use_linear",
                               layer.use_linear, False)
    if rank_bearing == 0:
        raise RuntimeError("MoSES policy contains no rank-bearing LoRA layer")
    return {
        "policy_test_decode_type": "greedy", "decode_temperature": 1.0,
        "multistart": True, "num_starts": problem_size,
        "start_selector": "all customers 1..N",
        "num_augmentations": 8, "augmentation": "dihedral8",
        "selection": "max start then max augmentation",
        "normalization": "rms", "encoder_use_prenorm": False,
        "encoder_use_post_layers_norm": False, "mlp_activation": "silu",
        "attn_sparse_ratio": 0.5, "sparse_applied_to_score": True,
        "lora_rank": [32] * 5, "lora_alpha": 1.0,
        "lora_act_func": "sigmoid", "lora_n_experts": 4,
        "lora_top_k": 4, "lora_temperature": 1.0,
        "lora_use_trainable_layer": True, "lora_use_dynamic_topK": False,
        "lora_use_basis_variants": False,
        "lora_use_basis_variants_as_input": False, "lora_use_linear": False,
        "gated_multi_lora_layer_count": len(gated),
        "rank_bearing_lora_layer_count": rank_bearing,
        "constructor_unused_kwargs": [],
    }


class Runtime:
    def __init__(self, upstream, checkpoint, problem_size, device, torch):
        self.upstream, self.problem_size, self.device, self.torch = (
            Path(upstream), problem_size, device, torch)
        with _official_module_context(self.upstream):
            from envs import MTVRPEnv
            from models import CadaMultiLoRAPolicy
            _require_official_module(
                sys.modules[MTVRPEnv.__module__], self.upstream, "environment module")
            _require_official_module(
                sys.modules[CadaMultiLoRAPolicy.__module__], self.upstream,
                "policy module")
            self.official_test = _load_official_test(self.upstream)
            self.policy = _construct_policy(CadaMultiLoRAPolicy)
            self.effective_runtime_protocol = effective_runtime_protocol(
                self.policy, problem_size=problem_size)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            parsed = checkpoint_metadata(payload)
            state = payload["state_dict"]
            policy_state = {
                name.removeprefix("policy."): value for name, value in state.items()}
            incompatible = self.policy.load_state_dict(policy_state, strict=False)
            self.env = MTVRPEnv()
            self.smoke_env = MTVRPEnv(
                generator_params={"num_loc": problem_size, "variant_preset": "vrptw"},
                check_solution=True)
        self.checkpoint_state = {
            "official_load_strict": False, **parsed,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "effective_runtime_protocol": self.effective_runtime_protocol,
        }
        if self.checkpoint_state["missing_keys"] or self.checkpoint_state["unexpected_keys"]:
            raise CheckpointCompatibilityError(self.checkpoint_state)
        self.policy = self.policy.to(device).eval()

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
        selections, elapsed = self.solve_batch(native, timed=timed)
        if len(selections) != 1:
            raise RuntimeError("MoSES solve() is the batch-one compatibility API")
        return selections[0], elapsed

    def solve_batch(self, native, *, timed=True):
        td = to_tensordict(native, torch=self.torch, device=self.device)

        def call():
            reset = self.env.reset(td)
            out = self.official_test(self.policy, reset, self.env, num_augment=8,
                                     augment_fn="dihedral8", num_starts=self.problem_size,
                                     device=self.device)
            return select_rl4co_batch_output(
                out, problem_size=self.problem_size, torch=self.torch)
        return timed_call(call, torch=self.torch, device=self.device, timed=timed)
