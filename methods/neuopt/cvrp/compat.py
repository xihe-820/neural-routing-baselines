"""Import-only compatibility helpers for the pinned official NeuOpt source."""
from __future__ import annotations

import importlib
import hashlib
import inspect
import sys
import textwrap
from types import ModuleType


COMPATIBILITY_REASON = (
    "official NeuOpt imports tensorboard_logger unconditionally, while formal "
    "eval_only/no_tb inference does not instantiate Logger"
)

BS1_COMPATIBILITY_REASON = (
    "pinned NeuOpt kopt_Decoder.forward removes the batch dimension from the "
    "stopped tensor with dimensionless squeeze when its internal batch is one"
)

_STOPPED_SQUEEZE_REPLACEMENTS = (
    (
        "stopped = stopped | (action == next_of_last_action).squeeze()",
        "stopped = stopped | (action == next_of_last_action).squeeze(-1)",
    ),
    (
        "stopped = (action == next_of_last_action).squeeze()",
        "stopped = (action == next_of_last_action).squeeze(-1)",
    ),
)


def ensure_tensorboard_logger(*, import_module=importlib.import_module):
    """Make NeuOpt's unconditional import succeed without faking logger behavior."""
    try:
        import_module("tensorboard_logger")
    except ModuleNotFoundError as exc:
        if exc.name != "tensorboard_logger":
            raise

        class GuardLogger:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "tensorboard_logger compatibility shim was instantiated; "
                    "no_tb inference unexpectedly entered TensorBoard path"
                )

        module = ModuleType("tensorboard_logger")
        module.Logger = GuardLogger
        sys.modules["tensorboard_logger"] = module
        return {
            "tensorboard_logger_available": False,
            "tensorboard_logger_import_shim": True,
            "compatibility_reason": COMPATIBILITY_REASON,
            "official_source_modified": False,
        }
    return {
        "tensorboard_logger_available": True,
        "tensorboard_logger_import_shim": False,
        "compatibility_reason": None,
        "official_source_modified": False,
    }


def dimension_preserving_decoder_source(source):
    """Patch exactly the two ``stopped`` squeezes and reject source drift."""
    patched = source
    for original, replacement in _STOPPED_SQUEEZE_REPLACEMENTS:
        if patched.count(original) != 1:
            raise RuntimeError(
                "pinned NeuOpt decoder source no longer has the expected BS1 squeeze")
        patched = patched.replace(original, replacement)
    return patched


def ensure_bs1_decoder_compatibility(decoder_class):
    """Install the guarded in-memory shape fix; never edit official source on disk."""
    current = decoder_class.forward
    prior = getattr(current, "_neuopt_bs1_compatibility", None)
    if prior is not None:
        return dict(prior)
    source = textwrap.dedent(inspect.getsource(current))
    patched_source = dimension_preserving_decoder_source(source)
    namespace = {}
    filename = inspect.getsourcefile(current) or "<pinned NeuOpt decoder>"
    exec(compile(patched_source, filename, "exec"), current.__globals__, namespace)
    patched = namespace["forward"]
    patched.__name__ = current.__name__
    patched.__qualname__ = current.__qualname__
    patched.__doc__ = current.__doc__
    provenance = {
        "bs1_shape_shim": True,
        "compatibility_reason": BS1_COMPATIBILITY_REASON,
        "patched_tensor": "stopped",
        "original_shape_at_internal_bs1": [],
        "patched_shape_at_internal_bs1": [1],
        "operation_change": "two stopped comparisons use squeeze(-1) instead of squeeze()",
        "original_forward_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "patched_forward_sha256": hashlib.sha256(patched_source.encode()).hexdigest(),
        "official_source_modified": False,
        "action_reward_logits_rng_budget_changed": False,
    }
    patched._neuopt_bs1_compatibility = provenance
    decoder_class.forward = patched
    return dict(provenance)
