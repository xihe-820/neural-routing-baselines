"""Import-only compatibility helpers for the pinned official NeuOpt source."""
from __future__ import annotations

import importlib
import sys
from types import ModuleType


COMPATIBILITY_REASON = (
    "official NeuOpt imports tensorboard_logger unconditionally, while formal "
    "eval_only/no_tb inference does not instantiate Logger"
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
