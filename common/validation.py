"""Numeric input checks only; no model, Kit, loader, or method adapter imports."""
import numpy as np


def points2d(value, name="points"):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or raw.shape[1] != 2 or len(raw) == 0:
        raise ValueError(f"{name} must be a nonempty real array [N,2]")
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def integer_ids(value):
    if isinstance(value, (list, tuple)) and any(isinstance(item, (bool, np.bool_)) for item in value):
        raise ValueError("boolean values are not node IDs")
    result = np.asarray(value)
    if result.ndim != 1 or result.dtype.kind not in "iu" or result.size == 0:
        raise ValueError("solution must be a nonempty 1D integer array; floats/bools are not IDs")
    return result


def scalar(value, name, *, positive=False):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real scalar")
    result = float(raw)
    if not np.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def vector(value, length, name):
    raw = np.asarray(value)
    if raw.shape != (length,) or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must have shape [{length}] with real entries")
    result = raw.astype(np.float64)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def failure(message):
    return {"feasible": False, "independent_objective": None,
            "constraint_details": {"input_error": str(message)}}
