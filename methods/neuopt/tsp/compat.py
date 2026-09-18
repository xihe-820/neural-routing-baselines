"""Replay-only compatibility for pinned NeuOpt TSP evidence recording."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import textwrap
from types import MethodType

from methods.neuopt.cvrp.compat import ensure_bs1_decoder_compatibility


PINNED_TSP_STEP_SHA256 = "62775b74cf54c4d22f7a34ef93e0f6b2d8e26a88b9e8a7e7845e419b7dd0448d"
PINNED_DECODER_FORWARD_SHA256 = "b4e959f046040b55db7a872d7d20b04ba2c4b9ae35fcba9aac0b3ea666221887"
RECORD_COMPATIBILITY_REASON = (
    "pinned TSP.step returns feasibility_history=None, while official PPO.rollout(record=True) "
    "indexes that bookkeeping value after each step"
)


def configure_decoder(decoder_class, *, original_batch_size, val_m=1):
    if val_m != 1 or original_batch_size not in (1, 16, 128):
        raise ValueError("formal NeuOpt TSP decoder supports only D2A=1 and BS=1/16/128")
    if original_batch_size == 1:
        provenance = ensure_bs1_decoder_compatibility(decoder_class)
        provenance.update({
            "original_batch_size": 1, "D2A": 1, "val_m": 1,
            "internal_decoder_batch_size": 1,
        })
        return provenance
    current = decoder_class.forward
    if getattr(current, "_neuopt_bs1_compatibility", None) is not None:
        raise RuntimeError("native BS16/128 refuses a decoder with the BS1 shim installed")
    source = textwrap.dedent(inspect.getsource(current))
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != PINNED_DECODER_FORWARD_SHA256:
        raise RuntimeError("pinned NeuOpt decoder source changed; native batch refused")
    return {
        "bs1_shape_shim": False,
        "compatibility_reason": None,
        "original_batch_size": original_batch_size,
        "D2A": 1, "val_m": 1,
        "internal_decoder_batch_size": original_batch_size,
        "unmodified_forward_sha256": digest,
        "official_source_modified": False,
        "action_reward_logits_rng_budget_changed": False,
    }


def record_compatibility_provenance(problem):
    source = textwrap.dedent(inspect.getsource(type(problem).step))
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != PINNED_TSP_STEP_SHA256:
        raise RuntimeError("pinned NeuOpt TSP.step source changed; replay shim refused")
    return {
        "record_step_shim": True,
        "compatibility_reason": RECORD_COMPATIBILITY_REASON,
        "pinned_step_sha256": digest,
        "timed_rollout_shim_active": False,
        "evidence_replay_shim_active": True,
        "preserved_value": "input feasibility_history bookkeeping tensor",
        "official_source_modified": False,
        "action_reward_logits_rng_budget_changed": False,
    }


@contextmanager
def replay_record_compatibility(problem):
    """Preserve TSP bookkeeping only during untimed record=True replay."""
    provenance = record_compatibility_provenance(problem)
    original = problem.step

    def wrapped(self, batch, rec, action, obj, feasible_history, t, weights=0):
        out = original(batch, rec, action, obj, feasible_history, t, weights=weights)
        if not isinstance(out, tuple) or len(out) != 7 or out[3] is not None:
            raise RuntimeError("pinned NeuOpt TSP.step output changed; replay shim refused")
        return out[:3] + (feasible_history,) + out[4:]

    problem.step = MethodType(wrapped, problem)
    try:
        yield provenance
    finally:
        del problem.__dict__["step"]
