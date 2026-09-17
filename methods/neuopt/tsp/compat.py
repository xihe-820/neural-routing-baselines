"""Replay-only compatibility for pinned NeuOpt TSP evidence recording."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import textwrap
from types import MethodType


PINNED_TSP_STEP_SHA256 = "62775b74cf54c4d22f7a34ef93e0f6b2d8e26a88b9e8a7e7845e419b7dd0448d"
RECORD_COMPATIBILITY_REASON = (
    "pinned TSP.step returns feasibility_history=None, while official PPO.rollout(record=True) "
    "indexes that bookkeeping value after each step"
)


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
