"""Frozen NeuOpt-GIRE CVRP paper protocol plus legacy calibration identity."""
from __future__ import annotations

import hashlib
import json


UPSTREAM_URL = "https://github.com/yining043/NeuOpt"
UPSTREAM_COMMIT = "ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b"
SEED = 6666
FORMAL_D2A = 1
FORMAL_T_VALUES = (20, 50)
FORMAL_BATCH_SIZES = (1, 100)
FORMAL_STALL_LIMIT = 10
FORMAL_K = 4

# Retained only so pre-existing D2A=5 calibration artifacts remain readable.
LEGACY_MANUSCRIPT_CANDIDATE_T = (1000, 5000)
LEGACY_CALIBRATION_TARGETS = {
    50: {
        "fewer_seconds": 0.197,
        "more_seconds": 0.804,
        "source": "superseded COReformer runtime-calibration comparison",
    },
    100: {
        "fewer_seconds": 0.307,
        "more_seconds": 1.193,
        "source": "superseded COReformer runtime-calibration comparison",
    },
}


def _problem_size(value):
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("NeuOpt paper evaluation supports only CVRP50 or CVRP100") from exc
    if size not in (50, 100):
        raise ValueError("NeuOpt paper evaluation supports only CVRP50 or CVRP100")
    return size


def paper_protocol(problem_size, *, T_max, batch_size, d2a=FORMAL_D2A,
                   stall_limit=FORMAL_STALL_LIMIT, k=FORMAL_K):
    """Return one of the eight senior-author-frozen production configurations."""
    size = _problem_size(problem_size)
    if d2a != FORMAL_D2A:
        raise ValueError("final NeuOpt paper protocol requires D2A=1")
    if T_max not in FORMAL_T_VALUES:
        raise ValueError("final NeuOpt paper protocol requires T_max 20 or 50")
    if batch_size not in FORMAL_BATCH_SIZES:
        raise ValueError("final NeuOpt paper protocol requires batch_size 1 or 100")
    if stall_limit != FORMAL_STALL_LIMIT or k != FORMAL_K:
        raise ValueError("final NeuOpt paper protocol requires stall_limit=10 and k=4")
    return {
        "protocol_status": "LATEST_SENIOR_AUTHOR_FROZEN",
        "method": "NeuOpt", "variant": "NeuOpt-GIRE", "problem": "CVRP",
        "problem_size": size, "D2A": FORMAL_D2A, "val_m": FORMAL_D2A,
        "T_max": T_max, "original_batch_size": batch_size,
        "internal_decoder_batch_size": batch_size,
        "stall_limit": FORMAL_STALL_LIMIT, "k": FORMAL_K,
        "init_val_met": "random", "seed": SEED,
        "eval_only": True, "no_tb": True, "no_saving": True,
        "with_bonus": True, "with_regular": True, "single_gpu": True,
        "training": False, "fine_tuning": False, "backward": False,
        "optimizer_created": False, "protobuf_workaround": False,
        "paper_table_role": "COMPLETE_RESULTS" if batch_size == 1 else "PARALLEL",
    }


def legacy_calibration_protocol(problem_size, *, T_max, d2a=5,
                                stall_limit=FORMAL_STALL_LIMIT, k=FORMAL_K):
    """Reconstruct superseded D2A=5 identity without admitting it as production."""
    size = _problem_size(problem_size)
    if d2a != 5 or stall_limit != FORMAL_STALL_LIMIT or k != FORMAL_K:
        raise ValueError("legacy NeuOpt calibration requires D2A=5, stall_limit=10, and k=4")
    if isinstance(T_max, bool) or not isinstance(T_max, int) or T_max <= 0:
        raise ValueError("legacy T_max must be a positive integer")
    return {
        "method": "NeuOpt", "variant": "NeuOpt-GIRE", "problem": "CVRP",
        "problem_size": size, "D2A": 5, "val_m": 5, "T_max": T_max,
        "stall_limit": FORMAL_STALL_LIMIT, "k": FORMAL_K,
        "init_val_met": "random", "original_batch_size": 1, "seed": SEED,
        "eval_only": True, "no_tb": True, "no_saving": True,
        "with_bonus": True, "with_regular": True, "training": False,
        "backward": False, "optimizer_created": False,
        "protobuf_workaround": False,
        "budget_status": (
            "MANUSCRIPT_CANDIDATE"
            if T_max in LEGACY_MANUSCRIPT_CANDIDATE_T else "CALIBRATION_CANDIDATE"
        ),
        "final_budget_status": "PENDING_CALIBRATION",
        "calibration_target": dict(LEGACY_CALIBRATION_TARGETS[size]),
    }


def protocol_fingerprint(protocol):
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def pending_final_t(*, cvrp50_fewer=None, cvrp50_more=None,
                    cvrp100_fewer=None, cvrp100_more=None):
    """Legacy calibration helper retained for old report readers."""
    return {
        "50": {"fewer": cvrp50_fewer, "more": cvrp50_more},
        "100": {"fewer": cvrp100_fewer, "more": cvrp100_more},
    }
