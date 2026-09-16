"""NeuOpt-GIRE CVRP paper-calibration protocol and manuscript targets."""
from __future__ import annotations

import hashlib
import json


UPSTREAM_URL = "https://github.com/yining043/NeuOpt"
UPSTREAM_COMMIT = "ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b"
SEED = 6666
FORMAL_D2A = 5
FORMAL_STALL_LIMIT = 10
FORMAL_K = 4
ORIGINAL_BATCH_SIZE = 1
MANUSCRIPT_CANDIDATE_T = (1000, 5000)

CALIBRATION_TARGETS = {
    50: {
        "fewer_seconds": 0.197,
        "more_seconds": 0.804,
        "source": (
            "latest manuscript CVRP-[Uniform-50], COReformer fewer/more rows"
        ),
    },
    100: {
        "fewer_seconds": 0.307,
        "more_seconds": 1.193,
        "source": (
            "latest manuscript CVRP-[Uniform-100], COReformer fewer/more rows"
        ),
    },
}


def paper_protocol(problem_size, *, T_max, d2a=FORMAL_D2A,
                   stall_limit=FORMAL_STALL_LIMIT, k=FORMAL_K):
    """Return one formal calibration candidate or reject protocol drift."""
    try:
        size = int(problem_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("NeuOpt paper calibration supports only CVRP50 or CVRP100") from exc
    if size not in CALIBRATION_TARGETS:
        raise ValueError("NeuOpt paper calibration supports only CVRP50 or CVRP100")
    values = {"D2A": d2a, "stall_limit": stall_limit, "k": k}
    expected = {"D2A": FORMAL_D2A, "stall_limit": FORMAL_STALL_LIMIT,
                "k": FORMAL_K}
    if values != expected:
        raise ValueError(
            "formal NeuOpt calibration requires D2A=5, stall_limit=10, and k=4")
    if isinstance(T_max, bool) or not isinstance(T_max, int) or T_max <= 0:
        raise ValueError("T_max must be a positive integer")
    return {
        "method": "NeuOpt",
        "variant": "NeuOpt-GIRE",
        "problem": "CVRP",
        "problem_size": size,
        "D2A": FORMAL_D2A,
        "val_m": FORMAL_D2A,
        "T_max": T_max,
        "stall_limit": FORMAL_STALL_LIMIT,
        "k": FORMAL_K,
        "init_val_met": "random",
        "original_batch_size": ORIGINAL_BATCH_SIZE,
        "seed": SEED,
        "eval_only": True,
        "no_tb": True,
        "no_saving": True,
        "with_bonus": True,
        "with_regular": True,
        "training": False,
        "backward": False,
        "optimizer_created": False,
        "protobuf_workaround": False,
        "budget_status": (
            "MANUSCRIPT_CANDIDATE"
            if T_max in MANUSCRIPT_CANDIDATE_T else "CALIBRATION_CANDIDATE"
        ),
        "final_budget_status": "PENDING_CALIBRATION",
        "calibration_target": dict(CALIBRATION_TARGETS[size]),
    }


def protocol_fingerprint(protocol):
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def pending_final_t(*, cvrp50_fewer=None, cvrp50_more=None,
                    cvrp100_fewer=None, cvrp100_more=None):
    """Represent independently selectable size/budget values without freezing them."""
    return {
        "50": {"fewer": cvrp50_fewer, "more": cvrp50_more},
        "100": {"fewer": cvrp100_fewer, "more": cvrp100_more},
    }
