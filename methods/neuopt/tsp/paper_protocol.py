"""Calibration-only NeuOpt TSP100 protocol; no final T is selected here."""
from __future__ import annotations

import hashlib
import json


UPSTREAM_URL = "https://github.com/yining043/NeuOpt"
UPSTREAM_COMMIT = "ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b"
SEED = 6666
GRAPH_SIZE = 100
D2A = 1
STALL_LIMIT = 10
K = 4
ORIGINAL_BATCH_SIZE = 1
CALIBRATION_INDICES = tuple(range(20))
FIRST_ROUND_T_VALUES = (1, 2, 5, 10, 20)
CALIBRATION_TARGETS = {
    "fewer_seconds": 0.035,
    "more_seconds": 0.117,
    "source": "current paper COReformer TSP100 BS1 runtime",
}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def calibration_protocol(problem_size, *, T_max, batch_size=ORIGINAL_BATCH_SIZE,
                         d2a=D2A, stall_limit=STALL_LIMIT, k=K):
    try:
        size = int(problem_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("NeuOpt TSP calibration supports only TSP100") from exc
    if size != GRAPH_SIZE:
        raise ValueError("NeuOpt TSP calibration supports only TSP100")
    _positive_integer(T_max, "T_max")
    if batch_size != ORIGINAL_BATCH_SIZE:
        raise ValueError("NeuOpt TSP calibration requires original batch size 1")
    if d2a != D2A:
        raise ValueError("NeuOpt TSP calibration requires D2A=1")
    if stall_limit != STALL_LIMIT or k != K:
        raise ValueError("NeuOpt TSP calibration requires stall_limit=10 and k=4")
    return {
        "protocol_status": "CALIBRATION_CANDIDATE_ONLY",
        "method": "NeuOpt", "variant": "NeuOpt-GIRE", "problem": "TSP",
        "problem_size": GRAPH_SIZE, "graph_size": GRAPH_SIZE,
        "D2A": D2A, "val_m": D2A, "T_max": T_max,
        "original_batch_size": ORIGINAL_BATCH_SIZE,
        "internal_decoder_batch_size": ORIGINAL_BATCH_SIZE,
        "stall_limit": STALL_LIMIT, "k": K, "init_val_met": "random",
        "seed": SEED, "eval_only": True, "no_tb": True, "no_saving": True,
        "training": False, "fine_tuning": False, "backward": False,
        "optimizer_created": False, "single_gpu": True,
        "first_round_candidate": T_max in FIRST_ROUND_T_VALUES,
        "final_T_status": "PENDING_USER_REVIEW",
        "paper_ready": False,
    }


def protocol_fingerprint(protocol):
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
