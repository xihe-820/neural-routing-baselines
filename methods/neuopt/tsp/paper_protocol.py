"""NeuOpt TSP100 calibration and human-frozen formal production protocols."""
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
FINAL_T_BY_BUDGET = {"fewer": 1, "more": 5}
FORMAL_BATCH_SIZES = (1, 16, 128)
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


def formal_protocol(problem_size, *, budget, batch_size, T_max,
                    d2a=D2A, stall_limit=STALL_LIMIT, k=K):
    try:
        size = int(problem_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("formal NeuOpt TSP production supports only TSP100") from exc
    if size != GRAPH_SIZE:
        raise ValueError("formal NeuOpt TSP production supports only TSP100")
    if budget not in FINAL_T_BY_BUDGET:
        raise ValueError("formal NeuOpt TSP budget must be fewer or more")
    expected_t = FINAL_T_BY_BUDGET[budget]
    if type(T_max) is not int or T_max != expected_t:
        raise ValueError(f"formal NeuOpt TSP {budget} requires T_max={expected_t}")
    if type(batch_size) is not int or batch_size not in FORMAL_BATCH_SIZES:
        raise ValueError("formal NeuOpt TSP batch_size must be 1, 16, or 128")
    if type(d2a) is not int or d2a != D2A:
        raise ValueError("formal NeuOpt TSP production requires D2A=1")
    if (type(stall_limit) is not int or type(k) is not int or
            stall_limit != STALL_LIMIT or k != K):
        raise ValueError("formal NeuOpt TSP production requires stall_limit=10 and k=4")
    return {
        "protocol_status": "HUMAN_FROZEN_FORMAL",
        "method": "NeuOpt", "variant": "NeuOpt-GIRE", "problem": "TSP",
        "problem_size": GRAPH_SIZE, "graph_size": GRAPH_SIZE,
        "budget": budget, "D2A": D2A, "val_m": D2A, "T_max": expected_t,
        "original_batch_size": batch_size,
        "internal_decoder_batch_size": batch_size,
        "stall_limit": STALL_LIMIT, "k": K, "init_val_met": "random",
        "seed": SEED, "eval_only": True, "no_tb": True, "no_saving": True,
        "training": False, "fine_tuning": False, "backward": False,
        "optimizer_created": False, "single_gpu": True,
        "budget_selection": "manual_after_RTX4090_first20_calibration",
        "paper_table_role": (
            "COMPLETE_RESULTS_AND_PARALLEL" if batch_size == 1 else "PARALLEL"
        ),
        "paper_ready_protocol": True,
    }


def protocol_fingerprint(protocol):
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
