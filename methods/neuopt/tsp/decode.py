"""Strictly decode NeuOpt's TSP successor cycle into a canonical tour."""
from __future__ import annotations

import numpy as np

from methods.neuopt.tsp.config import supported_config


def decode_successor(successor, *, problem_size=100):
    supported_config(problem_size)
    values = np.asarray(successor)
    if values.ndim != 1 or len(values) != problem_size:
        raise ValueError("successor must be a one-dimensional TSP100 vector")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("successor contains non-integer node ids")
    values = values.astype(np.int64)
    if (values < 0).any() or (values >= problem_size).any():
        raise ValueError("successor contains an out-of-range node id")
    if not np.array_equal(np.sort(values), np.arange(problem_size)):
        raise ValueError("successor is not a permutation; duplicate/missing nodes are rejected")

    order = [0]
    seen = {0}
    current = 0
    for _ in range(problem_size - 1):
        current = int(values[current])
        if current in seen:
            raise ValueError("successor contains a subtour before all nodes are visited")
        seen.add(current)
        order.append(current)
    if int(values[current]) != 0 or len(seen) != problem_size:
        raise ValueError("successor does not form one Hamiltonian cycle")
    return order + [0], {
        "internal_order": order,
        "root_node": 0,
        "canonicalization": "traverse the exact successor cycle from node 0 and append node 0",
        "repair": False,
    }


def extract_final_best(rollout_output, *, batch_size=1, val_m=1):
    if batch_size != 1 or val_m != 1:
        raise ValueError("NeuOpt TSP calibration extraction requires BS=1 and D2A=1")
    best_objective, obj_history, _, record = rollout_output
    if record is None or len(record) != 3:
        raise ValueError("official rollout did not return record=True histories")
    solution_best_history = record[1]
    if len(solution_best_history) != obj_history.shape[1]:
        raise ValueError("best-solution and objective histories have different lengths")
    final = solution_best_history[-1]
    if final.ndim != 2 or tuple(final.shape) != (batch_size, 100):
        raise ValueError("final recorded successor has an unexpected shape")
    final_history_best = obj_history[:, -1, 1]
    import torch
    if not torch.equal(best_objective, final_history_best):
        raise ValueError("official out[0] does not correspond to final recorded best solution")
    return best_objective, final
