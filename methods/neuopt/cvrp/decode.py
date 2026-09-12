"""Decode NeuOpt's successor Hamiltonian cycle into a canonical CVRP route."""
from __future__ import annotations

import numpy as np

from methods.neuopt.cvrp.config import supported_config


def successor_to_order(successor, *, sequence_length):
    successor = np.asarray(successor)
    if successor.ndim != 1 or len(successor) != sequence_length:
        raise ValueError("successor must be a one-dimensional full-length vector")
    if not np.issubdtype(successor.dtype, np.integer):
        if not np.isfinite(successor).all() or not np.equal(successor, np.floor(successor)).all():
            raise ValueError("successor contains non-integer node ids")
    successor = successor.astype(np.int64)
    if (successor < 0).any() or (successor >= sequence_length).any():
        raise ValueError("successor contains an out-of-range node id")
    if not np.array_equal(np.sort(successor), np.arange(sequence_length)):
        raise ValueError("successor is not a permutation; duplicate/missing nodes are not repaired")
    order = [0]
    seen = {0}
    current = 0
    for _ in range(sequence_length - 1):
        current = int(successor[current])
        if current in seen:
            raise ValueError("successor closes a cycle before visiting every node")
        seen.add(current)
        order.append(current)
    if int(successor[current]) != 0 or len(seen) != sequence_length:
        raise ValueError("successor does not define one Hamiltonian cycle rooted at internal node 0")
    return order


def canonicalize_internal_order(order, *, problem_size, dummy_size):
    sequence_length = problem_size + dummy_size
    order = np.asarray(order)
    if order.ndim != 1 or len(order) != sequence_length:
        raise ValueError("internal order must contain every internal node exactly once")
    if not np.issubdtype(order.dtype, np.integer):
        raise ValueError("internal order contains non-integer node ids")
    order = order.astype(np.int64)
    if not np.array_equal(np.sort(order), np.arange(sequence_length)):
        raise ValueError("internal order has duplicate/missing nodes; it is not repaired")
    mapped = [0 if node < dummy_size else int(node - dummy_size + 1) for node in order]
    canonical = []
    for node in mapped:
        if node != 0 or not canonical or canonical[-1] != 0:
            canonical.append(node)
    if not canonical or canonical[0] != 0:
        canonical.insert(0, 0)
    if canonical[-1] != 0:
        canonical.append(0)
    return canonical


def decode_successor(successor, *, problem_size):
    config = supported_config(problem_size)
    order = successor_to_order(successor, sequence_length=config["sequence_length"])
    canonical = canonicalize_internal_order(
        order, problem_size=problem_size, dummy_size=config["dummy_size"]
    )
    return canonical, {
        "internal_order": order,
        "dummy_size": config["dummy_size"],
        "consecutive_dummy_policy": "collapse adjacent depot separators only; customer nodes are never repaired",
    }


def extract_final_best(rollout_output, *, batch_size, val_m=1):
    """Extract the recorded solution that corresponds to official rollout out[0]."""
    if val_m != 1:
        raise ValueError("formal NeuOpt extraction requires val_m=1")
    best_objective, obj_history, _, record = rollout_output
    if record is None or len(record) != 3:
        raise ValueError("official rollout did not return record=True histories")
    solution_best_history = record[1]
    if len(solution_best_history) != obj_history.shape[1]:
        raise ValueError("best-solution and objective histories have different lengths")
    final = solution_best_history[-1]
    if tuple(final.shape)[0] != batch_size:
        raise ValueError("final best successor batch does not match original batch")
    final_history_best = obj_history[:, -1, 1]
    import torch
    if not torch.equal(best_objective, final_history_best):
        raise ValueError("official out[0] does not correspond to final obj-history best column")
    return best_objective, final
