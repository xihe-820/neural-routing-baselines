"""Select and canonicalize the exact MVMoE VRPTW reward candidate."""
from __future__ import annotations

import numpy as np

from methods.mvmoe.cvrptw.config import PROBLEM_SIZE


def decode_selected_nodes(selected_nodes):
    """Collapse only terminal finished-POMO depot padding; never repair routes."""
    raw = np.asarray(selected_nodes)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.size == 0:
        raise ValueError("selected nodes must be a nonempty 1D integer sequence")
    if (raw < 0).any() or (raw > PROBLEM_SIZE).any():
        raise ValueError("selected node ID outside 0..50")
    if raw[0] != 0:
        raise ValueError("official MVMoE rollout must start at depot 0")
    end = len(raw)
    while end > 0 and raw[end - 1] == 0:
        end -= 1
    if end == 0:
        canonical = [0]
    else:
        canonical = raw[:end].astype(int).tolist()
        canonical.append(0)
    return canonical, {
        "raw_steps": int(len(raw)),
        "trailing_depot_count": int(len(raw) - end),
        "removed_finish_padding": max(0, int(len(raw) - end) - 1),
    }


def select_best_candidates(reward, selected_node_list, *, aug_factor, batch_size):
    """Mirror official max(POMO), then max(augmentation), gathering the same route."""
    rewards = np.asarray(reward)
    selected = np.asarray(selected_node_list)
    if rewards.ndim != 2 or selected.ndim != 3 or selected.shape[:2] != rewards.shape:
        raise ValueError("reward [A*B,P] and selected [A*B,P,T] shapes must agree")
    if rewards.shape[0] != aug_factor * batch_size:
        raise ValueError("augmented batch does not match aug_factor * batch_size")
    pomo = rewards.shape[1]
    aug_rewards = rewards.reshape(aug_factor, batch_size, pomo)
    aug_selected = selected.reshape(aug_factor, batch_size, pomo, selected.shape[2])
    best_pomo_per_aug = aug_rewards.argmax(axis=2)
    best_reward_per_aug = aug_rewards.max(axis=2)
    best_aug = best_reward_per_aug.argmax(axis=0)
    output = []
    for batch_index in range(batch_size):
        aug_index = int(best_aug[batch_index])
        pomo_index = int(best_pomo_per_aug[aug_index, batch_index])
        candidate_reward = float(aug_rewards[aug_index, batch_index, pomo_index])
        canonical, decoding = decode_selected_nodes(
            aug_selected[aug_index, batch_index, pomo_index])
        output.append({
            "best_aug_idx": aug_index,
            "best_pomo_idx": pomo_index,
            "candidate_reward": candidate_reward,
            "reported_objective": -candidate_reward,
            "canonical_solution": canonical,
            "decoding": decoding,
            "reshape_order": "augmentation-major [aug_factor,batch,pomo]",
            "node_id_mapping": "8-fold coordinate augmentation preserves original node order",
        })
    return output
