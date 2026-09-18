"""Runtime primitives shared without importing any baseline until server execution."""
from __future__ import annotations

import time


def require_cuda_4090(device_name: str):
    if "RTX 4090" not in device_name:
        raise RuntimeError(f"formal CVRPTW protocol requires RTX 4090; observed {device_name!r}")


def to_tensordict(native: dict, *, torch, device):
    from tensordict import TensorDict
    tensors = {}
    for key, value in native.items():
        tensors[key] = torch.as_tensor(value, device=device)
    return TensorDict(tensors, batch_size=[1], device=device)


def timed_call(call, *, torch, device, timed=True):
    if timed:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
    value = call()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started if timed else None
    return value, elapsed


def select_rl4co_output(out, *, problem_size: int, torch):
    """Audit official two-level selection and return its exact selected result."""
    reward = out["reward"]
    if reward.ndim == 1:
        reward = reward.reshape(1, 8, problem_size)
    if reward.ndim != 3 or tuple(reward.shape[1:]) != (8, problem_size):
        raise RuntimeError(f"unexpected official reward shape {tuple(reward.shape)}")
    actions = out["actions"]
    if actions.ndim != 4 or tuple(actions.shape[1:3]) != (8, problem_size):
        raise RuntimeError(f"unexpected official action shape {tuple(actions.shape)}")
    best_start_reward, start_indices = reward.max(dim=-1)
    best_reward, aug_indices = best_start_reward.max(dim=1)
    aug = int(aug_indices[0].item())
    start = int(start_indices[0, aug].item())
    selected = actions[0, aug, start]
    if "max_aug_reward" not in out or not torch.equal(best_reward, out["max_aug_reward"]):
        raise RuntimeError("recomputed official reward selection mismatch")
    if "best_aug_actions" not in out or not torch.equal(selected, out["best_aug_actions"][0]):
        raise RuntimeError("recomputed official action selection mismatch")
    return {
        "raw_action": selected.detach().cpu().tolist(),
        "official_reward": float(best_reward[0].detach().cpu().item()),
        "selected_candidate": {"augmentation_index": aug, "start_index": start,
                               "flat_index": aug * problem_size + start,
                               "official_flat_index": aug * problem_size + start,
                               "layout": "augmentation-major"},
    }


def select_cada_output(reward, captured_actions, *, problem_size: int, torch):
    """Apply the exact CaDA POMO-then-augmentation max order for batch one."""
    rewards = reward.reshape(problem_size, 8, 1)
    actions = captured_actions.reshape(problem_size, 8, 1, -1)
    best_start_reward, start_indices = rewards.max(dim=0)
    best_reward, aug_indices = best_start_reward.max(dim=0)
    aug = int(aug_indices[0].item())
    start = int(start_indices[aug, 0].item())
    selected = actions[start, aug, 0]
    return {
        "raw_action": selected.detach().cpu().tolist(),
        "official_reward": float(best_reward[0].detach().cpu().item()),
        "selected_candidate": {"augmentation_index": aug, "start_index": start,
                               "flat_index": aug * problem_size + start,
                               "official_flat_index": start * 8 + aug,
                               "layout": "CaDA start-major; flat_index is shared aug-major identity"},
    }
