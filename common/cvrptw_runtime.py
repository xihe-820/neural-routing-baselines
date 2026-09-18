"""Runtime primitives shared without importing any baseline until server execution."""
from __future__ import annotations

import time

import numpy as np


def require_cuda_4090(device_name: str):
    if "RTX 4090" not in device_name:
        raise RuntimeError(f"formal CVRPTW protocol requires RTX 4090; observed {device_name!r}")


def stack_native(native_instances):
    """Concatenate adapter outputs with their existing singleton batch axis."""
    if not native_instances:
        raise ValueError("cannot build an empty CVRPTW native batch")
    keys = tuple(native_instances[0])
    if any(tuple(native) != keys for native in native_instances):
        raise ValueError("CVRPTW native instances have different fields")
    batch = {}
    for key in keys:
        values = [np.asarray(native[key]) for native in native_instances]
        if any(value.ndim == 0 or value.shape[0] != 1 for value in values):
            raise ValueError(f"native field {key!r} lacks a singleton batch axis")
        trailing = values[0].shape[1:]
        if any(value.shape[1:] != trailing or value.dtype != values[0].dtype
               for value in values):
            raise ValueError(f"native field {key!r} has inconsistent shape/dtype")
        batch[key] = np.concatenate(values, axis=0)
    return batch


def to_tensordict(native: dict, *, torch, device, expected_batch_size=None):
    from tensordict import TensorDict
    tensors = {}
    batch_size = None
    for key, value in native.items():
        tensor = torch.as_tensor(value, device=device)
        if tensor.ndim == 0:
            raise ValueError(f"native field {key!r} lacks an original-instance axis")
        if batch_size is None:
            batch_size = int(tensor.shape[0])
        elif int(tensor.shape[0]) != batch_size:
            raise ValueError("native fields disagree on original-instance batch size")
        tensors[key] = tensor
    if batch_size is None or batch_size <= 0:
        raise ValueError("native CVRPTW batch is empty")
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise ValueError(
            f"native batch size {batch_size} differs from expected {expected_batch_size}")
    return TensorDict(tensors, batch_size=[batch_size], device=device)


def timed_call(call, *, torch, device, timed=True):
    if timed:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
    value = call()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started if timed else None
    return value, elapsed


def rl4co_unbatchify_tensor(value, shape):
    """Exact tensor layout used by pinned RL4CO 0.6 ``unbatchify``.

    This intentionally mirrors ``rl4co.utils.ops._unbatchify_single`` rather
    than treating the policy's flat first dimension as reshape-compatible.
    """
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    result = value
    for repeats in reversed(shape):
        if (isinstance(repeats, bool) or not isinstance(repeats, int) or
                repeats <= 0 or result.shape[0] % repeats):
            raise RuntimeError("invalid RL4CO unbatchify shape")
        current = result.shape
        result = result.view(
            repeats, current[0] // repeats, *current[1:]
        ).permute(1, 0, *range(2, len(current) + 1))
    return result


def _require_official_tensor(out, name, expected, *, torch):
    observed = out.get(name)
    if (observed is None or tuple(observed.shape) != tuple(expected.shape) or
            not torch.equal(observed, expected)):
        raise RuntimeError(f"recomputed official {name} mismatch")


def select_rl4co_batch_output(out, *, problem_size: int, torch):
    """Audit official two-level selection independently for every original instance."""
    flat_reward = out["reward"]
    width = 8 * problem_size
    if flat_reward.ndim != 1 or flat_reward.numel() % width:
        raise RuntimeError(f"unexpected official reward shape {tuple(flat_reward.shape)}")
    reward = rl4co_unbatchify_tensor(flat_reward, (8, problem_size))
    if reward.ndim != 3 or tuple(reward.shape[1:]) != (8, problem_size):
        raise RuntimeError(f"unexpected unbatchified reward shape {tuple(reward.shape)}")
    actions = out["actions"]
    batch_size = int(reward.shape[0])
    if (actions.ndim != 4 or int(actions.shape[0]) != batch_size or
            tuple(actions.shape[1:3]) != (8, problem_size)):
        raise RuntimeError(f"unexpected official action shape {tuple(actions.shape)}")
    best_start_reward, start_indices = reward.max(dim=-1)
    _require_official_tensor(
        out, "max_reward", best_start_reward, torch=torch)
    batch_indices = torch.arange(batch_size, device=actions.device)[:, None]
    augmentation_indices = torch.arange(8, device=actions.device)[None, :]
    best_multistart_actions = actions[
        batch_indices, augmentation_indices, start_indices]
    _require_official_tensor(
        out, "best_multistart_actions", best_multistart_actions, torch=torch)
    best_reward, aug_indices = best_start_reward.max(dim=1)
    _require_official_tensor(out, "max_aug_reward", best_reward, torch=torch)
    best_aug_actions = best_multistart_actions[
        torch.arange(batch_size, device=actions.device), aug_indices]
    _require_official_tensor(
        out, "best_aug_actions", best_aug_actions, torch=torch)
    selected_actions = []
    selections = []
    for batch_index in range(batch_size):
        aug = int(aug_indices[batch_index].item())
        start = int(start_indices[batch_index, aug].item())
        selected = actions[batch_index, aug, start]
        selected_actions.append(selected)
        selections.append({
            "raw_action": selected.detach().cpu().tolist(),
            "official_reward": float(best_reward[batch_index].detach().cpu().item()),
            "selected_candidate": {
                "augmentation_index": aug, "start_index": start,
                "flat_index": aug * problem_size + start,
                "official_flat_index": aug * problem_size + start,
                "layout": (
                    "logical augmentation-major candidate identity after official "
                    "unbatchify; not the flat reward storage offset"
                ),
            },
        })
    selected_tensor = torch.stack(selected_actions, dim=0)
    if not torch.equal(selected_tensor, best_aug_actions):
        raise RuntimeError("internal selected action audit mismatch")
    return selections


def select_rl4co_output(out, *, problem_size: int, torch):
    """Backward-compatible batch-one wrapper."""
    selections = select_rl4co_batch_output(
        out, problem_size=problem_size, torch=torch)
    if len(selections) != 1:
        raise RuntimeError("batch-one RL4CO selector received multiple original instances")
    return selections[0]


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
