"""Small runtime helpers shared by the two MVMoE paper evaluators."""
from __future__ import annotations

import random
import time

import numpy as np


def cuda_device(value, torch):
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal MVMoE paper evaluation requires an available CUDA device")
    gpu = torch.cuda.get_device_name(device)
    if "RTX 4090" not in gpu:
        raise RuntimeError(f"formal protocol requires RTX 4090; observed {gpu!r}")
    return device


def seed_official_inference(torch, seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def solve_one(model, env, native, *, selector, problem_size, device, torch,
              timed):
    """Run one original instance with official Aug8/POMO and select its route."""
    selections, elapsed = solve_batch(
        model, env, native, selector=selector, problem_size=problem_size,
        batch_size=1, device=device, torch=torch, timed=timed)
    return selections[0], elapsed


def solve_batch(model, env, native, *, selector, problem_size, batch_size,
                device, torch, timed, depot_windows=None):
    """Run one native original-instance batch through official Aug8/POMO."""
    if batch_size not in (1, 10):
        raise ValueError("MVMoE formal original batch size is 1 or 10")
    augmented = None
    if depot_windows is not None:
        windows = np.asarray(depot_windows, dtype=np.float32)
        if windows.shape != (batch_size, 2) or not np.all(windows[:, 0] == 0):
            raise ValueError("scaled depot windows must have shape [B,2] and start at 0")
        augmented = torch.as_tensor(
            np.tile(windows, (8, 1)), dtype=torch.float32, device=device)
        # load_problems constructs depot columns from scalar attributes; the exact
        # per-instance values replace those columns immediately afterwards.
        env.depot_start, env.depot_end = 0.0, float(windows[0, 1])
    if timed:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
    with torch.no_grad():
        env.load_problems(batch_size, problems=native, aug_factor=8)
        if augmented is not None:
            env.depot_node_tw_start[:, 0] = augmented[:, 0]
            env.depot_node_tw_end[:, 0] = augmented[:, 1]
            # Official step() broadcasts this [8B,1,1] bound independently.
            env.depot_end = augmented[:, 1].reshape(8 * batch_size, 1, 1)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)
    torch.cuda.synchronize(device)
    selections = selector(
        reward.detach().cpu().numpy(),
        env.selected_node_list.detach().cpu().numpy(),
        aug_factor=8,
        batch_size=batch_size,
        problem_size=problem_size,
    )
    if len(selections) != batch_size:
        raise RuntimeError("MVMoE selector returned the wrong original batch size")
    elapsed = time.perf_counter() - started if timed else None
    return selections, elapsed


def compact_constraint_details(details, *, problem):
    """Keep audit gates while omitting large reconstructible CVRPTW timelines."""
    keys = (
        "id_range", "depot_endpoints", "no_empty_routes", "capacity",
        "capacity_tolerance", "each_customer_once", "missing_customers",
        "duplicate_customers", "route_loads", "capacity_ok",
    )
    compact = {key: details[key] for key in keys if key in details}
    if problem == "CVRPTW":
        for key in ("speed", "start_time", "time_tolerance", "time_windows_ok",
                    "time_violations"):
            if key in details:
                compact[key] = details[key]
        timelines = details.get("route_timelines", [])
        compact["depot_arrivals"] = [route["depot_arrival"] for route in timelines]
        compact["all_depot_returns_ok"] = all(
            route["depot_return_ok"] for route in timelines)
    return compact
