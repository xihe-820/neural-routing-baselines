"""Read-only invocation of pinned LEHD testers and exact solution capture."""
from __future__ import annotations

from contextlib import contextmanager
import gc
from pathlib import Path
import random
import sys
import time

import numpy as np

from methods.lehd.config import LEHD_SUBTREE, PROJECT_SEED, resolve_config
from methods.lehd.paper_results import capture_rng_state, restore_rng_state


@contextmanager
def capture_official_solution(env, *, problem: str, problem_size: int):
    """Observe official objective calls, retaining only the last eligible full solution."""
    original = env._get_travel_distance_2
    observed = {"solution": None, "objective": None, "eligible_calls": 0}
    expected = (1, problem_size) if problem == "tsp" else (1, problem_size, 2)

    def wrapped(problems, solution, *args, **kwargs):
        result = original(problems, solution, *args, **kwargs)
        shape = tuple(getattr(solution, "shape", ())) if solution is not None else ()
        if shape == expected and hasattr(result, "shape"):
            observed["solution"] = solution
            observed["objective"] = result
            observed["eligible_calls"] += 1
        return result

    env._get_travel_distance_2 = wrapped
    try:
        yield observed
    finally:
        env._get_travel_distance_2 = original


def _clear_official_namespaces():
    for name in list(sys.modules):
        if name == "LEHD" or name.startswith("LEHD."):
            del sys.modules[name]


def seed_project_rng(torch, seed: int = PROJECT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tester(*, problem: str, problem_size: int, protocol_label: str,
                 upstream: Path, checkpoint: Path, device, torch):
    config = resolve_config(problem, problem_size, protocol_label)
    upstream = Path(upstream).resolve()
    lehd_root = upstream / LEHD_SUBTREE
    if not lehd_root.is_dir():
        raise ValueError(f"official LEHD subtree is missing: {lehd_root}")
    _clear_official_namespaces()
    sys.path.insert(0, str(lehd_root.parent))
    env_params = {
        "mode": "test", "data_path": None, "sub_path": False,
        "RRC_budget": config["RRC_budget"],
    }
    if problem == "tsp":
        from LEHD.TSP.TSPTester import TSPTester
        tester_cls = TSPTester
    elif problem == "cvrp":
        from LEHD.CVRP.VRPTester import VRPTester
        tester_cls = VRPTester
    else:
        raise ValueError(problem)
    model_params = dict(config["model"])
    tester_params = {
        "use_cuda": True,
        "cuda_device_num": device.index or 0,
        "model_load": {
            "path": str(Path(checkpoint).resolve().parent),
            "epoch": config["checkpoint"]["epoch"],
        },
        "test_episodes": 1,
        "test_batch_size": 1,
    }
    tester = tester_cls(
        env_params=env_params, model_params=model_params, tester_params=tester_params)
    tester.time_estimator_2.reset()
    return tester, config


def solve_one(tester, *, problem: str, problem_size: int, inject, device, torch,
              timed: bool = True):
    """Run exactly one official BS1 batch and return the captured final incumbent."""
    inject(tester.env)
    tester.time_estimator_2.reset()
    with capture_official_solution(
            tester.env, problem=problem, problem_size=problem_size) as captured:
        if timed:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
        returned = tester._test_one_batch(
            0, 1, clock=tester.time_estimator_2,
            **({"logger": tester.logger} if problem == "cvrp" else {}))
        torch.cuda.synchronize(device)
        runtime = time.perf_counter() - started if timed else None
    if captured["solution"] is None or captured["objective"] is None:
        raise RuntimeError("official LEHD tester did not expose a final full solution")
    solution = captured["solution"].detach().cpu().numpy().copy()[0]
    official = float(captured["objective"].detach().cpu().reshape(-1)[0].item())
    if not np.isclose(float(returned[1]), official, rtol=0.0, atol=0.0):
        raise RuntimeError("captured final LEHD objective differs from official tester return")
    return {
        "solution": solution,
        "official_objective": official,
        "runtime_seconds": runtime,
        "capture": {
            "hook": "temporary instance-level wrapper of env._get_travel_distance_2",
            "selection": "last eligible full-solution objective call",
            "eligible_official_calls_observed": captured["eligible_calls"],
            "extra_solver_or_objective_calls": 0,
            "original_call_return_forwarded_unchanged": True,
            "solution_clone_excluded_from_timing": True,
        },
    }


def run_isolated_warmup(build, solve, *, torch):
    """Warm up a disposable tester and restore every RNG stream exactly."""
    formal_start = capture_rng_state(torch)
    tester = None
    try:
        tester = build()
        solve(tester)
    finally:
        tester = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        restore_rng_state(formal_start, torch)
    if capture_rng_state(torch) != formal_start:
        raise RuntimeError("LEHD isolated warm-up RNG restore was not exact")
