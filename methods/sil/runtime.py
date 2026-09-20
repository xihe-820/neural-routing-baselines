"""Isolated invocation of pinned SIL testers with a side-effect-free solution hook."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import time

import numpy as np

from methods.sil.config import resolve_config


@contextmanager
def capture_official_solution(env, *, problem: str, problem_size: int):
    """Observe existing official objective calls without adding or changing one."""
    original = env._get_travel_distance_2
    observed = {"solution": None, "objective": None, "eligible_calls": 0}
    expected = (1, problem_size) if problem == "tsp" else (1, problem_size, 2)

    def wrapped(problems, solution, *args, **kwargs):
        result = original(problems, solution, *args, **kwargs)
        need_optimal = bool(kwargs.get("need_optimal", False))
        shape = tuple(getattr(solution, "shape", ())) if solution is not None else ()
        if not need_optimal and shape == expected and hasattr(result, "shape"):
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
        if (name in {"TSP", "CVRP", "utils"} or name.startswith("TSP.")
                or name.startswith("CVRP.") or name.startswith("utils.")):
            del sys.modules[name]


def build_tester(*, problem: str, problem_size: int, budget_label: str,
                 upstream: Path, checkpoint: Path, device, torch):
    config = resolve_config(problem, problem_size, budget_label)
    _clear_official_namespaces()
    upstream = Path(upstream).resolve()
    sys.path.insert(0, str(upstream))
    env_params = {
        "pomo_size": config["pomo_size"], "k_nearest": config["k_nearest"],
        "beam_width": config["beam_width"], "decode_method": config["decode_method"],
        "mode": "test", "data_path": None, "load_way": "allin", "sub_path": False,
        "budget": config["budget"], "PRC": config["PRC"],
        "repair_max_sub_length": config["repair_max_sub_length"],
        "random_insertion": config["random_insertion"],
    }
    model_params = dict(config["model"])
    if problem == "tsp":
        env_params.update(test_in_tsplib=False, tsplib_path=None)
        from TSP.Test_All.TSPTester import TSPTester
        tester_cls = TSPTester
    elif problem == "cvrp":
        env_params.update(test_in_vrplib=False, vrplib_path=None)
        from CVRP.Test_All.Tester import VRPTester
        tester_cls = VRPTester
    else:
        raise ValueError(problem)
    tester_params = {
        "use_cuda": True, "cuda_device_num": device.index or 0,
        "model_load": {"path": str(Path(checkpoint).resolve())},
        "test_episodes": 1, "test_batch_size": 1,
    }
    if problem == "cvrp":
        tester_params.update(augmentation_enable=False, aug_factor=1, aug_batch_size=1)
    tester = tester_cls(env_params=env_params, model_params=model_params,
                        tester_params=tester_params)
    tester.time_estimator_2.reset()
    return tester, config


def solve_one(tester, *, problem: str, problem_size: int, inject, device, torch):
    """Time one official BS1 solve and return the exact last full-solution argument."""
    inject(tester.env)
    tester.time_estimator_2.reset()
    with capture_official_solution(
            tester.env, problem=problem, problem_size=problem_size) as captured:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        returned = tester._test_one_batch(
            0, 1, tester.env_params["k_nearest"],
            tester.env_params["decode_method"], clock=tester.time_estimator_2,
            **({"logger": tester.logger} if problem == "cvrp" else {}))
        torch.cuda.synchronize(device)
        runtime = time.perf_counter() - started
    if captured["solution"] is None or captured["objective"] is None:
        raise RuntimeError("official SIL tester did not expose a final full solution")
    solution = captured["solution"].detach().cpu().numpy().copy()[0]
    official = float(captured["objective"].detach().cpu().reshape(-1)[0].item())
    if not np.isclose(float(returned[1]), official, rtol=0.0, atol=0.0):
        raise RuntimeError("captured final SIL objective differs from official tester return")
    return {
        "solution": solution,
        "official_objective": official,
        "runtime_seconds": runtime,
        "capture": {
            "hook": "temporary instance-level wrapper of env._get_travel_distance_2",
            "eligible_official_calls_observed": captured["eligible_calls"],
            "extra_solver_or_objective_calls": 0,
            "original_call_return_forwarded_unchanged": True,
            "solution_clone_excluded_from_timing": True,
        },
    }
