"""Exact ML4CO-Kit validation helper for NeuOpt TSP100 evidence."""
from __future__ import annotations

import numpy as np

from common.objective_agreement import objective_agrees


def validate_task(task, canonical_tour, *, independent_objective, reference_objective,
                  kit_module):
    if type(task) is not kit_module.TSPTask:
        raise ValueError("benchmark wrapper returned a non-exact ML4CO TSPTask")
    solution = np.asarray(canonical_tour, dtype=np.int64)
    feasible = bool(task.check_constraints(solution))
    objective = float(task.evaluate(solution))
    kit_reference = float(task.evaluate(task.ref_sol))
    if not feasible or not objective_agrees(objective, independent_objective):
        raise RuntimeError("ML4CO-Kit validation disagrees with independent validation")
    if not objective_agrees(kit_reference, reference_objective):
        raise ValueError("ML4CO-Kit reference objective differs from pinned dataset evidence")
    return {
        "kit_feasible": True,
        "kit_objective": objective,
        "kit_reference_objective": kit_reference,
        "kit_objective_agrees": True,
    }
