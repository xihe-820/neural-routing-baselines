from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from common.objective_agreement import objective_agrees
from methods.mvmoe.cvrptw.adapter import adapt_batch
from methods.mvmoe.cvrptw.scaling import assert_continuous_env, scale_instance
from methods.mvmoe.cvrptw.scaling_audit import (
    _slack_diagnostics, summarize, verify_project_provenance)
from problems.cvrp.objective import route_distance
from problems.cvrptw.validate import validate


def instance(problem_size=2):
    depot = np.asarray([0.0, 0.0], dtype=np.float32)
    points = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    if problem_size != 2:
        points = np.stack((np.linspace(0.01, 0.99, problem_size),
                           np.linspace(0.99, 0.01, problem_size)), axis=1).astype(np.float32)
    demands = np.full(problem_size, 8.0, dtype=np.float32)
    tw = np.zeros((problem_size + 1, 2), dtype=np.float32)
    tw[0] = [0.0, 4.6]
    tw[1:, 1] = 4.0
    service = np.zeros(problem_size + 1, dtype=np.float32)
    service[1:] = 0.16
    return depot, points, demands, np.float32(40.0), tw, service


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True,
        text=True).stdout.strip()


def make_git_repo():
    temporary = tempfile.TemporaryDirectory()
    repo = Path(temporary.name)
    git(repo, "init", "-q", "-b", "master")
    git(repo, "config", "user.email", "audit-test@example.invalid")
    git(repo, "config", "user.name", "Audit Test")
    git(repo, "remote", "add", "origin", "https://github.com/test/audit.git")
    git(repo, "commit", "--allow-empty", "-q", "-m", "formal base")
    base = git(repo, "rev-parse", "HEAD")
    allowed = repo / "methods/mvmoe/cvrptw/scaling_audit.py"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("audit\n")
    git(repo, "add", str(allowed.relative_to(repo)))
    git(repo, "commit", "-q", "-m", "audit implementation")
    return temporary, repo, base


class ContinuousScalingTests(unittest.TestCase):
    def test_uniform_scaling_and_depot_horizon(self):
        data = instance()
        scaled = scale_instance(*data)
        expected = np.float32(4.6) / 3.0
        self.assertAlmostEqual(scaled["scaler"], expected)
        np.testing.assert_allclose(scaled["depot"], data[0] / expected)
        np.testing.assert_allclose(scaled["points"], data[1] / expected)
        np.testing.assert_allclose(scaled["time_windows"], data[4] / expected)
        np.testing.assert_allclose(scaled["service_times"], data[5] / expected)
        self.assertAlmostEqual(float(scaled["time_windows"][0, 1]), 3.0, places=6)

    def test_demand_is_not_scaled_and_adapter_normalizes_once(self):
        data = instance(problem_size=50)
        scaled = scale_instance(*data)
        np.testing.assert_array_equal(scaled["raw_demands"], data[2])
        native, _, mapping = adapt_batch(
            scaled["depot"][None], scaled["points"][None],
            scaled["raw_demands"][None], np.asarray([scaled["raw_capacity"]]),
            scaled["time_windows"][None], scaled["service_times"][None],
            problem_size=50, device="cpu")
        np.testing.assert_allclose(native[2].numpy(), np.full((1, 50), 0.2))
        self.assertIn("exactly once", mapping["normalization"])

    def test_loc_scaler_must_be_none(self):
        assert_continuous_env(SimpleNamespace(loc_scaler=None))
        with self.assertRaisesRegex(RuntimeError, "loc_scaler is None"):
            assert_continuous_env(SimpleNamespace(loc_scaler=1.5))
        with self.assertRaisesRegex(RuntimeError, "loc_scaler is None"):
            assert_continuous_env(SimpleNamespace())

    def test_objective_equivalence_and_node_order(self):
        data = instance()
        scaled = scale_instance(*data)
        route = [0, 1, 0, 2, 0]
        original = route_distance(data[0], data[1], route)
        scaled_distance = route_distance(scaled["depot"], scaled["points"], route)
        self.assertTrue(objective_agrees(scaled_distance * scaled["scaler"], original))
        np.testing.assert_allclose(
            scaled["points"] * scaled["scaler"], data[1], rtol=1e-6, atol=1e-6)
        self.assertEqual(route, [0, 1, 0, 2, 0])

    def test_waiting_route_feasibility_is_equivalent(self):
        depot = np.asarray([0.0, 0.0], dtype=np.float32)
        points = np.asarray([[1.0, 0.0]], dtype=np.float32)
        demands = np.asarray([1.0], dtype=np.float32)
        tw = np.asarray([[0.0, 5.0], [2.0, 3.0]], dtype=np.float32)
        service = np.asarray([0.0, 0.5], dtype=np.float32)
        route = [0, 1, 0]
        scaled = scale_instance(depot, points, demands, 10.0, tw, service)
        original_result = validate(
            depot, points, demands, 10.0, tw, service, route,
            speed=1.0, time_tolerance=0.0)
        scaled_result = validate(
            scaled["depot"], scaled["points"], demands, 10.0,
            scaled["time_windows"], scaled["service_times"], route,
            speed=1.0, time_tolerance=0.0)
        self.assertTrue(original_result["feasible"])
        self.assertTrue(scaled_result["feasible"])
        self.assertGreater(
            original_result["constraint_details"]["route_timelines"][0]["events"][0]["waiting"],
            0.0)

    def test_official_coordinate_maximum_dominates_when_larger(self):
        data = list(instance())
        data[1][0, 0] = 6.0
        scaled = scale_instance(*data)
        self.assertEqual(scaled["scaler"], 6.0)
        self.assertAlmostEqual(float(scaled["points"].max()), 1.0)

    def test_summary_is_paired_and_epsilon_diagnostic_is_explicit(self):
        def record(a, b, same, independent, kit, error, runtime, near):
            return {
                "comparison_to_A": {
                    "A_original_objective": a, "B_original_objective": b,
                    "A_gap": a, "B_gap": b, "same_route_as_A": same,
                    "A_runtime_seconds": 0.5,
                },
                "validation": {
                    "independent_feasible": independent, "kit_feasible": kit},
                "objectives": {"scaled_times_s_original_abs_error": error},
                "runtime_seconds": runtime,
                "epsilon_diagnostics": {"within_10x_official_epsilon": near},
            }
        result = summarize([
            record(10.0, 9.0, False, True, True, 2e-6, 0.4, False),
            record(10.0, 10.0, True, True, True, 1e-6, 0.6, True),
        ])
        self.assertEqual(result["B_better_count"], 1)
        self.assertEqual(result["tie_count"], 1)
        self.assertEqual(result["same_route_count"], 1)
        self.assertEqual(result["B_independent_feasible_count"], 2)
        self.assertEqual(result["B_kit_feasible_count"], 2)
        self.assertEqual(result["max_scaled_times_s_original_abs_error"], 2e-6)
        self.assertEqual(result["B_mean_runtime_seconds"], 0.5)
        self.assertEqual(result["epsilon_boundary_observation_count"], 1)

        validation = {"constraint_details": {"route_timelines": [{
            "depot_tw_end": 4.6, "depot_arrival": 4.5,
            "events": [{"tw_end": 2.0, "service_start": 1.99999}],
        }]}}
        diagnostic = _slack_diagnostics(validation, 4.6 / 3.0)
        self.assertFalse(diagnostic["uses_positive_official_epsilon_to_pass"])
        self.assertTrue(diagnostic["within_10x_official_epsilon"])


class ProjectProvenanceGateTests(unittest.TestCase):
    ALLOWLIST = {"methods/mvmoe/cvrptw/scaling_audit.py"}

    def test_ancestor_with_only_allowlisted_delta_passes(self):
        temporary, repo, base = make_git_repo()
        self.addCleanup(temporary.cleanup)
        project, gate = verify_project_provenance(
            repo, formal_base_commit=base, allowed_paths=self.ALLOWLIST)
        self.assertFalse(project["dirty"])
        self.assertEqual(gate["formal_pipeline_base_commit"], base)
        self.assertEqual(gate["audit_implementation_commit"], project["commit"])
        self.assertEqual(
            gate["base_to_head_changed_paths"], sorted(self.ALLOWLIST))

    def test_non_audit_changed_path_fails(self):
        temporary, repo, base = make_git_repo()
        self.addCleanup(temporary.cleanup)
        (repo / "README.md").write_text("formal change\n")
        git(repo, "add", "README.md")
        git(repo, "commit", "-q", "-m", "non-audit change")
        with self.assertRaisesRegex(ValueError, "non-audit paths"):
            verify_project_provenance(
                repo, formal_base_commit=base, allowed_paths=self.ALLOWLIST)

    def test_dirty_project_fails(self):
        temporary, repo, base = make_git_repo()
        self.addCleanup(temporary.cleanup)
        path = repo / "methods/mvmoe/cvrptw/scaling_audit.py"
        path.write_text("uncommitted\n")
        with self.assertRaisesRegex(ValueError, "clean project working tree"):
            verify_project_provenance(
                repo, formal_base_commit=base, allowed_paths=self.ALLOWLIST)

    def test_non_ancestor_base_fails(self):
        temporary, repo, base = make_git_repo()
        self.addCleanup(temporary.cleanup)
        git(repo, "branch", "side", base)
        git(repo, "checkout", "-q", "side")
        git(repo, "commit", "--allow-empty", "-q", "-m", "side commit")
        side = git(repo, "rev-parse", "HEAD")
        git(repo, "checkout", "-q", "master")
        with self.assertRaisesRegex(ValueError, "not an ancestor"):
            verify_project_provenance(
                repo, formal_base_commit=side, allowed_paths=self.ALLOWLIST)


if __name__ == "__main__":
    unittest.main()
