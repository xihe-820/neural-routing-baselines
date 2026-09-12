import unittest
import numpy as np

from problems.tsp.validate import validate as tsp, decode_successors
from problems.cvrp.validate import validate as cvrp
from problems.cvrptw.validate import validate as cvrptw


class TSPTests(unittest.TestCase):
    points = [[0, 0], [1, 0], [1, 1], [0, 1]]

    def test_valid_implicit_and_explicit(self):
        for tour in ([0, 1, 2, 3], [2, 3, 0, 1, 2]):
            result = tsp(self.points, tour)
            self.assertTrue(result["feasible"])
            self.assertEqual(result["independent_objective"], 4.0)

    def test_duplicate(self):
        self.assertFalse(tsp(self.points, [0, 1, 1, 3])["feasible"])

    def test_missing(self):
        self.assertFalse(tsp(self.points, [0, 1, 2])["feasible"])

    def test_invalid_ids(self):
        for tour in ([0, 1, 2, 4], [-1, 1, 2, 3], [0., 1., 2., 3.], [False, True, False, True], [0, True, 2, 3]):
            self.assertFalse(tsp(self.points, tour)["feasible"])

    def test_bad_closure(self):
        self.assertFalse(tsp(self.points, [0, 1, 2, 3, 1])["feasible"])

    def test_single_successor_cycle(self):
        self.assertEqual(decode_successors([1, 2, 3, 0]), [0, 1, 2, 3, 0])
        self.assertTrue(tsp(self.points, [1, 2, 3, 0], representation="successors")["feasible"])

    def test_disjoint_subtours(self):
        self.assertFalse(tsp(self.points, [1, 0, 3, 2], representation="successors")["feasible"])

    def test_nonfinite_or_bad_shape(self):
        for points in ([[0, np.nan]], [[0, 0, 0]], []):
            self.assertFalse(tsp(points, [0])["feasible"])


class CVRPTests(unittest.TestCase):
    def run_case(self, solution, demands=(2, 3), capacity=5):
        return cvrp([0, 0], [[1, 0], [2, 0]], demands, capacity, solution)

    def test_valid_one_route_exact_capacity(self):
        r = self.run_case([0, 1, 2, 0])
        self.assertTrue(r["feasible"])
        self.assertEqual(r["independent_objective"], 4.)
        self.assertEqual(r["constraint_details"]["route_loads"], [5.])

    def test_valid_multi_route(self):
        r = self.run_case([0, 1, 0, 2, 0], capacity=3)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["independent_objective"], 6.)

    def test_duplicate(self):
        r = self.run_case([0, 1, 2, 1, 0], capacity=10)
        self.assertFalse(r["feasible"])
        self.assertEqual(r["constraint_details"]["duplicate_customers"], [1])

    def test_missing(self):
        r = self.run_case([0, 1, 0])
        self.assertFalse(r["feasible"])
        self.assertEqual(r["constraint_details"]["missing_customers"], [2])

    def test_capacity_violation_uses_raw_units(self):
        self.assertFalse(self.run_case([0, 1, 2, 0], capacity=4.99)["feasible"])

    def test_explicit_capacity_tolerance(self):
        r = cvrp([0, 0], [[1, 0]], [1.000001], 1, [0, 1, 0], capacity_tolerance=1e-5)
        self.assertTrue(r["feasible"])

    def test_bad_depot_boundaries_or_ids(self):
        for sol in ([1, 2, 0], [0, 1, 2], [0, 1, 0, 0, 2, 0], [0, 1, -1, 2, 0], [0, 1, 3, 0]):
            self.assertFalse(self.run_case(sol)["feasible"])

    def test_bad_instance(self):
        for demands, capacity in [([1], 5), ([2, -1], 5), ([2, np.nan], 5), ([2, 3], 0)]:
            self.assertFalse(self.run_case([0, 1, 2, 0], demands, capacity)["feasible"])


class CVRPTWTests(unittest.TestCase):
    def run_case(self, *, tw=None, service=(0, 0, 0), solution=(0, 1, 2, 0),
                 demands=(1, 1), capacity=2, speed=1):
        return cvrptw([0, 0], [[1, 0], [2, 0]], demands, capacity,
                      tw if tw is not None else [[0, 20], [0, 20], [0, 20]],
                      service, solution, speed=speed)

    def test_waiting_and_early_arrival(self):
        r = self.run_case(tw=[[0, 20], [3, 4], [0, 20]])
        self.assertTrue(r["feasible"])
        events = r["constraint_details"]["route_timelines"][0]["events"]
        self.assertEqual(events[0]["arrival"], 1)
        self.assertEqual(events[0]["waiting"], 2)
        self.assertEqual(events[0]["service_start"], 3)
        self.assertEqual(events[1]["arrival"], 4)
        self.assertEqual(r["independent_objective"], 4)  # Waiting is not distance.

    def test_exact_boundary(self):
        r = self.run_case(tw=[[0, 4], [1, 1], [2, 2]])
        self.assertTrue(r["feasible"])
        self.assertEqual(r["constraint_details"]["route_timelines"][0]["depot_arrival"], 4)

    def test_late_service_start(self):
        r = self.run_case(tw=[[0, 20], [0, .9], [0, 20]])
        self.assertFalse(r["feasible"])
        self.assertEqual(r["constraint_details"]["time_violations"][0]["type"], "late_service_start")

    def test_service_causes_downstream_violation(self):
        tw = [[0, 20], [0, 2], [0, 3]]
        self.assertTrue(self.run_case(tw=tw)["feasible"])
        self.assertFalse(self.run_case(tw=tw, service=[0, 2, 0])["feasible"])

    def test_service_may_finish_after_customer_tw_end(self):
        self.assertTrue(self.run_case(tw=[[0, 20], [0, 1], [0, 20]], service=[0, 2, 0])["feasible"])

    def test_depot_return_violation(self):
        r = self.run_case(tw=[[0, 3.9], [0, 20], [0, 20]])
        self.assertFalse(r["feasible"])
        self.assertEqual(r["constraint_details"]["time_violations"][-1]["type"], "late_depot_return")

    def test_capacity_duplicate_missing(self):
        self.assertFalse(self.run_case(capacity=1)["feasible"])
        self.assertFalse(self.run_case(solution=[0, 1, 1, 2, 0], capacity=10)["feasible"])
        self.assertFalse(self.run_case(solution=[0, 1, 0])["feasible"])

    def test_speed_changes_time_not_objective(self):
        r = self.run_case(speed=2, tw=[[0, 2], [0, 1], [0, 1]])
        self.assertTrue(r["feasible"])
        self.assertEqual(r["independent_objective"], 4)

    def test_each_route_restarts_clock(self):
        r = self.run_case(solution=[0, 1, 0, 2, 0], tw=[[0, 4], [0, 1], [0, 2]])
        self.assertTrue(r["feasible"])

    def test_malformed_time_fields(self):
        for kw in [dict(tw=[[0, 5], [0, 5]]), dict(service=[0, -1, 0]), dict(speed=0),
                   dict(tw=[[0, 20], [5, 4], [0, 20]])]:
            self.assertFalse(self.run_case(**kw)["feasible"])


if __name__ == "__main__":
    unittest.main()
