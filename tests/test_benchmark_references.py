"""Opt-in tests against the six hash-pinned public benchmark files, first 5 each."""
import json
import os
from pathlib import Path
import unittest
import numpy as np

from common.hashing import sha256_file
from problems.tsp.validate import validate as tsp
from problems.cvrp.validate import validate as cvrp
from problems.cvrptw.validate import validate as cvrptw

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("ML4CO_REFERENCE_TESTS") == "1", "enable ML4CO_REFERENCE_TESTS=1 in an existing Kit environment")
class PublicReferenceTests(unittest.TestCase):
    def test_six_datasets_first_five_references(self):
        import ml4co_kit as kit
        data_root = Path(os.environ.get("ML4CO_DATA_ROOT", ROOT / "datasets/ML4CO-Bench-101-SL"))
        entries = json.loads((ROOT / "manifests/public_datasets.json").read_text())["files"]
        self.assertEqual(len(entries), 6)
        for entry in entries:
            problem, n = entry["problem"], entry["target_size"]
            path = data_root / entry["filename"]
            with self.subTest(problem=problem, size=n):
                self.assertEqual(sha256_file(path), entry["sha256"])
                wrapper = getattr(kit, problem + "Wrapper")()
                wrapper.from_pickle(path)
                self.assertGreaterEqual(len(wrapper.task_list), 5)
                for i, task in enumerate(wrapper.task_list[:5]):
                    with self.subTest(instance=i):
                        self.assertEqual(task.points.shape, (n, 2))
                        if problem == "TSP":
                            result = tsp(task.points, task.ref_sol)
                        elif problem == "CVRP":
                            result = cvrp(task.depots, task.points, task.demands, task.capacity, task.ref_sol)
                        else:
                            result = cvrptw(task.depots, task.points, task.demands, task.capacity,
                                            task.tw, task.service, task.ref_sol,
                                            speed=1.0, time_tolerance=float(task.threshold))
                        self.assertTrue(result["feasible"], result["constraint_details"])
                        self.assertTrue(task.check_constraints(task.ref_sol))
                        self.assertTrue(np.isclose(result["independent_objective"], float(task.evaluate(task.ref_sol)), rtol=1e-6, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
