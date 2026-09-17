import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common.hashing import sha256_file
from methods.udc.adapter import (adapt_cvrp_task, adapt_tsp_task,
                                 decode_cyclic_routes,
                                 validate_cvrp_population,
                                 validate_tsp_population)
from methods.udc.protocol import (DATASET_FILENAMES, OFFICIAL_COMMIT,
                                  S2_SCRIPT_SHA256, discover_datasets,
                                  s2_gate)
from methods.udc.s3_eval import prior_our2_gate


class Task:
    pass


class UdcS3AdapterTests(unittest.TestCase):
    def test_tsp_adapter_preserves_order_and_dtype_audit(self):
        task = Task()
        task.points = np.arange(1000, dtype=np.float64).reshape(500, 2) / 1000
        native, evidence = adapt_tsp_task(task)
        np.testing.assert_array_equal(native, task.points.astype(np.float32))
        self.assertEqual(evidence["transformation"], "none")
        self.assertEqual(evidence["node_order"], "unchanged")

    def test_cvrp_raw_demand_is_normalized_exactly_once(self):
        task = Task()
        task.depots = np.array([[0.5, 0.5]])
        task.points = np.zeros((500, 2), dtype=np.float32)
        task.demands = np.arange(500) % 10 + 1
        task.capacity = 50
        coordinates, demand, evidence = adapt_cvrp_task(task)
        self.assertEqual(coordinates.shape, (501, 2))
        np.testing.assert_allclose(demand[1:], task.demands / 50)
        self.assertEqual(evidence["input_demand_representation"], "raw_integer")
        self.assertEqual(evidence["normalization_applied"],
                         "raw_demand / capacity exactly once")

    def test_normalized_cvrp_input_fails_closed(self):
        task = Task()
        task.depots = np.zeros((1, 2)); task.points = np.zeros((500, 2))
        task.demands = np.full(500, 0.1); task.capacity = 1
        with self.assertRaises(ValueError):
            adapt_cvrp_task(task)

    def test_cyclic_boundary_semantics(self):
        self.assertEqual(decode_cyclic_routes([1, 2, 3], [1, 0, 1]),
                         [[1], [2, 3]])

    def test_tsp_best_feasible_alpha(self):
        points = np.zeros((500, 2))
        population = np.tile(np.arange(500), (50, 1))
        population[0, 0] = 1
        result = validate_tsp_population(points, population)
        self.assertEqual(result["best_alpha"], 1)
        self.assertFalse(result["candidate_feasible"][0])
        self.assertEqual(result["solution"][0], 0)
        self.assertEqual(result["solution"][-1], 0)
        self.assertEqual(len(result["solution"]), 501)

    def test_cvrp_population_selects_feasible(self):
        points = np.column_stack((np.arange(1, 501), np.zeros(500)))
        solutions = np.tile(np.arange(1, 501), (50, 1))
        flags = np.ones((50, 500), dtype=int)
        solutions[0, 0] = 2
        result = validate_cvrp_population(
            np.array([0, 0]), points, np.ones(500), 1, solutions, flags)
        self.assertEqual(result["best_alpha"], 1)
        self.assertEqual(result["route_count"], 500)


class UdcS3ProvenanceTests(unittest.TestCase):
    def test_discovery_requires_exactly_one_file_per_family(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            for filename in DATASET_FILENAMES.values():
                (root / filename).touch()
            found = discover_datasets(root)
            self.assertEqual(set(found), {"tsp", "cvrp"})
            duplicate = root / "copy" / DATASET_FILENAMES["tsp"]
            duplicate.parent.mkdir(); duplicate.touch()
            with self.assertRaises(ValueError):
                discover_datasets(root)

    def _s2(self):
        return {
            "schema": "udc_s2_official_smoke.v1",
            "ready_for_stage_s3_ml4co_adapter": "YES",
            "script_sha256": S2_SCRIPT_SHA256,
            "project_pre": {"head": "15d02f5b95ecb418c19f12dacbe6a2acc7516fed", "pass": True},
            "project_post": {"head": "15d02f5b95ecb418c19f12dacbe6a2acc7516fed", "pass": True},
            "official_pre": {"head": OFFICIAL_COMMIT, "pass": True},
            "official_post": {"head": OFFICIAL_COMMIT, "pass": True},
            "tsp": {"final_pass": True}, "cvrp": {"final_pass": True},
        }

    def test_s2_authoritative_exact_gate(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value) / "s2_official_smoke_v3"; root.mkdir()
            (root / "metadata.json").write_text(json.dumps(self._s2()))
            self.assertTrue(s2_gate(root)["pass"])

    def test_s2_wrong_status_fails_closed(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value) / "s2_official_smoke_v3"; root.mkdir()
            payload = self._s2(); payload["cvrp"]["final_pass"] = False
            (root / "metadata.json").write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                s2_gate(root)

    def test_our5_requires_untampered_our2_artifacts(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            datasets = {}
            for family in ("tsp", "cvrp"):
                dataset = root / DATASET_FILENAMES[family]
                dataset.write_text(family)
                datasets[family] = dataset
            our2 = root / "our_2"; our2.mkdir()
            summaries = {}
            for family in ("tsp", "cvrp"):
                records = our2 / f"{family}500_records.jsonl"
                records.write_text("{}\n")
                summary = {"status": "KIT_VALIDATED", "validated_count": 2,
                           "records_sha256": sha256_file(records)}
                (our2 / f"{family}500_summary.json").write_text(json.dumps(summary))
                summaries[family] = summary
            metadata = {"state": "KIT_VALIDATED", "count": 2,
                        "script_sha256": "abc", "summaries": summaries,
                        "datasets": {family: {"sha256": sha256_file(path)}
                                     for family, path in datasets.items()}}
            (our2 / "metadata.json").write_text(json.dumps(metadata))
            self.assertTrue(prior_our2_gate(our2, datasets, "abc")["pass"])
            (our2 / "tsp500_records.jsonl").write_text("tampered\n")
            with self.assertRaises(ValueError):
                prior_our2_gate(our2, datasets, "abc")


if __name__ == "__main__":
    unittest.main()
