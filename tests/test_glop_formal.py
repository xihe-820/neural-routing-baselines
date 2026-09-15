import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common.hashing import sha256_file
from methods.glop.cvrp.adapter import validate_instance
from methods.glop.cvrp.decode import decode_subtour_coordinates
from methods.glop.paper_protocol import formal_protocol
from methods.glop.paper_results import (METADATA_FILE, SCHEMA_VERSION,
                                        VALIDATED_RECORDS_FILE, fingerprint,
                                        summarize_chunks)
from methods.glop.tsp.adapter import adapt_points
from problems.cvrp.validate import validate as validate_cvrp
from problems.tsp.validate import validate as validate_tsp


def _record(index):
    return {
        "dataset_instance_index": index, "instance_id": f"i{index}",
        "canonical_solution": [0, 1, 0], "reported_objective": 2.0,
        "independent_objective": 2.0, "reference_objective": 1.0,
        "gap_percent": 100.0, "runtime_seconds": 0.1,
        "independent_feasible": True, "reported_objective_agrees": True,
        "evidence_status": "INDEPENDENT_VERIFIED", "kit_feasible": True,
        "kit_objective": 2.0, "kit_objective_agrees": True,
    }


def _identity(root, name, indices, prepared, *, gpu="NVIDIA GeForce RTX 4090"):
    dataset = root / "dataset.pkl"
    asset = root / "reviser.pt"
    args = root / "args.json"
    for path, value in ((dataset, b"dataset"), (asset, b"asset"), (args, b"args")):
        if not path.exists():
            path.write_bytes(value)
    metadata = prepared.with_suffix(prepared.suffix + ".json")
    metadata.write_text("{}\n")
    protocol = formal_protocol("TSP", 500, name)
    return {
        "method": "GLOP", "variant": name, "problem": "TSP",
        "problem_size": 500, "official_protocol_name": name,
        "paper_protocol": protocol,
        "project": {"commit": "a" * 40, "dirty": False, "url": "project"},
        "upstream": {"commit": "b" * 40, "dirty": False, "url": "upstream"},
        "assets": {"revisers": [{
            "checkpoint_path": str(asset), "checkpoint_sha256": sha256_file(asset),
            "args_path": str(args), "args_sha256": sha256_file(args)}]},
        "dataset": {"path": str(dataset), "sha256": sha256_file(dataset),
                    "size_bytes": 1, "count": 2, "task_class": "TSPTask",
                    "coordinate_shape": [500, 2], "coordinate_range": [0, 1],
                    "reference_source": "test"},
        "prepared_input": {
            "path": str(prepared), "sha256": sha256_file(prepared),
            "metadata_path": str(metadata),
            "metadata_sha256": sha256_file(metadata),
        },
        "chunk": {"offset": min(indices), "count": len(indices),
                  "expected_indices": indices},
        "warmup": {"instances": 2, "policy": "test"},
        "rng": {**protocol["rng_semantics"],
                "shared_ri_orders_fingerprint": "0" * 64},
        "environment": {"gpu": gpu, "device": "cuda:0"},
        "source_provenance": [{
            "path": "methods/glop/paper_results.py",
            "sha256": sha256_file(
                Path(__file__).resolve().parents[1] /
                "methods/glop/paper_results.py"),
        }],
        "timing_semantics": "single-original-instance test",
    }


def _chunk(root, name, protocol, index, *, gpu="NVIDIA GeForce RTX 4090"):
    directory = root / name
    directory.mkdir()
    record = _record(index)
    path = directory / VALIDATED_RECORDS_FILE
    path.write_text(json.dumps(record) + "\n")
    prepared = root / f"{name}.npz"
    prepared.write_bytes(name.encode())
    identity = _identity(root, protocol, [index], prepared, gpu=gpu)
    metadata = {
        "schema_version": SCHEMA_VERSION, "state": "KIT_VALIDATED",
        "resume_identity": identity, "resume_fingerprint": fingerprint(identity),
        "validated_records_file": VALIDATED_RECORDS_FILE,
        "validated_records_sha256": sha256_file(path),
    }
    (directory / METADATA_FILE).write_text(json.dumps(metadata) + "\n")
    return directory


class GLOPExactMappingTests(unittest.TestCase):
    def test_tsp_duplicate_coordinate_identity_fails(self):
        points = np.zeros((1, 500, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            adapt_points(points, problem_size=500,
                         top_level_transforms=("identity",))

    def test_cvrp_duplicate_coordinate_identity_fails(self):
        depot = np.asarray([0.0, 0.0], dtype=np.float32)
        points = np.arange(2000, dtype=np.float32).reshape(1000, 2)
        points[0] = depot
        with self.assertRaisesRegex(ValueError, "AMBIGUOUS"):
            validate_instance(depot, points, np.ones(1000), 100,
                              problem_size=1000)

    def test_cvrp_preflatten_decode_preserves_routes_and_padding(self):
        depot = np.asarray([0.0, 0.0], dtype=np.float32)
        points = np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.float32)
        subtours = np.asarray([
            [[0, 0], [1, 0], [2, 0], [0, 0], [0, 0]],
            [[3, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        ], dtype=np.float32)
        canonical, routes = decode_subtour_coordinates(subtours, depot, points)
        self.assertEqual(routes, [[0, 1, 2, 0], [0, 3, 0]])
        self.assertEqual(canonical, [0, 1, 2, 0, 3, 0])

    def test_invalid_tsp_and_cvrp_solutions_fail(self):
        tsp = validate_tsp(np.arange(1000).reshape(500, 2), [0, 1, 0])
        self.assertFalse(tsp["feasible"])
        depot = np.asarray([0, 0])
        points = np.asarray([[1, 0], [2, 0]])
        duplicate = validate_cvrp(depot, points, [2, 2], 3, [0, 1, 1, 0])
        missing = validate_cvrp(depot, points, [2, 2], 3, [0, 1, 0])
        capacity = validate_cvrp(depot, points, [2, 2], 3, [0, 1, 2, 0])
        self.assertFalse(duplicate["feasible"])
        self.assertFalse(missing["feasible"])
        self.assertFalse(capacity["feasible"])


class GLOPAggregationTests(unittest.TestCase):
    def test_mixed_standard_and_more_chunks_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            standard = _chunk(root, "standard", "official_standard", 0)
            more = _chunk(root, "more", "official_more", 1)
            with self.assertRaisesRegex(ValueError, "mixed GLOP protocol"):
                summarize_chunks([standard, more])

    def test_complete_consistent_chunks_compute_per_instance_means(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _chunk(root, "first", "official_standard", 0)
            second = _chunk(root, "second", "official_standard", 1)
            result = summarize_chunks([first, second])
            self.assertEqual(result["status"], "PAPER_READY")
            self.assertNotIn("manuscript_hardware_consistency", result)
            self.assertEqual(result["drop_mean_per_instance_gap_percent"], 100.0)

    def test_non_rtx4090_hardware_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _chunk(
                root, "first", "official_standard", 0,
                gpu="NVIDIA A100-SXM4-80GB")
            with self.assertRaisesRegex(ValueError, "RTX 4090"):
                summarize_chunks([first])


if __name__ == "__main__":
    unittest.main()
