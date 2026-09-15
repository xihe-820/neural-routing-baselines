import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common.hashing import sha256_file
from methods.glop.cvrp.adapter import validate_instance
from methods.glop.cvrp.decode import decode_subtour_coordinates
from methods.glop.paper_protocol import formal_protocol
from methods.glop.paper_results import (METADATA_FILE, RECORDS_FILE,
                                        SCHEMA_VERSION,
                                        TSP_TIMING_SEMANTICS,
                                        VALIDATED_RECORDS_FILE, append_record,
                                        finalize_chunk, fingerprint,
                                        initialize_chunk, read_jsonl,
                                        summarize_chunks,
                                        tsp_runtime_accounting)
from methods.glop.tsp.adapter import adapt_points
from problems.cvrp.objective import route_distance
from problems.cvrp.validate import validate as validate_cvrp
from problems.tsp.validate import validate as validate_tsp


def _record(index, *, shared_setup=None, runtime=0.1):
    if shared_setup is None:
        shared_setup = 0.025 if index == 0 else 0.0
    solve = runtime - shared_setup
    return {
        "dataset_instance_index": index, "instance_id": f"i{index}",
        "canonical_solution": [0, 1, 0], "reported_objective": 2.0,
        "independent_objective": 2.0, "reference_objective": 1.0,
        "gap_percent": 100.0, "runtime_seconds": runtime,
        "runtime_components": {
            "per_instance_solve_seconds": solve,
            "shared_ri_order_generation_seconds": shared_setup,
        },
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
        "timing_semantics": TSP_TIMING_SEMANTICS,
    }


def _chunk(root, name, protocol, index, *, gpu="NVIDIA GeForce RTX 4090",
           shared_setup=None, runtime=0.1):
    directory = root / name
    directory.mkdir()
    record = _record(index, shared_setup=shared_setup, runtime=runtime)
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
    @staticmethod
    def _cvrp_subtour(ids):
        depot = np.asarray([0.0, 0.0], dtype=np.float32)
        points = np.asarray([
            [1.0, 0.0], [2.0, 0.0], [0.0, 1.0],
            [0.0, 2.0], [1.0, 1.0], [2.0, 1.0],
        ], dtype=np.float32)
        coordinates = np.concatenate((depot[None], points), axis=0)
        return depot, points, coordinates[np.asarray(ids, dtype=np.int64)][None]

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

    def test_cvrp_separated_depots_preserve_cyclic_customer_runs(self):
        depot, points, subtour = self._cvrp_subtour(
            [5, 6, 0, 0, 1, 2, 0, 0, 0, 3, 4])
        canonical, routes = decode_subtour_coordinates(subtour, depot, points)
        self.assertEqual(routes, [[0, 1, 2, 0], [0, 3, 4, 5, 6, 0]])
        self.assertEqual(canonical, [0, 1, 2, 0, 3, 4, 5, 6, 0])
        self.assertNotIn([0, 0], routes)

    def test_cvrp_consecutive_depots_do_not_emit_empty_routes(self):
        depot, points, subtour = self._cvrp_subtour(
            [0, 0, 0, 1, 0, 0, 2, 0])
        canonical, routes = decode_subtour_coordinates(subtour, depot, points)
        self.assertEqual(routes, [[0, 1, 0], [0, 2, 0]])
        self.assertEqual(canonical, [0, 1, 0, 2, 0])

    def test_cvrp_all_zero_subtour_fails_closed(self):
        depot, points, subtour = self._cvrp_subtour([0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "no customer"):
            decode_subtour_coordinates(subtour, depot, points)

    def test_cvrp_subtour_without_depot_fails_closed(self):
        depot, points, subtour = self._cvrp_subtour([1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "no depot"):
            decode_subtour_coordinates(subtour, depot, points)

    def test_cvrp_depot_split_preserves_official_cycle_objective(self):
        depot, points, subtour = self._cvrp_subtour(
            [5, 6, 0, 0, 1, 2, 0, 0, 0, 3, 4])
        canonical, _ = decode_subtour_coordinates(subtour, depot, points)
        row = subtour[0]
        official_cycle = float(np.linalg.norm(
            np.roll(row, -1, axis=0) - row, axis=1).sum(dtype=np.float64))
        self.assertAlmostEqual(
            official_cycle, route_distance(depot, points, canonical), places=7)

    def test_cvrp_real_failure_zero_run_topology_yields_two_routes(self):
        depot, points, subtour = self._cvrp_subtour(
            [5, 6, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 3, 4])
        canonical, routes = decode_subtour_coordinates(subtour, depot, points)
        self.assertEqual(routes, [[0, 1, 2, 0], [0, 3, 4, 5, 6, 0]])
        self.assertEqual(len(routes), 2)
        self.assertEqual(canonical, [0, 1, 2, 0, 3, 4, 5, 6, 0])

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
    def test_tsp_runtime_accounting_charges_only_index_zero(self):
        zero_runtime, zero_components = tsp_runtime_accounting(0, 0.8, 0.2)
        later_runtime, later_components = tsp_runtime_accounting(1, 0.8, 0.2)
        self.assertEqual(zero_runtime, 1.0)
        self.assertEqual(
            zero_components["shared_ri_order_generation_seconds"], 0.2)
        self.assertEqual(later_runtime, 0.8)
        self.assertEqual(
            later_components["shared_ri_order_generation_seconds"], 0.0)

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
            self.assertEqual(result["time_mean_single_instance_seconds"], 0.1)

    def test_nonzero_chunk_shared_order_charge_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _chunk(root, "first", "official_standard", 0)
            second = _chunk(
                root, "second", "official_standard", 1,
                shared_setup=0.01)
            with self.assertRaisesRegex(ValueError, "only be charged"):
                summarize_chunks([first, second])

    def test_runtime_component_sum_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _chunk(root, "first", "official_standard", 0)
            second = _chunk(root, "second", "official_standard", 1)
            path = second / VALIDATED_RECORDS_FILE
            record = json.loads(path.read_text())
            record["runtime_components"]["per_instance_solve_seconds"] = 0.2
            path.write_text(json.dumps(record) + "\n")
            metadata_path = second / METADATA_FILE
            metadata = json.loads(metadata_path.read_text())
            metadata["validated_records_sha256"] = sha256_file(path)
            metadata_path.write_text(json.dumps(metadata) + "\n")
            with self.assertRaisesRegex(ValueError, "runtime_components sum"):
                summarize_chunks([first, second])

    def test_resume_with_completed_index_zero_preserves_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared.npz"
            prepared.write_bytes(b"prepared")
            identity = _identity(
                root, "official_standard", [0], prepared)
            output = root / "output"
            initialize_chunk(output, identity)
            append_record(output, _record(0, shared_setup=0.03, runtime=0.11))
            before = (output / RECORDS_FILE).read_bytes()
            _, completed = initialize_chunk(output, identity)
            self.assertEqual(completed, {0})
            finalize_chunk(output)
            self.assertEqual((output / RECORDS_FILE).read_bytes(), before)
            resumed = read_jsonl(output / RECORDS_FILE)[0]
            self.assertEqual(resumed["runtime_seconds"], 0.11)
            self.assertEqual(
                resumed["runtime_components"][
                    "shared_ri_order_generation_seconds"], 0.03)

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
