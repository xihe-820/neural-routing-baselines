import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from common.cvrptw_artifacts import append_batch, finalize, initialize
from common.cvrptw_artifacts import scope_instance_count
from common.cvrptw_formal import DATASETS
from common.cvrptw_runtime import stack_native
from methods.symnco.cvrptw.adapter import (
    adapt_instance, historical_raw_instance_id, raw_ids_from_native)
from methods.symnco.cvrptw.config import (
    BASE_IDENTITY, CHECKPOINTS, CHECKPOINT_HASHES, SNAPSHOT_FILES,
    SNAPSHOT_MANIFEST_SHA256, TIMING_SEMANTICS, protocol,
    validate_historical_record, validate_historical_report)
from methods.symnco.cvrptw.decode import decode_selected_action
from methods.symnco.cvrptw.official_runtime import select_e1_batch


def instance(size, offset=0.0):
    capacity = DATASETS[size]["capacity"]
    depot = np.asarray([0.2 + offset, 0.3], dtype=np.float32)
    points = np.linspace(0.01, 0.99, size * 2, dtype=np.float32).reshape(size, 2)
    demand = np.resize(np.arange(1, 10, dtype=np.float32), size)
    tw = np.zeros((size + 1, 2), dtype=np.float32)
    tw[:, 1] = np.float32(4.6)
    service = np.zeros(size + 1, dtype=np.float32)
    service[1:] = np.float32(0.15)
    return depot, points, demand, capacity, tw, service


def symnco_record(index, size=50, augmentation=0, runtime=0.1):
    customers = list(range(1, size + 1))
    return {
        "dataset_instance_index": index, "instance_id": f"raw-{index}",
        "raw_official_action": [0, *customers, 0],
        "canonical_solution": [0, *customers, 0],
        "selected_candidate": {
            "augmentation_index": augmentation, "candidate_index": 0,
            "rollout_index": 0, "flat_index": augmentation,
        },
        "official_reward": -2.0, "reported_objective": 2.0,
        "independent_objective": 2.0, "kit_objective": 2.0,
        "reference_objective": 1.0, "instance_drop_percent": 100.0,
        "runtime_seconds": runtime, "independent_feasible": True,
        "kit_feasible": True, "reported_objective_agrees": True,
        "kit_objective_agrees": True, "route_count": 1,
        "route_loads": [1.0], "route_timelines": [], "status": "KIT_VALIDATED",
    }


class SymNCOConfigAdapterTests(unittest.TestCase):
    def test_dataset_and_checkpoint_identities(self):
        self.assertEqual(DATASETS[200], {
            "filename": "cvrptw200_pyvrp-60s_41.597.pkl",
            "sha256": "8cd550a9316d8e15832780b7c5c0f26ed539bde0b4df4a5f99918ee57ec16c8d",
            "count": 100, "capacity": 80.0,
        })
        self.assertEqual(set(CHECKPOINTS), {50, 100, 200})
        self.assertEqual(CHECKPOINT_HASHES[50], "6be6ed8c5b40330db0f2605f2cf354ee006cf8d5d6564726fd904a0b2bcf1395")
        self.assertEqual(CHECKPOINT_HASHES[100], "e4aad3807cf75864e561913905cf350cd986be01aea714c8cab8a6f1a27e8144")
        self.assertEqual(CHECKPOINT_HASHES[200], "23c18b2ab2ff966c90a07bae51f63fcf1f5c33e9a56018c0d972204a32207479")
        self.assertEqual(scope_instance_count("production", 1, DATASETS[200]["count"]), 100)
        self.assertEqual(scope_instance_count("production", 10, DATASETS[200]["count"]) // 10, 10)

    def test_e1_protocol_is_frozen_for_all_sizes_and_batches(self):
        for size in (50, 100, 200):
            for batch_size in (1, 10):
                value = protocol(size, batch_size)
                self.assertEqual(value["original_instance_batch_size"], batch_size)
                self.assertEqual(value["num_augmentations"], 4)
                self.assertEqual(value["num_rollouts"], 1)
                self.assertEqual(value["decode_type"], "greedy")
                self.assertEqual(value["seed"], 1234)
                self.assertEqual(value["forced_customer_starts"], 0)
                self.assertIs(value["local_search"], False)
        with self.assertRaises(ValueError):
            protocol(200, 2)

    def test_adapter_preserves_raw_fields_and_true_batch_axis(self):
        for size in (50, 100, 200):
            native, mapping = adapt_instance(*instance(size), problem_size=size)
            self.assertEqual(native["coords"].shape, (1, size + 1, 2))
            self.assertEqual(native["demand_raw"].shape, (1, size + 1))
            self.assertEqual(native["capacity"].shape, (1,))
            np.testing.assert_array_equal(native["time_windows"][0], instance(size)[4])
            np.testing.assert_allclose(
                native["demand_norm"][0, 1:], instance(size)[2] / instance(size)[3])
            self.assertIn("exactly once", mapping["demand"])
        rows = [adapt_instance(*instance(50, index / 100), problem_size=50)[0]
                for index in range(10)]
        batch = stack_native(rows)
        self.assertEqual(batch["coords"].shape, (10, 51, 2))
        self.assertEqual(len(set(raw_ids_from_native(batch))), 10)

    def test_historical_raw_id_matches_txt_numeric_hash_rule(self):
        values = instance(50)
        observed = historical_raw_instance_id(*values)
        digest = hashlib.sha256()
        for field in (*values[:3], np.asarray([values[3]]), *values[4:]):
            parsed_txt_numbers = np.asarray(
                [float(str(value)) for value in np.asarray(field).reshape(-1)], dtype="<f8")
            digest.update(parsed_txt_numbers.tobytes())
        self.assertEqual(observed, digest.hexdigest())

    def test_snapshot_manifest_and_timing_identity_are_stable(self):
        encoded = json.dumps(SNAPSHOT_FILES, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), SNAPSHOT_MANIFEST_SHA256)
        self.assertIn("138c95e", BASE_IDENTITY)
        self.assertIn("identity plus three", TIMING_SEMANTICS)
        self.assertIn("candidate selection", TIMING_SEMANTICS)

    def test_decoder_neither_repairs_nor_rewraps_routes(self):
        route = [0, 1, 0, 2, 0]
        self.assertEqual(decode_selected_action(route, 2), route)
        with self.assertRaises(ValueError):
            decode_selected_action([1, 2, 0], 2)


class SymNCOSelectionRegressionTests(unittest.TestCase):
    def test_candidates_are_isolated_per_original(self):
        import torch
        batch_size, steps = 10, 5
        actions = torch.zeros(batch_size, 4, 1, steps, dtype=torch.long)
        lengths = torch.full((batch_size, 4, 1), steps, dtype=torch.long)
        expected = []
        samples = []
        for batch_index in range(batch_size):
            target = batch_index % 4
            expected.append(target)
            samples.append({"instance_id": f"id-{batch_index}"})
            for augmentation in range(4):
                actions[batch_index, augmentation, 0, 0] = 100 + augmentation
            actions[batch_index, target, 0, 0] = batch_index

        def fake_select(sample, row_actions, row_lengths):
            del row_lengths
            candidates = [(int(row_actions[a, 0, 0]), a) for a in range(4)]
            cost, augmentation = min(candidates)
            return {"feasible": True, "cost": float(cost),
                    "tour": [0, 1, 0], "candidate": [augmentation, 0, 0]}

        selected = select_e1_batch(samples, actions, lengths, fake_select)
        self.assertEqual(len(selected), batch_size)
        for index, row in enumerate(selected):
            self.assertEqual(row["instance_id"], f"id-{index}")
            self.assertEqual(row["selected_candidate"]["augmentation_index"], expected[index])
            self.assertEqual(row["selected_candidate"]["flat_index"], expected[index])

    def test_historical_bs1_record_regression_is_route_level(self):
        record = symnco_record(0)
        report = {
            "protocol": "E1", "decode": "greedy", "A": 4, "K": 1,
            "local_search": False, "seed": 1234, "forced_customer_starts": 0,
            "instances": 1,
            "rows": [{"row_index": 0, "instance_id": "raw-0", "feasible": True,
                      "tour": record["canonical_solution"], "candidate": [0, 0, 0],
                      "cost": 2.0}],
        }
        validate_historical_report(report, problem_size=50, dataset_count=1)
        validate_historical_record(report, record, dataset_index=0)
        changed = json.loads(json.dumps(record))
        changed["canonical_solution"][1:3] = reversed(changed["canonical_solution"][1:3])
        with self.assertRaisesRegex(RuntimeError, "tour"):
            validate_historical_record(report, changed, dataset_index=0)


class SymNCOArtifactTests(unittest.TestCase):
    def identity(self, *, batch_size=10, checkpoint="a", source="source-a", protocol_seed=1234):
        count = 20
        return {
            "method": "SymNCO", "variant": "E1", "problem": "CVRPTW",
            "problem_size": 50, "scope": "small_gate",
            "dataset": {"sha256": DATASETS[50]["sha256"], "count": 1000},
            "checkpoint": {"sha256": checkpoint * 64},
            "chunk": {"expected_indices": list(range(count))},
            "project": {"commit": "p", "dirty": False},
            "upstream": {"manifest_sha256": source, "dirty": False},
            "protocol": {**protocol(50, batch_size), "seed": protocol_seed},
            "environment": {"gpu": "RTX 4090"}, "source_provenance": [],
            "timing_semantics": TIMING_SEMANTICS,
        }

    def test_complete_batch_timing_and_resume_fingerprint(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            identity = self.identity()
            initialize(path, identity)
            for batch_index, runtime in enumerate((0.2, 0.4)):
                indices = list(range(batch_index * 10, (batch_index + 1) * 10))
                records = [symnco_record(i, augmentation=i % 4, runtime=runtime)
                           for i in indices]
                append_batch(path, records, {
                    "batch_index": batch_index, "dataset_indices": indices,
                    "batch_size": 10, "runtime_seconds": runtime})
            summary = finalize(path)
            self.assertEqual(summary["num_batches"], 2)
            self.assertAlmostEqual(summary["mean_batch_runtime_seconds"], 0.3)
            self.assertAlmostEqual(summary["total_runtime_seconds"], 0.6)
            initialize(path, identity)
            for changed in (
                    self.identity(batch_size=1), self.identity(checkpoint="b"),
                    self.identity(source="source-b"), self.identity(protocol_seed=7)):
                with self.assertRaisesRegex(ValueError, "resume refused"):
                    initialize(path, changed)


if __name__ == "__main__":
    unittest.main()
