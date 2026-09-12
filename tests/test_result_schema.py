import json
from pathlib import Path
import tempfile
import unittest

from common.result_schema import new_result, read_result_bundle, write_result_bundle


def identity(**updates):
    values = {
        "method": "MVMoE", "variant": "MOE/4E", "problem": "CVRP",
        "problem_size": 50, "instance_id": "x", "project_repo_commit": "a" * 40,
        "project_repo_dirty": True, "upstream_url": "https://example.test/repo",
        "upstream_commit": "b" * 40, "upstream_dirty": False,
        "checkpoint_path": "/checkpoint.pt", "checkpoint_sha256": "c" * 64,
        "dataset_path": "/dataset.pkl", "dataset_sha256": "d" * 64,
        "dataset_instance_index": 0, "adapter_provenance": {"sha256": "e" * 64},
    }
    values.update(updates)
    return values


class ResultSchemaTests(unittest.TestCase):
    def test_required_identity_fields(self):
        values = identity()
        values.pop("checkpoint_sha256")
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            new_result(**values)

    def test_json_round_trip_keeps_explicit_not_run(self):
        row = new_result(**identity())
        self.assertEqual(row["evidence_status"], "NOT_RUN")
        self.assertIsNone(row["independent_feasible"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_result_bundle(path, [row], run_metadata={"purpose": "test"})
            self.assertEqual(read_result_bundle(path)["results"], [row])
            json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    def test_nan_and_inf_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "NaN or Inf"):
                new_result(**identity(runtime_seconds=value))

    def test_failure_records_error(self):
        row = new_result(**identity(evidence_status="FAILED", error="rollout failed"))
        self.assertEqual(row["error"], "rollout failed")
        with self.assertRaisesRegex(ValueError, "nonempty error"):
            new_result(**identity(evidence_status="FAILED"))

    def test_missing_validation_cannot_claim_feasible(self):
        with self.assertRaisesRegex(ValueError, "requires objective"):
            new_result(**identity(independent_feasible=True))


if __name__ == "__main__":
    unittest.main()
