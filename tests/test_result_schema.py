import json
from pathlib import Path
import tempfile
import unittest

from common.provenance import normalize_git_repository_identity
from common.result_schema import (complete_validation, make_run_metadata, new_result,
                                  read_result_bundle, write_result_bundle)


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
                new_result(**identity(runtime_seconds=value, runtime_semantics="amortized"))

    def test_failure_records_error(self):
        row = new_result(**identity(evidence_status="FAILED", error="rollout failed"))
        self.assertEqual(row["error"], "rollout failed")
        with self.assertRaisesRegex(ValueError, "nonempty error"):
            new_result(**identity(evidence_status="FAILED"))

    def test_missing_validation_cannot_claim_feasible(self):
        with self.assertRaisesRegex(ValueError, "requires objective"):
            new_result(**identity(independent_feasible=True))

    def validated_row(self, **updates):
        values = identity(
            independent_feasible=True, independent_objective=10.0,
            constraint_details={"each_customer_once": True},
            reported_objective=10.0, reported_objective_agrees=True,
            kit_feasible=True, kit_objective=10.0, kit_objective_agrees=True,
        )
        values.update(updates)
        return new_result(**values)

    def test_reported_mismatch_cannot_claim_full_local_verified(self):
        with self.assertRaisesRegex(ValueError, "completion gates"):
            self.validated_row(evidence_status="LOCAL_VERIFIED", reported_objective=12.0,
                               reported_objective_agrees=False)

    def test_reported_mismatch_cannot_claim_agreement_flag(self):
        with self.assertRaisesRegex(ValueError, "contradicts objective values"):
            self.validated_row(reported_objective=12.0,
                               reported_objective_agrees=True)

    def test_reported_mismatch_fails_even_when_independent_and_kit_agree(self):
        row = self.validated_row(reported_objective=12.0,
                                 reported_objective_agrees=False)
        complete_validation(row)
        self.assertEqual(row["evidence_status"], "FAILED")
        self.assertIn("reported_objective_agrees", row["error"])

    def test_all_four_completion_gates_produce_local_verified(self):
        row = complete_validation(self.validated_row())
        self.assertEqual(row["evidence_status"], "LOCAL_VERIFIED")
        self.assertIsNone(row["error"])

    def test_rollout_timestamps_and_runtime_semantics(self):
        metadata = make_run_metadata(
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            total_runtime_seconds=1.0, batch_size=2)
        self.assertIn("amortized_batch_runtime_seconds",
                      metadata["per_row_runtime_semantics"])
        with self.assertRaisesRegex(ValueError, "precedes"):
            make_run_metadata(started_at="2026-01-01T00:00:01+00:00",
                              finished_at="2026-01-01T00:00:00+00:00",
                              total_runtime_seconds=1.0, batch_size=2)


class GitIdentityTests(unittest.TestCase):
    def test_common_github_https_and_ssh_forms_match(self):
        expected = "github.com/OWNER/REPO"
        for url in (
            "https://github.com/OWNER/REPO",
            "https://github.com/OWNER/REPO.git",
            "git@github.com:OWNER/REPO.git",
            "ssh://git@github.com/OWNER/REPO.git",
        ):
            with self.subTest(url=url):
                self.assertEqual(normalize_git_repository_identity(url), expected)

    def test_non_github_remote_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "GitHub"):
            normalize_git_repository_identity("https://example.com/OWNER/REPO.git")


if __name__ == "__main__":
    unittest.main()
