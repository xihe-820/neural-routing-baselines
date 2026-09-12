"""Failure-path regression tests; no official imports, installs, GPU or commits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file


class AuditTests(unittest.TestCase):
    def run_script(self, name, *args):
        return subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / name), *map(str, args)],
                              capture_output=True, text=True, timeout=30,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=True)

    def test_hash_known_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bytes"
            path.write_bytes(b"abc")
            self.assertEqual(sha256_file(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_lfs_pointer_and_missing_checkpoint_never_pass_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "upstream/GLOP"
            official.mkdir(parents=True)
            pointer = official / "weights.pt"
            payload = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 999999\n"
            pointer.write_bytes(payload)
            report = root / "audit.json"
            self.run_script("audit_assets.py", "--upstream-root", root / "upstream", "--output", report,
                            "--load-checkpoints", "--checkpoint", "GLOP", root / "missing.pt")
            r = json.loads(report.read_text())
            self.assertFalse(r["repos"]["UDC"]["exists"])
            records = r["repos"]["GLOP"]["checkpoints"]
            self.assertEqual(len(records), 2)
            lfs = next(c for c in records if c.get("lfs_pointer"))
            self.assertFalse(lfs["materialized"])
            self.assertIsNone(lfs["load"]["deserialize_ok"])
            missing = next(c for c in records if c.get("error") == "CHECKPOINT_NOT_FOUND")
            self.assertIsNone(missing["sha256"])
            self.assertIsNone(r["repos"]["GLOP"]["dirty"])
            self.assertEqual(pointer.read_bytes(), payload)

    def test_missing_dataset_root_reports_empty_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "audit.json"
            self.run_script("audit_datasets.py", "--dataset-root", root / "absent", "--output", report)
            r = json.loads(report.read_text())
            self.assertFalse(r["root_exists"])
            self.assertEqual(len(r["coverage"]), 6)
            self.assertTrue(all(not c["verified_files"] for c in r["coverage"]))

    def test_missing_explicit_dataset_needs_no_kit_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "audit.json"
            self.run_script("audit_datasets.py", "--dataset", "CVRPTW", "100", root / "missing.pkl", "--output", report)
            row = json.loads(report.read_text())["datasets"][0]
            self.assertFalse(row["exists"])
            self.assertNotIn("reference", row)

    def test_model_probe_disallows_second_repository_path(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/audit_model_load.py"),
                                   "--repo", directory, "--module-path", "../another-repo", "--output", directory + "/out.json"],
                                  capture_output=True, text=True, timeout=10)
            self.assertEqual(proc.returncode, 2)
            self.assertFalse((Path(directory) / "out.json").exists())


if __name__ == "__main__":
    unittest.main()
