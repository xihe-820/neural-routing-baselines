import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from methods.glop.paper_protocol import (CVRP_PROTOCOLS, PARTITIONER_ASSETS,
                                         TSP_PROTOCOLS, audit_entry,
                                         formal_protocol)
from methods.glop.paper_results import fingerprint, initialize_chunk
from methods.glop.runtime import verify_file

ROOT = Path(__file__).resolve().parents[1]


class GLOPPaperProtocolTests(unittest.TestCase):
    def test_tsp500_standard_and_more_are_exact(self):
        standard = formal_protocol("TSP", 500, "official_standard")
        more = formal_protocol("TSP", 500, "official_more")
        self.assertEqual(standard["revision_lens"], [100, 50, 20])
        self.assertEqual(standard["revision_iters"], [20, 25, 5])
        self.assertEqual((standard["internal_width"], more["internal_width"]), (1, 10))
        self.assertEqual(more["revision_iters"], [20, 25, 5])

    def test_tsp1k_standard_and_more_are_exact(self):
        standard = formal_protocol("TSP", 1000, "official_standard")
        more = formal_protocol("TSP", 1000, "official_more")
        self.assertEqual(standard["revision_lens"], [100, 50, 20])
        self.assertEqual(standard["revision_iters"], [20, 25, 5])
        self.assertEqual((standard["internal_width"], more["internal_width"]), (1, 10))

    def test_tsp10k_standard_and_more_are_exact(self):
        standard = formal_protocol("TSP", 10000, "official_standard")
        more = formal_protocol("TSP", 10000, "official_more")
        self.assertEqual(standard["revision_iters"], [10, 20, 5])
        self.assertEqual(more["revision_iters"], [50, 25, 5])
        self.assertEqual((standard["internal_width"], more["internal_width"]), (1, 1))

    def test_cvrp1k_and_2k_official_single_are_exact(self):
        one = formal_protocol("CVRP", 1000, "official_single")
        two = formal_protocol("CVRP", 2000, "official_single")
        self.assertEqual((one["partitioner_size"], one["k_sparse"]), (1000, 100))
        self.assertEqual((one["revision_lens"], one["revision_iters"]), ([20], [5]))
        self.assertEqual((two["partitioner_size"], two["k_sparse"]), (2000, 200))
        self.assertEqual((two["revision_lens"], two["revision_iters"]), ([50, 20], [5, 5]))
        self.assertEqual(one["global_decode_strategy"], "greedy")
        self.assertEqual(one["local_decode_strategy"], "sampling")

    def test_all_enabled_protocols_freeze_bs1_seed_and_search_flags(self):
        cases = [(p, n, name) for p, table in (
            ("TSP", TSP_PROTOCOLS), ("CVRP", CVRP_PROTOCOLS))
                 for n, entry in table.items() if entry["formal_enabled"]
                 for name in entry["configs"]]
        self.assertEqual(len(cases), 8)
        for problem, size, name in cases:
            with self.subTest(problem=problem, size=size, name=name):
                config = formal_protocol(problem, size, name)
                self.assertEqual(config["original_batch_size"], 1)
                self.assertEqual(config["seed"], 1)
                self.assertTrue(config["local_augmentation"])
                self.assertTrue(config["pruning"])
                self.assertFalse(config["training"])
                self.assertFalse(config["fine_tuning"])

    def test_disabled_sizes_fail_closed(self):
        cases = (("TSP", 100, "official_standard"),
                 ("TSP", 2000, "official_standard"),
                 ("TSP", 5000, "official_more"),
                 ("CVRP", 500, "official_single"))
        for problem, size, name in cases:
            with self.subTest(problem=problem, size=size):
                with self.assertRaises(ValueError):
                    formal_protocol(problem, size, name)
        self.assertEqual(
            audit_entry("TSP", 100)["official_config_status"],
            "OFFICIAL_CONFIG_EXISTS_EXECUTION_AMBIGUOUS")
        self.assertEqual(
            audit_entry("TSP", 2000)["official_config_status"],
            "NO_OFFICIAL_PRIMARY_CONFIG")
        self.assertEqual(
            audit_entry("CVRP", 500)["official_config_status"],
            "UNSUPPORTED_RELEASED_SYNTHETIC_PATH")

    def test_tsp100_records_official_configs_and_execution_ambiguity(self):
        entry = audit_entry("TSP", 100)
        standard = entry["configs"]["official_standard"]
        more = entry["configs"]["official_more"]
        self.assertEqual((standard["paper_width"] // 4) * 4, 32)
        self.assertNotEqual((standard["paper_width"] // 4) * 4,
                            standard["paper_width"])
        self.assertEqual((more["paper_width"] // 4) * 4, 140)
        self.assertEqual(
            entry["blocking_issue"],
            "PAPER_CODE_WIDTH_AND_TOP_LEVEL_AUGMENTATION_AMBIGUITY")

    def test_manuscript_mapping_is_separate_from_algorithm_status(self):
        entry = audit_entry("TSP", 500)
        self.assertEqual(entry["official_config_status"], "FROZEN_OFFICIAL")
        self.assertEqual(entry["paper_row_mapping_status"],
                         "UNRESOLVED_MANUSCRIPT_MAPPING")
        self.assertEqual(
            audit_entry("CVRP", 1000)["official_config_status"],
            "FROZEN_OFFICIAL_SINGLE_CONFIG")

    def test_standard_and_more_resume_identities_differ(self):
        standard = formal_protocol("TSP", 500, "official_standard")
        more = formal_protocol("TSP", 500, "official_more")
        self.assertNotEqual(fingerprint(standard), fingerprint(more))
        with tempfile.TemporaryDirectory() as directory:
            base = {
                "chunk": {"expected_indices": [0]},
                "official_protocol_name": "official_standard",
                "paper_protocol": standard,
            }
            initialize_chunk(directory, base)
            changed = dict(base, official_protocol_name="official_more",
                           paper_protocol=more)
            with self.assertRaisesRegex(ValueError, "resume refused"):
                initialize_chunk(directory, changed)

    def test_wrong_reviser_and_partitioner_sha_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.pt"
            path.write_bytes(b"checkpoint")
            for kind in ("reviser", "partitioner"):
                with self.subTest(kind=kind):
                    with self.assertRaisesRegex(ValueError, "SHA256"):
                        verify_file(path, {"sha256": "0" * 64,
                                           "size_bytes": len(b"checkpoint")})

    def test_partitioner_identities_and_source_have_no_cvrp500(self):
        self.assertEqual(PARTITIONER_ASSETS[1000]["k_sparse"], 100)
        self.assertEqual(PARTITIONER_ASSETS[2000]["k_sparse"], 200)
        source = ROOT / "external/GLOP/heatmap/cvrp/infer.py"
        tree = ast.parse(source.read_text(), filename=str(source))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "K_SPARSE"
                                  for target in node.targets))
        self.assertEqual(set(ast.literal_eval(assignment.value)),
                         {1000, 2000, 5000, 7000})
        self.assertFalse((ROOT / "checkpoints/glop/pretrained/Partitioner/cvrp/"
                          "cvrp-500.pt").exists())

    def test_readme_commands_anchor_enabled_official_configs(self):
        readme = (ROOT / "external/GLOP/README.md").read_text()
        for command_fragment in (
                "--problem_size 500 --revision_iters 20 25 5 "
                "--revision_lens 100 50 20 --width 10",
                "--problem_size 1000 --revision_iters 20 25 5 "
                "--revision_lens 100 50 20 --width 10",
                "--problem_size 10000 --revision_iters 50 25 5 "
                "--revision_lens 100 50 20 --width 1",
                "--problem_type cvrp --problem_size 1000 "
                "--revision_lens 20 --revision_iters 5",
                "--problem_type cvrp --problem_size 2000 "
                "--revision_lens 50 20 --revision_iters 5 5"):
            self.assertIn(command_fragment, readme)

    def test_missing_dataset_root_reports_all_nine_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            proc = subprocess.run(
                [sys.executable, "-B",
                 str(ROOT / "scripts/audit_glop_paper_datasets.py"),
                 "--dataset-root", str(Path(directory) / "missing"),
                 "--output", str(output)],
                capture_output=True, text=True, timeout=30,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=True)
            self.assertFalse(proc.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(len(report["coverage"]), 9)
            self.assertFalse(report["complete"])
            self.assertTrue(all(row["status"] == "MISSING"
                                for row in report["coverage"]))


if __name__ == "__main__":
    unittest.main()
