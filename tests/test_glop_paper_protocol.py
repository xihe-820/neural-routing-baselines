from pathlib import Path
import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest

from methods.glop.paper_protocol import (CVRP_OFFICIAL, PARTITIONER_ASSETS,
                                         TSP_OFFICIAL, paper_config)

ROOT = Path(__file__).resolve().parents[1]


class GLOPPaperProtocolTests(unittest.TestCase):
    def test_exact_manuscript_scope_and_fail_closed_lookup(self):
        self.assertEqual(set(TSP_OFFICIAL), {100, 500, 1000, 2000, 5000, 10000})
        self.assertEqual(set(CVRP_OFFICIAL), {500, 1000, 2000})
        with self.assertRaisesRegex(ValueError, "unsupported"):
            paper_config("TSP", 50)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            paper_config("CVRPTW", 100)

    def test_official_tsp_table_nine_values_are_not_interpolated(self):
        self.assertEqual(TSP_OFFICIAL[500]["standard"]["width"], 1)
        self.assertEqual(TSP_OFFICIAL[500]["more"]["width"], 10)
        self.assertEqual(TSP_OFFICIAL[10000]["standard"]["revision_iters"], [10, 20, 5])
        self.assertEqual(TSP_OFFICIAL[10000]["more"]["revision_iters"], [50, 25, 5])
        self.assertIsNone(TSP_OFFICIAL[2000]["standard"])
        self.assertIsNone(TSP_OFFICIAL[5000]["more"])

    def test_cvrp_is_single_config_and_500_fails_closed(self):
        self.assertIsNone(CVRP_OFFICIAL[500]["config"])
        self.assertEqual(CVRP_OFFICIAL[500]["status"], "BLOCKED")
        self.assertEqual(CVRP_OFFICIAL[1000]["config"]["revision_lens"], [20])
        self.assertEqual(CVRP_OFFICIAL[2000]["config"]["revision_lens"], [50, 20])
        self.assertEqual(CVRP_OFFICIAL[1000]["config"]["global_decode"], "greedy")
        self.assertEqual(CVRP_OFFICIAL[1000]["config"]["local_decode_strategy"], "sampling")

    def test_pinned_source_has_no_cvrp500_graph_or_checkpoint_mapping(self):
        source = ROOT / "external/GLOP/heatmap/cvrp/infer.py"
        tree = ast.parse(source.read_text(), filename=str(source))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "K_SPARSE"
                                  for target in node.targets))
        self.assertEqual(set(ast.literal_eval(assignment.value)), {1000, 2000, 5000, 7000})
        self.assertFalse((ROOT / "checkpoints/glop/pretrained/Partitioner/cvrp/cvrp-500.pt").exists())

    def test_readme_commands_match_recorded_official_more_and_cvrp_configs(self):
        readme = (ROOT / "external/GLOP/README.md").read_text()
        for command_fragment in (
                "--problem_size 500 --revision_iters 20 25 5 --revision_lens 100 50 20 --width 10",
                "--problem_size 1000 --revision_iters 20 25 5 --revision_lens 100 50 20 --width 10",
                "--problem_size 10000 --revision_iters 50 25 5 --revision_lens 100 50 20 --width 1",
                "--problem_type cvrp --problem_size 1000 --revision_lens 20 --revision_iters 5",
                "--problem_type cvrp --problem_size 2000 --revision_lens 50 20 --revision_iters 5 5"):
            self.assertIn(command_fragment, readme)

    def test_tsp100_small_size_branch_exposes_width_mismatch(self):
        standard = TSP_OFFICIAL[100]["standard"]
        more = TSP_OFFICIAL[100]["more"]
        self.assertEqual((standard["width"] // 4) * 4, 32)
        self.assertNotEqual((standard["width"] // 4) * 4, standard["width"])
        self.assertEqual((more["width"] // 4) * 4, 140)
        self.assertEqual(more["top_level_augmentation"], "CODE_FORCES_FOUR_REFLECTIONS")

    def test_partitioner_identities(self):
        self.assertEqual(PARTITIONER_ASSETS[1000]["k_sparse"], 100)
        self.assertEqual(PARTITIONER_ASSETS[2000]["k_sparse"], 200)
        self.assertEqual(PARTITIONER_ASSETS[1000]["parameter_count"], 148513)

    def test_missing_dataset_root_reports_all_nine_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            proc = subprocess.run(
                [sys.executable, "-B", str(ROOT / "scripts/audit_glop_paper_datasets.py"),
                 "--dataset-root", str(Path(directory) / "missing"), "--output", str(output)],
                capture_output=True, text=True, timeout=30,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=True)
            self.assertFalse(proc.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(len(report["coverage"]), 9)
            self.assertFalse(report["complete"])
            self.assertTrue(all(row["status"] == "MISSING" for row in report["coverage"]))


if __name__ == "__main__":
    unittest.main()
