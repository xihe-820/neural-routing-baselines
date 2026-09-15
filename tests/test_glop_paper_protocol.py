import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from methods.glop.paper_protocol import (CVRP_PROTOCOLS,
                                         PAPER_DATASET_FILENAMES,
                                         PARTITIONER_ASSETS, TSP_PROTOCOLS,
                                         audit_entry, formal_protocol)
from methods.glop.paper_results import fingerprint, initialize_chunk
from methods.glop.runtime import verify_file
from methods.glop.tsp.adapter import apply_top_level_reflections
from scripts import audit_glop_paper_datasets

ROOT = Path(__file__).resolve().parents[1]


class GLOPPaperProtocolTests(unittest.TestCase):
    def test_tsp500_standard_and_more_are_exact(self):
        standard = formal_protocol("TSP", 500, "official_standard")
        more = formal_protocol("TSP", 500, "official_more")
        self.assertEqual(standard["revision_lens"], [100, 50, 20])
        self.assertEqual(standard["revision_iters"], [20, 25, 5])
        self.assertEqual((standard["internal_width"], more["internal_width"]), (1, 10))
        self.assertEqual(more["revision_iters"], [20, 25, 5])
        self.assertEqual(
            standard["shared_ri_order_timing_policy"],
            "charged exactly once to dataset index 0")

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

    def test_tsp2k_is_author_approved_1k_adaptation(self):
        for name in ("official_standard", "official_more"):
            with self.subTest(name=name):
                source = formal_protocol("TSP", 1000, name)
                adapted = formal_protocol("TSP", 2000, name)
                keys = ("revision_lens", "revision_iters", "ri_order_width",
                        "decode_strategy", "local_reconnect_augmentation",
                        "pruning", "seed")
                self.assertEqual({k: adapted[k] for k in keys},
                                 {k: source[k] for k in keys})
                self.assertEqual(adapted["problem_size"], 2000)
                self.assertEqual(adapted["adapted_from_size"], 1000)
                self.assertEqual(adapted["config_origin"],
                                 "author_approved_adaptation")

    def test_tsp5k_is_author_approved_10k_adaptation(self):
        for name in ("official_standard", "official_more"):
            with self.subTest(name=name):
                source = formal_protocol("TSP", 10000, name)
                adapted = formal_protocol("TSP", 5000, name)
                keys = ("revision_lens", "revision_iters", "ri_order_width",
                        "decode_strategy", "local_reconnect_augmentation",
                        "pruning", "seed")
                self.assertEqual({k: adapted[k] for k in keys},
                                 {k: source[k] for k in keys})
                self.assertEqual(adapted["adapted_from_size"], 10000)
                self.assertEqual(adapted["config_origin"],
                                 "author_approved_adaptation")

    def test_cvrp1k_and_2k_official_standard_are_exact(self):
        one = formal_protocol("CVRP", 1000, "official_standard")
        two = formal_protocol("CVRP", 2000, "official_standard")
        self.assertEqual((one["partitioner_size"], one["k_sparse"]), (1000, 100))
        self.assertEqual((one["revision_lens"], one["revision_iters"]), ([20], [5]))
        self.assertEqual((two["partitioner_size"], two["k_sparse"]), (2000, 200))
        self.assertEqual((two["revision_lens"], two["revision_iters"]), ([50, 20], [5, 5]))
        self.assertEqual(one["global_decode_strategy"], "greedy")
        self.assertEqual(one["local_decode_strategy"], "sampling")

    def test_cvrp500_uses_1k_partitioner_without_500_asset_lookup(self):
        standard = formal_protocol("CVRP", 500, "official_standard")
        more = formal_protocol("CVRP", 500, "project_more_revisions")
        for config, iters in ((standard, [5]), (more, [10])):
            self.assertEqual(config["problem_size"], 500)
            self.assertEqual(config["partitioner_source_size"], 1000)
            self.assertEqual(config["adapted_from_size"], 1000)
            self.assertEqual(config["partitioner_path"],
                             "Partitioner/cvrp/cvrp-1000.pt")
            self.assertEqual(config["k_sparse"], 100)
            self.assertEqual(config["partitioner_depth"], 12)
            self.assertEqual(config["revision_lens"], [20])
            self.assertEqual(config["revision_iters"], iters)
        self.assertNotIn(500, PARTITIONER_ASSETS)

    def test_cvrp_project_more_only_changes_algorithm_revision_iters(self):
        algorithm_keys = (
            "partitioner_source_size", "partitioner_path", "k_sparse",
            "partitioner_depth", "revision_lens", "global_decode_strategy",
            "local_decode_strategy", "internal_width", "n_partition",
            "local_augmentation", "pruning", "seed", "original_batch_size")
        for size, standard_iters, more_iters in (
                (500, [5], [10]), (1000, [5], [10]),
                (2000, [5, 5], [10, 10])):
            with self.subTest(size=size):
                standard = formal_protocol("CVRP", size, "official_standard")
                more = formal_protocol("CVRP", size, "project_more_revisions")
                self.assertEqual(standard["revision_iters"], standard_iters)
                self.assertEqual(more["revision_iters"], more_iters)
                self.assertEqual({key: standard[key] for key in algorithm_keys},
                                 {key: more[key] for key in algorithm_keys})
                self.assertEqual(more["config_origin"], "project_defined_budget")
                self.assertEqual(more["base_protocol"], "official_standard")
                self.assertEqual(
                    more["budget_rule"],
                    "double every official revision iteration count")
                self.assertNotEqual(fingerprint(standard), fingerprint(more))

    def test_all_enabled_protocols_freeze_bs1_seed_and_search_flags(self):
        cases = [(p, n, name) for p, table in (
            ("TSP", TSP_PROTOCOLS), ("CVRP", CVRP_PROTOCOLS))
                 for n, entry in table.items() if entry["formal_enabled"]
                 for name in entry["configs"]]
        self.assertEqual(len(cases), 18)
        for problem, size, name in cases:
            with self.subTest(problem=problem, size=size, name=name):
                config = formal_protocol(problem, size, name)
                self.assertEqual(config["original_batch_size"], 1)
                self.assertEqual(config["seed"], 1)
                self.assertNotIn("seed_scope", config)
                self.assertFalse(
                    config["rng_semantics"]["per_instance_reseed"])
                self.assertTrue(
                    config["rng_semantics"]["warmup_rng_restored"])
                if not (problem == "TSP" and size == 100):
                    self.assertTrue(config["local_augmentation"])
                    self.assertTrue(config["pruning"])
                self.assertFalse(config["training"])
                self.assertFalse(config["fine_tuning"])

    def test_tsp100_width_layers_and_augmentation_are_explicit(self):
        standard = formal_protocol("TSP", 100, "official_standard")
        more = formal_protocol("TSP", 100, "official_more")
        self.assertEqual(
            (standard["paper_nominal_width"], standard["ri_order_width"],
             standard["top_level_reflection_factor"],
             standard["effective_candidate_count"]), (35, 35, 1, 35))
        self.assertEqual(
            (more["paper_nominal_width"], more["ri_order_width"],
             more["top_level_reflection_factor"],
             more["effective_candidate_count"]), (140, 35, 4, 140))
        self.assertFalse(standard["top_level_augmentation"])
        self.assertTrue(more["top_level_augmentation"])
        self.assertFalse(standard["local_reconnect_augmentation"])
        self.assertFalse(more["local_reconnect_augmentation"])
        self.assertFalse(standard["pruning"])
        self.assertFalse(more["pruning"])
        self.assertEqual(standard["decode_strategy"], "sampling")
        self.assertEqual(standard["rng_semantics"]["sampling_rng_stream"],
                         "continuous across original instances")
        self.assertEqual(standard["rng_semantics"]["resume_rng_policy"],
                         "completed-prefix replay")
        self.assertEqual(
            standard["released_code_alternative"]["effective_candidate_count"],
            32)

    def test_tsp100_candidate_orchestration_matches_frozen_counts(self):
        seeds = torch.arange(35 * 100 * 2, dtype=torch.float32).reshape(35, 100, 2)
        for name, expected in (("official_standard", 35),
                               ("official_more", 140)):
            with self.subTest(name=name):
                protocol = formal_protocol("TSP", 100, name)
                candidates = apply_top_level_reflections(
                    seeds, transforms=protocol["top_level_transforms"],
                    expected_candidate_count=protocol["effective_candidate_count"])
                self.assertEqual(tuple(candidates.shape), (expected, 100, 2))
                self.assertTrue(torch.equal(candidates[:35], seeds))
        more = formal_protocol("TSP", 100, "official_more")
        candidates = apply_top_level_reflections(
            seeds, transforms=more["top_level_transforms"],
            expected_candidate_count=140)
        self.assertTrue(torch.equal(candidates[35:70, :, 0], 1 - seeds[:, :, 0]))
        self.assertTrue(torch.equal(candidates[35:70, :, 1], seeds[:, :, 1]))

    def test_manuscript_mapping_is_separate_from_algorithm_status(self):
        entry = audit_entry("TSP", 500)
        self.assertEqual(entry["official_config_status"], "FROZEN_OFFICIAL")
        self.assertEqual(entry["paper_row_mapping_status"],
                         "PROJECT_FROZEN")
        self.assertEqual(entry["paper_display_mapping"]["GLOP (fewer)"],
                         "official_standard")
        self.assertEqual(entry["paper_display_mapping"]["GLOP (more)"],
                         "official_more")
        self.assertEqual(entry["paper_display_mapping"]["Appendix GLOP"],
                         "official_standard")
        self.assertEqual(
            audit_entry("CVRP", 1000)["official_config_status"],
            "FROZEN_OFFICIAL_BASE_WITH_PROJECT_MORE")
        self.assertEqual(
            audit_entry("CVRP", 1000)["paper_display_mapping"]["GLOP (more)"],
            "project_more_revisions")

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

    def test_cvrp_standard_and_more_resume_identities_differ(self):
        standard = formal_protocol("CVRP", 1000, "official_standard")
        more = formal_protocol("CVRP", 1000, "project_more_revisions")
        self.assertNotEqual(fingerprint(standard), fingerprint(more))
        with tempfile.TemporaryDirectory() as directory:
            base = {"chunk": {"expected_indices": [0]},
                    "official_protocol_name": "official_standard",
                    "paper_protocol": standard}
            initialize_chunk(directory, base)
            changed = dict(
                base, official_protocol_name="project_more_revisions",
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
            self.assertEqual(
                {(row["problem"], row["size"]): row["expected_filename"]
                 for row in report["coverage"]}, PAPER_DATASET_FILENAMES)

    def test_dataset_audit_filename_match_uses_realpath_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / PAPER_DATASET_FILENAMES[("TSP", 100)]
            output = root / "audit.json"
            audited_row = {
                "realpath": str(dataset.resolve()),
                "all_expected_size": True,
                "all_expected_task_class": True,
                "error": None,
            }
            argv = [
                "audit_glop_paper_datasets.py",
                "--dataset-root", str(root / "missing"),
                "--dataset", "TSP", "100", str(dataset),
                "--output", str(output),
            ]
            with mock.patch.object(
                    audit_glop_paper_datasets, "audit",
                    return_value=audited_row), mock.patch.object(sys, "argv", argv):
                audit_glop_paper_datasets.main()
            row = json.loads(output.read_text())["datasets"][0]
            self.assertNotIn("path", row)
            self.assertTrue(row["filename_matches"])
            self.assertEqual(row["expected_filename"], dataset.name)

    def test_dataset_registry_uses_exact_paper_target_names(self):
        self.assertEqual(PAPER_DATASET_FILENAMES, {
            ("TSP", 100): "tsp100_concorde_7.756.pkl",
            ("TSP", 500): "tsp500_concorde_16.546.pkl",
            ("TSP", 1000): "tsp1000_concorde_23.118.pkl",
            ("TSP", 2000): "tsp2000_lkh_500_32.436.pkl",
            ("TSP", 5000): "tsp5000_lkh_500_50.968.pkl",
            ("TSP", 10000): "tsp10000_lkh_500_71.782.pkl",
            ("CVRP", 500): "cvrp500_hgs-300s_37.154.pkl",
            ("CVRP", 1000): "cvrp1000_hgs-360s_41.171.pkl",
            ("CVRP", 2000): "cvrp2000_hgs-360s_57.181.pkl",
        })


if __name__ == "__main__":
    unittest.main()
