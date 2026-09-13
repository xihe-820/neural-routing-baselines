import ast
from pathlib import Path
import unittest

import numpy as np

from methods.glop.tsp.adapter import (adapt_points, apply_top_level_transform,
                                      validate_initial_permutations)
from methods.glop.tsp.config import (supported_config,
                                     validate_reviser_schedule)
from methods.glop.tsp.decode import (decode_coordinate_tour,
                                     select_best_candidates)
from problems.tsp.objective import cycle_length
from problems.tsp.validate import validate

ROOT = Path(__file__).resolve().parents[1]


class GLOPTSPConfigAdapterTests(unittest.TestCase):
    def test_dataset_and_formal_config_identities(self):
        n50 = supported_config(50)
        n100 = supported_config(100)
        self.assertEqual((n50["dataset_count"], n50["required_revisers"],
                          n50["revision_iters"], n50["width_requested"], n50["tsp_aug"]),
                         (1280, [20], [10], 1, False))
        self.assertEqual(n50["dataset_sha256"],
                         "1ede2b289d2e6fbfe614219a86d1ba6dfca2a726925a8fe8cef9c8196ee0b213")
        self.assertEqual((n100["required_revisers"], n100["revision_iters"],
                          n100["width_requested"], n100["width_after_small_size_branch"],
                          n100["tsp_aug"], n100["no_aug"], n100["no_prune"]),
                         ([100, 50, 20, 10], [20, 10, 10, 5], 140, 35, True, True, True))
        self.assertEqual(n100["dataset_sha256"],
                         "a2bfe99857b8072bdba051f6ae402b7e241f01b0462c5f379ed0aa03786406a0")

    def test_reviser_larger_than_instance_is_rejected(self):
        self.assertTrue(validate_reviser_schedule(50, [20], [10]))
        self.assertTrue(validate_reviser_schedule(100, [100, 50, 20, 10], [20, 10, 10, 5]))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            validate_reviser_schedule(50, [100], [20])

    def test_neutral_adapter_and_initial_permutation_provenance(self):
        points = np.arange(200, dtype=np.float32).reshape(2, 50, 2) / 201
        native, mapping = adapt_points(points, problem_size=50, device="cpu")
        self.assertEqual(tuple(native.shape), (2, 50, 2))
        np.testing.assert_array_equal(native.numpy(), points)
        self.assertFalse(mapping["posthoc_tolerance_matching"])
        permutations = np.stack((np.tile(np.arange(50), (2, 1)),), axis=0)
        self.assertTrue(validate_initial_permutations(
            permutations, problem_size=50, width=1, batch_size=2
        ))
        permutations[0, 0, -1] = 0
        with self.assertRaisesRegex(ValueError, "duplicate or missing"):
            validate_initial_permutations(permutations, problem_size=50, width=1, batch_size=2)


class GLOPTSPDecoderTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.points = rng.random((100, 2), dtype=np.float32)
        self.permutation = np.roll(np.arange(100), 17)

    def test_exact_inverse_mapping_for_each_top_level_augmentation(self):
        for transform in ("identity", "reflect_x", "reflect_y", "reflect_xy"):
            with self.subTest(transform=transform):
                transformed = apply_top_level_transform(self.points, transform)
                output = transformed[self.permutation]
                route, metadata = decode_coordinate_tour(
                    output, self.points,
                    allowed_transforms=("identity", "reflect_x", "reflect_y", "reflect_xy"),
                )
                expected = list(range(100))
                self.assertEqual(route, expected + [0])
                self.assertEqual(metadata["top_level_transform"], transform)
                self.assertTrue(metadata["exact_coordinate_bit_match"])

    def test_duplicate_or_missing_output_is_rejected_without_repair(self):
        output = self.points.copy()
        output[-1] = output[-2]
        with self.assertRaisesRegex(ValueError, "no unique exact"):
            decode_coordinate_tour(output, self.points, allowed_transforms=("identity",))

    def test_objective_correspondence_after_reflection_and_rotation(self):
        output = apply_top_level_transform(self.points, "reflect_xy")[self.permutation]
        route, _ = decode_coordinate_tour(
            output, self.points,
            allowed_transforms=("identity", "reflect_x", "reflect_y", "reflect_xy"),
        )
        official_coordinate_cost = cycle_length(output, np.arange(len(output)))
        independent = validate(self.points, route)
        self.assertTrue(independent["feasible"])
        self.assertAlmostEqual(official_coordinate_cost, independent["independent_objective"], places=6)

    def test_candidate_selection_matches_first_argmin(self):
        tours = np.stack((self.points, self.points[::-1]))[:, None]
        selected, reported, indices = select_best_candidates(
            np.asarray([[8.0], [7.0]]), tours
        )
        np.testing.assert_array_equal(selected[0], self.points[::-1])
        self.assertEqual(reported.tolist(), [7.0])
        self.assertEqual(indices.tolist(), [1])


class GLOPCVRPSourceBlockerTests(unittest.TestCase):
    def test_pinned_k_sparse_has_no_50_or_100_entry(self):
        source_path = ROOT / "external/GLOP/heatmap/cvrp/infer.py"
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "K_SPARSE"
                    for target in node.targets)
        )
        keys = set(ast.literal_eval(assignment.value))
        self.assertEqual(keys, {1000, 2000, 5000, 7000})
        self.assertNotIn(50, keys)
        self.assertNotIn(100, keys)


if __name__ == "__main__":
    unittest.main()
