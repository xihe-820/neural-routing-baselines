"""Pinned dataset identities and source-supported GLOP TSP configurations."""

SUPPORTED = {
    50: {
        "dataset_filename": "tsp50_concorde_5.688.pkl",
        "dataset_sha256": "1ede2b289d2e6fbfe614219a86d1ba6dfca2a726925a8fe8cef9c8196ee0b213",
        "dataset_count": 1280,
        "required_revisers": [20],
        "revision_iters": [10],
        "width_requested": 1,
        "width_after_small_size_branch": 1,
        "tsp_aug": False,
        "top_level_transforms": ["identity"],
        "decode_strategy": "sampling",
        "no_aug": False,
        "no_prune": False,
        "seed": 1,
        "configuration_evidence": (
            "official own-dataset instruction plus main.py generic inference defaults "
            "and the explicit problem_size<=100 branch"
        ),
    },
    100: {
        "dataset_filename": "tsp100_concorde_7.756.pkl",
        "dataset_sha256": "a2bfe99857b8072bdba051f6ae402b7e241f01b0462c5f379ed0aa03786406a0",
        "dataset_count": 1280,
        "required_revisers": [100, 50, 20, 10],
        "revision_iters": [20, 10, 10, 5],
        "width_requested": 140,
        "width_after_small_size_branch": 35,
        "tsp_aug": True,
        "top_level_transforms": ["identity", "reflect_x", "reflect_y", "reflect_xy"],
        "decode_strategy": "sampling",
        "no_aug": True,
        "no_prune": True,
        "seed": 1,
        "configuration_evidence": "exact official README cross-distribution uniform TSP100 command",
    },
}


def supported_config(problem_size):
    try:
        return SUPPORTED[int(problem_size)]
    except (KeyError, ValueError) as exc:
        raise ValueError("GLOP/TSP integration scope contains only problem_size 50 or 100") from exc


def validate_reviser_schedule(problem_size, revision_lens, revision_iters):
    if len(revision_lens) != len(revision_iters) or not revision_lens:
        raise ValueError("revision lengths and iterations must be nonempty and aligned")
    if any(int(length) <= 0 or int(length) > problem_size for length in revision_lens):
        raise ValueError("a reviser length exceeds the actual TSP size")
    if any(int(value) <= 0 for value in revision_iters):
        raise ValueError("revision iterations must be positive")
    return True
