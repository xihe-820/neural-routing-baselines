"""Pinned dataset, asset, and source-supported GLOP TSP configurations."""

REVISER_ASSETS = {
    10: {
        "checkpoint_sha256": "41bd9e05d5f623a6a7978be0063354d75f6f8df62e0ed368789ca449f41922f4",
        "args_sha256": "e21195ed71321b91ca2517e49b4a7556c27239ed1017d603e34d25a49742879c",
        "checkpoint_size_bytes": 24394377,
        "args_size_bytes": 1004,
    },
    20: {
        "checkpoint_sha256": "6771bf6b955fe26004f378c1ab0a2068c3048d717325a62b85a279e0ec22a865",
        "args_sha256": "66171fc5178ee7fc2b8ddcda8a7c90e804a0e9c9eb960375f82df40cbc228ef6",
        "checkpoint_size_bytes": 25034377,
        "args_size_bytes": 1004,
    },
    50: {
        "checkpoint_sha256": "25189e74e1e0323ced3016e9c7495c2c8d8ae082961e8696dffff79f5db1d4a6",
        "args_sha256": "ae311d53fe1e36573a609cc7bab75be1f346a577576c36a1d30ff799cbdcc76c",
        "checkpoint_size_bytes": 27594377,
        "args_size_bytes": 1004,
    },
    100: {
        "checkpoint_sha256": "3810b460f210de35b5d4bd1f680f505ff4619823880652be1c7f35f320584451",
        "args_sha256": "b99400a52c2dd4b6bdbd221f17432ad65d0e9d585207032ee65126b78c0354d9",
        "checkpoint_size_bytes": 31434377,
        "args_size_bytes": 1022,
    },
}

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


def reviser_paths(asset_root, reviser_size):
    """Return the official bundle paths without assuming an upstream layout."""
    from pathlib import Path

    if int(reviser_size) not in REVISER_ASSETS:
        raise ValueError("unknown pinned GLOP reviser asset")
    directory = Path(asset_root) / "Reviser-stage2" / f"reviser_{int(reviser_size)}"
    return directory / "epoch-299.pt", directory / "args.json"
