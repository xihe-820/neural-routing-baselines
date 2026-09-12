"""Pinned CVRP50/100 identities and size-specific NeuOpt settings."""

SUPPORTED = {
    50: {
        "dataset_filename": "cvrp50_hgs-1s_10.366.pkl",
        "dataset_sha256": "eea12fbefe9c1bcc008d56ecfc1c50dadd64ac774f3547774c9fade8a7baa6c2",
        "dataset_count": 10000,
        "capacity": 40.0,
        "checkpoint_relative_path": "pre-trained/cvrp50.pt",
        "checkpoint_sha256": "1cd201ca47888e51068a157389460641c81d71054f064d9c8ea1743312289e3a",
        "dummy_rate": 0.4,
        "dummy_size": 20,
        "sequence_length": 70,
        "inference_evidence": "official generic inference defaults plus size-specific CVRP50 settings",
    },
    100: {
        "dataset_filename": "cvrp100_hgs-20s_15.563.pkl",
        "dataset_sha256": "bb47d5a113848e5a404edefc562d5d2ef6b0ade1aafc287828bdf60364e23532",
        "dataset_count": 10000,
        "capacity": 50.0,
        "checkpoint_relative_path": "pre-trained/cvrp100.pt",
        "checkpoint_sha256": "502a5904182306c1a3f65f7b1a8503a609ff2a044db054abcf93690af36594fb",
        "dummy_rate": 0.2,
        "dummy_size": 20,
        "sequence_length": 120,
        "inference_evidence": "explicit official README CVRP100 inference example and official defaults",
    },
}


def supported_config(problem_size):
    try:
        return SUPPORTED[int(problem_size)]
    except (KeyError, ValueError) as exc:
        raise ValueError("NeuOpt/CVRP supports only problem_size 50 or 100") from exc
