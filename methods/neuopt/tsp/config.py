"""Pinned NeuOpt TSP100 benchmark and checkpoint identities."""


SUPPORTED = {
    100: {
        "dataset_filename": "tsp100_concorde_7.756.pkl",
        "dataset_sha256": "a2bfe99857b8072bdba051f6ae402b7e241f01b0462c5f379ed0aa03786406a0",
        "dataset_count": 1280,
        "checkpoint_relative_path": "pre-trained/tsp100.pt",
        "checkpoint_sha256": "37a306d0974aed9e42da3aede9acbceab995585f25047dd059e211432afcd44d",
        "inference_evidence": (
            "pinned NeuOpt README TSP100 inference example with the official "
            "pre-trained/tsp100.pt checkpoint"
        ),
    },
}


def supported_config(problem_size):
    try:
        return SUPPORTED[int(problem_size)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("NeuOpt/TSP integration supports only problem_size 100") from exc
