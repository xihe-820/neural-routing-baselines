"""Exact official dataset/checkpoint identities supported by this integration."""

SUPPORTED = {
    50: {
        "dataset_sha256": "eea12fbefe9c1bcc008d56ecfc1c50dadd64ac774f3547774c9fade8a7baa6c2",
        "checkpoint_sha256": "3417f302fbddf232fd19a2a886cd1c7f44290b6d8c7280fcb0ae3777eeed3192",
        "capacity": 40.0,
        "checkpoint_relative_path": "pretrained/mvmoe_4e_n50/epoch-5000.pt",
        "dataset_filename": "cvrp50_hgs-1s_10.366.pkl",
        "dataset_count": 10000,
    },
    100: {
        "dataset_sha256": "bb47d5a113848e5a404edefc562d5d2ef6b0ade1aafc287828bdf60364e23532",
        "checkpoint_sha256": "554d6daea825e17d62c1b9db40d56869312923848504bdf4970663673c971bdc",
        "capacity": 50.0,
        "checkpoint_relative_path": "pretrained/mvmoe_4e_n100/epoch-5000.pt",
        "dataset_filename": "cvrp100_hgs-20s_15.563.pkl",
        "dataset_count": 10000,
    },
}


def supported_config(problem_size):
    try:
        return SUPPORTED[int(problem_size)]
    except (KeyError, ValueError) as exc:
        raise ValueError("MVMoE/CVRP supports only problem_size 50 or 100") from exc
