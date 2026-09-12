"""Exact official identities supported by the MVMoE CVRPTW integration."""

SUPPORTED_SIZES = (50, 100)

SIZE_CONFIGS = {
    50: {
        "dataset_sha256": "a16975d9dd242739973191256e1cdbd8166c4405759ef195b985a8c3ec49df40",
        "dataset_filename": "cvrptw50_pyvrp-10s_16.038.pkl",
        "dataset_count": 1000,
        "capacity": 40.0,
        "checkpoint_sha256": "3417f302fbddf232fd19a2a886cd1c7f44290b6d8c7280fcb0ae3777eeed3192",
        "checkpoint_relative_path": "pretrained/mvmoe_4e_n50/epoch-5000.pt",
    },
    100: {
        "dataset_sha256": "3b74fa520f42f7aa607a5bd00a7d4aaa118e0715ca1672ee854fc850bac67867",
        "dataset_filename": "cvrptw100_pyvrp-20s_25.431.pkl",
        "dataset_count": 1000,
        "capacity": 50.0,
        "checkpoint_sha256": "554d6daea825e17d62c1b9db40d56869312923848504bdf4970663673c971bdc",
        "checkpoint_relative_path": "pretrained/mvmoe_4e_n100/epoch-5000.pt",
    },
}


def get_size_config(problem_size):
    """Return the pinned identities for one supported formal size."""
    try:
        size = int(problem_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("problem_size must be 50 or 100") from exc
    if size not in SIZE_CONFIGS:
        raise ValueError("MVMoE/CVRPTW integration supports only problem_size 50 or 100")
    return SIZE_CONFIGS[size]
