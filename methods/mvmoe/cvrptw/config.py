"""Exact official identities supported by the MVMoE CVRPTW50 integration."""

PROBLEM_SIZE = 50
DATASET_SHA256 = "a16975d9dd242739973191256e1cdbd8166c4405759ef195b985a8c3ec49df40"
DATASET_FILENAME = "cvrptw50_pyvrp-10s_16.038.pkl"
DATASET_COUNT = 1000
CAPACITY = 40.0
CHECKPOINT_SHA256 = "3417f302fbddf232fd19a2a886cd1c7f44290b6d8c7280fcb0ae3777eeed3192"
CHECKPOINT_RELATIVE_PATH = "pretrained/mvmoe_4e_n50/epoch-5000.pt"


def require_problem_size(problem_size):
    if int(problem_size) != PROBLEM_SIZE:
        raise ValueError("MVMoE/CVRPTW integration supports only problem_size 50")
    return PROBLEM_SIZE
