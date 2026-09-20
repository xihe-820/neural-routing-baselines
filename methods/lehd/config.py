"""Frozen official source, assets, sizes, and paper protocols for LEHD."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path


UPSTREAM_URL = "https://github.com/CIAM-Group/NCO_code"
UPSTREAM_COMMIT = "274df3c4975384592b60fe7f79fbb2441ce11c15"
LEHD_SUBTREE = Path("single_objective/LEHD")
FORMAL_BATCH_SIZE = 1
PROJECT_SEED = 123
FORMAL_SIZES = {
    "tsp": (100, 500, 1000),
    "cvrp": (50, 100, 200, 500, 1000, 2000),
}
FORMAL_PROTOCOLS = ("greedy", "fewer", "more")
RRC_BUDGETS = {"greedy": 0, "fewer": 50, "more": 500}
HARDWARE_PROTOCOL_PENDING = True
MANUSCRIPT_HARDWARE = "H800"
CURRENT_SERVER_HARDWARE = "NVIDIA RTX 4090"

WARMUP_POLICY = {
    "batches": 1,
    "tester_isolation": "dedicated tester discarded before formal tester construction",
    "excluded_from_timing": True,
    "excluded_from_records": True,
    "rng_state_restored": True,
    "resume_policy": (
        "skip after any completed record; an empty resumed run may repeat the "
        "isolated RNG-neutral warm-up"
    ),
}

MODEL_PARAMS = {
    "mode": "test",
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "decoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "ff_hidden_dim": 512,
}

CHECKPOINTS = {
    "tsp": {
        "filename": "checkpoint-150.pt",
        "relative_path": (
            "single_objective/LEHD/TSP/result/20230509_153705_train/"
            "checkpoint-150.pt"
        ),
        "epoch": 150,
        "trained_on_size": 100,
        "training_data_path": "data/train_TSP100_n100w.txt",
        "official_source": "pinned official repository",
        "actual_sha256": None,
    },
    "cvrp": {
        "filename": "checkpoint-40.pt",
        "relative_path": (
            "single_objective/LEHD/CVRP/result/20230817_235537_train/"
            "checkpoint-40.pt"
        ),
        "epoch": 40,
        "trained_on_size": 100,
        "training_data_path": "data/vrp100_hgs_train_100w.txt",
        "official_source": "pinned official repository",
        "actual_sha256": None,
    },
}

DATASET_FILENAMES = {
    ("tsp", 100): "tsp100_concorde_7.756.pkl",
    ("tsp", 500): "tsp500_concorde_16.546.pkl",
    ("tsp", 1000): "tsp1000_concorde_23.118.pkl",
    ("cvrp", 50): "cvrp50_hgs-1s_10.366.pkl",
    ("cvrp", 100): "cvrp100_hgs-20s_15.563.pkl",
    ("cvrp", 200): "cvrp200_hgs-60s_19.630.pkl",
    ("cvrp", 500): "cvrp500_hgs-300s_37.154.pkl",
    ("cvrp", 1000): "cvrp1000_hgs-360s_41.171.pkl",
    ("cvrp", 2000): "cvrp2000_hgs-360s_57.181.pkl",
}

SIZE_ORIGINS = {
    ("tsp", 100): "official_native_training_size",
    ("tsp", 500): "official_paper_generalization_size",
    ("tsp", 1000): "official_paper_generalization_size",
    ("cvrp", 50): "senior_approved_scale_adaptation",
    ("cvrp", 100): "official_native_training_size",
    ("cvrp", 200): "official_paper_generalization_size",
    ("cvrp", 500): "official_paper_generalization_size",
    ("cvrp", 1000): "official_paper_generalization_size",
    ("cvrp", 2000): "senior_approved_scale_adaptation",
}


def resolve_config(problem: str, problem_size: int, protocol: str,
                   *, batch_size: int = FORMAL_BATCH_SIZE) -> dict:
    problem = problem.lower()
    size = int(problem_size)
    key = (problem, size)
    if key not in SIZE_ORIGINS:
        raise ValueError(f"unsupported formal LEHD problem/size: {problem}/{size}")
    if protocol not in FORMAL_PROTOCOLS:
        raise ValueError(f"unsupported formal LEHD protocol: {protocol}")
    if batch_size != FORMAL_BATCH_SIZE:
        raise ValueError("formal LEHD requires original-instance batch size 1")
    checkpoint = deepcopy(CHECKPOINTS[problem])
    return {
        "method": "LEHD",
        "problem": problem.upper(),
        "actual_problem_size": size,
        "trained_on_size": checkpoint["trained_on_size"],
        "config_origin": SIZE_ORIGINS[key],
        "expected_dataset_filename": DATASET_FILENAMES[key],
        "checkpoint": checkpoint,
        "protocol_label": protocol,
        "RRC_budget": RRC_BUDGETS[protocol],
        "budget_mapping_origin": (
            "project_protocol_mapping_of_author_reported_budgets"
        ),
        "inference_algorithm": (
            "official greedy construction followed by exactly RRC_budget "
            "official random reconstruction cycles with improvement-only acceptance"
        ),
        "model": deepcopy(MODEL_PARAMS),
        "original_instance_batch_size": FORMAL_BATCH_SIZE,
        "project_seed": PROJECT_SEED,
        "seed_origin": "project_reproducibility_policy",
        "official_rng_semantics": (
            "no test-time seed reset in TSPTester._test_one_batch"
            if problem == "tsp" else
            "VRPTester._test_one_batch calls torch.manual_seed(12) for every batch"
        ),
        "rng_resume_semantics": (
            "ordered BS1 processing; Python, NumPy, Torch CPU and all CUDA RNG states "
            "are checkpointed after each validated instance"
        ),
        "warmup": deepcopy(WARMUP_POLICY),
        "hardware_protocol_pending": HARDWARE_PROTOCOL_PENDING,
        "manuscript_hardware_statement": MANUSCRIPT_HARDWARE,
        "current_correctness_server_hardware": CURRENT_SERVER_HARDWARE,
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "official_subtree": str(LEHD_SUBTREE),
    }


def validate_checkpoint_path(config: dict, path: Path) -> None:
    observed = Path(path).name
    expected = config["checkpoint"]["filename"]
    if observed != expected:
        raise ValueError(f"LEHD checkpoint filename must be {expected}; observed {observed}")


def validate_checkpoint_location(config: dict, path: Path, upstream: Path) -> None:
    validate_checkpoint_path(config, path)
    expected = (Path(upstream).resolve() / config["checkpoint"]["relative_path"]).resolve()
    if Path(path).resolve() != expected:
        raise ValueError(f"LEHD checkpoint must be the pinned repository asset: {expected}")


def validate_dataset_path(config: dict, path: Path) -> None:
    observed = Path(path).name
    expected = config["expected_dataset_filename"]
    if observed != expected:
        raise ValueError(f"LEHD dataset filename must be {expected}; observed {observed}")
