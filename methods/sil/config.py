"""Frozen SIL source, checkpoint, size, and evaluation protocol registry."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

UPSTREAM_URL = "https://github.com/CIAM-Group/SIL"
UPSTREAM_COMMIT = "9ec783e90a1631f7b95f84eb20f8f9751cb45c10"
SEED = 123
FORMAL_BATCH_SIZE = 1

CHECKPOINTS = {
    "tsp1k": {
        "filename": "checkpoint-tsp1k.pt",
        "drive_file_id": "1nT2_JkghMjasHpZY2i_dhrGnhvbOlxTV",
        "official_source": "SIL README Google Drive release folder",
        "actual_sha256": None,
    },
    "tsp5k": {
        "filename": "checkpoint-tsp5k.pt",
        "drive_file_id": "1v2WHGKi8T-lTqE7NJsHRBno8AyKUY0n3",
        "official_source": "SIL README Google Drive release folder",
        "actual_sha256": None,
    },
    "tsp10k": {
        "filename": "checkpoint-tsp10k.pt",
        "drive_file_id": "1UTKPLAOqw2Lco-Zx2HZ64qYn2O3GrrQ5",
        "official_source": "SIL README Google Drive release folder",
        "actual_sha256": None,
    },
    "cvrp1k": {
        "filename": "checkpoint-cvrp1k.pt",
        "drive_file_id": "1yNUPciWXF-gSnsnRxXpMjO5rqAt0OOHj",
        "official_source": "SIL README Google Drive release folder",
        "actual_sha256": None,
    },
}

SIZE_REGISTRY = {
    ("tsp", 500): ("tsp1k", 1000, "senior_approved_adaptation"),
    ("tsp", 1000): ("tsp1k", 1000, "official_native"),
    ("tsp", 2000): ("tsp1k", 1000, "senior_approved_adaptation"),
    ("tsp", 5000): ("tsp5k", 5000, "official_native"),
    ("tsp", 10000): ("tsp10k", 10000, "official_native"),
    ("cvrp", 500): ("cvrp1k", 1000, "senior_approved_adaptation"),
    ("cvrp", 1000): ("cvrp1k", 1000, "official_native"),
    ("cvrp", 2000): ("cvrp1k", 1000, "senior_approved_adaptation"),
}

PAPER_BUDGETS = {
    "fewer": 50,
    "more": 500,
}


def resolve_config(problem: str, problem_size: int, budget: str,
                   *, batch_size: int = FORMAL_BATCH_SIZE) -> dict:
    problem = problem.lower()
    key = (problem, int(problem_size))
    if key not in SIZE_REGISTRY:
        raise ValueError(f"unsupported SIL problem/size: {problem}/{problem_size}")
    if batch_size != FORMAL_BATCH_SIZE:
        raise ValueError("formal SIL integration requires original-instance batch size 1")
    if budget not in (*PAPER_BUDGETS, "greedy_diagnostic"):
        raise ValueError(f"unsupported SIL budget: {budget}")
    checkpoint_key, adapted_from, origin = SIZE_REGISTRY[key]
    is_greedy = budget == "greedy_diagnostic"
    protocol = {
        "method": "SIL",
        "problem": problem.upper(),
        "actual_problem_size": int(problem_size),
        "config_origin": origin,
        "adapted_from_size": adapted_from,
        "checkpoint_key": checkpoint_key,
        "checkpoint": deepcopy(CHECKPOINTS[checkpoint_key]),
        "budget_label": budget,
        "budget": 0 if is_greedy else PAPER_BUDGETS[budget],
        "protocol_pending": False,
        "evaluation_mapping": (
            "diagnostic official greedy path" if is_greedy else
            f"project mapping to paper-reported PRC{PAPER_BUDGETS[budget]}"
        ),
        "random_insertion": not is_greedy,
        "PRC": True,
        "repair_max_sub_length": 1000,
        "pomo_size": 1,
        "beam_width": 16,
        "decode_method": "greedy",
        "k_nearest": 1,
        "k_nearest_num": 1000,
        "use_k_nearest": not is_greedy,
        "knn_activation_condition": "remaining/new data length > k_nearest_num",
        "initial_knn_path_active": (not is_greedy and int(problem_size) > 1000),
        "seed": SEED,
        "original_instance_batch_size": FORMAL_BATCH_SIZE,
        "rng_semantics": (
            "official tester seed=123; one persistent tester processes ordered BS1 instances; "
            "resume restores Python/NumPy/Torch CPU/CUDA states"
        ),
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    if problem == "tsp":
        protocol["model"] = {
            "embedding_dim": 128, "encoder_layer_num": 6, "qkv_dim": 16,
            "head_num": 8, "logit_clipping": 10, "ff_hidden_dim": 512,
            "eval_type": "argmax", "use_k_nearest": not is_greedy,
            "k_nearest_num": 1000,
        }
    else:
        protocol["model"] = {
            "embedding_dim": 128, "decoder_layer_num": 6, "qkv_dim": 16,
            "head_num": 8, "logit_clipping": 10, "ff_hidden_dim": 512,
            "eval_type": "argmax", "use_k_nearest": not is_greedy,
            "k_nearest_num": 1000,
        }
    return protocol


def validate_checkpoint_path(config: dict, path: Path) -> None:
    if Path(path).name != config["checkpoint"]["filename"]:
        raise ValueError(
            f"checkpoint filename must be {config['checkpoint']['filename']}")
