"""Frozen SIL source, checkpoint, size, and evaluation protocol registry."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from methods.glop.paper_protocol import expected_dataset_filename

UPSTREAM_URL = "https://github.com/CIAM-Group/SIL"
UPSTREAM_COMMIT = "9ec783e90a1631f7b95f84eb20f8f9751cb45c10"
SEED = 123
FORMAL_BATCH_SIZE = 1
FORMAL_SIZES = {
    "tsp": (1000, 2000, 5000, 10000),
    "cvrp": (1000, 2000),
}
FORMAL_PROTOCOLS = ("greedy", "fewer", "more")
DIAGNOSTIC_PROTOCOLS = ("greedy_diagnostic",)
HARDWARE_PROTOCOL_PENDING = True
MANUSCRIPT_HARDWARE = "H800"
CURRENT_SERVER_HARDWARE = "NVIDIA RTX 4090"
WARMUP_POLICY = {
    "batches": 1,
    "excluded_from_timing": True,
    "excluded_from_records": True,
    "rng_state_restored": True,
    "resume_policy": (
        "skip after any completed record; an empty resumed run may repeat the "
        "RNG-neutral warm-up"
    ),
}

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
    ("tsp", 1000): {
        "checkpoint_key": "tsp1k", "setting_size": 1000,
        "adapted_from_size": None, "config_origin": "official_native",
    },
    ("tsp", 2000): {
        "checkpoint_key": "tsp1k", "setting_size": 1000,
        "adapted_from_size": 1000, "config_origin": "senior_approved_adaptation",
    },
    ("tsp", 5000): {
        "checkpoint_key": "tsp5k", "setting_size": 5000,
        "adapted_from_size": None, "config_origin": "official_native",
    },
    ("tsp", 10000): {
        "checkpoint_key": "tsp10k", "setting_size": 10000,
        "adapted_from_size": None, "config_origin": "official_native",
    },
    ("cvrp", 1000): {
        "checkpoint_key": "cvrp1k", "setting_size": 1000,
        "adapted_from_size": None, "config_origin": "official_native",
    },
    ("cvrp", 2000): {
        "checkpoint_key": "cvrp1k", "setting_size": 1000,
        "adapted_from_size": 1000, "config_origin": "senior_approved_adaptation",
    },
}

PAPER_BUDGETS = {
    "greedy": 0,
    "fewer": 50,
    "more": 500,
}


def effective_repair_max(problem_size: int, nominal_max: int = 1000) -> int:
    problem_size, nominal_max = int(problem_size), int(nominal_max)
    if problem_size < 4 or nominal_max < 4:
        raise ValueError("repair maximum and problem size must both be at least 4")
    return min(problem_size, nominal_max)


def resolve_config(problem: str, problem_size: int, budget: str,
                   *, batch_size: int = FORMAL_BATCH_SIZE,
                   allow_diagnostic: bool = False) -> dict:
    problem = problem.lower()
    key = (problem, int(problem_size))
    if key not in SIZE_REGISTRY:
        raise ValueError(f"unsupported SIL problem/size: {problem}/{problem_size}")
    if batch_size != FORMAL_BATCH_SIZE:
        raise ValueError("formal SIL integration requires original-instance batch size 1")
    allowed = FORMAL_PROTOCOLS + (DIAGNOSTIC_PROTOCOLS if allow_diagnostic else ())
    if budget not in allowed:
        raise ValueError(f"unsupported SIL budget: {budget}")
    size_config = SIZE_REGISTRY[key]
    checkpoint_key = size_config["checkpoint_key"]
    is_greedy = budget in {"greedy", "greedy_diagnostic"}
    is_diagnostic = budget in DIAGNOSTIC_PROTOCOLS
    protocol = {
        "method": "SIL",
        "problem": problem.upper(),
        "actual_problem_size": int(problem_size),
        "setting_size": size_config["setting_size"],
        "config_origin": size_config["config_origin"],
        "adapted_from_size": size_config["adapted_from_size"],
        "expected_dataset_filename": expected_dataset_filename(problem, problem_size),
        "checkpoint_key": checkpoint_key,
        "checkpoint": deepcopy(CHECKPOINTS[checkpoint_key]),
        "budget_label": budget,
        "budget": 0 if is_diagnostic else PAPER_BUDGETS[budget],
        "artifact_class": "diagnostic" if is_diagnostic else "formal_paper_protocol",
        "paper_result_eligible": not is_diagnostic,
        "protocol_pending": False,
        "evaluation_mapping": (
            "diagnostic official native greedy path" if is_diagnostic else
            "manuscript SIL (Greedy) using official pure greedy path" if is_greedy else
            f"project mapping to paper-reported PRC{PAPER_BUDGETS[budget]}"
        ),
        "random_insertion": not is_greedy,
        "PRC": True,
        "repair_max_sub_length_nominal": 1000,
        "repair_max_sub_length_effective": effective_repair_max(problem_size, 1000),
        "repair_max_rule": "min(actual_problem_size, nominal_max)",
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
        "warmup": deepcopy(WARMUP_POLICY),
        "hardware_protocol_pending": HARDWARE_PROTOCOL_PENDING,
        "manuscript_hardware_statement": MANUSCRIPT_HARDWARE,
        "current_correctness_server_hardware": CURRENT_SERVER_HARDWARE,
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


def validate_dataset_path(config: dict, path: Path) -> None:
    actual = Path(path).name
    expected = config["expected_dataset_filename"]
    if actual != expected:
        raise ValueError(f"dataset filename must be {expected}; observed {actual}")
