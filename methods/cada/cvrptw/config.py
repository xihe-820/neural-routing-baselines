"""Frozen CaDA CVRPTW50/100 official protocol identity."""
UPSTREAM_URL = "https://github.com/CIAM-Group/CaDA"
UPSTREAM_COMMIT = "b9868e1e09b3a1a3754960d830e801e51c7cb38d"
CHECKPOINTS = {50: "50/result/2024-1111-1139/checkpoint-300.pt",
               100: "100/result/2024-1121-1355/checkpoint-300.pt"}


def protocol(problem_size):
    if problem_size not in CHECKPOINTS:
        raise ValueError("CaDA formal scope is CVRPTW50/100")
    return {
        "method": "CaDA", "architecture": "official size-specific CaDA",
        "problem": "CVRPTW", "problem_size": problem_size,
        "original_instance_batch_size": 1, "num_augmentations": 8,
        "augmentation": "official StateAugmentation/dihedral8",
        "num_starts": problem_size, "start_selector": "all customers 1..N",
        "decode_type": "greedy", "temperature": None,
        "return_actions": False, "action_capture": "temporary env.get_reward wrapper",
        "selection": "max POMO then max augmentation",
        "official_seed": 7, "numpy_seed_injected": False,
        "task_prompt": [1, 0, 1, 0, 0],
        "selection_origin": "official size-specific trainer validation",
    }
