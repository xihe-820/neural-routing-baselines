"""Frozen RF-TE CVRPTW50/100 official protocol identity."""
UPSTREAM_URL = "https://github.com/ai4co/routefinder"
UPSTREAM_COMMIT = "fe0e45b6df118af03c5f42db8b93a351f7629131"
RELEASE = "v0.4.0"
HF_REVISION = "4edf73d68a5915787dd2f81ba177d77e13317190"
CHECKPOINTS = {50: "checkpoints/50/rf-transformer.ckpt",
               100: "checkpoints/100/rf-transformer.ckpt"}


def protocol(problem_size):
    if problem_size not in CHECKPOINTS:
        raise ValueError("RF-TE formal scope is CVRPTW50/100")
    return {
        "method": "RF-TE", "architecture": "RouteFinderBase/Transformer",
        "problem": "CVRPTW", "problem_size": problem_size,
        "original_instance_batch_size": 1, "num_augmentations": 8,
        "augmentation": "dihedral8", "num_starts": problem_size,
        "start_selector": "all customers 1..N", "decode_type": "checkpoint test greedy",
        "temperature": "checkpoint inherited; server preflight requires 1.0",
        "return_actions": True, "selection": "max start then max augmentation",
        "cuda_autocast": True, "float32_matmul_precision": "medium",
        "official_test_seed": None, "seed_injected": False,
        "selection_origin": "official release test.py and rf-transformer checkpoint",
    }
