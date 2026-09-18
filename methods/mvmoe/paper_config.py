"""Immutable MVMoE/4E settings for formal paper evaluation."""
from __future__ import annotations


UPSTREAM_URL = "https://github.com/RoyalSkye/Routing-MVMoE"
UPSTREAM_COMMIT = "af29e5af0595f94f3ecc3bc46d72df1089a62682"

MODEL_CONFIG = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "encoder_layer_num": 6,
    "decoder_layer_num": 1,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "num_experts": 4,
    "eval_type": "argmax",
    "norm": "instance",
    "norm_loc": "norm_last",
    "expert_loc": ["Enc0", "Enc1", "Enc2", "Enc3", "Enc4", "Enc5", "Dec"],
    "problem": "Train_ALL",
    "topk": 2,
    "routing_level": "node",
    "routing_method": "input_choice",
}


def paper_inference_config(problem_size, *, problem, original_batch_size=1):
    """Return the non-overridable formal inference protocol."""
    if int(problem_size) not in (50, 100):
        raise ValueError("MVMoE paper evaluation supports only problem_size 50 or 100")
    if problem not in ("CVRP", "CVRPTW"):
        raise ValueError("MVMoE paper evaluation supports CVRP or CVRPTW")
    if original_batch_size not in (1, 10):
        raise ValueError("MVMoE formal original batch size is 1 or 10")
    config = {
        "variant": "MVMoE/4E",
        "model_type": "MOE",
        "num_experts": 4,
        "routing_level": "node",
        "routing_method": "input_choice",
        "problem_size": int(problem_size),
        "pomo_size": int(problem_size),
        "original_batch_size": original_batch_size,
        "aug_factor": 8,
        "eval_type": "argmax",
        "seed": 2024,
        "fine_tune_epochs": 0,
        "training": False,
        "backward": False,
        "optimizer_created": False,
        "model": MODEL_CONFIG,
    }
    if problem == "CVRPTW":
        config.update(loc_scaler=None, speed=1.0)
    return config
