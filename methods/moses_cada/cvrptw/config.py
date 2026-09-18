"""Frozen MoSES(CaDA) CVRPTW50/100 official protocol identity."""
UPSTREAM_URL = "https://github.com/panyxy/moses_vrp"
UPSTREAM_COMMIT = "e301478b7a5df6d7b0b10a0543f4dee5e3c027a8"
CHECKPOINTS = {
    50: {"path": "pretrained_moses_model/cada/50/multilora_denseroute_sigmoid.ckpt",
         "sha256": "1aa499f3fce5d3412c2544c9632bbb9709309a7299b4298b535fa7e9014ef803"},
    100: {"path": "pretrained_moses_model/cada/100/multilora_denseroute_sigmoid.ckpt",
          "sha256": "2eac9b038ae4655581aa73e4dbe8ad529aefd1963368c9a92d254b6269f8aabf"},
}


def protocol(problem_size):
    if problem_size not in CHECKPOINTS:
        raise ValueError("MoSES(CaDA) formal scope is CVRPTW50/100")
    return {
        "method": "MoSES(CaDA)", "backbone": "CaDA",
        "model_name": "cada_multilora", "problem": "CVRPTW",
        "problem_size": problem_size, "original_instance_batch_size": 1,
        "num_augmentations": 8, "augmentation": "dihedral8",
        "first_augmentation_identity": True, "num_starts": problem_size,
        "policy_test_decode_type": "greedy", "decode_type": "greedy",
        "multistart": True, "start_selector": "all customers 1..N",
        "return_actions": True, "selection": "max start then max augmentation",
        "lora_activation": "sigmoid", "lora_rank": [32] * 5,
        "lora_alpha": 1.0, "lora_n_experts": 4, "lora_top_k": 4,
        "lora_temperature": 1.0, "lora_use_trainable_layer": True,
        "lora_use_dynamic_topK": False, "lora_use_basis_variants": False,
        "lora_use_basis_variants_as_input": False, "lora_use_linear": False,
        "normalization": "rms", "encoder_use_prenorm": False,
        "encoder_use_post_layers_norm": False, "mlp_activation": "silu",
        "attn_sparse_ratio": 0.5, "sparse_applied_to_score": True,
        "cuda_autocast": True, "float32_matmul_precision": "medium",
        "official_test_seed": None, "seed_injected": False,
        "selection_origin": "official scripts/test_script.sh",
    }
