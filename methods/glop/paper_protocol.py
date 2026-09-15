"""Fail-closed registry for source-audited formal GLOP protocols."""
from __future__ import annotations

from copy import deepcopy

UPSTREAM_URL = "https://github.com/henry-yeh/GLOP"
UPSTREAM_COMMIT = "e540bc0153a0598e923e35116deeaecaf9c1cfff"
SEED = 1
ORIGINAL_BATCH_SIZE = 1
ROW_MAPPING_STATUS = "UNRESOLVED_MANUSCRIPT_MAPPING"

REVISER_ASSETS = {
    10: {"path": "Reviser-stage2/reviser_10/epoch-299.pt", "args_path": "Reviser-stage2/reviser_10/args.json", "sha256": "41bd9e05d5f623a6a7978be0063354d75f6f8df62e0ed368789ca449f41922f4", "args_sha256": "e21195ed71321b91ca2517e49b4a7556c27239ed1017d603e34d25a49742879c", "size_bytes": 24394377, "args_size_bytes": 1004, "parameter_count": 1301888},
    20: {"path": "Reviser-stage2/reviser_20/epoch-299.pt", "args_path": "Reviser-stage2/reviser_20/args.json", "sha256": "6771bf6b955fe26004f378c1ab0a2068c3048d717325a62b85a279e0ec22a865", "args_sha256": "66171fc5178ee7fc2b8ddcda8a7c90e804a0e9c9eb960375f82df40cbc228ef6", "size_bytes": 25034377, "args_size_bytes": 1004, "parameter_count": 1301888},
    50: {"path": "Reviser-stage2/reviser_50/epoch-299.pt", "args_path": "Reviser-stage2/reviser_50/args.json", "sha256": "25189e74e1e0323ced3016e9c7495c2c8d8ae082961e8696dffff79f5db1d4a6", "args_sha256": "ae311d53fe1e36573a609cc7bab75be1f346a577576c36a1d30ff799cbdcc76c", "size_bytes": 27594377, "args_size_bytes": 1004, "parameter_count": 1301888},
    100: {"path": "Reviser-stage2/reviser_100/epoch-299.pt", "args_path": "Reviser-stage2/reviser_100/args.json", "sha256": "3810b460f210de35b5d4bd1f680f505ff4619823880652be1c7f35f320584451", "args_sha256": "b99400a52c2dd4b6bdbd221f17432ad65d0e9d585207032ee65126b78c0354d9", "size_bytes": 31434377, "args_size_bytes": 1022, "parameter_count": 1301888},
}

PARTITIONER_ASSETS = {
    1000: {"path": "Partitioner/cvrp/cvrp-1000.pt", "sha256": "3dd5ad73da831a69f65674cf30ebe910f2b74fcfeb0d57e42d0a0efb3139663b", "size_bytes": 1951194, "k_sparse": 100, "depth": 12, "parameter_count": 148513},
    2000: {"path": "Partitioner/cvrp/cvrp-2000.pt", "sha256": "988458678a06297edb767bb1c710074e3ad2d0a14f7f5b85934530f14c8ab89b", "size_bytes": 1951194, "k_sparse": 200, "depth": 12, "parameter_count": 148513},
}


def _tsp(name, revision_iters, width):
    return {
        "official_protocol_name": name,
        "problem": "TSP", "revision_lens": [100, 50, 20],
        "revision_iters": list(revision_iters), "internal_width": width,
        "decode_strategy": "greedy", "local_augmentation": True,
        "top_level_augmentation": False, "pruning": True,
        "seed": SEED,
        "rng_semantics": {
            "seed_placement": "once before official reviser construction/loading",
            "rng_seed_scope": "once per formal evaluation",
            "ri_order_scope": "one shared order set across original instances",
            "per_instance_reseed": False,
            "warmup_rng_restored": True,
        },
        "random_insertion_min_version": "0.3.0",
        "original_batch_size": ORIGINAL_BATCH_SIZE,
        "training": False, "fine_tuning": False,
    }


def _cvrp(size, revision_lens, revision_iters):
    asset = PARTITIONER_ASSETS[size]
    return {
        "official_protocol_name": "official_single", "problem": "CVRP",
        "partitioner_size": size, "partitioner_path": asset["path"],
        "k_sparse": asset["k_sparse"], "partitioner_depth": asset["depth"],
        "global_decode_strategy": "greedy",
        "local_decode_strategy": "sampling",
        "revision_lens": list(revision_lens),
        "revision_iters": list(revision_iters),
        "internal_width": 1, "n_partition": 1,
        "local_augmentation": True, "pruning": True,
        "seed": SEED,
        "rng_semantics": {
            "seed_placement": (
                "once before official reviser then partitioner "
                "construction/loading"),
            "rng_seed_scope": "once per formal sequential evaluation",
            "sampling_rng_stream": "continuous across original instances",
            "per_instance_reseed": False,
            "warmup_rng_restored": True,
            "resume_rng_policy": "completed-prefix replay",
            "prepared_offset_policy": "offset zero only",
        },
        "random_insertion_min_version": "0.3.0",
        "original_batch_size": ORIGINAL_BATCH_SIZE,
        "training": False, "fine_tuning": False,
    }


TSP_PROTOCOLS = {
    100: {
        "official_config_status": "OFFICIAL_CONFIG_EXISTS_EXECUTION_AMBIGUOUS",
        "official_source": "GLOP paper Appendix Table 9 and TSP100 cross-distribution evaluation",
        "paper_row_mapping_status": ROW_MAPPING_STATUS,
        "formal_enabled": False,
        "configs": {
            "official_standard": {"revision_lens": [100, 50, 20, 10], "revision_iters": [20, 10, 10, 5], "paper_width": 35, "decode_strategy": "sampling"},
            "official_more": {"revision_lens": [100, 50, 20, 10], "revision_iters": [20, 10, 10, 5], "paper_width": 140, "decode_strategy": "sampling"},
        },
        "blocking_issue": "PAPER_CODE_WIDTH_AND_TOP_LEVEL_AUGMENTATION_AMBIGUITY",
    },
    500: {
        "official_config_status": "FROZEN_OFFICIAL",
        "official_source": "GLOP paper Tables 1/9 and official README",
        "paper_row_mapping_status": ROW_MAPPING_STATUS, "formal_enabled": True,
        "configs": {"official_standard": _tsp("official_standard", [20, 25, 5], 1),
                    "official_more": _tsp("official_more", [20, 25, 5], 10)},
    },
    1000: {
        "official_config_status": "FROZEN_OFFICIAL",
        "official_source": "GLOP paper Tables 1/9 and official README",
        "paper_row_mapping_status": ROW_MAPPING_STATUS, "formal_enabled": True,
        "configs": {"official_standard": _tsp("official_standard", [20, 25, 5], 1),
                    "official_more": _tsp("official_more", [20, 25, 5], 10)},
    },
    2000: {"official_config_status": "NO_OFFICIAL_PRIMARY_CONFIG",
           "official_source": "Table 16 is a time-complexity experiment, not the primary result protocol",
           "paper_row_mapping_status": ROW_MAPPING_STATUS, "formal_enabled": False,
           "configs": {}, "blocking_issue": "NO_OFFICIAL_PRIMARY_CONFIG"},
    5000: {"official_config_status": "NO_OFFICIAL_PRIMARY_CONFIG",
           "official_source": "Table 16 is a time-complexity experiment, not the primary result protocol",
           "paper_row_mapping_status": ROW_MAPPING_STATUS, "formal_enabled": False,
           "configs": {}, "blocking_issue": "NO_OFFICIAL_PRIMARY_CONFIG"},
    10000: {
        "official_config_status": "FROZEN_OFFICIAL",
        "official_source": "GLOP paper Tables 1/9 and official README",
        "paper_row_mapping_status": ROW_MAPPING_STATUS, "formal_enabled": True,
        "configs": {"official_standard": _tsp("official_standard", [10, 20, 5], 1),
                    "official_more": _tsp("official_more", [50, 25, 5], 1)},
    },
}

CVRP_PROTOCOLS = {
    500: {"official_config_status": "UNSUPPORTED_RELEASED_SYNTHETIC_PATH",
          "official_source": "no released CVRP500 K_SPARSE entry, checkpoint, command, or result",
          "paper_row_mapping_status": ROW_MAPPING_STATUS, "formal_enabled": False,
          "configs": {}, "blocking_issue": "BLOCKED_OFFICIAL_SYNTHETIC_CONFIG_MISSING"},
    1000: {"official_config_status": "FROZEN_OFFICIAL_SINGLE_CONFIG",
           "official_source": "GLOP paper Tables 6/10 and official README neural path",
           "paper_row_mapping_status": ROW_MAPPING_STATUS, "formal_enabled": True,
           "configs": {"official_single": _cvrp(1000, [20], [5])}},
    2000: {"official_config_status": "FROZEN_OFFICIAL_SINGLE_CONFIG",
           "official_source": "GLOP paper Tables 6/10 and official README neural path",
           "paper_row_mapping_status": ROW_MAPPING_STATUS, "formal_enabled": True,
           "configs": {"official_single": _cvrp(2000, [50, 20], [5, 5])}},
}


def audit_entry(problem, size):
    table = TSP_PROTOCOLS if problem.upper() == "TSP" else CVRP_PROTOCOLS if problem.upper() == "CVRP" else None
    if table is None or int(size) not in table:
        raise ValueError("unsupported GLOP manuscript problem/size")
    return deepcopy(table[int(size)])


def formal_protocol(problem, size, protocol_name):
    """Return one enabled official protocol or reject it without adaptation."""
    entry = audit_entry(problem, size)
    if not entry["formal_enabled"]:
        raise ValueError(entry.get("blocking_issue", "GLOP formal protocol disabled"))
    try:
        config = entry["configs"][protocol_name]
    except KeyError as exc:
        raise ValueError("official protocol name is not enabled for this problem/size") from exc
    result = deepcopy(config)
    result["problem_size"] = int(size)
    result["official_config_status"] = entry["official_config_status"]
    result["official_source"] = entry["official_source"]
    result["paper_row_mapping_status"] = entry["paper_row_mapping_status"]
    return result


# Compatibility for audit callers; formal execution must use formal_protocol.
paper_config = audit_entry
