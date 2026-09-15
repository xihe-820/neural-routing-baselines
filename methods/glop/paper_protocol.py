"""Fail-closed registry for source-audited formal GLOP protocols."""
from __future__ import annotations

from copy import deepcopy

UPSTREAM_URL = "https://github.com/henry-yeh/GLOP"
UPSTREAM_COMMIT = "e540bc0153a0598e923e35116deeaecaf9c1cfff"
SEED = 1
ORIGINAL_BATCH_SIZE = 1
ROW_MAPPING_STATUS = "PROJECT_FROZEN"

TSP_PAPER_DISPLAY_MAPPING = {
    "GLOP (fewer)": "official_standard",
    "GLOP (more)": "official_more",
    "Appendix GLOP": "official_standard",
}
CVRP_PAPER_DISPLAY_MAPPING = {
    "GLOP (fewer)": "official_standard",
    "GLOP (more)": "project_more_revisions",
    "Appendix GLOP": "official_standard",
}

PAPER_DATASET_FILENAMES = {
    ("TSP", 100): "tsp100_concorde_7.756.pkl",
    ("TSP", 500): "tsp500_concorde_16.546.pkl",
    ("TSP", 1000): "tsp1000_concorde_23.118.pkl",
    ("TSP", 2000): "tsp2000_lkh_500_32.436.pkl",
    ("TSP", 5000): "tsp5000_lkh_500_50.968.pkl",
    ("TSP", 10000): "tsp10000_lkh_500_71.782.pkl",
    ("CVRP", 500): "cvrp500_hgs-300s_37.154.pkl",
    ("CVRP", 1000): "cvrp1000_hgs-360s_41.171.pkl",
    ("CVRP", 2000): "cvrp2000_hgs-360s_57.181.pkl",
}

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


def expected_dataset_filename(problem, size):
    try:
        return PAPER_DATASET_FILENAMES[(problem.upper(), int(size))]
    except KeyError as exc:
        raise ValueError("unsupported GLOP paper dataset problem/size") from exc


def _tsp(name, revision_iters, ri_order_width, *, decode_strategy="greedy",
         local_reconnect_augmentation=True, pruning=True,
         top_level_transforms=("identity",), paper_nominal_width=None,
         config_origin="official_release", adapted_from_size=None,
         released_code_alternative=None):
    transforms = list(top_level_transforms)
    reflection_factor = len(transforms)
    effective_count = ri_order_width * reflection_factor
    nominal = effective_count if paper_nominal_width is None else paper_nominal_width
    if nominal != effective_count:
        raise ValueError("paper width and effective TSP candidate count differ")
    result = {
        "official_protocol_name": name,
        "problem": "TSP", "revision_lens": [100, 50, 20],
        "revision_iters": list(revision_iters),
        "paper_nominal_width": nominal,
        "ri_order_width": ri_order_width,
        "internal_width": ri_order_width,
        "top_level_reflection_factor": reflection_factor,
        "effective_candidate_count": effective_count,
        "top_level_transforms": transforms,
        "decode_strategy": decode_strategy,
        "top_level_augmentation": reflection_factor > 1,
        "local_reconnect_augmentation": local_reconnect_augmentation,
        "local_augmentation": local_reconnect_augmentation,
        "pruning": pruning, "seed": SEED,
        "config_origin": config_origin,
        "rng_semantics": {
            "seed_placement": "once before official reviser construction/loading",
            "rng_seed_scope": "once per formal evaluation",
            "ri_order_scope": "one shared order set across original instances",
            "per_instance_reseed": False,
            "warmup_rng_restored": True,
        },
        "shared_ri_order_timing_policy": "charged exactly once to dataset index 0",
        "random_insertion_min_version": "0.3.0",
        "original_batch_size": ORIGINAL_BATCH_SIZE,
        "training": False, "fine_tuning": False,
    }
    if adapted_from_size is not None:
        result["adapted_from_size"] = adapted_from_size
    if released_code_alternative is not None:
        result["released_code_alternative"] = deepcopy(released_code_alternative)
    return result


def _tsp100(name, paper_width):
    if paper_width == 35:
        ri_width = 35
        transforms = ("identity",)
        alternative = {
            "released_cli_width": 35,
            "ri_order_width_after_integer_division": 8,
            "top_level_reflection_factor": 4,
            "effective_candidate_count": 32,
            "formal_registry_status": (
                "excluded_because_it_does_not_preserve_paper_nominal_width"),
        }
    elif paper_width == 140:
        ri_width = 35
        transforms = ("identity", "reflect_x", "reflect_y", "reflect_xy")
        alternative = None
    else:
        raise ValueError("unsupported TSP100 paper width")
    result = _tsp(
        name, [20, 10, 10, 5], ri_width,
        decode_strategy="sampling", local_reconnect_augmentation=False,
        pruning=False, top_level_transforms=transforms,
        paper_nominal_width=paper_width,
        config_origin="paper_faithful_candidate_orchestration",
        released_code_alternative=alternative)
    result["revision_lens"] = [100, 50, 20, 10]
    result["rng_semantics"].update({
        "sampling_rng_stream": "continuous across original instances",
        "resume_rng_policy": "completed-prefix replay",
    })
    return result


def _cvrp(problem_size, partitioner_source_size, revision_lens,
          revision_iters, name, *, config_origin, adapted_from_size=None,
          base_protocol=None, budget_rule=None):
    asset = PARTITIONER_ASSETS[partitioner_source_size]
    result = {
        "official_protocol_name": name, "problem": "CVRP",
        "problem_size": problem_size,
        "partitioner_source_size": partitioner_source_size,
        "partitioner_size": partitioner_source_size,
        "partitioner_path": asset["path"],
        "k_sparse": asset["k_sparse"], "partitioner_depth": asset["depth"],
        "global_decode_strategy": "greedy",
        "local_decode_strategy": "sampling",
        "revision_lens": list(revision_lens),
        "revision_iters": list(revision_iters),
        "internal_width": 1, "n_partition": 1,
        "local_augmentation": True, "pruning": True,
        "seed": SEED, "config_origin": config_origin,
        "rng_semantics": {
            "seed_placement": (
                "once before official reviser then partitioner construction/loading"),
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
    if adapted_from_size is not None:
        result["adapted_from_size"] = adapted_from_size
    if base_protocol is not None:
        result["base_protocol"] = base_protocol
    if budget_rule is not None:
        result["budget_rule"] = budget_rule
    return result


def _cvrp_pair(size, source_size, revision_lens, official_iters,
               *, adapted_from_size=None):
    standard_origin = (
        "author_approved_adaptation" if adapted_from_size is not None
        else "official_release")
    standard = _cvrp(
        size, source_size, revision_lens, official_iters, "official_standard",
        config_origin=standard_origin, adapted_from_size=adapted_from_size)
    more = _cvrp(
        size, source_size, revision_lens,
        [value * 2 for value in official_iters], "project_more_revisions",
        config_origin="project_defined_budget",
        adapted_from_size=adapted_from_size,
        base_protocol="official_standard",
        budget_rule="double every official revision iteration count")
    return {"official_standard": standard, "project_more_revisions": more}


def _entry(status, source, configs, mapping):
    return {
        "official_config_status": status,
        "official_source": source,
        "paper_row_mapping_status": ROW_MAPPING_STATUS,
        "paper_display_mapping": deepcopy(mapping),
        "formal_enabled": True,
        "configs": configs,
    }


TSP_PROTOCOLS = {
    100: _entry(
        "PROJECT_FROZEN_PAPER_FAITHFUL",
        "GLOP paper Appendix Table 9, Appendix A.6, official README and main.py",
        {"official_standard": _tsp100("official_standard", 35),
         "official_more": _tsp100("official_more", 140)},
        TSP_PAPER_DISPLAY_MAPPING),
    500: _entry(
        "FROZEN_OFFICIAL", "GLOP paper Tables 1/9 and official README",
        {"official_standard": _tsp("official_standard", [20, 25, 5], 1),
         "official_more": _tsp("official_more", [20, 25, 5], 10)},
        TSP_PAPER_DISPLAY_MAPPING),
    1000: _entry(
        "FROZEN_OFFICIAL", "GLOP paper Tables 1/9 and official README",
        {"official_standard": _tsp("official_standard", [20, 25, 5], 1),
         "official_more": _tsp("official_more", [20, 25, 5], 10)},
        TSP_PAPER_DISPLAY_MAPPING),
    2000: _entry(
        "AUTHOR_APPROVED_ADAPTATION",
        "author-approved adaptation of the frozen TSP1K protocols",
        {"official_standard": _tsp(
             "official_standard", [20, 25, 5], 1,
             config_origin="author_approved_adaptation", adapted_from_size=1000),
         "official_more": _tsp(
             "official_more", [20, 25, 5], 10,
             config_origin="author_approved_adaptation", adapted_from_size=1000)},
        TSP_PAPER_DISPLAY_MAPPING),
    5000: _entry(
        "AUTHOR_APPROVED_ADAPTATION",
        "author-approved adaptation of the frozen TSP10K protocols",
        {"official_standard": _tsp(
             "official_standard", [10, 20, 5], 1,
             config_origin="author_approved_adaptation", adapted_from_size=10000),
         "official_more": _tsp(
             "official_more", [50, 25, 5], 1,
             config_origin="author_approved_adaptation", adapted_from_size=10000)},
        TSP_PAPER_DISPLAY_MAPPING),
    10000: _entry(
        "FROZEN_OFFICIAL", "GLOP paper Tables 1/9 and official README",
        {"official_standard": _tsp("official_standard", [10, 20, 5], 1),
         "official_more": _tsp("official_more", [50, 25, 5], 1)},
        TSP_PAPER_DISPLAY_MAPPING),
}

CVRP_PROTOCOLS = {
    500: _entry(
        "AUTHOR_APPROVED_ADAPTATION_WITH_PROJECT_MORE",
        "author-approved adaptation of CVRP1K using the released 1K partitioner",
        _cvrp_pair(500, 1000, [20], [5], adapted_from_size=1000),
        CVRP_PAPER_DISPLAY_MAPPING),
    1000: _entry(
        "FROZEN_OFFICIAL_BASE_WITH_PROJECT_MORE",
        "GLOP paper Tables 6/10 and official README neural path",
        _cvrp_pair(1000, 1000, [20], [5]),
        CVRP_PAPER_DISPLAY_MAPPING),
    2000: _entry(
        "FROZEN_OFFICIAL_BASE_WITH_PROJECT_MORE",
        "GLOP paper Tables 6/10 and official README neural path",
        _cvrp_pair(2000, 2000, [50, 20], [5, 5]),
        CVRP_PAPER_DISPLAY_MAPPING),
}


def audit_entry(problem, size):
    table = (TSP_PROTOCOLS if problem.upper() == "TSP" else
             CVRP_PROTOCOLS if problem.upper() == "CVRP" else None)
    if table is None or int(size) not in table:
        raise ValueError("unsupported GLOP manuscript problem/size")
    return deepcopy(table[int(size)])


def formal_protocol(problem, size, protocol_name):
    """Return one enabled frozen protocol or reject it without adaptation."""
    entry = audit_entry(problem, size)
    if not entry["formal_enabled"]:
        raise ValueError(entry.get("blocking_issue", "GLOP formal protocol disabled"))
    try:
        config = entry["configs"][protocol_name]
    except KeyError as exc:
        raise ValueError("protocol name is not enabled for this problem/size") from exc
    result = deepcopy(config)
    result["problem_size"] = int(size)
    result["official_config_status"] = entry["official_config_status"]
    result["official_source"] = entry["official_source"]
    result["paper_row_mapping_status"] = entry["paper_row_mapping_status"]
    result["paper_display_mapping"] = deepcopy(entry["paper_display_mapping"])
    result["expected_dataset_filename"] = expected_dataset_filename(problem, size)
    return result


# Compatibility for audit callers; formal execution must use formal_protocol.
paper_config = audit_entry
