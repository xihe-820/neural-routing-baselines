"""Source-audited GLOP paper configurations; no inference is performed here."""

UPSTREAM_URL = "https://github.com/henry-yeh/GLOP"
UPSTREAM_COMMIT = "e540bc0153a0598e923e35116deeaecaf9c1cfff"

REVISER_SHA256 = {
    10: "41bd9e05d5f623a6a7978be0063354d75f6f8df62e0ed368789ca449f41922f4",
    20: "6771bf6b955fe26004f378c1ab0a2068c3048d717325a62b85a279e0ec22a865",
    50: "25189e74e1e0323ced3016e9c7495c2c8d8ae082961e8696dffff79f5db1d4a6",
    100: "3810b460f210de35b5d4bd1f680f505ff4619823880652be1c7f35f320584451",
}

PARTITIONER_ASSETS = {
    1000: {
        "path": "Partitioner/cvrp/cvrp-1000.pt",
        "sha256": "3dd5ad73da831a69f65674cf30ebe910f2b74fcfeb0d57e42d0a0efb3139663b",
        "size_bytes": 1951194,
        "k_sparse": 100,
        "depth": 12,
        "parameter_count": 148513,
    },
    2000: {
        "path": "Partitioner/cvrp/cvrp-2000.pt",
        "sha256": "988458678a06297edb767bb1c710074e3ad2d0a14f7f5b85934530f14c8ab89b",
        "size_bytes": 1951194,
        "k_sparse": 200,
        "depth": 12,
        "parameter_count": 148513,
    },
}


def _tsp(revision_lens, revision_iters, width, *, decode="greedy"):
    return {
        "revision_lens": list(revision_lens),
        "revision_iters": list(revision_iters),
        "width": width,
        "decode_strategy": decode,
        "local_augmentation": True,
        "top_level_augmentation": False,
        "pruning": True,
        "original_batch_size": 1,
    }


# Table 9 in the official paper. "standard" is the paper's plain GLOP row;
# "more" is its GLOP (more revisions) row. The manuscript's appendix does not
# say which one its single GLOP row denotes, so these are evidence, not a final
# manuscript-row selection.
TSP_OFFICIAL = {
    100: {
        "standard": {
            **_tsp([100, 50, 20, 10], [20, 10, 10, 5], 35, decode="sampling"),
            "local_augmentation": False,
            "top_level_augmentation": "UNRESOLVED_PAPER_CODE_MISMATCH",
            "pruning": None,
        },
        "more": {
            **_tsp([100, 50, 20, 10], [20, 10, 10, 5], 140, decode="sampling"),
            "local_augmentation": False,
            "top_level_augmentation": "CODE_FORCES_FOUR_REFLECTIONS",
            "pruning": False,
        },
        "status": "NEED_AUTHOR_DECISION",
        "reason": "cross-distribution-only official setting and paper/code width-augmentation ambiguity",
    },
    500: {
        "standard": _tsp([100, 50, 20], [20, 25, 5], 1),
        "more": _tsp([100, 50, 20], [20, 25, 5], 10),
        "status": "NEED_AUTHOR_DECISION",
        "reason": "appendix single-row selection is undefined",
    },
    1000: {
        "standard": _tsp([100, 50, 20], [20, 25, 5], 1),
        "more": _tsp([100, 50, 20], [20, 25, 5], 10),
        "status": "NEED_AUTHOR_DECISION",
        "reason": "appendix single-row selection is undefined",
    },
    2000: {
        "standard": None,
        "more": None,
        "status": "NEED_AUTHOR_DECISION",
        "reason": "no official paper, README, script, or released result configuration",
    },
    5000: {
        "standard": None,
        "more": None,
        "status": "NEED_AUTHOR_DECISION",
        "reason": "no official paper, README, script, or released result configuration",
    },
    10000: {
        "standard": _tsp([100, 50, 20], [10, 20, 5], 1),
        "more": _tsp([100, 50, 20], [50, 25, 5], 1),
        "status": "NEED_AUTHOR_DECISION",
        "reason": "appendix label differs from its first half and the main/appendix selection is not stated",
    },
}


CVRP_OFFICIAL = {
    500: {
        "config": None,
        "status": "BLOCKED",
        "reason": "official K_SPARSE/checkpoint/config/result coverage has no CVRP500",
    },
    1000: {
        "config": {
            "partitioner": 1000,
            "global_decode": "greedy",
            "revision_lens": [20],
            "revision_iters": [5],
            "local_decode_strategy": "sampling",
            "local_augmentation": True,
            "pruning": True,
            "width": 1,
            "n_partition": 1,
            "original_batch_size": 1,
        },
        "status": "NEED_AUTHOR_DECISION",
        "reason": "official neural GLOP-G is single-config; manuscript asks for fewer and more",
    },
    2000: {
        "config": {
            "partitioner": 2000,
            "global_decode": "greedy",
            "revision_lens": [50, 20],
            "revision_iters": [5, 5],
            "local_decode_strategy": "sampling",
            "local_augmentation": True,
            "pruning": True,
            "width": 1,
            "n_partition": 1,
            "original_batch_size": 1,
        },
        "status": "NEED_AUTHOR_DECISION",
        "reason": "official neural GLOP-G is single-config; manuscript asks for fewer and more",
    },
}


def paper_config(problem, size):
    """Return audited evidence, rejecting sizes outside the manuscript scope."""
    table = TSP_OFFICIAL if problem.upper() == "TSP" else CVRP_OFFICIAL if problem.upper() == "CVRP" else None
    if table is None or int(size) not in table:
        raise ValueError("unsupported GLOP manuscript problem/size")
    return table[int(size)]
