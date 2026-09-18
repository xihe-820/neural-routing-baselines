"""Frozen identity and protocol for the standalone CVRPTW SymNCO E1 runs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from common.hashing import sha256_file


BASE_IDENTITY = "RS4CO 138c95efb34a5910ab618f68987098a7312965dd"
SERVER_ROOT = Path("/inspire/hdd/global_user/majiale-253108540229/zhang")
SOURCE_SNAPSHOT = SERVER_ROOT / "RS4CO_symnco_baseline_138c95e"
RUN_ROOT = SERVER_ROOT / "cvrptw_symnco_runs"

CHECKPOINTS = {
    50: str(RUN_ROOT / "train50-cache-b512-seed1234/best.pt"),
    100: str(RUN_ROOT / "train100-cache-b512-seed1234/best.pt"),
    200: str(RUN_ROOT / "train200-cache-b256-seed1234/best.pt"),
}
CHECKPOINT_HASHES = {
    50: "6be6ed8c5b40330db0f2605f2cf354ee006cf8d5d6564726fd904a0b2bcf1395",
    100: "e4aad3807cf75864e561913905cf350cd986be01aea714c8cab8a6f1a27e8144",
    200: "23c18b2ab2ff966c90a07bae51f63fcf1f5c33e9a56018c0d972204a32207479",
}
HISTORICAL_RESULTS = {
    50: str(RUN_ROOT / "train50-cache-b512-seed1234/test50-E1.json"),
    100: str(RUN_ROOT / "train100-cache-b512-seed1234/test100-E1.json"),
    200: str(RUN_ROOT / "train200-cache-b256-seed1234/test200-E1.json"),
}

SNAPSHOT_FILES = {
    "baselines/cvrptw_symnco/PERFORMANCE.md": "5f6ae50cd97092b58a26accf0d0858474815d149e8590c686c7d7a3d2bcbb054",
    "baselines/cvrptw_symnco/README.md": "a947e3205e44c84cd4f4ce987dc1f332f1e7e78feeeda94d10596787c85947a1",
    "baselines/cvrptw_symnco/__init__.py": "99ab9ae66cb06fe5e266a12bc2d91ab570af861c16f1c3aec2168fe8afd4a0c0",
    "baselines/cvrptw_symnco/benchmark_data.py": "3d6ee0daf79b867c39f55118d26b77b006a666ab0ccccc23a8f436f02c057162",
    "baselines/cvrptw_symnco/check.py": "8fe286fd16380920aaa022ec7cd9f7ca6ee2a314114dca52ee6d3d5b37b8921c",
    "baselines/cvrptw_symnco/data.py": "ebe7e1f13c4cf45b1270905354134f36f4a15257e1f4ff058acf4996f88ef1b7",
    "baselines/cvrptw_symnco/env.py": "6613a32f95c88101e552969b4527aa666422d33050abe1255a51389512e9ec3b",
    "baselines/cvrptw_symnco/evaluate.py": "479a39cd03e9e0435fca01b7823ca8f2cd26eb7479985e495028708e344da81d",
    "baselines/cvrptw_symnco/model.py": "b5b20a3fd1f3f112a2d8130b94954fbc0d179bf0976da4e8ca8cc74ddba9b50a",
    "baselines/cvrptw_symnco/nn/__init__.py": "5f5bddf1d31f7003223dc67a9ed3fe9d9c7960451e075da942bd66663353ec6e",
    "baselines/cvrptw_symnco/nn/gat_layer.py": "7119495d518b3f57967119c1cedbd0412eb4af2af1d4d94df0c9c16f4829266c",
    "baselines/cvrptw_symnco/nn/mha.py": "10b13f317a914113242fa25aba0cbae8e6f96e57530249d600c276b6f3a8d077",
    "baselines/cvrptw_symnco/nn/mlp.py": "d8802ba587ca684af043c9ec27f40e2c6a928ce4d2cf57c9d4acf1d85a1a9025",
    "baselines/cvrptw_symnco/nn/ops.py": "3c267dec6b6585adf3a241a6dff831cd52e67c14121e7c02b49f7914ece09834",
    "baselines/cvrptw_symnco/paper_timing.py": "ca50c5a7d6a78ce755b6d4237f5769ea7cce75404b8667aefbd0db1a24c2d5c7",
    "baselines/cvrptw_symnco/pipeline.py": "49fe4751fc81e467f82f4e98ae0debf072b66b69b187d29c03c0571453b0f257",
    "baselines/cvrptw_symnco/tensors.py": "807a47c8c5585cbb14038fca991d2a11656a5a548e35cc955f2874b8ec72ee78",
}
SNAPSHOT_MANIFEST_SHA256 = "9c5f2c9cacafb4a60e25b99161a1027ff35cff64e1c6af70ff6b93a60a446a2f"

TIMING_SEMANTICS = (
    "native original-instance inference-batch wall-clock seconds; batch size is recorded "
    "explicitly and latency is never divided by batch size; host-to-device input transfer is "
    "outside the timed region; CUDA synchronize precedes perf_counter; the timed region contains "
    "E1 identity plus three per-instance seeded rigid geometry views, model forward, greedy "
    "decode, actions/lengths device-to-host transfer, and independent per-original feasible "
    "candidate selection by checked closed Euclidean cost, then CUDA synchronize; checkpoint/model/"
    "dataset loading, E0 warm-up, hashing, post-timing shared validation, ML4CO-Kit validation, "
    "artifact I/O, and aggregation are excluded"
)


def protocol(problem_size, batch_size=1):
    if problem_size not in CHECKPOINTS:
        raise ValueError("SymNCO formal scope is CVRPTW50/100/200")
    if batch_size not in (1, 10):
        raise ValueError("SymNCO original-instance batch size is 1 or 10")
    return {
        "method": "SymNCO", "problem": "CVRPTW", "problem_size": problem_size,
        "identity": "standalone raw CVRPTW SymNCO E1",
        "base_identity": BASE_IDENTITY,
        "model_architecture": {
            "hidden_dim": 128, "num_heads": 8, "num_layers": 3,
            "feedforward_hidden": 512, "normalization": "batch",
            "activation": "ReLU", "customer_input": 6, "depot_input": 5,
            "context_input": 130, "tanh_clipping": 10.0,
        },
        "original_instance_batch_size": batch_size,
        "num_augmentations": 4,
        "augmentation": "identity plus 3 SHA256(seed:raw_instance_id)-seeded rigid views",
        "first_augmentation_identity": True,
        "num_rollouts": 1, "candidate_axis": "rollout_index",
        "forced_customer_starts": 0, "decode_type": "greedy",
        "temperature": 1.0, "tanh_clipping": 10.0, "seed": 1234,
        "selection": "minimum (independently checked cost, augmentation, rollout) per original",
        "local_search": False, "repair": False,
        "objective": "closed raw-coordinate Euclidean route length",
        "input": "raw coordinates/TW/service; demand divided by raw capacity once for neural feature",
    }


def validate_snapshot(root):
    root = Path(root).expanduser().resolve()
    observed = {}
    for relative, expected in SNAPSHOT_FILES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"SymNCO snapshot file missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SymNCO snapshot SHA256 mismatch: {relative}")
        observed[relative] = actual
    encoded = json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()
    manifest_hash = hashlib.sha256(encoded).hexdigest()
    if manifest_hash != SNAPSHOT_MANIFEST_SHA256:
        raise ValueError("SymNCO snapshot manifest identity mismatch")
    return {
        "kind": "read-only source snapshot", "path": str(root), "dirty": False,
        "base_identity": BASE_IDENTITY, "manifest_sha256": manifest_hash,
        "files": [{"path": name, "sha256": observed[name]} for name in sorted(observed)],
    }


def validate_historical_report(report, *, problem_size, dataset_count):
    expected = {
        "protocol": "E1", "decode": "greedy", "A": 4, "K": 1,
        "local_search": False, "seed": 1234,
    }
    mismatched = [key for key, value in expected.items() if report.get(key) != value]
    rows = report.get("rows")
    if mismatched or report.get("instances") != dataset_count or not isinstance(rows, list) \
            or len(rows) != dataset_count:
        raise ValueError(
            f"historical CVRPTW{problem_size} E1 report identity/rows mismatch: {mismatched}")
    if report.get("forced_customer_starts") != 0:
        raise ValueError("historical E1 report unexpectedly used forced customer starts")


def validate_historical_record(report, record, *, dataset_index):
    row = report["rows"][dataset_index]
    candidate = record["selected_candidate"]
    expected_candidate = [candidate["augmentation_index"], candidate["candidate_index"], 0]
    differences = []
    if row.get("row_index") != dataset_index:
        differences.append("row_index")
    if row.get("instance_id") != record["instance_id"]:
        differences.append("instance_id")
    if row.get("tour") != record["canonical_solution"]:
        differences.append("tour")
    if row.get("candidate") != expected_candidate:
        differences.append("candidate")
    if not math.isclose(float(row.get("cost", math.nan)), record["reported_objective"],
                        rel_tol=1e-12, abs_tol=1e-9):
        differences.append("cost")
    if row.get("feasible") is not True or differences:
        raise RuntimeError(
            f"BS1 historical E1 regression failed at dataset index {dataset_index}: {differences}")
