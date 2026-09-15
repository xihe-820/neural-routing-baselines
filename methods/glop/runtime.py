"""Shared read-only asset and runtime checks for formal GLOP runners."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys

from common.hashing import sha256_file
from common.provenance import (git_provenance,
                               normalize_git_repository_identity)
from methods.glop.paper_protocol import (REVISER_ASSETS, UPSTREAM_COMMIT,
                                         UPSTREAM_URL)


def verify_file(path, spec, *, sha_key="sha256", size_key="size_bytes"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != spec[size_key]:
        raise ValueError(f"unexpected GLOP asset size: {path}")
    if path.read_bytes()[:128].startswith(b"version https://git-lfs.github.com/spec"):
        raise ValueError(f"GLOP asset is a Git LFS pointer: {path}")
    actual = sha256_file(path)
    if actual != spec[sha_key]:
        raise ValueError(f"unexpected GLOP asset SHA256: {path}")
    return actual


def verify_upstream(path):
    evidence = git_provenance(path)
    if (evidence["commit"] != UPSTREAM_COMMIT or evidence["dirty"] or
            normalize_git_repository_identity(evidence["url"]) !=
            normalize_git_repository_identity(UPSTREAM_URL)):
        raise ValueError("official GLOP checkout identity/cleanliness mismatch")
    return evidence


def activate_upstream(path):
    """Make pinned top-level official packages win in this fresh runner."""
    for name in list(sys.modules):
        if name in ("problems", "utils", "nets", "heatmap") or name.startswith(
                ("problems.", "utils.", "nets.", "heatmap.")):
            del sys.modules[name]
    sys.path.insert(0, str(Path(path).resolve()))


def random_insertion_identity():
    version = importlib.metadata.version("random-insertion")
    core = version.split(".post", 1)[0]
    if tuple(int(value) for value in core.split(".")) < (0, 3, 0):
        raise RuntimeError("GLOP requires random-insertion >= 0.3.0")
    import random_insertion
    return {"version": version, "module": random_insertion.__file__}


def load_revisers(asset_root, protocol, *, device, torch, load_model):
    models, evidence = [], []
    for size in protocol["revision_lens"]:
        spec = REVISER_ASSETS[size]
        checkpoint = Path(asset_root) / spec["path"]
        args_path = Path(asset_root) / spec["args_path"]
        checkpoint_hash = verify_file(checkpoint, spec)
        args_hash = verify_file(
            args_path, spec, sha_key="args_sha256", size_key="args_size_bytes")
        args = json.loads(args_path.read_text())
        if args.get("problem") != "local" or args.get("graph_size") != size:
            raise ValueError(f"reviser_{size} args identity mismatch")
        model, loaded_args = load_model(str(checkpoint.resolve()), is_local=True)
        if loaded_args != args:
            raise ValueError(f"reviser_{size} loader args mismatch")
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("model", payload) if isinstance(payload, dict) else payload.state_dict()
        result = model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"reviser_{size} strict load mismatch")
        if sum(parameter.numel() for parameter in model.parameters()) != spec["parameter_count"]:
            raise ValueError(f"reviser_{size} parameter count mismatch")
        model.to(device).eval()
        model.set_decode_type(protocol.get(
            "decode_strategy", protocol.get("local_decode_strategy")))
        models.append(model)
        evidence.append({
            "reviser_size": size, "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash, "checkpoint_size_bytes": checkpoint.stat().st_size,
            "args_path": str(args_path.resolve()), "args_sha256": args_hash,
            "args_size_bytes": args_path.stat().st_size, "strict_load": True,
            "parameter_count": spec["parameter_count"],
        })
    return models, evidence


def cuda_device(value, torch):
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal GLOP evaluation requires an available CUDA device")
    return device
