#!/usr/bin/env python3
"""CPU import/model/checkpoint audit in one selected method environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.hashing import sha256_file
from common.paper_results import utc_now, write_json
from common.provenance import git_provenance, normalize_git_repository_identity

METHODS = {
    "rfte": {
        "name": "RF-TE", "url": "https://github.com/ai4co/routefinder",
        "commit": "fe0e45b6df118af03c5f42db8b93a351f7629131",
        "runtime": "methods.rfte.cvrptw.official_runtime", "checkpoints": {
            50: "checkpoints/50/rf-transformer.ckpt",
            100: "checkpoints/100/rf-transformer.ckpt"}, "hashes": {},
    },
    "cada": {
        "name": "CaDA", "url": "https://github.com/CIAM-Group/CaDA",
        "commit": "b9868e1e09b3a1a3754960d830e801e51c7cb38d",
        "runtime": "methods.cada.cvrptw.official_runtime", "checkpoints": {
            50: "50/result/2024-1111-1139/checkpoint-300.pt",
            100: "100/result/2024-1121-1355/checkpoint-300.pt"}, "hashes": {},
    },
    "moses_cada": {
        "name": "MoSES(CaDA)", "url": "https://github.com/panyxy/moses_vrp",
        "commit": "e301478b7a5df6d7b0b10a0543f4dee5e3c027a8",
        "runtime": "methods.moses_cada.cvrptw.official_runtime", "checkpoints": {
            50: "pretrained_moses_model/cada/50/multilora_denseroute_sigmoid.ckpt",
            100: "pretrained_moses_model/cada/100/multilora_denseroute_sigmoid.ckpt"},
        "hashes": {
            50: "1aa499f3fce5d3412c2544c9632bbb9709309a7299b4298b535fa7e9014ef803",
            100: "2eac9b038ae4655581aa73e4dbe8ad529aefd1963368c9a92d254b6269f8aabf"},
    },
}


def versions():
    result = {}
    for name in ("torch", "numpy", "torch-geometric", "rl4co", "tensordict",
                 "torchrl", "lightning", "pytorch-lightning", "einops", "entmax"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--problem-size", type=int, choices=(50, 100), required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = METHODS[args.method]
    report = {
        "schema": "remaining-cvrptw-method-environment-audit-v1",
        "created_at": utc_now(), "method": cfg["name"],
        "problem_size": args.problem_size, "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(),
        "packages": versions(), "status": "FAIL",
    }
    try:
        upstream = git_provenance(args.upstream)
        if (upstream["dirty"] or upstream["commit"] != cfg["commit"] or
                normalize_git_repository_identity(upstream["url"]) !=
                normalize_git_repository_identity(cfg["url"])):
            raise ValueError("official upstream identity/cleanliness mismatch")
        relative = str(args.checkpoint.resolve().relative_to(args.upstream.resolve()))
        if relative != cfg["checkpoints"][args.problem_size]:
            raise ValueError("checkpoint path is not the exact size-specific official path")
        checkpoint_hash = sha256_file(args.checkpoint)
        if checkpoint_hash != args.expected_checkpoint_sha256:
            raise ValueError("checkpoint SHA256 differs from expected audit identity")
        if (args.problem_size in cfg["hashes"] and
                checkpoint_hash != cfg["hashes"][args.problem_size]):
            raise ValueError("checkpoint SHA256 differs from repository-pinned official asset")
        import importlib
        import torch
        runtime_module = importlib.import_module(cfg["runtime"])
        runtime = runtime_module.Runtime(
            args.upstream, args.checkpoint, args.problem_size,
            torch.device("cpu"), torch)
        report.update(
            upstream=upstream,
            checkpoint={"path": str(args.checkpoint.resolve()),
                        "relative_path": relative, "sha256": checkpoint_hash,
                        "size_bytes": args.checkpoint.stat().st_size},
            torch_cuda_build=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            checkpoint_state=runtime.checkpoint_state,
            import_smoke=True, cpu_model_construction=True,
            audited_checkpoint_load=True, status="PASS")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    write_json(args.output, report)
    print(args.output)
    if report["status"] != "PASS":
        raise SystemExit(report["error"])


if __name__ == "__main__":
    main()
