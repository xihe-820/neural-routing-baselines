#!/usr/bin/env python3
"""Gate and launch one remaining-CVRPTW production cell; dry-run by default."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.cvrptw_artifacts import require_our5_gate
from common.cvrptw_formal import dataset_config
from common.hashing import sha256_file
from common.provenance import (git_provenance,
                               normalize_git_repository_identity)

RUNNERS = {
    "rfte": ("RF-TE", "methods/rfte/cvrptw/paper_eval.py",
             "https://github.com/ai4co/routefinder",
             "fe0e45b6df118af03c5f42db8b93a351f7629131",
             {50: "checkpoints/50/rf-transformer.ckpt",
              100: "checkpoints/100/rf-transformer.ckpt"}),
    "cada": ("CaDA", "methods/cada/cvrptw/paper_eval.py",
             "https://github.com/CIAM-Group/CaDA",
             "b9868e1e09b3a1a3754960d830e801e51c7cb38d",
             {50: "50/result/2024-1111-1139/checkpoint-300.pt",
              100: "100/result/2024-1121-1355/checkpoint-300.pt"}),
    "moses_cada": ("MoSES(CaDA)", "methods/moses_cada/cvrptw/paper_eval.py",
                   "https://github.com/panyxy/moses_vrp",
                   "e301478b7a5df6d7b0b10a0543f4dee5e3c027a8",
                   {50: "pretrained_moses_model/cada/50/multilora_denseroute_sigmoid.ckpt",
                    100: "pretrained_moses_model/cada/100/multilora_denseroute_sigmoid.ckpt"}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(RUNNERS), required=True)
    parser.add_argument("--problem-size", type=int, choices=(50, 100), required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--our5-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    method_name, runner, upstream_url, upstream_commit, checkpoint_paths = RUNNERS[args.method]
    cfg = dataset_config(args.problem_size)
    project = git_provenance(ROOT)
    upstream = git_provenance(args.upstream)
    if project["dirty"]:
        raise ValueError("production launcher requires a clean project checkout")
    if (upstream["dirty"] or upstream["commit"] != upstream_commit or
            normalize_git_repository_identity(upstream["url"]) !=
            normalize_git_repository_identity(upstream_url)):
        raise ValueError("production launcher official upstream gate failed")
    if args.dataset.name != cfg["filename"] or sha256_file(args.dataset) != cfg["sha256"]:
        raise ValueError("production launcher pinned dataset gate failed")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("production launcher checkpoint SHA256 gate failed")
    if str(args.checkpoint.resolve().relative_to(args.upstream.resolve())) != checkpoint_paths[args.problem_size]:
        raise ValueError("production launcher checkpoint path gate failed")
    if not args.input.is_file() or not args.input.with_suffix(args.input.suffix + ".json").is_file():
        raise ValueError("production launcher prepared input or metadata is missing")
    require_our5_gate(args.our5_evidence, method=method_name,
                      problem_size=args.problem_size, dataset_sha256=cfg["sha256"],
                      checkpoint_sha256=args.expected_checkpoint_sha256)
    command = [
        str(args.python), "-B", str(ROOT / runner), "--scope", "production",
        "--problem-size", str(args.problem_size), "--input", str(args.input),
        "--dataset", str(args.dataset), "--upstream", str(args.upstream),
        "--checkpoint", str(args.checkpoint),
        "--expected-checkpoint-sha256", args.expected_checkpoint_sha256,
        "--our5-evidence", str(args.our5_evidence),
        "--output-dir", str(args.output_dir),
    ]
    print(json.dumps({"gate": "PASS", "execute": args.execute, "command": command}, indent=2))
    if args.execute:
        raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)


if __name__ == "__main__":
    main()
