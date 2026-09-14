#!/usr/bin/env python3
"""Strictly aggregate complete, independently and Kit-validated paper chunks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.paper_results import METADATA_FILE, summarize_chunks, write_json
from methods.mvmoe.cvrptw.paper_protocol import (
    scaled_paper_inference_config, unscaled_control_inference_config)


def summarize_for_protocol(chunk_dirs):
    """Dispatch scaled CVRPTW without changing the frozen CVRP aggregator."""
    paths = [Path(path) for path in chunk_dirs]
    if not paths:
        raise ValueError("at least one paper chunk is required")
    metadata = json.loads((paths[0] / METADATA_FILE).read_text())
    identity = metadata.get("resume_identity", {})
    if identity.get("problem") == "CVRPTW":
        size = identity.get("problem_size")
        protocol = identity.get("paper_protocol")
        if protocol == scaled_paper_inference_config(size):
            from methods.mvmoe.cvrptw.paper_results import summarize_scaled_chunks
            return summarize_scaled_chunks(paths)
        if protocol == unscaled_control_inference_config(size):
            raise ValueError(
                "verified unscaled CVRPTW control is not the canonical scaled paper protocol")
    return summarize_chunks(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_for_protocol(args.chunk_dirs)
    write_json(args.output, summary)
    print(args.output)


if __name__ == "__main__":
    main()
