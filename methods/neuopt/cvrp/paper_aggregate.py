#!/usr/bin/env python3
"""Aggregate strict NeuOpt production chunks over an exact expected range."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from methods.neuopt.cvrp.production_results import aggregate_chunks, write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-offset", type=int, default=0)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--scope", choices=["fullset", "timing_subset"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate_chunks(
        args.chunk_dirs, expected_offset=args.expected_offset,
        expected_count=args.expected_count, scope=args.scope)
    write_summary(args.output, summary)
    print(args.output)


if __name__ == "__main__":
    main()
