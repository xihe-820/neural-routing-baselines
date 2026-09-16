#!/usr/bin/env python3
"""Build an explicit BS100-quality plus BS1-time NeuOpt hybrid candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from methods.neuopt.cvrp.production_results import build_hybrid_summary, write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-summary", type=Path, required=True)
    parser.add_argument("--time-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hybrid-quality-warning", action="store_true",
        help="record HYBRID_QUALITY_WARNING after human review of same-subset deltas")
    args = parser.parse_args()
    quality = json.loads(args.quality_summary.read_text())
    timing = json.loads(args.time_summary.read_text())
    write_summary(
        args.output,
        build_hybrid_summary(
            quality, timing,
            hybrid_quality_warning=args.hybrid_quality_warning))
    print(args.output)


if __name__ == "__main__":
    main()
