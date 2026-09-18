#!/usr/bin/env python3
"""Create one fail-closed paper result from a formal NeuOpt TSP100 full set."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from methods.neuopt.tsp.production_results import build_paper_result, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_new_json(args.output, build_paper_result(args.run_dir))
    print(args.output)


if __name__ == "__main__":
    main()
