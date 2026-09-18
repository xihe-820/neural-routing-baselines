#!/usr/bin/env python3
"""Read all six formal NeuOpt TSP100 results into one strict paper matrix."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from methods.neuopt.tsp.production_results import build_result_matrix, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_new_json(args.output, build_result_matrix(args.result_files))
    print(args.output)


if __name__ == "__main__":
    main()
