#!/usr/bin/env python3
"""Build a non-freezing NeuOpt runtime calibration report from candidate artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from methods.neuopt.cvrp.paper_results import (build_calibration_report,
                                               write_calibration_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-size", type=int, choices=[50, 100], required=True)
    parser.add_argument("--candidate-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_calibration_report(
        args.candidate_dirs, problem_size=args.problem_size)
    write_calibration_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
