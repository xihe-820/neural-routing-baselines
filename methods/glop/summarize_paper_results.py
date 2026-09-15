#!/usr/bin/env python3
"""Strictly aggregate complete Kit-validated formal GLOP chunks."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from methods.glop.paper_results import summarize_chunks, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, summarize_chunks(args.chunk_dirs))
    print(args.output)


if __name__ == "__main__":
    main()
