#!/usr/bin/env python3
"""Strictly aggregate complete, independently and Kit-validated paper chunks."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.paper_results import summarize_chunks, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_chunks(args.chunk_dirs)
    write_json(args.output, summary)
    print(args.output)


if __name__ == "__main__":
    main()
