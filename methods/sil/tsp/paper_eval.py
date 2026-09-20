#!/usr/bin/env python3
"""Problem-specific entrypoint for SIL TSP evaluation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from methods.sil.paper_eval import main

if __name__ == "__main__":
    raise SystemExit(main(["--problem", "tsp", *sys.argv[1:]]))
