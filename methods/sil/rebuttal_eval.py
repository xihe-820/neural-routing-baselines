#!/usr/bin/env python3
"""Explicit PRC integer rebuttal evaluation through the pinned SIL tester."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rebuttal.tsp2k.runner import main

if __name__ == "__main__":
    raise SystemExit(main(method="SIL"))
