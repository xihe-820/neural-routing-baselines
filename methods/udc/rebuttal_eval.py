#!/usr/bin/env python3
"""TSP2K rebuttal budget runner using the existing UDC solver."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rebuttal.tsp2k.runner import main

if __name__ == "__main__":
    raise SystemExit(main(method="UDC"))
