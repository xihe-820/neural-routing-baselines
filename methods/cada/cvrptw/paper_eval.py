#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.cvrptw_evaluator import run
from methods.cada.cvrptw.adapter import adapt_instance
from methods.cada.cvrptw.config import CHECKPOINTS, UPSTREAM_COMMIT, UPSTREAM_URL, protocol
from methods.cada.cvrptw.decode import decode_selected_action
from methods.cada.cvrptw.official_runtime import Runtime

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    run({"name": "CaDA", "variant": "official CaDA", "root": ROOT,
         "upstream_url": UPSTREAM_URL, "upstream_commit": UPSTREAM_COMMIT,
         "checkpoints": CHECKPOINTS, "protocol": protocol}, adapt_instance,
        decode_selected_action, Runtime,
        source_files=[Path(__file__), base / "config.py", base / "adapter.py",
                      base / "decode.py", base / "official_runtime.py"])
