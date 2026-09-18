#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.cvrptw_evaluator import run
from methods.rfte.cvrptw.adapter import adapt_instance
from methods.rfte.cvrptw.config import CHECKPOINTS, UPSTREAM_COMMIT, UPSTREAM_URL, protocol
from methods.rfte.cvrptw.decode import decode_selected_action
from methods.rfte.cvrptw.official_runtime import Runtime

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    run({"name": "RF-TE", "variant": "RouteFinder Transformer", "root": ROOT,
         "formal_batch_sizes": (1, 10), "batch_aware_protocol": True,
         "upstream_url": UPSTREAM_URL, "upstream_commit": UPSTREAM_COMMIT,
         "checkpoints": CHECKPOINTS, "protocol": protocol}, adapt_instance,
        decode_selected_action, Runtime,
        source_files=[Path(__file__), base / "config.py", base / "adapter.py",
                      base / "decode.py", base / "official_runtime.py"])
