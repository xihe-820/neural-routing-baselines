#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.cvrptw_evaluator import run
from methods.symnco.cvrptw.adapter import adapt_instance
from methods.symnco.cvrptw.config import (
    CHECKPOINTS, CHECKPOINT_HASHES, HISTORICAL_RESULTS, TIMING_SEMANTICS,
    protocol, validate_historical_record, validate_historical_report,
    validate_snapshot,
)
from methods.symnco.cvrptw.decode import decode_selected_action
from methods.symnco.cvrptw.official_runtime import Runtime


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    run({
        "name": "SymNCO", "variant": "standalone raw CVRPTW E1", "root": ROOT,
        "problem_sizes": (50, 100, 200), "formal_batch_sizes": (1, 10),
        "batch_aware_protocol": True, "protocol": protocol,
        "source_kind": "snapshot", "snapshot_validator": validate_snapshot,
        "checkpoints": CHECKPOINTS, "checkpoint_hashes": CHECKPOINT_HASHES,
        "checkpoint_path_mode": "absolute", "historical_results": HISTORICAL_RESULTS,
        "validate_historical_report": validate_historical_report,
        "validate_historical_record": validate_historical_record,
        "default_warmup_batches": 1, "timing_semantics": TIMING_SEMANTICS,
    }, adapt_instance, decode_selected_action, Runtime,
       source_files=[Path(__file__), base / "config.py", base / "adapter.py",
                     base / "decode.py", base / "official_runtime.py"])
