#!/usr/bin/env python3
"""Compare a scaled formal first-two chunk with the completed scaling audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.hashing import sha256_file
from common.objective_agreement import objective_agrees
from common.paper_results import (METADATA_FILE, RECORDS_FILE, SCHEMA_VERSION,
                                  json_fingerprint, read_jsonl)
from methods.mvmoe.cvrptw.paper_protocol import scaled_paper_inference_config


EXPECTED_INDICES = [0, 1]


def compare_records(paper_records, audit_records):
    """Require formal B to reproduce the audit B result for both instances."""
    paper = {record["dataset_instance_index"]: record for record in paper_records}
    audit = {record["dataset_instance_index"]: record for record in audit_records}
    if (len(paper_records) != len(paper) or len(paper) != 2 or
            len(audit_records) != len(audit) or
            sorted(paper) != EXPECTED_INDICES or
            sorted(audit) != list(range(20))):
        raise ValueError(
            "preflight comparison requires paper indices 0,1 and audit indices 0..19")
    comparisons = []
    for index in EXPECTED_INDICES:
        formal = paper[index]
        experimental = audit[index]
        exact_fields = (
            ("canonical_solution", formal["canonical_solution"],
             experimental["decode"]["canonical_solution"]),
            ("best_aug_idx", formal["selection"]["best_aug_idx"],
             experimental["decode"]["best_aug_idx"]),
            ("best_pomo_idx", formal["selection"]["best_pomo_idx"],
             experimental["decode"]["best_pomo_idx"]),
        )
        for field, actual, expected in exact_fields:
            if actual != expected:
                raise ValueError(f"preflight index {index} differs in {field}")
        numeric_fields = (
            ("original_objective", formal["independent_objective"],
             experimental["objectives"]["independent_original_objective"]),
            ("gap_percent", formal["gap_percent"],
             experimental["objectives"]["original_domain_gap_percent"]),
            ("scaler", formal["scaler"], experimental["scaler"]),
        )
        for field, actual, expected in numeric_fields:
            if not objective_agrees(actual, expected):
                raise ValueError(f"preflight index {index} differs in {field}")
        comparisons.append({
            "dataset_instance_index": index,
            "canonical_solution_matches": True,
            "best_aug_idx_matches": True,
            "best_pomo_idx_matches": True,
            "original_objective_matches": True,
            "gap_percent_matches": True,
            "scaler_matches": True,
        })
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-chunk", type=Path, required=True)
    parser.add_argument("--scaling-audit-dir", type=Path, required=True)
    args = parser.parse_args()

    paper_metadata = json.loads((args.paper_chunk / METADATA_FILE).read_text())
    if (paper_metadata.get("schema_version") != SCHEMA_VERSION or
            paper_metadata.get("state") not in ("INFERENCE_COMPLETE", "KIT_VALIDATED") or
            paper_metadata.get("resume_fingerprint") != json_fingerprint(
                paper_metadata.get("resume_identity"))):
        raise ValueError("formal preflight metadata/state/fingerprint mismatch")
    identity = paper_metadata["resume_identity"]
    if (identity.get("problem") != "CVRPTW" or
            identity.get("chunk", {}).get("expected_indices") != EXPECTED_INDICES or
            identity.get("paper_protocol") != scaled_paper_inference_config(
                identity.get("problem_size"))):
        raise ValueError("formal preflight is not the exact scaled CVRPTW first-two protocol")
    paper_path = args.paper_chunk / RECORDS_FILE
    if paper_metadata.get("inference_records_sha256") != sha256_file(paper_path):
        raise ValueError("formal preflight inference-record hash mismatch")

    audit_metadata = json.loads((args.scaling_audit_dir / METADATA_FILE).read_text())
    records_info = audit_metadata.get("records", {})
    audit_path = args.scaling_audit_dir / records_info.get("path", "")
    if (audit_metadata.get("status") != "SCALING_AUDIT_IMPLEMENTED_AND_RUN" or
            not audit_path.is_file() or
            records_info.get("sha256") != sha256_file(audit_path)):
        raise ValueError("scaling-audit metadata/status/record hash mismatch")

    comparisons = compare_records(read_jsonl(paper_path), read_jsonl(audit_path))
    print(json.dumps({
        "status": "PASS",
        "problem_size": identity["problem_size"],
        "indices": EXPECTED_INDICES,
        "comparisons": comparisons,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
