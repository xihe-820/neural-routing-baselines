#!/usr/bin/env python3
"""Fetch only explicitly manifested public files, verifying bytes and SHA256.

Existing matching files may be reused. Never overwrite a mismatching file or
install dependencies. Downloaded bytes are never deserialized by this script.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.hashing import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = json.loads(args.manifest.read_text())["files"]
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "files": []}
    for item in files:
        row = dict(item)
        report["files"].append(row)
        target = (root / item["filename"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("manifest path escapes destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        row["local_path"] = str(target)
        try:
            if target.exists():
                if target.stat().st_size != item["bytes"] or sha256_file(target) != item["sha256"]:
                    raise ValueError("existing destination differs; refusing overwrite")
                row["action"] = "existing_verified"
            else:
                reuse = args.reuse_root / item["filename"] if args.reuse_root else None
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".part", delete=False) as stream:
                    temporary = Path(stream.name)
                    try:
                        if reuse and reuse.is_file() and reuse.stat().st_size == item["bytes"] and sha256_file(reuse) == item["sha256"]:
                            with reuse.open("rb") as source:
                                shutil.copyfileobj(source, stream)
                            row["action"] = "reused_hash_verified_public_copy"
                        else:
                            if not item["url"].startswith("https://"):
                                raise ValueError("public manifest URL must use HTTPS")
                            with urllib.request.urlopen(item["url"], timeout=60) as source:
                                total = 0
                                while block := source.read(1024 * 1024):
                                    total += len(block)
                                    if total > item["bytes"]:
                                        raise ValueError("download exceeds expected byte count")
                                    stream.write(block)
                            row["action"] = "downloaded"
                        stream.flush()
                        if temporary.stat().st_size != item["bytes"] or sha256_file(temporary) != item["sha256"]:
                            raise ValueError("download size/hash mismatch")
                        temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)
            row["evidence_status"] = "LOCAL_VERIFIED"
        except Exception as exc:
            row.update(evidence_status="BLOCKED", error=f"{type(exc).__name__}: {exc}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(item["filename"], row["evidence_status"], row.get("error", ""), flush=True)
    if any(r["evidence_status"] != "LOCAL_VERIFIED" for r in report["files"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
