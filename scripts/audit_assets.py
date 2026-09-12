#!/usr/bin/env python3
"""Read-only checkout/checkpoint inventory. No downloads, inference or training.

Only deserialize trusted official files. Each CPU load is isolated and timed;
failure is recorded instead of being misclassified as a missing checkpoint.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import pickletools
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.hashing import sha256_file

REPOS = {
    "GLOP": ("GLOP", "https://github.com/henry-yeh/GLOP", "."),
    "UDC": ("NCO_code", "https://github.com/CIAM-Group/NCO_code", "single_objective/UDC-Large-scale-CO-master/UDC"),
    "CaDA": ("CaDA", "https://github.com/CIAM-Group/CaDA", "."),
    "MVMoE": ("Routing-MVMoE", "https://github.com/RoyalSkye/Routing-MVMoE", "."),
    "RF-TE": ("routefinder", "https://github.com/ai4co/routefinder", "."),
    "MoSES(CaDA)": ("moses_vrp", "https://github.com/panyxy/moses_vrp", "."),
    "NeuOpt": ("NeuOpt", "https://github.com/yining043/NeuOpt", "."),
}


def git(path, *args):
    proc = subprocess.run(["git", "--no-optional-locks", "-C", str(path), *args],
                          capture_output=True, text=True, timeout=30)
    return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def inspect_checkpoint(path):
    import torch
    # Official artifacts are trusted by task scope; preserve legacy payloads,
    # including optimizer/RNG/Lightning metadata. Never construct a trainer.
    obj = torch.load(path, map_location="cpu", weights_only=False)
    result = {"deserialize_ok": True, "torch": torch.__version__, "python": sys.version,
              "top_level_type": type(obj).__name__, "tensors": {}, "metadata": {}}
    def walk(value, prefix="", depth=0):
        if torch.is_tensor(value):
            result["tensors"][prefix] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, dict) and depth < 4:
            for key, child in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                if name.split('.')[0] not in ("optimizer", "optimizer_state_dict", "optimizer_states", "lr_schedulers"):
                    walk(child, name, depth + 1)
        elif value is None or isinstance(value, (str, int, float, bool)):
            result["metadata"][prefix] = value if not isinstance(value, str) else value[:500]
        elif depth < 2:
            result["metadata"][prefix] = {"type": type(value).__name__}
    walk(obj)
    result["top_level_keys"] = list(map(str, obj.keys())) if isinstance(obj, dict) else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", "--upstream-root", dest="external_root", type=Path,
                        default=Path(os.environ.get("BASELINE_UPSTREAM_ROOT", "external")))
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("BASELINE_ARTIFACT_ROOT", "artifacts")))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=Path(os.environ.get("BASELINE_CHECKPOINT_ROOT", ".")))
    parser.add_argument("--checkpoint", nargs=2, action="append", default=[], metavar=("METHOD", "PATH"),
                        help="Additional official checkpoint, absolute or relative to --checkpoint-root; repeatable")
    parser.add_argument("--checkpoint-python", default=sys.executable)
    parser.add_argument("--load-checkpoints", action="store_true")
    parser.add_argument("--compare-manifest", type=Path, help="Compare binaries with LOCAL_VERIFIED wsl_evidence in checkpoint manifest")
    parser.add_argument("--load-one", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--load-repo", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output or args.artifact_root / "audit/assets.json"
    if any(name not in REPOS for name, _ in args.checkpoint):
        parser.error("--checkpoint METHOD must be one of: " + ", ".join(REPOS))
    if args.load_one:
        if args.load_repo:
            # Exactly one official repository per disposable interpreter.
            sys.path.insert(0, str(args.load_repo.resolve()))
        try:
            result = inspect_checkpoint(args.load_one)
        except Exception as exc:
            result = {"deserialize_ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print("AUDIT_JSON:" + json.dumps(result))
        return
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
              "external_root": str(args.external_root.resolve()), "repos": {},
              "audit_script_sha256": sha256_file(__file__),
              "checkpoint_python": args.checkpoint_python}
    for name, (directory, url, subtree) in REPOS.items():
        path = (args.external_root / directory).resolve()
        entry = {"path": str(path), "expected_url": url, "exists": path.is_dir(), "checkpoints": []}
        report["repos"][name] = entry
        if not path.is_dir():
            continue
        entry["git"] = {key: git(path, *command) for key, command in {
            "origin": ["remote", "get-url", "origin"], "head": ["rev-parse", "HEAD"],
            "branch": ["branch", "--show-current"], "status": ["status", "--porcelain", "--untracked-files=all"],
            "tags": ["tag", "--sort=-version:refname"], "head_tags": ["tag", "--points-at", "HEAD"],
            "shallow": ["rev-parse", "--is-shallow-repository"],
            "last_commit": ["log", "-1", "--format=%H %cI %s"],
        }.items()}
        status = entry["git"]["status"]
        entry["dirty"] = bool(status["stdout"]) if status["returncode"] == 0 else None
        scope = path / subtree
        files = sorted(p for p in scope.rglob("*") if p.is_file() and '.git' not in p.parts and p.suffix in (".pt", ".pth", ".ckpt"))
        explicit_files = {(args.checkpoint_root / filename).resolve() for method, filename in args.checkpoint if method == name}
        files = sorted(set(files) | explicit_files)
        for file in files:
            # The monorepo's other solvers are explicitly out of scope.
            if name == "UDC" and file not in explicit_files and not any(part in file.parts for part in ("TSP-AGNN-ICAM", "CVRP-AGNN-ICAM")):
                continue
            relative = os.path.relpath(file, path)
            selected = (file in explicit_files or name not in ("MVMoE", "MoSES(CaDA)", "NeuOpt") or
                        (name == "MVMoE" and any(f"mvmoe_4e_n{n}/" in relative for n in (50, 100))) or
                        (name == "MoSES(CaDA)" and "/cada/" in relative and file.name == "multilora_denseroute_sigmoid.ckpt") or
                        (name == "NeuOpt" and file.name in ("cvrp50.pt", "cvrp100.pt")))
            if not file.is_file():
                entry["checkpoints"].append({"path": str(file), "relative_path": relative,
                                             "materialized": False, "exists": False, "selected": selected,
                                             "size_bytes": None, "sha256": None, "lfs_pointer": None,
                                             "load": {"deserialize_ok": None}, "error": "CHECKPOINT_NOT_FOUND"})
                continue
            with file.open("rb") as stream:
                prefix = stream.read(200)
            pointer = prefix.startswith(b"version https://git-lfs.github.com/spec/v1")
            checkpoint = {"path": str(file), "relative_path": relative, "size_bytes": file.stat().st_size,
                          "sha256": sha256_file(file), "lfs_pointer": pointer, "materialized": not pointer,
                          "format": "git-lfs-pointer" if pointer else ("zip" if zipfile.is_zipfile(file) else "legacy-or-other"),
                          "selected": selected, "load": {"deserialize_ok": None}}
            if name == "GLOP" and "reviser" in str(file).lower():
                config = file.parent / "args.json"
                checkpoint["companion_config"] = {"path": str(config), "exists": config.is_file(),
                                                   "sha256": sha256_file(config) if config.is_file() else None}
                if config.is_file():
                    try:
                        checkpoint["companion_config"]["content"] = json.loads(config.read_text())
                    except Exception as exc:
                        checkpoint["companion_config"]["error"] = f"{type(exc).__name__}: {exc}"
            if selected and checkpoint["format"] == "zip":
                try:
                    with zipfile.ZipFile(file) as archive:
                        member = next(info for info in archive.infolist() if info.filename.endswith("/data.pkl"))
                        if member.file_size > 16 * 1024 * 1024:
                            raise ValueError("pickle metadata exceeds 16 MiB static-inspection limit")
                        operations = list(pickletools.genops(archive.read(member)))
                    strings = [arg for op, arg, _ in operations if op.name in ("BINUNICODE", "SHORT_BINUNICODE")]
                    checkpoint["static_pickle"] = {
                        "note": "pickle opcode inspection only, not deserialization or tensor-shape verification",
                        "globals": sorted({str(arg) for op, arg, _ in operations if op.name == "GLOBAL"}),
                        "initial_strings": strings[:30],
                        "parameter_key_examples": [s for s in strings if s.startswith("policy.")][:20],
                    }
                except Exception as exc:
                    checkpoint["static_pickle"] = {"error": f"{type(exc).__name__}: {exc}"}
            if selected and args.load_checkpoints and not pointer:
                try:
                    proc = subprocess.run([args.checkpoint_python, "-B", str(Path(__file__).resolve()), "--load-one", str(file),
                                           "--load-repo", str(path)],
                                          capture_output=True, text=True, timeout=90,
                                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
                    lines = [line[11:] for line in proc.stdout.splitlines() if line.startswith("AUDIT_JSON:")]
                    checkpoint["load"] = json.loads(lines[-1]) if lines else {"deserialize_ok": False, "error": proc.stderr[-3000:], "returncode": proc.returncode}
                except Exception as exc:
                    checkpoint["load"] = {"deserialize_ok": False, "error": f"{type(exc).__name__}: {exc}"}
            entry["checkpoints"].append(checkpoint)
    if args.compare_manifest:
        expected = json.loads(args.compare_manifest.read_text())["checkpoints"]
        for name, repo in report["repos"].items():
            known = {e["sha256"] for row in expected if row["method"] == name for e in row.get("wsl_evidence", [])}
            for checkpoint in repo["checkpoints"]:
                checkpoint["manifest_identity_match"] = checkpoint.get("sha256") in known if known else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
