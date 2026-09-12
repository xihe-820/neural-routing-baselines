#!/usr/bin/env python3
"""Read-only dependency audit; each import uses a separate timed subprocess.

Only the explicitly requested report files are written. Run with the actual
target interpreter; a local report is never evidence for cp311_base.
"""
import argparse
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

PACKAGES = {
    "torch": "torch", "numpy": "numpy", "scipy": "scipy",
    "networkx": "networkx", "torch-geometric": "torch_geometric",
    "torch-scatter": "torch_scatter", "torch-sparse": "torch_sparse",
    "torch-cluster": "torch_cluster", "torch-spline-conv": "torch_spline_conv",
    "tensordict": "tensordict", "torchrl": "torchrl", "rl4co": "rl4co",
    "lightning": "lightning", "pytorch-lightning": "pytorch_lightning",
    "random-insertion": "random_insertion", "protobuf": "google.protobuf",
    "hydra-core": "hydra", "omegaconf": "omegaconf",
    "huggingface-hub": "huggingface_hub", "pyvrp": "pyvrp", "vrplib": "vrplib",
    "ml4co-kit": "ml4co_kit", "tensorboard-logger": "tensorboard_logger",
}
MARKER = "AUDIT_JSON:"


def probe(distribution):
    result = dict(installed=False, version=None, import_ok=False, error=None)
    try:
        result["version"] = metadata.version(distribution)
        result["installed"] = True
    except metadata.PackageNotFoundError:
        pass
    try:
        result["installed"] |= importlib.util.find_spec(PACKAGES[distribution]) is not None
        module = importlib.import_module(PACKAGES[distribution])
        result["import_ok"] = True
        result["module_path"] = getattr(module, "__file__", None)
        if distribution == "torch":
            result["cuda"] = {"torch_version_cuda": module.version.cuda}
            try:
                available = module.cuda.is_available()
                result["cuda"].update(available=available, gpu_names=[
                    module.cuda.get_device_name(i) for i in range(module.cuda.device_count())
                ] if available else [])
            except Exception as exc:
                result["cuda"]["error"] = f"{type(exc).__name__}: {exc}"
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("BASELINE_ARTIFACT_ROOT", "artifacts")))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--label", default="local-unclassified")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--probe", choices=PACKAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output or args.artifact_root / "environment/environment_audit.json"
    if args.probe:
        print(MARKER + json.dumps(probe(args.probe)))
        return
    report = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "hostname": platform.node(), "label": args.label,
              "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
              "packages": {}, "compiled_cuda_ops_tested": False}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for distribution in PACKAGES:
        try:
            proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                   "--probe", distribution], capture_output=True, text=True,
                                  timeout=args.timeout, env=env)
            lines = [line[len(MARKER):] for line in proc.stdout.splitlines() if line.startswith(MARKER)]
            if not lines:
                raise RuntimeError(f"worker exit={proc.returncode}: {proc.stderr[-2000:]}")
            result = json.loads(lines[-1])
            result["stderr"] = proc.stderr[-4000:]
        except Exception as exc:
            try:
                version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                version = None
            result = dict(installed=version is not None, version=version, import_ok=False,
                          error=f"{type(exc).__name__}: {exc}")
        report["packages"][distribution] = result
    try:
        proc = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True,
                              text=True, timeout=args.timeout, env=env)
        report["pip_check"] = dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except Exception as exc:
        report["pip_check"] = dict(returncode=None, error=f"{type(exc).__name__}: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown:
        rows = ["# Environment audit", "", f"Environment label: `{args.label}`.",
                "This report describes only the interpreter and host recorded below. Imports do not prove compiled CUDA operations or method compatibility.",
                "", f"- Timestamp: {report['timestamp']}", f"- Host: {report['hostname']}",
                f"- Python: {sys.version.split()[0]}", f"- Executable: `{sys.executable}`",
                f"- Platform: {report['platform']}", "", "| Package | Installed | Version | Import | Error |",
                "|---|---|---|---|---|"]
        for name, item in report["packages"].items():
            error = (item["error"] or "").replace("\n", " ").replace("|", "/")
            rows.append(f"| {name} | {item['installed']} | {item['version']} | {item['import_ok']} | {error} |")
        rows += ["", "CUDA discovery:", "```json", json.dumps(report['packages']['torch'].get('cuda'), indent=2),
                 "```", "", "pip check:", "```json", json.dumps(report['pip_check'], indent=2), "```", ""]
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(rows))
    print(args.output)


if __name__ == "__main__":
    main()
