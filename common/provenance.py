"""Read-only provenance helpers used by method runners."""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
import re
from urllib.parse import urlsplit

from common.hashing import sha256_file


def _git(repo, *args):
    proc = subprocess.run(["git", "-C", str(Path(repo).resolve()), *args],
                          capture_output=True, text=True, check=False)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "git command failed")
    return proc.stdout.strip()


def git_provenance(repo):
    return {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
        "url": _git(repo, "remote", "get-url", "origin"),
    }


def normalize_git_repository_identity(url):
    """Normalize common HTTPS/SSH GitHub remotes without weakening commit checks."""
    value = url.strip()
    scp = re.fullmatch(r"(?:[^@]+@)?([^:]+):(.+)", value)
    if scp and "://" not in value:
        host, path = scp.groups()
    else:
        parsed = urlsplit(value)
        if not parsed.hostname:
            raise ValueError(f"unsupported git remote URL: {url}")
        host, path = parsed.hostname, parsed.path
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if host.lower() != "github.com" or len(path.split("/")) != 2:
        raise ValueError(f"expected a GitHub OWNER/REPO remote: {url}")
    return f"github.com/{path}"


def source_provenance(paths, *, root):
    base = Path(root).resolve()
    records = []
    for value in paths:
        path = Path(value).resolve()
        try:
            relative = str(path.relative_to(base))
        except ValueError as exc:
            raise ValueError(f"source path is outside provenance root: {path}") from exc
        records.append({"path": relative, "sha256": sha256_file(path)})
    return records


def environment_provenance(device):
    import numpy
    import torch
    gpu = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
    return {
        "hostname": platform.node(), "platform": platform.platform(),
        "python": sys.version, "executable": sys.executable,
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "numpy": numpy.__version__, "device": str(device), "gpu": gpu,
    }
