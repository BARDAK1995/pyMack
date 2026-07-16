"""Reusable provenance records for generated verification artifacts."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* as lowercase hexadecimal."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_files(paths: Iterable[Path], *, root: Path) -> dict[str, str]:
    """Hash files under *root*, keyed by stable POSIX-style relative paths."""
    root = Path(root).resolve()
    return {
        path.resolve().relative_to(root).as_posix(): sha256_file(path)
        for path in (Path(item) for item in paths)
    }


def capture_source(repo: Path) -> dict:
    """Capture the repository commit and pre-generation tracked-tree state."""
    repo = Path(repo).resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo, text=True,
    )
    return {"commit": commit, "tracked_tree_dirty": bool(status.strip())}


def exact_command() -> str:
    """Return the current Python invocation in Windows command-line syntax."""
    return subprocess.list2cmdline([sys.executable, *sys.argv])


def build_provenance(
    *,
    source: dict,
    command: str,
    effective_parameters: dict,
    sha256: dict[str, str],
) -> dict:
    """Build the versioned provenance block embedded in a verdict artifact."""
    return {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "source": source,
        "execution": {
            "command": command,
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "PYMACK_SWEEP_BACKEND",
                    "PYMACK_NVCP_SYSMEM_FALLBACK",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
        "effective_parameters": effective_parameters,
        "sha256": sha256,
    }
