"""Shared exclusive-machine lock for CPU benchmark measurements.

The lock is an atomic directory. It never removes or rewrites another owner's
lock and waits in 60-second intervals until it can create the directory itself.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import os
import tempfile
import time
import uuid
from pathlib import Path


LOCK_DIR = Path(os.environ.get(
    "PYMACK_MEASURE_LOCK",
    str(Path(tempfile.gettempdir()) / "pymack" / "MEASURE_LOCK"),
))
OWNER_PREFIX = "cpu-benchmark"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_owner() -> str:
    try:
        return (LOCK_DIR / "owner.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return "<owner unavailable>"


@contextlib.contextmanager
def measurement_lock(
    *,
    run_label: str,
    poll_seconds: float = 60.0,
    owner_prefix: str = OWNER_PREFIX,
):
    """Acquire and release the shared measurement lock for exactly one run."""
    LOCK_DIR.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    token = uuid.uuid4().hex
    acquired_at = None
    owner = None
    while True:
        try:
            LOCK_DIR.mkdir()
        except FileExistsError:
            print(
                f"measurement lock busy owner={_read_owner()!r}; "
                f"{owner_prefix} waits {poll_seconds:.0f}s",
                flush=True,
            )
            time.sleep(poll_seconds)
            continue
        acquired_at = _utc_now()
        owner = f"{owner_prefix} {acquired_at} {token} {run_label}"
        try:
            (LOCK_DIR / "owner.txt").write_text(owner + "\n", encoding="utf-8")
        except BaseException:
            try:
                LOCK_DIR.rmdir()
            finally:
                raise
        break

    event = {
        "protocol": "atomic mkdir; low-priority poll; never steal",
        "lock_dir": str(LOCK_DIR),
        "owner": owner,
        "run_label": run_label,
        "waited_s": time.monotonic() - started,
        "acquired_at_utc": acquired_at,
        "released_at_utc": None,
    }
    try:
        yield event
    finally:
        current = _read_owner()
        if current != owner:
            raise RuntimeError(
                "refusing to release measurement lock whose owner changed: "
                f"expected={owner!r}, observed={current!r}"
            )
        try:
            os.remove(LOCK_DIR / "owner.txt")
            os.rmdir(LOCK_DIR)
        finally:
            event["released_at_utc"] = _utc_now()


def wait_for_priority_lanes_quiet(*, quiet_seconds: float = 120.0) -> dict:
    """Wait for an uninterrupted lock-free interval before long capped M10 rows."""
    began = time.monotonic()
    quiet_since = None
    while True:
        if LOCK_DIR.exists():
            quiet_since = None
            print(
                f"priority-lane lock active owner={_read_owner()!r}; "
                "deferring capped M10 row 60s",
                flush=True,
            )
            time.sleep(60.0)
            continue
        now = time.monotonic()
        if quiet_since is None:
            quiet_since = now
        elapsed = now - quiet_since
        if elapsed >= quiet_seconds:
            return {
                "quiet_interval_required_s": quiet_seconds,
                "quiet_interval_observed_s": elapsed,
                "total_wait_s": now - began,
                "completed_at_utc": _utc_now(),
            }
        remaining = quiet_seconds - elapsed
        print(
            f"priority lock absent; confirming quiet for another {remaining:.0f}s "
            "before capped M10",
            flush=True,
        )
        time.sleep(min(60.0, remaining))
