"""Run one fresh child process under the shared machine-measurement lock."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

from measurement_lock import OWNER_PREFIX, measurement_lock


REPO = Path(__file__).resolve().parents[2]


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--owner-prefix", default=OWNER_PREFIX)
    ap.add_argument("--event-output", type=Path, required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        ap.error("a child command is required after --")
    event = None
    started = _utc_now()
    with measurement_lock(
        run_label=args.label,
        owner_prefix=args.owner_prefix,
    ) as lock_event:
        event = lock_event
        print(f"locked command: {subprocess.list2cmdline(command)}", flush=True)
        completed = subprocess.run(command, cwd=REPO, env=os.environ.copy())
        ended = _utc_now()
        event.update(
            {
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
                ).strip(),
                "command": command,
                "command_line": subprocess.list2cmdline(command),
                "cwd": str(REPO),
                "child_started_at_utc": started,
                "child_ended_at_utc": ended,
                "exit_code": int(completed.returncode),
            }
        )
    args.event_output.parent.mkdir(parents=True, exist_ok=True)
    args.event_output.write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(event, indent=2), flush=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
