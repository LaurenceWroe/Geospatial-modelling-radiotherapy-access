"""
Run a command under a memory watchdog so a heavy job can never take the
machine down.

    python scripts/run_capped.py [--max-gb 6] [--min-free-gb 3] -- <command ...>

The child (and anything it spawns) is killed if its total resident memory
exceeds --max-gb, or if system available memory falls below --min-free-gb.
Only one capped job may run at a time: a lock file in the system temp dir
refuses a second launch while the first is alive.

macOS does not honour ``ulimit -v`` for Mach tasks, so the cap is enforced by
polling with psutil rather than by rlimits. Exit codes: child's code on
success, 137 when the watchdog killed it, 75 when another capped job holds
the lock.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

LOCK = Path(tempfile.gettempdir()) / "radmaps_heavy_job.lock"


def _tree_rss_gb(proc: psutil.Process) -> float:
    total = 0
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.Error:
        return 0.0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total / 1e9


def _kill_tree(proc: psutil.Process) -> None:
    try:
        kids = proc.children(recursive=True)
    except psutil.Error:
        kids = []
    for p in kids + [proc]:
        try:
            p.kill()
        except psutil.Error:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-gb", type=float, default=6.0, help="kill if the job's RSS exceeds this")
    ap.add_argument("--min-free-gb", type=float, default=3.0, help="kill if system available memory drops below this")
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between checks")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run (prefix with --)")
    args = ap.parse_args()
    cmd = [c for c in args.cmd if c != "--"]
    if not cmd:
        ap.error("no command given")

    if LOCK.exists():
        try:
            other = int(LOCK.read_text().strip())
            if psutil.pid_exists(other):
                print(f"run_capped: another capped job (pid {other}) is still running; refusing to start", file=sys.stderr)
                return 75
        except ValueError:
            pass
    avail = psutil.virtual_memory().available / 1e9
    if avail < args.min_free_gb + 1.0:
        print(f"run_capped: only {avail:.1f} GB available before start; need {args.min_free_gb + 1.0:.1f} GB", file=sys.stderr)
        return 75

    child = subprocess.Popen(cmd)
    LOCK.write_text(str(child.pid))
    proc = psutil.Process(child.pid)
    peak = 0.0
    killed = None
    try:
        while child.poll() is None:
            time.sleep(args.poll)
            rss = _tree_rss_gb(proc)
            peak = max(peak, rss)
            free = psutil.virtual_memory().available / 1e9
            if rss > args.max_gb:
                killed = f"job RSS {rss:.1f} GB exceeded cap {args.max_gb:.1f} GB"
            elif free < args.min_free_gb:
                killed = f"system available memory {free:.1f} GB fell below {args.min_free_gb:.1f} GB"
            if killed:
                _kill_tree(proc)
                break
    finally:
        try:
            LOCK.unlink()
        except FileNotFoundError:
            pass
    if killed:
        print(f"run_capped: KILLED — {killed} (peak {peak:.1f} GB)", file=sys.stderr)
        return 137
    print(f"run_capped: finished with exit {child.returncode} (peak RSS {peak:.1f} GB)", file=sys.stderr)
    return child.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
