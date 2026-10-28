#!/usr/bin/env bash
# Run a heavy RadMaps job on the remote server instead of the laptop.
#
#   scripts/remote_run.sh <command ...>
#
# 1. rsyncs the working tree (code + data caches, no .git) to $REMOTE:$REMOTE_DIR
# 2. runs the command there inside the project venv, under scripts/run_capped.py
#    (memory watchdog, one job at a time), with nohup so it survives disconnects
# 3. waits for it, then rsyncs back the result directories (figures/, cache/,
#    travel_time_cache/, H3_region_cache/, dashboard_store/)
#
# Examples
#   scripts/remote_run.sh python -m pytest tests -q
#   scripts/remote_run.sh python scripts/global_results.py
#   scripts/remote_run.sh python scripts/build_dashboard_world.py
#
# Override the host with REMOTE=user@host (default: the 'cybench-box' ssh alias).
set -euo pipefail

REMOTE="${REMOTE:-cybench-box}"
REMOTE_DIR="${REMOTE_DIR:-/root/radmaps}"
VENV="${VENV:-/root/radmaps_venv}"
MAX_GB="${MAX_GB:-8}"
MIN_FREE_GB="${MIN_FREE_GB:-2}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if [ $# -eq 0 ]; then
  echo "usage: scripts/remote_run.sh <command ...>" >&2
  exit 2
fi

echo "→ syncing working tree to $REMOTE:$REMOTE_DIR"
rsync -az --delete-excluded \
  --exclude .git --exclude __pycache__ --exclude '*.pyc' \
  --exclude H3_pop_density_maps --exclude .streamlit/secrets.toml --exclude Old_Tests \
  ./ "$REMOTE:$REMOTE_DIR/"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$REMOTE_DIR/logs/job_$STAMP.log"
# Quote the command for the remote shell.
CMD=$(printf '%q ' "$@")

echo "→ launching on $REMOTE (log: $LOG)"
ssh -o BatchMode=yes "$REMOTE" "mkdir -p $REMOTE_DIR/logs && cd $REMOTE_DIR && \
  PATH=$VENV/bin:\$PATH MPLBACKEND=Agg nohup $VENV/bin/python scripts/run_capped.py \
  --max-gb $MAX_GB --min-free-gb $MIN_FREE_GB -- $CMD > $LOG 2>&1 & echo \$! > $REMOTE_DIR/logs/job_$STAMP.pid"

echo "→ waiting (tail of remote log follows; Ctrl-C only stops the tail, not the job)"
ssh -o BatchMode=yes "$REMOTE" "pid=\$(cat $REMOTE_DIR/logs/job_$STAMP.pid); \
  while kill -0 \$pid 2>/dev/null; do sleep 15; done; tail -n 40 $LOG"

echo "→ syncing results back"
for d in figures cache H3_region_cache travel_time_cache tt_jobs dashboard_store; do
  rsync -az "$REMOTE:$REMOTE_DIR/$d/" "./$d/" 2>/dev/null || true
done
rsync -az "$REMOTE:$REMOTE_DIR/data/country_metrics.parquet" ./data/ 2>/dev/null || true
mkdir -p logs && rsync -az "$REMOTE:$LOG" ./logs/ 2>/dev/null || true
echo "✓ done — remote log copied to logs/job_$STAMP.log"
