#!/usr/bin/env bash
# Run planning status every minute until done or stuck. Agent can run this and review output to decide changes.
# Usage: bash scripts/agent_check_planning.sh
# Or:    while true; do poetry run python scripts/planning_status.py; sleep 60; done

set -e
cd "$(dirname "$0")/.."
INTERVAL="${PLANNING_CHECK_INTERVAL:-60}"

while true; do
  echo "--- $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
  if poetry run python scripts/planning_status.py; then
    echo "Both runs done. Exiting."
    exit 0
  fi
  EXIT=$?
  if [ "$EXIT" -eq 2 ]; then
    echo ">>> STUCK? Check terminal logs for Azure/Ollama runs and consider restarting failed models."
  fi
  echo "Next check in ${INTERVAL}s..."
  sleep "$INTERVAL"
done
