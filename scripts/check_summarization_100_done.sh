#!/usr/bin/env bash
# Poll summarization run until all 13 models have 100 tasks. Usage: ./scripts/check_summarization_100_done.sh
RESULTS_ROOT="results/summarization"
RUN_PATTERN="_20260225_042134"
while true; do
  incomplete=0
  for d in "$RESULTS_ROOT"/*${RUN_PATTERN}/OneShotAgent; do
    [ -f "$d/results.jsonl" ] || continue
    n=$(wc -l < "$d/results.jsonl")
    model=$(basename "$(dirname "$d")" | sed "s/${RUN_PATTERN}//")
    printf "%s: %s\n" "$model" "$n"
    [ "$n" -lt 100 ] && incomplete=1
  done
  if ! pgrep -f "benchmark_runner.py" >/dev/null 2>&1; then
    echo "ENDED"
    break
  fi
  [ "$incomplete" -eq 0 ] && echo "ALL_100" && break
  echo "RUNNING"
  sleep 30
done
