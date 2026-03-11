#!/bin/bash
# BIG-bench sweep status check with ETA estimates.
# Usage: bash scripts/check_sweep_status.sh
cd "$(dirname "$0")/.."
.venv/bin/python scripts/check_sweep_status.py
