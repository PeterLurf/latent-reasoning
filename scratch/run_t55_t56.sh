#!/usr/bin/env bash
# THEORY §55 (exp27) then §56 (exp28), handed to Experiments; CPU, 3 threads (Idle priority set by the launcher).
cd "$(dirname "$0")/.."
THEORY_THREADS=3 python -u theory/exp27_fallback_kill.py > logs/t55_fallback_kill.log 2>&1
echo "$(date +%T) T55_DONE rc $?" >> logs/t55_fallback_kill.log
THEORY_THREADS=3 python -u theory/exp28_interchange.py > logs/t56_interchange.log 2>&1
echo "$(date +%T) T56_DONE rc $?" >> logs/t56_interchange.log
