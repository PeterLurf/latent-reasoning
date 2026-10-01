#!/usr/bin/env bash
# THEORY §53 (F2 bidirectional, registered 54ca518), handed to Experiments: CPU only, 3 threads, Idle priority (set by the launcher).
cd "$(dirname "$0")/.."
THEORY_THREADS=3 python -u theory/exp26_bidirectional.py > logs/t53_bidirectional.log 2>&1
echo "$(date +%T) T53_DONE rc $?" >> logs/t53_bidirectional.log
