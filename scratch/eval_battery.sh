#!/usr/bin/env bash
# PROTOCOL4 addendum-D scoring battery for one run:  bash scratch/eval_battery.sh <run> <device> [clock]
# Writes results/<experiment>_<run>.json; skips outputs that already exist and parse. "clock" adds clock_freeze (P17).
set -u
r=$1; dev=$2; clock=${3:-}
cd "$(dirname "$0")/.." && mkdir -p results logs
PY=$(command -v python || command -v python3)
ok() { [ -s "$1" ] && "$PY" -c "import json,sys;json.load(open(sys.argv[1]))" "$1" 2>/dev/null; }
step() { out=$1; shift; if ok "$out"; then echo "skip $out"; else echo "run  $out"; "$PY" -m "$@" --out "$out" --device "$dev" >> "logs/eval_$r.log" 2>&1 || echo "FAIL $out"; fi; }
[ -f "runs/$r/model.pt" ] || { echo "no model for $r"; exit 1; }
step results/competence_$r.json          latent.competence runs/$r --n 4096
step results/competence_matched_$r.json  latent.competence_matched runs/$r
step results/length_gen_$r.json          latent.length_gen runs/$r
step results/delayed_trace_$r.json       latent.upfront_trace runs/$r --delayed --n 400
step results/budget_sweep_$r.json        latent.budget_sweep runs/$r
[ "$clock" = clock ] && step results/clock_freeze_$r.json latent.clock_freeze runs/$r
echo "EVAL_DONE $r"
