#!/usr/bin/env bash
# v6 filler fixture: scratch/s271_fill.sh (template v6) in a fake git root with stubs (offer_pick prints nothing, rebuild_live_hosts no-op). Cases on one cell b901:
#   A  the queue has no '# CAPS' header                 -> 'REFUSED b901: REFUSED: no single # CAPS ...', rc 2
#   B  env EVAL_H=0.9 vs the header's 0.3              -> REFUSED (env != CAPS), rc 2
#   C  header + no env caps                            -> 'caps from the queue header' and 'no RTX_3090 offer' (GPU from the header; the filler then sleeps: killed)
#   D  the header's wallw has 2 weights, the queue 3 lines -> REFUSED, rc 2
# + the chained top line sets ADMLOG / BADLOG / GPU (the inline-comment trap): checked in C (the start log names the ledgers via the 'start:' line + no bash error).
set -u
cd "$(dirname "$0")/../.." || exit 2
SRC=$(pwd); R=$SRC/scratch/_tmp/fillv6fx; rm -rf "$R"; mkdir -p "$R/scratch" "$R/boxes" "$R/logs"
cp scratch/s271_fill.sh scratch/caps_check.py scratch/check_queue_ships.py scratch/check_py311.py "$R/scratch/"
printf 'def tok_aw():\n    pass\n' > "$R/boxes/inject_aw.py"; echo "# eval" > "$R/boxes/inject_aw_eval.py"
printf 'import sys\n' > "$R/scratch/offer_pick.py"; printf '#!/usr/bin/env bash\nexit 0\n' > "$R/scratch/rebuild_live_hosts.sh"
echo '{}' > "$R/scratch/s271_args_frozen_xrun.json"
C="# CAPS wallw=0.1,0.1,1.0 eval_h=0.3 gates_h=0 sps_est=1.4 hostprobe=0.854 class=RTX_3090 fetch_gb=0.5"
mkq() { printf '%s\n' "# THEORY fx" "# HOSTPROBE" "$1" "# P=1" "boxes/runs_fsq/xrun_mem boxes.inject_aw --mem_check" "boxes/runs_fsq/xrun_gtime boxes.inject_aw --time_steps 60" \
  "boxes/runs_fsq/xrun scratch.s257_guarded --mode train -- --steps 2000" | grep -v '^$' > "$R/scratch/gen_q_b901.txt"; (cd "$R" && git add -A > /dev/null && git commit -qm q > /dev/null); }
(cd "$R" && git init -q && git config user.email fx@x && git config user.name fx && git add -A && git commit -qm init) > /dev/null
run() {   # run <case> <env assignments...>
  local c=$1; shift; : > "$R/out_$c.log"
  (cd "$R" && env -u WALLW -u EVAL_H -u GATES_H -u SPS_EST -u GPU "$@" CAPF=1.0 QS="901:xrun:0.42:0.47:2000:0.4" bash scratch/s271_fill.sh >> "$R/out_$c.log" 2>&1) & local pid=$!
  for i in $(seq 1 30); do sleep 1; kill -0 $pid 2>/dev/null || break; grep -q "no RTX_3090 offer\|no  offer" "$R/out_$c.log" && break; done
  kill $pid 2>/dev/null; for p_ in $(ps -ef | grep 's271_fill.sh' | grep -v grep | awk '{print $2}'); do kill $p_ 2>/dev/null; done; wait $pid 2>/dev/null; }
ok=1
mkq "# (no caps)"; run A; grep -q "REFUSED b901: REFUSED: no single" "$R/out_A.log" || { echo "A FAIL"; tail -n 3 "$R/out_A.log"; ok=0; }
mkq "$C"; run B EVAL_H=0.9; grep -q "env EVAL_H=0.9 != CAPS eval_h=0.3" "$R/out_B.log" || { echo "B FAIL"; tail -n 3 "$R/out_B.log"; ok=0; }
mkq "$C"; run C; grep -q "caps from the queue header" "$R/out_C.log" && grep -q "no RTX_3090 offer" "$R/out_C.log" || { echo "C FAIL"; tail -n 4 "$R/out_C.log"; ok=0; }
grep -qi "command not found\|unbound\|syntax error" "$R/out_C.log" && { echo "C FAIL: bash errors"; ok=0; }
[ -s "$R/logs/s271_admitted.log" ] && [ -s "$R/logs/s271_bad_cost.log" ] || { echo "C FAIL: the ledgers were not initialised (ADMLOG / BADLOG unset?)"; ok=0; }
mkq "${C/wallw=0.1,0.1,1.0/wallw=0.1,1.0}"; run D; grep -q "CAPS wallw has 2 weights, the queue 3 lines" "$R/out_D.log" || { echo "D FAIL"; tail -n 3 "$R/out_D.log"; ok=0; }
[ $ok = 1 ] && echo "FIXTURE filler_v6 PASS" || echo "FIXTURE filler_v6 FAIL"
