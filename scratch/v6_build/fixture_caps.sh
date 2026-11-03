#!/usr/bin/env bash
# v6 item 5 fixture: scratch/caps_check.py on a 3-line fake queue. Expect: a match -> rc 0 and the assignments (WALLW_G = the last weight + fetch_gb / 10);
# env mismatch (EVAL_H / WALLW / GPU) -> rc 2; no header -> rc 2; wrong weight count -> rc 2; hostprobe vs '# HOSTPROBE' -> rc 2; PMAX > admit / hours -> rc 2.
set -u
cd "$(dirname "$0")/../.." || exit 2
R=scratch/_tmp/capsfx; rm -rf "$R"; mkdir -p "$R"
mk() { printf '%s\n' "# THEORY fx" "# HOSTPROBE" "$1" "# P=1" "boxes/runs_fsq/x_mem boxes.inject_aw --mem_check" "boxes/runs_fsq/x scratch.s257_guarded --mode train -- --steps 2000" \
  "boxes/runs_fsq/x scratch.s268_stopland --step 2000" > "$R/q.txt"; }
C="# CAPS wallw=0.1,1.0,0.3 eval_h=0.3 gates_h=0 sps_est=1.4 hostprobe=0.854 class=RTX_3090 fetch_gb=1.0"
ok=1
t() {   # t <name> <want rc> <env...> ; runs caps_check
  local n=$1 w=$2; shift 2; out=$(env -u WALLW -u EVAL_H -u GATES_H -u SPS_EST -u GPU "$@" python scratch/caps_check.py "$R/q.txt" 2000 0.45 0.21 0.4); rc=$?
  [ $rc = $w ] && echo "case $n OK (rc $rc) $out" | cut -c1-160 || { echo "case $n FAIL (rc $rc, want $w): $out"; ok=0; }; }
mk "$C"; t match 0; echo "$out" | grep -q "WALLW_G=0.1,1.0,0.4" || { echo "FAIL: WALLW_G"; ok=0; }
t env_same 0 EVAL_H=0.30 GPU=RTX_3090
t env_eval 2 EVAL_H=0.5
t env_wallw 2 WALLW=0.1,1.0,0.5
t env_gpu 2 GPU=RTX_A6000
mk "# no caps"; t no_header 2
mk "${C/wallw=0.1,1.0,0.3/wallw=0.1,1.0}"; t weights 2
mk "${C/hostprobe=0.854/hostprobe=1.15}"; t hostprobe 2
mk "$C"; out=$(env -u WALLW -u EVAL_H -u GATES_H -u SPS_EST -u GPU python scratch/caps_check.py "$R/q.txt" 2000 0.35 0.25 0.4); rc=$?
[ $rc = 2 ] && echo "case pmax OK (rc 2)" || { echo "case pmax FAIL (rc $rc): $out"; ok=0; }
python -c "import sys; sys.path.insert(0, 'scratch'); import caps_check as C; q = open('$R/q.txt').read(); print('fetch_gb', C.fetch_gb(q)); sys.exit(0 if C.fetch_gb(q) == 0.51 else 1)" || { echo "FAIL: fetch_gb"; ok=0; }
[ $ok = 1 ] && echo "FIXTURE caps PASS" || echo "FIXTURE caps FAIL"
