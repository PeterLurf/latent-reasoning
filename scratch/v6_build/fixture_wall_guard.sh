#!/usr/bin/env bash
# v6 item 1 fixture: scratch/wall_guard_v6.py with stubbed ssh / vast CLI / vdestroy (a fake host dir), poll 2 s. Scenarios (cap $0.50, ceiling x2 = $1.00):
#   A  QUEUE_DONE, spent $0.60, drain running -> ALERT DRAINING, no vdestroy; the drain log then says ALL_DRAINED and the instance goes -> "guard done", still no vdestroy
#   B  no QUEUE_DONE, spent $0.60                -> STOP (vdestroy called)
#   C  QUEUE_DONE, the drain log says DRAIN FAILED -> STOP
#   D  QUEUE_DONE, spent $1.20 >= the ceiling     -> STOP
set -u
cd "$(dirname "$0")/../.." || exit 2
R=$(pwd)/scratch/_tmp/wgv6; rm -rf "$R"; mkdir -p "$R"
cat > "$R/ssh.sh" <<EOF
#!/usr/bin/env bash
cmd="\${@: -1}"; cmd="\${cmd//\/workspace\/Latent/$R/host}"; bash -c "\$cmd"
EOF
cat > "$R/vast.sh" <<EOF
#!/usr/bin/env bash
[ -f "$R/present" ] && echo '[{"id": 777}]' || echo '[]'
EOF
cat > "$R/vdestroy.sh" <<EOF
#!/usr/bin/env bash
echo "DESTROY \$1" >> "$R/destroyed.log"; rm -f "$R/present"; echo "DESTROYED_VERIFIED \$1"
EOF
chmod +x "$R"/*.sh
run() {   # run <scenario> <queue_done 0/1> <hours ago> ; prints the guard pid
  mkdir -p "$R/host/logs"; printf '00:00:01 evaluated a rc 0\n00:00:02 evaluated b rc 0\n' > "$R/host/logs/queue_gen_bfx.out"
  [ "$2" = 1 ] && echo "00:00:03 QUEUE_DONE" >> "$R/host/logs/queue_gen_bfx.out"
  : > "$R/drain.out"; : > "$R/destroyed.log"; touch "$R/present"; t0=$(python -c "import time; print(time.time() - $3 * 3600)")
  GUARD_STOP=1 GUARD_WEIGHTS=1,1 GUARD_SSH="bash $R/ssh.sh" GUARD_VASTCLI="$R/vast.sh" GUARD_VDESTROY="$R/vdestroy.sh" GUARD_POLL_S=2 GUARD_RETRY_S=1 GUARD_DRAINLOG="$R/drain.out" \
    python -u scratch/wall_guard_v6.py "fx_$1" 0.50 777 fakehost 22 0.10 "$t0" gen_bfx 2 0.40 > "$R/guard_$1.out" 2>&1 &
  echo $!
}
ok=1
pid=$(run A 1 6); sleep 7; [ -s "$R/destroyed.log" ] && { echo "A FAIL: destroyed while draining"; ok=0; }; grep -q "ALERT DRAINING" "$R/guard_A.out" || { echo "A FAIL: no DRAINING alert"; ok=0; }
echo "12:00:00 ALL_DRAINED" >> "$R/drain.out"; rm -f "$R/present"; sleep 6; grep -q "guard done" "$R/guard_A.out" || { echo "A FAIL: no exit after the drain"; ok=0; }
[ -s "$R/destroyed.log" ] && { echo "A FAIL: destroyed"; ok=0; }; kill $pid 2>/dev/null
pid=$(run B 0 6); sleep 6; grep -q "DESTROY 777" "$R/destroyed.log" || { echo "B FAIL: no STOP without QUEUE_DONE"; ok=0; }; kill $pid 2>/dev/null
pid=$(run C 1 6); sleep 4; echo "12:00:00 gen_bfx DRAIN FAILED (x)" >> "$R/drain.out"; sleep 6; grep -q "DESTROY 777" "$R/destroyed.log" || { echo "C FAIL: no STOP after DRAIN FAILED"; ok=0; }; kill $pid 2>/dev/null
pid=$(run D 1 12); sleep 6; grep -q "DESTROY 777" "$R/destroyed.log" || { echo "D FAIL: no STOP at the ceiling"; ok=0; }; kill $pid 2>/dev/null
[ $ok = 1 ] && echo "FIXTURE wall_guard_v6 PASS" || { echo "FIXTURE wall_guard_v6 FAIL"; for f_ in "$R"/guard_*.out; do tail -n 3 "$f_"; done; }
