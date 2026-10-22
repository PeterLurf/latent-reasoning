#!/usr/bin/env bash
# §224 Stage-1 per-cell checks after admission (started by scratch/s224_fill.sh with nohup, so the filler moves on to the next cell):
#   (1) RD / FH cells: G-FH-EQ (Theory decision 3) - wait (<= 120 min) for <run>_eqA / _eqB log.jsonl to hold step 1200, fetch both read-only, run
#       scratch/s224_eq_check.py (s1 cells also vs boxes/runs_fsq/s215_ftctrl_s1/log.jsonl): CODE-MISMATCH / UNREADABLE -> HOLD; HOST-NOTE -> logged;
#   (2) every cell: wait (<= 150 min) for the cell's host args.json -> scratch/s224_args_check.py check vs scratch/s224_args_frozen_<run>.json (BAD -> HOLD:
#       rule b, an ARGS-DIFF voids every reading).
#   HOLD = ALERT line (logs/s224_alerts.log + gen_deploy.log), the host's gen8 drain watcher killed, the instance destroyed (vdestroy), the filler stopped
#   (scratch/fill_s224.stop). Read-only ssh otherwise.   bash scratch/s224_post_admit.sh <n> <run> <host> <port> <instance> <far 0|1> <seed>
cd "$(dirname "$0")/.." || exit 2
n=$1; run=$2; h=$3; p=$4; iid=$5; far=$6; seed=$7; W=/workspace/Latent/boxes/runs_fsq; T=scratch/_tmp/s224_eq/$run; mkdir -p "$T"
log() { echo "$(date +%T) s224_post b$n $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') b$n $run $*" >> logs/s224_post_admit.log; }
sq() { timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$p" "root@$h" "$1" 2>/dev/null; }
hold() {
  echo "$(date '+%F %T') ALERT b$n $run: $* - HOLD (instance $iid destroyed, filler stopped); tell Audit / Theory" >> logs/s224_alerts.log; log "ALERT $* - HOLD"
  for dp in $(ps -ef | grep "auto_drain_gen8_s224.sh $h $p gen_b$n $iid" | grep -v grep | awk '{print $2}'); do kill "$dp"; done
  touch scratch/fill_s224.stop; log "$(bash scratch/vdestroy.sh "$iid" | tail -1)"; exit 3
}
if [ "$far" = 1 ]; then
  ok=""; for i in $(seq 1 24); do
    ok=$(sq "grep -c '\"step\": 1200,' $W/${run}_eqA/log.jsonl $W/${run}_eqB/log.jsonl 2>/dev/null | awk -F: '{s += \$2} END {print s}'"); [ "$ok" = 2 ] && break; sleep 300; done
  [ "$ok" = 2 ] || hold "G-FH-EQ: eqA / eqB did not reach step 1200 in 120 min ($(sq "tail -n 2 /workspace/Latent/logs/gen_${run}_eqA.log" | tr '\n' ' ' | cut -c1-200))"
  for e in eqA eqB; do
    scp -q -o ConnectTimeout=20 -i "$HOME/.ssh/runpod_ed25519" -P "$p" "root@$h:$W/${run}_$e/log.jsonl" "$T/$e.jsonl" || hold "G-FH-EQ: fetch of ${run}_$e/log.jsonl failed"
  done
  ref=""; [ "$seed" = 1 ] && ref=boxes/runs_fsq/s215_ftctrl_s1/log.jsonl
  r=$(python scratch/s224_eq_check.py "$T/eqA.jsonl" "$T/eqB.jsonl" $ref); rc=$?; log "G-FH-EQ: $r"
  [ $rc = 0 ] || hold "G-FH-EQ $r"
fi
ca=""; for i in $(seq 1 30); do ca=$(sq "cat $W/$run/args.json"); [ -n "$ca" ] && break; sleep 300; done
[ -n "$ca" ] || hold "no cell args.json after 150 min"
ck=$(echo "$ca" | python scratch/s224_args_check.py check - "scratch/s224_args_frozen_$run.json"); log "host cell args.json: $ck"
echo "$ck" | grep -q "^ARGS-CHECK OK" || hold "args $ck"
log "POSTCHECK OK"
