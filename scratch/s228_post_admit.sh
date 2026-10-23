#!/usr/bin/env bash
# §228 per-cell check after admission (a copy of scratch/s224_post_admit.sh minus G-FH-EQ; started by scratch/s228_fill.sh with nohup): wait (<= 150 min)
# for the cell's host args.json -> scratch/s228_args_check.py check vs scratch/s228_args_frozen_<run>.json (incl. the shipped module shas); BAD -> HOLD:
# ALERT line (logs/s228_alerts.log + gen_deploy.log), the host's §224-chain drain watcher killed, the instance destroyed, the filler stopped
# (scratch/fill_s228.stop). Read-only ssh otherwise.   bash scratch/s228_post_admit.sh <n> <run> <host> <port> <instance>
cd "$(dirname "$0")/.." || exit 2
STOPF=scratch/fill_s228.stop; ALERTS=logs/s228_alerts.log; PALOG=logs/s228_post_admit.log
n=$1; run=$2; h=$3; p=$4; iid=$5; W=/workspace/Latent/boxes/runs_fsq
log() { echo "$(date +%T) s228_post b$n $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') b$n $run $*" >> "$PALOG"; }
sq() { timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$p" "root@$h" "$1" 2>/dev/null; }
hold() {
  echo "$(date '+%F %T') ALERT b$n $run: $* - HOLD (instance $iid destroyed, filler stopped); tell Audit / Theory" >> "$ALERTS"; log "ALERT $* - HOLD"
  for dp in $(ps -ef | grep "auto_drain_gen8_s224.sh $h $p gen_b$n $iid" | grep -v grep | awk '{print $2}'); do kill "$dp"; done
  touch "$STOPF"; log "$(bash scratch/vdestroy.sh "$iid" | tail -1)"; exit 3
}
ca=""; for i in $(seq 1 30); do ca=$(sq "cat $W/$run/args.json"); [ -n "$ca" ] && break; sleep 300; done
[ -n "$ca" ] || hold "no cell args.json after 150 min"
ck=$(echo "$ca" | python scratch/s228_args_check.py check - "scratch/s228_args_frozen_$run.json"); log "host cell args.json: $ck"
echo "$ck" | grep -q "^ARGS-CHECK OK" || hold "args $ck"
log "POSTCHECK OK"
