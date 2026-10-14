#!/usr/bin/env bash
# §199 s2 / s3 paired filler (Audit 2026-10-14): ONE filler that checks BOTH control drivers each loop (b285 s3 595.84, b284 s2 570.172.08; <= $0.33)
# and deploys whichever matches first. After a STARTED the queue file is moved aside until the host's HOSTPROBE resolves, so requeue_watch3 (which
# would refill WITHOUT the driver pin) skips that failure; a slow / failed host stays pending here and is retried PAIRED. The one-time try on
# 194.228.55.129 (excluded BAD HOST, un-excluded by Audit for one s3 deploy) is recorded in vast_exclude_ips.txt either way.
#   nohup bash scratch/s199_dual_fill.sh >> logs/fill_s199_dual.out 2>&1 &
cd "$(dirname "$0")/.."
log() { echo "$(date +%T) fill_s199_dual $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
declare -A DRV=( [285]=595.84 [284]=570.172.08 ); SEED=( [285]=3 [284]=2 ); TRYIP=194.228.55.129
# Audit 09:0x cap: a seed stops after BADMAX more BAD HOSTs (this run), or everything stops when FAMILY_BASE + the billed BAD-HOST total > FAMILY_CAP
declare -A BADN=( [285]=0 [284]=0 ); BADMAX=4; FAMILY_BASE=8.0; FAMILY_CAP=9.50; BADLOG=logs/s199_bad_cost.log
badtotal() { awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0; }
probe() {   # $1 n, $2 host, $3 port -> "OK" / "SLOW" / "" (unknown)
  local r; r=$(timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$3" "root@$2" "grep -h 'HOSTPROBE' /workspace/Latent/logs/queue_gen_b$1.out | tail -n 1" 2>/dev/null)
  echo "$r" | grep -q "HOSTPROBE_SLOW" && { echo SLOW; return; }; echo "$r" | grep -q "s/step" && echo OK
}
reexclude() { sed -i "s|^# $TRYIP .*|$TRYIP # BAD HOST b68; re-excluded $(date '+%F %T') after the one §199 s3 try: $1|" scratch/vast_exclude_ips.txt; log "re-excluded $TRYIP ($1)"; }
while [ ${#DRV[@]} -gt 0 ] && [ ! -f scratch/fill_s199_dual.stop ]; do
  bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
  for n in 285 284; do
    [ -n "${DRV[$n]:-}" ] || continue
    read -r o ip dph <<<"$(python scratch/offer_pick.py "${DRV[$n]}" 0.33)"
    if [ -z "${o:-}" ]; then log "b$n: no matching offer (driver ${DRV[$n]}, <= \$0.33); retry in 10 min"; continue; fi
    log "b$n: deploying on offer $o ($ip, driver ${DRV[$n]}, \$$dph/h)"; t0=$(date +%s); bash scratch/gen_deploy_120.sh "$o $n"
    line=$(grep "gen host b$n offer $o:" logs/gen_deploy.log | tail -1)
    if ! echo "$line" | grep -q STARTED; then
      mins=$(( ($(date +%s) - t0) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * ${dph:-0.33}, 4))"); BADN[$n]=$(( BADN[$n] + 1 ))
      echo "$(date '+%F %T') b$n offer $o $ip ${mins}min \$${dph:-0.33}/h $c" >> "$BADLOG"; tot=$(badtotal)
      log "b$n: did not start (BAD HOST ${BADN[$n]} / $BADMAX this run; billed ~\$$c; §199 BAD-HOST total \$$tot)"; [ "$ip" = "$TRYIP" ] && reexclude "never started / left loading"
      if python -c "import sys; sys.exit(0 if $FAMILY_BASE + $tot > $FAMILY_CAP else 1)"; then log "CAP: family base \$$FAMILY_BASE + BAD-HOST \$$tot > \$$FAMILY_CAP - ALL seeds held; tell Audit / Theory"; exit 3; fi
      [ "${BADN[$n]}" -ge "$BADMAX" ] && { log "CAP: b$n reached $BADMAX BAD HOSTs - seed ${SEED[$n]} held; tell Audit / Theory"; unset "DRV[$n]"; }
      continue
    fi
    log "b$n: STARTED"; echo "$(date +%T) fill_s199_b$n b$n: STARTED" >> logs/gen_deploy.log   # the A4 auto-arm keys on this line
    mv "scratch/gen_q_b$n.txt" "scratch/gen_q_b$n.txt.norequeue"
    read -r h p <<<"$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="INSTANCE") print $(i+2)" "$(i+3)}')"
    res=""; for i in $(seq 1 40); do res=$(probe "$n" "$h" "$p"); [ -n "$res" ] && break; sleep 60; done
    if [ "$res" = "OK" ]; then
      log "b$n: HOSTPROBE OK on $h:$p - seed ${SEED[$n]} running paired"; [ "$ip" = "$TRYIP" ] && sed -i "s|^# $TRYIP .*|# $TRYIP (BAD HOST b68; one §199 s3 try $(date '+%F'): STARTED + HOSTPROBE OK)|" scratch/vast_exclude_ips.txt
      unset "DRV[$n]"
    else
      log "b$n: HOSTPROBE ${res:-unknown after 40 min} on $h:$p - stays pending (paired retry)"; [ "$ip" = "$TRYIP" ] && reexclude "HOSTPROBE ${res:-unknown}"
    fi
    sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"   # requeue_watch3 (5-min polls) has seen and skipped the failure file by now
  done
  [ ${#DRV[@]} -gt 0 ] && sleep 600
done
log "done (pending: ${!DRV[*]})"
