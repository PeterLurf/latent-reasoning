#!/usr/bin/env bash
# s205_fill_s1 (2026-10-15 04:4x, Audit-approved spend cut): s205_fill restricted to the pending seed 1 (b296, driver 570.172.08) with the offer
# price cap PMAX (default $0.24/h, was $0.28): s2 / s3 project $5.94 + $5.99 at their post-gamma rates, so s1 must stay <= ~$6 for the $18 family cap.
#   PMAX=0.24 nohup bash scratch/s205_fill_s1.sh >> logs/fill_s205.out 2>&1 &
# --- the s205_fill text follows (296 only) ---
# §205 IIT-SERIAL 3-seed paired filler (from scratch/s199_dual_fill.sh; Theory de56ca30, user cap $18 via Audit): ONE filler that checks each pending
# seed's control driver every loop (b296 s1 / b297 s2 on 570.172.08, b298 s3 on 595.84; <= $0.28) and deploys whichever matches; the queue file is
# moved aside until HOSTPROBE resolves (requeue_watch3 would refill without the driver pin); a slow host stays pending and is retried paired.
# Caps: a seed stops after 4 BAD HOSTs this run; everything stops if FAMILY_BASE ($15, ~3 x the G-TIME projection) + the BAD-HOST total > $18.
#   nohup bash scratch/s205_fill.sh >> logs/fill_s205.out 2>&1 &
cd "$(dirname "$0")/.."
PMAX=${PMAX:-0.24}
log() { echo "$(date +%T) fill_s205s1 $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
declare -A DRV=( [296]=570.172.08 ); declare -A SEED=( [296]=1 ); TRYIP=none.none
# Audit 09:0x cap: a seed stops after BADMAX more BAD HOSTs (this run), or everything stops when FAMILY_BASE + the billed BAD-HOST total > FAMILY_CAP
declare -A BADN=( [296]=0 ); BADMAX=4; FAMILY_BASE=15.0; FAMILY_CAP=18.0; BADLOG=logs/s205_bad_cost.log
badtotal() { awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0; }
probe() {   # $1 n, $2 host, $3 port -> "OK" / "SLOW" / "" (unknown)
  local r; r=$(timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$3" "root@$2" "grep -h 'HOSTPROBE' /workspace/Latent/logs/queue_gen_b$1.out | tail -n 1" 2>/dev/null)
  echo "$r" | grep -q "HOSTPROBE_SLOW" && { echo SLOW; return; }; echo "$r" | grep -q "s/step" && echo OK
}
reexclude() { sed -i "s|^# $TRYIP .*|$TRYIP # BAD HOST b68; re-excluded $(date '+%F %T') after the one §199 s3 try: $1|" scratch/vast_exclude_ips.txt; log "re-excluded $TRYIP ($1)"; }
while [ ${#DRV[@]} -gt 0 ] && [ ! -f scratch/fill_s205.stop ]; do
  bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
  for n in 296; do
    [ -n "${DRV[$n]:-}" ] || continue
    read -r o ip dph <<<"$(python scratch/offer_pick.py "${DRV[$n]}" "$PMAX")"
    if [ -z "${o:-}" ]; then log "b$n: no matching offer (driver ${DRV[$n]}, <= \$$PMAX); retry in 10 min"; continue; fi
    log "b$n: deploying on offer $o ($ip, driver ${DRV[$n]}, \$$dph/h)"; t0=$(date +%s); bash scratch/gen_deploy_120.sh "$o $n"
    line=$(grep "gen host b$n offer $o:" logs/gen_deploy.log | tail -1)
    if ! echo "$line" | grep -q STARTED; then
      mins=$(( ($(date +%s) - t0) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * ${dph:-$PMAX}, 4))"); BADN[$n]=$(( BADN[$n] + 1 ))
      echo "$(date '+%F %T') b$n offer $o $ip ${mins}min \$${dph:-$PMAX}/h $c" >> "$BADLOG"; tot=$(badtotal)
      log "b$n: did not start (BAD HOST ${BADN[$n]} / $BADMAX this run; billed ~\$$c; §205 BAD-HOST total \$$tot)"; [ "$ip" = "$TRYIP" ] && reexclude "never started / left loading"
      if python -c "import sys; sys.exit(0 if $FAMILY_BASE + $tot > $FAMILY_CAP else 1)"; then log "CAP: family base \$$FAMILY_BASE + BAD-HOST \$$tot > \$$FAMILY_CAP - ALL seeds held; tell Audit / Theory"; exit 3; fi
      [ "${BADN[$n]}" -ge "$BADMAX" ] && { log "CAP: b$n reached $BADMAX BAD HOSTs - seed ${SEED[$n]} held; tell Audit / Theory"; unset "DRV[$n]"; }
      continue
    fi
    log "b$n: STARTED"; echo "$(date +%T) fill_s205_b$n b$n: STARTED" >> logs/gen_deploy.log
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
