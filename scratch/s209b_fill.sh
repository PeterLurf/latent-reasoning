#!/usr/bin/env bash
# §209 step 2 filler (Audit 20:3x 10-15): one cell, b303 gsm_codi_20ep_s1 on an A100 SXM4 (GPU_NAME=A100_SXM4), offer cap PMAX (default $0.60/h), never 580.x
# (offer_pick). At HOSTPROBE OK the run is admitted only if its projection <= CAP ($12):
#   hours = probe s/step x RATIO x STEPS / 3600 + 0.45 (setup + 20 valid evals + final eval / ablation), cost = hours x the host's $/h;
#   RATIO = train s/step / probe s/step: 3.22 from the 3090 (b302: 2.11 / 0.6561) until an A100 host has measured its own (logged, then used).
# A refused host: vdestroy, its gen8 drain watcher stopped by pid, drained_hosts + exclude list, billed minutes logged; stays pending. 4 BAD / refused hosts -> held.
#   PMAX=0.60 nohup bash scratch/s209b_fill.sh >> logs/fill_s209b.out 2>&1 &
cd "$(dirname "$0")/.."
PMAX=${PMAX:-0.60}; CAP=12.0; STEPS=59822; RATIO=${RATIO:-3.22}; n=303; BADN=0; BADMAX=4; BADLOG=logs/s209b_bad_cost.log
export GPU_NAME=A100_SXM4; export EXTRA_120="$(tr '\n' ' ' < scratch/gen_q_b303.extra)"
log() { echo "$(date +%T) fill_s209b $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
probe() {   # host port -> "OK <s/step>" / "SLOW" / ""
  local r; r=$(timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$2" "root@$1" "grep -h 'HOSTPROBE' /workspace/Latent/logs/queue_gen_b$n.out | tail -n 1" 2>/dev/null)
  echo "$r" | grep -q "HOSTPROBE_SLOW" && { echo SLOW; return; }; echo "$r" | grep -q "s/step" && echo "OK $(echo "$r" | awk '{for(i=1;i<=NF;i++) if($i=="HOSTPROBE") print $(i+1)}')"
}
[ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }
while [ ! -f scratch/fill_s209b.stop ]; do
  bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
  read -r o ip dph <<<"$(python scratch/offer_pick.py - "$PMAX")"
  if [ -z "${o:-}" ]; then log "b$n: no A100 SXM4 offer (non-580, <= \$$PMAX); retry in 10 min"; sleep 600; continue; fi
  log "b$n: deploying on offer $o ($ip, A100 SXM4, \$$dph/h)"; t0=$(date +%s); bash scratch/gen_deploy_120.sh "$o $n"
  line=$(grep "gen host b$n offer $o:" logs/gen_deploy.log | tail -1)
  if ! echo "$line" | grep -q STARTED; then
    mins=$(( ($(date +%s) - t0) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $dph, 4))"); BADN=$((BADN + 1))
    echo "$(date '+%F %T') b$n offer $o $ip ${mins}min \$$dph/h $c" >> "$BADLOG"; log "b$n: did not start (BAD HOST $BADN / $BADMAX; billed ~\$$c)"
    [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
    continue
  fi
  log "b$n: STARTED"; mv "scratch/gen_q_b$n.txt" "scratch/gen_q_b$n.txt.norequeue"
  read -r h p iid <<<"$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="INSTANCE") print $(i+2)" "$(i+3)" "$(i+1)}')"
  res=""; for i in $(seq 1 40); do res=$(probe "$h" "$p"); [ -n "$res" ] && break; sleep 60; done
  read -r st pv <<<"$res"
  if [ "$st" = "OK" ]; then
    proj=$(python -c "print(round(($pv * $RATIO * $STEPS / 3600 + 0.45) * $dph, 2))")
    if python -c "import sys; sys.exit(0 if $proj <= $CAP else 1)"; then
      log "b$n: HOSTPROBE OK ($pv s/step) on $h:$p - ADMIT: projection \$$proj (ratio $RATIO, $STEPS steps, \$$dph/h) <= \$$CAP"
      sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; log "b$n: done (admitted)"; exit 0
    fi
    log "b$n: HOSTPROBE OK ($pv s/step) but projection \$$proj > \$$CAP - refusing $h:$p (instance $iid)"
  else
    log "b$n: HOSTPROBE ${res:-unknown after 40 min} on $h:$p - refusing (instance $iid)"
  fi
  for dp in $(ps -ef | grep "auto_drain_gen8.sh $h $p gen_b$n $iid" | grep -v grep | awk '{print $2}'); do [ "$dp" != "$$" ] && kill "$dp"; done
  echo "$h:$p" >> logs/drained_hosts.txt; log "b$n: $(bash scratch/vdestroy.sh "$iid" | tail -1)"
  mins=$(( ($(date +%s) - t0) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $dph, 4))"); BADN=$((BADN + 1))
  echo "$(date '+%F %T') b$n offer $o $ip ${mins}min \$$dph/h (refused) $c" >> "$BADLOG"; echo "$ip # §209b refused $(date '+%F %T')" >> scratch/vast_exclude_ips.txt
  mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"
  [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
done
log "stopped (scratch/fill_s209b.stop)"
