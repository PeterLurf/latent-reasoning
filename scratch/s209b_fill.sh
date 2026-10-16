#!/usr/bin/env bash
# §210 filler (user 21:5x 'go back to 3090s': RTX 3090 by default, PMAX $0.25; GPUS env lists card classes). Was the §209 step 2 A100 filler (Audit 20:3x 10-15): one cell, b303 gsm_codi_20ep_s1 on an A100 SXM4 (GPU_NAME=A100_SXM4), offer cap PMAX (default $0.60/h), never 580.x
# (offer_pick). At HOSTPROBE OK the run is admitted only if its projection <= CAP ($12):
#   hours = probe s/step x RATIO x STEPS / 3600 + 0.45 (setup + 20 valid evals + final eval / ablation), cost = hours x the host's $/h;
#   RATIO = train s/step / probe s/step: 3.22 from the 3090 (b302: 2.11 / 0.6561) until an A100 host has measured its own (logged, then used).
# A refused host: vdestroy, its gen8 drain watcher stopped by pid, drained_hosts + exclude list, billed minutes logged; stays pending. 4 BAD / refused hosts -> held.
#   PMAX=0.60 nohup bash scratch/s209b_fill.sh >> logs/fill_s209b.out 2>&1 &
cd "$(dirname "$0")/.."
PMAX=${PMAX:-0.60}; CAP=11.5; STEPS=59822; RATIO=${RATIO:-3.22}; n=303; BADN=${BAD0:-0}; BADMAX=4; BADLOG=logs/s209b_bad_cost.log
export EXTRA_120="$(tr '\n' ' ' < scratch/gen_q_b303.extra)"
log() { echo "$(date +%T) fill_s209b $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
probe() {   # host port -> "OK <s/step>" / "SLOW" / ""
  local r; r=$(timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$2" "root@$1" "grep -h 'HOSTPROBE' /workspace/Latent/logs/queue_gen_b$n.out | tail -n 1" 2>/dev/null)
  echo "$r" | grep -q "HOSTPROBE_SLOW" && { echo SLOW; return; }; echo "$r" | grep -q "s/step" && echo "OK $(echo "$r" | awk '{for(i=1;i<=NF;i++) if($i=="HOSTPROBE") print $(i+1)}')"
}
g1check() {   # §210 G1': the released CODI ckpt re-scored on this host (queue line 1) must be within 0.010 of 0.425, else stop the host and report
  local r=""; for i in $(seq 1 30); do
    r=$(timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$2" "root@$1" "cat /workspace/Latent/boxes/runs_fsq/gsm_codi_20ep_s1_g1/eval.json 2>/dev/null" 2>/dev/null)
    [ -n "$r" ] && break; sleep 60
  done
  if echo "$r" | python -c "import json,sys; d=json.load(sys.stdin); print('G1PRIME acc', d['acc'], 'pass', d['pass'], d.get('card')); sys.exit(0 if d['pass'] else 1)" >> logs/fill_s209b.out 2>&1; then
    log "b$n: G1' PASS on $1:$2"; return 0
  fi
  log "b$n: G1' FAIL or missing on $1:$2 - stopping the host (instance $3) and reporting (no retry)"
  for dp in $(ps -ef | grep "auto_drain_gen8.sh $1 $2 gen_b$n $3" | grep -v grep | awk '{print $2}'); do [ "$dp" != "$$" ] && kill "$dp"; done
  echo "$1:$2" >> logs/drained_hosts.txt; log "b$n: $(bash scratch/vdestroy.sh "$3" | tail -1)"; exit 4
}
[ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }
while [ ! -f scratch/fill_s209b.stop ]; do
  bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
  bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0)       # Audit 21:4x: cumulative BAD-host spend cap $1.00
  if python -c "import sys; sys.exit(0 if $bt > 1.0 else 1)"; then log "CAP: §210 BAD-host spend \$$bt > \$1.00 - held; tell Audit"; exit 3; fi
  best=""; for gn in ${GPUS:-RTX_3090}; do                                   # user 21:5x 'go back to 3090s' (was A100_SXM4 A100_PCIE; GPUS env overrides)                            # Audit 21:4x: SXM4 or PCIe 40 / 80 GB (same sm80 chip), the cheapest qualifying offer
    r_=$(GPU_NAME=$gn python scratch/offer_pick.py - "$PMAX"); [ -z "$r_" ] && continue
    if [ -z "$best" ] || python -c "import sys; sys.exit(0 if float('$(echo $r_ | cut -d' ' -f3)') < float('$(echo $best | cut -d' ' -f3)') else 1)"; then best="$r_"; bgn=$gn; fi
  done
  read -r o ip dph <<<"$best"
  if [ -z "${o:-}" ]; then log "b$n: no ${GPUS:-RTX_3090} offer (non-580, <= \$$PMAX); retry in 10 min"; sleep 600; continue; fi
  log "b$n: deploying on offer $o ($ip, $bgn, \$$dph/h)"; t0=$(date +%s); bash scratch/gen_deploy_120.sh "$o $n"
  line=$(grep "gen host b$n offer $o:" logs/gen_deploy.log | tail -1)
  if ! echo "$line" | grep -q STARTED; then
    mins=$(( ($(date +%s) - t0) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $dph, 4))"); BADN=$((BADN + 1))
    echo "$(date '+%F %T') b$n offer $o $ip ${mins}min \$$dph/h $c" >> "$BADLOG"; log "b$n: did not start (BAD HOST $BADN / $BADMAX; billed ~\$$c)"
    [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
    continue
  fi
  log "b$n: STARTED - UPLOAD_OK of the 90 MB bundle (the real test of the 21:0x fix; $bgn)"; mv "scratch/gen_q_b$n.txt" "scratch/gen_q_b$n.txt.norequeue"
  read -r h p iid <<<"$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="INSTANCE") print $(i+2)" "$(i+3)" "$(i+1)}')"
  res=""; for i in $(seq 1 40); do res=$(probe "$h" "$p"); [ -n "$res" ] && break; sleep 60; done
  read -r st pv <<<"$res"
  if [ "$st" = "OK" ]; then
    proj=$(python -c "print(round(($pv * $RATIO * $STEPS / 3600 + 0.45) * $dph, 2))")
    if python -c "import sys; sys.exit(0 if $proj <= $CAP else 1)"; then
      log "b$n: HOSTPROBE OK ($pv s/step) on $h:$p - ADMIT: projection \$$proj (ratio $RATIO, $STEPS steps, \$$dph/h) <= \$$CAP"
      g1check "$h" "$p" "$iid"; sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; log "b$n: done (admitted)"; exit 0
    fi
    log "b$n: HOSTPROBE OK ($pv s/step) but projection \$$proj > \$$CAP on ratio $RATIO"
    if [ ! -s logs/s209b_ratio.txt ]; then                               # Audit: the 3090 ratio may not transfer -> measure the real run's s/step at step >= 200 first
      sps=""; for i in $(seq 1 60); do
        sps=$(timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$p" "root@$h" "grep -o '\"step\": [0-9]*, \"loss\": [^,]*, \"sec\": [0-9]*' /workspace/Latent/boxes/runs_fsq/gsm_codi_20ep_s1/log.jsonl | tail -n 1" 2>/dev/null \
              | python -c "import re,sys; m_=re.search(r'\"step\": (\d+), \"loss\": [^,]*, \"sec\": (\d+)', sys.stdin.read()); print(int(m_.group(2)) / int(m_.group(1)) if m_ and int(m_.group(1)) >= 200 else '')")
        [ -n "$sps" ] && break; sleep 60
      done
      if [ -n "$sps" ]; then
        RATIO=$(python -c "print(round($sps / $pv, 3))"); echo "$(date '+%F %T') A100 probe $pv s/step, train $sps s/step at step >= 200 -> ratio $RATIO ($h:$p)" >> logs/s209b_ratio.txt
        proj=$(python -c "print(round(($sps * $STEPS / 3600 + 0.45) * $dph, 2))"); log "b$n: measured train $sps s/step (ratio $RATIO) -> projection \$$proj"
        if python -c "import sys; sys.exit(0 if $proj <= $CAP else 1)"; then
          log "b$n: ADMIT on the measured rate: projection \$$proj <= \$$CAP"; g1check "$h" "$p" "$iid"; sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; log "b$n: done (admitted)"; exit 0
        fi
      else
        log "b$n: no step-200 record within 60 min - treating as refused"
      fi
    fi
    log "b$n: refusing $h:$p (instance $iid); projection \$$proj > \$$CAP - tell Audit before the $BADMAX-refusal hold"
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
