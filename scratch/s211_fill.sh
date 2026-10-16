#!/usr/bin/env bash
# §211 filler (registered 5e554c39; user-approved Stage 0+1 via Audit, family cap $9, per-cell guard $3; Audit reviews the vbank diff before this runs).
# ONE filler, the three cells in order b304 N2 (main) -> b305 N1 -> b306 N3, each on its own RTX 3090 (GPUS env), offer <= PMAX (default $0.25/h), never
# 580.x (offer_pick). Per cell: deploy (gen_deploy_120 also arms the gen8 drain watcher) -> HOSTPROBE OK -> queue line 1 = G-TIME (60 forced final-stage
# steps: n <= 8, K 12, --auto_micro) -> admission: proj = (gtime s/step x 48000 / 3600 + SETUP_H) x the host's $/h must be <= $3.00 (per cell) AND the
# admitted family sum + this proj (+ BAD-host spend) <= $9.00; then the off-host ckpt mirror (scratch/ckpt_mirror.py, every 2 h) and a per-cell
# cost_guard (cap $3, alert only) start, and on the first admit the FAMILY guard s211fam ($9, GUARD_STOP=1; Audit 23:4x); marks 1k / 4k / 12k / 20k
# (the cumulative rate under-projects while the curriculum is in its cheap stages, REDTEAM F10). A refused host: vdestroy, drain watcher stopped by pid, drained_hosts + exclude list, billed minutes logged.
# A family projection above $9 is never resolved by dropping a cell: the filler holds and says so (Audit / the user decide). BAD / refused hosts: 4 -> held;
# cumulative BAD-host spend > $1.00 -> held.
#   nohup bash scratch/s211_fill.sh >> logs/fill_s211.out 2>&1 &
cd "$(dirname "$0")/.."
PMAX=${PMAX:-0.25}; CAPC=3.00; CAPF=9.00; STEPS=48000; SETUP_H=0.5; BADN=${BAD0:-0}; BADMAX=4; BADLOG=logs/s211_bad_cost.log; ADMLOG=logs/s211_admitted.log
QS="${QS:-304:s211_vb_pa_s1 305:s211_vb_carry_s1 306:s211_vb_pamct_s1}"
log() { echo "$(date +%T) fill_s211 $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
sq() { timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$2" "root@$1" "$3" 2>/dev/null; }
fam() { python -c "import sys; s = 0.0
for f in ('$ADMLOG', '$BADLOG'):
    try: s += sum(float(l.split()[-1]) for l in open(f) if l.strip() and not l.startswith('#'))
    except OSError: pass
print(round(s, 4))"; }
refuse() {   # host port iid offer ip dph t0 why
  for dp in $(ps -ef | grep "auto_drain_gen8.sh $1 $2 gen_b$n $3" | grep -v grep | awk '{print $2}'); do [ "$dp" != "$$" ] && kill "$dp"; done
  echo "$1:$2" >> logs/drained_hosts.txt; log "b$n: $(bash scratch/vdestroy.sh "$3" | tail -1)"
  local mins c; mins=$(( ($(date +%s) - $7) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $6, 4))"); BADN=$((BADN + 1))
  echo "$(date '+%F %T') b$n offer $4 $5 ${mins}min \$$6/h ($8) $c" >> "$BADLOG"; echo "$5 # §211 refused ($8) $(date '+%F %T')" >> scratch/vast_exclude_ips.txt
  mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"
  [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
}
for cell in $QS; do
  n=${cell%%:*}; run=${cell#*:}
  grep -q "^$n:$run " "$ADMLOG" 2>/dev/null && { log "b$n: $run already admitted - next"; continue; }
  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }
  while true; do
    [ -f scratch/fill_s211.stop ] && { log "stopped (scratch/fill_s211.stop)"; exit 0; }
    bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
    bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0)
    if python -c "import sys; sys.exit(0 if $bt > 1.0 else 1)"; then log "CAP: §211 BAD-host spend \$$bt > \$1.00 - held; tell Audit"; exit 3; fi
    best=""; for gn in ${GPUS:-RTX_3090}; do
      r_=$(GPU_NAME=$gn python scratch/offer_pick.py - "$PMAX"); [ -z "$r_" ] && continue
      if [ -z "$best" ] || python -c "import sys; sys.exit(0 if float('$(echo $r_ | cut -d' ' -f3)') < float('$(echo $best | cut -d' ' -f3)') else 1)"; then best="$r_"; bgn=$gn; fi
    done
    read -r o ip dph <<<"$best"
    if [ -z "${o:-}" ]; then log "b$n: no ${GPUS:-RTX_3090} offer (non-580, <= \$$PMAX); retry in 10 min"; sleep 600; continue; fi
    log "b$n: deploying $run on offer $o ($ip, $bgn, \$$dph/h)"; t0=$(date +%s); bash scratch/gen_deploy_120.sh "$o $n"
    line=$(grep "gen host b$n offer $o:" logs/gen_deploy.log | tail -1)
    if ! echo "$line" | grep -q STARTED; then
      mins=$(( ($(date +%s) - t0) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $dph, 4))"); BADN=$((BADN + 1))
      echo "$(date '+%F %T') b$n offer $o $ip ${mins}min \$$dph/h $c" >> "$BADLOG"; log "b$n: did not start (BAD HOST $BADN / $BADMAX; billed ~\$$c)"
      [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
      continue
    fi
    mv "scratch/gen_q_b$n.txt" "scratch/gen_q_b$n.txt.norequeue"
    read -r h p iid <<<"$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="INSTANCE") print $(i+2)" "$(i+3)" "$(i+1)}')"
    pr=""; for i in $(seq 1 40); do pr=$(sq "$h" "$p" "grep -h 'HOSTPROBE' /workspace/Latent/logs/queue_gen_b$n.out | tail -n 1"); echo "$pr" | grep -q "s/step" && break; pr=""; sleep 60; done
    if [ -z "$pr" ] || echo "$pr" | grep -q HOSTPROBE_SLOW; then log "b$n: HOSTPROBE ${pr:-none after 40 min} on $h:$p - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "hostprobe"; continue; fi
    log "b$n: $pr on $h:$p"
    gt=""; for i in $(seq 1 45); do gt=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_gtime/timing.json"); [ -n "$gt" ] && break; sleep 60; done
    sps=$(echo "$gt" | python -c "import json,sys; print(json.load(sys.stdin)['sec_per_step'])" 2>/dev/null)
    am=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_gtime/auto_micro.json" | tr -d '\n ')
    if [ -z "$sps" ]; then log "b$n: no G-TIME timing.json after 45 min on $h:$p ($(sq "$h" "$p" "tail -n 3 /workspace/Latent/logs/gen_${run}_gtime.log" | tr '\n' ' ' | cut -c1-300)) - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "no-gtime"; continue; fi
    proj=$(python -c "print(round(($sps * $STEPS / 3600 + $SETUP_H) * $dph, 2))"); fs=$(fam)
    log "b$n: G-TIME $sps s/step ($am) -> cell projection \$$proj (cap \$$CAPC); family so far \$$fs (admitted + BAD) of \$$CAPF"
    if python -c "import sys; sys.exit(0 if $proj <= $CAPC and $fs + $proj <= $CAPF else 1)"; then
      echo "$n:$run $h:$p $iid \$$dph/h sps $sps $proj" >> "$ADMLOG"; log "b$n: ADMIT $run on $h:$p (instance $iid): \$$proj"
      nohup python scratch/ckpt_mirror.py "$n:$run" --every_h 2 > "logs/ckpt_mirror_b$n.out" 2>&1 &
      GUARD_MARKS=1000,4000,12000,20000 nohup python scratch/cost_guard.py "s211_b$n" 3.00 "$STEPS" "$n:$run:$proj" > "logs/s211_guard_b$n.out" 2>&1 &
      log "b$n: ckpt mirror + cost guard (\$3, alert only) started"
      if ! ps -ef | grep "cost_guard.py s211fam " | grep -v grep > /dev/null; then   # Audit 23:4x: the FAMILY \$9 guard with GUARD_STOP=1 (BAD hosts included)
        GUARD_STOP=1 GUARD_MARKS=1000,4000,12000,20000 nohup python scratch/cost_guard.py s211fam 9.00 "$STEPS" 304:s211_vb_pa_s1:3.0 305:s211_vb_carry_s1:3.0 306:s211_vb_pamct_s1:3.0 > logs/s211fam_guard.out 2>&1 &
        log "family guard s211fam (\$9, GUARD_STOP=1; unstarted cells at \$3) started"
      fi
      sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; break
    fi
    log "b$n: projection \$$proj (cell cap \$$CAPC; family \$$fs + \$$proj vs \$$CAPF) NOT admitted on $h:$p - HOLD: no cell is dropped silently; tell Audit / the user"
    for dp in $(ps -ef | grep "auto_drain_gen8.sh $h $p gen_b$n $iid" | grep -v grep | awk '{print $2}'); do [ "$dp" != "$$" ] && kill "$dp"; done
    echo "$h:$p" >> logs/drained_hosts.txt; log "b$n: $(bash scratch/vdestroy.sh "$iid" | tail -1)"
    mins=$(( ($(date +%s) - t0) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $dph, 4))"); echo "$(date '+%F %T') b$n offer $o $ip ${mins}min \$$dph/h (projection hold) $c" >> "$BADLOG"
    mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; exit 5
  done
done
log "all §211 cells admitted: $(cat $ADMLOG | tr '\n' ';')"
