#!/usr/bin/env bash
# §216 SD2 filler (registered 25016c72; SD2_v1.md at e9ade2ec §8; user-approved 01:31 10-18 via Audit, family cap $6.50 under GUARD_STOP). Copied from
# scratch/s215_fill.sh; changed: cells n:run:admission:guard:steps (QS), per-cell steps in the projection and guards, the family guard s216fam $6.50
# (GUARD_STOP=1, per-cell steps; unstarted cells at their guard), per-cell EXTRA_120 (cell_extra), and note 215-2's try rule: only a MEASURED G-TIME
# refusal uses one of HOSTMAX tries (BAD / non-start / hostprobe / no-gtime hosts do not, but count toward the cell's guard and the family).
# ONE filler, cells in QS order, each on its own RTX 3090, offer <= PMAX, never 580.x. Per cell: deploy (gen_deploy_120 arms the gen8 drain watcher) ->
# HOSTPROBE -> line 1 mem_check -> line 2 G-TIME -> admission: (gtime s/step x steps / 3600 + SETUP_H) x $/h <= the cell's bound AND admitted + BAD + proj
# <= $6.50; then the off-host mirror (~2k steps), the cell guard (guard - the cell's BAD spend, GUARD_STOP=1) and, once, s216fam.
#   QS="312:s216_d3t_s2:1.75:2.00:48000" nohup bash scratch/s216_fill.sh >> logs/fill_s216.out 2>&1 &
cd "$(dirname "$0")/.."
PMAX=${PMAX:-0.25}; CAPF=6.50; SETUP_H=0.4; BADN=${BAD0:-0}; BADMAX=4; HOSTMAX=${HOSTMAX:-3}; BADLOG=logs/s216_bad_cost.log; ADMLOG=logs/s216_admitted.log
QS="${QS:-312:s216_d3t_s2:1.75:2.00:48000}"
FAMC="312:s216_d3t_s2:2.00:48000 313:s216_r2_s1:1.70:12000 314:s216_r1_s2:1.05:12000 315:s216_r2_s2:1.70:12000"   # unstarted cells project at their guard
cell_extra() {   # per-cell shipped files beside the queue's modules (auto_ship) - the ARGS-DIFF ref, the init model + calibration, the pool
  case "$1" in
    312) echo "scratch/s214_args_ref_d3t.json" ;;
    313) echo "scratch/s216_args_ref_ftself.json boxes/runs_fsq/s215_ftself_s1/model.pt boxes/runs_fsq/s215_ftself_s1/calibration.json boxes/runs_fsq/s216_pool_s1/pool_r2.jsonl boxes/runs_fsq/s216_pool_s1/pool_meta.json" ;;
    314) echo "scratch/s216_args_ref_ftself.json boxes/runs_fsq/s216_d3t_s2/model.pt boxes/runs_fsq/s216_d3t_s2/calibration.json boxes/runs_fsq/s216_pool_r1_s2/pool_r1.jsonl boxes/runs_fsq/s216_pool_r1_s2/pool_meta.json" ;;
    315) echo "scratch/s216_args_ref_ftself.json boxes/runs_fsq/s216_r1_s2/model.pt boxes/runs_fsq/s216_r1_s2/calibration.json boxes/runs_fsq/s216_pool_s2/pool_r2.jsonl boxes/runs_fsq/s216_pool_s2/pool_meta.json" ;;
  esac
}
log() { echo "$(date +%T) fill_s216 $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
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
  echo "$(date '+%F %T') b$n offer $4 $5 ${mins}min \$$6/h ($8) $c" >> "$BADLOG"; echo "$5 # §216 refused ($8) $(date '+%F %T')" >> scratch/vast_exclude_ips.txt
  mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"
  [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
}
for cell in $QS; do
  IFS=: read -r n run CAPC GSTOP STEPS <<<"$cell"; tries=0; export EXTRA_120="$(cell_extra "$n")"
  grep -q "^$n:$run " "$ADMLOG" 2>/dev/null && { log "b$n: $run already admitted - next"; continue; }
  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }
  while true; do
    [ -f scratch/fill_s216.stop ] && { log "stopped (scratch/fill_s216.stop)"; exit 0; }
    bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
    bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0)
    if python -c "import sys; sys.exit(0 if $bt > 0.50 else 1)"; then log "CAP: §216 BAD-host spend \$$bt > \$0.50 - held; tell Audit"; exit 3; fi
    [ "$tries" -ge "$HOSTMAX" ] && { log "b$n: $run NOT RUN - no admitting host in $HOSTMAX tries (G-S10); tell Audit / the user"; continue 2; }
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
      evh=$(python -c "print(round(max(0.25, 2000 * $sps / 3600), 2))")       # the mirror every ~2k steps (SELFDEPTH_v1 §2.3)
      nohup python scratch/ckpt_mirror.py "$n:$run" --every_h "$evh" > "logs/ckpt_mirror_b$n.out" 2>&1 &
      cb=$(awk -v n="b$n" '!/^#/ && $3 == n {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0); gl=$(python -c "print(round($GSTOP - $cb, 4))")
      GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000,16000,32000 nohup python scratch/cost_guard.py "s216_b$n" "$gl" "$STEPS" "$n:$run:$proj:$STEPS" > "logs/s216_guard_b$n.out" 2>&1 &
      log "b$n: ckpt mirror (every $evh h) + cell guard (\$$GSTOP - rejected-host \$$cb = \$$gl, GUARD_STOP=1) started"
      if ! ps -ef | grep "cost_guard.py s216fam " | grep -v grep > /dev/null; then   # the FAMILY \$6.50 guard with GUARD_STOP=1 (BAD hosts included)
        GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000,16000,32000 nohup python scratch/cost_guard.py s216fam 6.50 48000 $FAMC > logs/s216fam_guard.out 2>&1 &
        log "guard s216fam (\$6.50, GUARD_STOP=1; $FAMC) started"
      fi
      sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; break
    fi
    log "b$n: projection \$$proj (cell cap \$$CAPC; pair \$$fs + \$$proj vs \$$CAPF) NOT admitted on $h:$p (measured refusal $((tries + 1)) / $HOSTMAX) - destroying, next host"
    tries=$((tries + 1)); refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "projection"
  done
done
log "§216 filler done: $(cat $ADMLOG 2>/dev/null | tr '\n' ';')"
