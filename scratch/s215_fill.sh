#!/usr/bin/env bash
# §215 SELF-DEPTH-CHAIN round 1 filler (registered c995604b; SELFDEPTH_v1.md at 5ded8703 §8; user-approved via Audit 11:58 10-17, pair cap $1.5, its own
# guards - not poolable with §214; Audit reviews the vbank diff (85e49560) and these queues before the .hold files are renamed; GATE-1 must PASS first).
# Copied from scratch/s214_fill.sh; changed: two cells b310 FT-SELF -> b311 FT-CTRL with per-cell admission caps ($0.95 / $0.40, G-S10) and per-cell
# GUARD_STOP=1 guards ($1.05 / $0.45 minus that cell's rejected-host spend), the pair guard s215fam ($1.50, GUARD_STOP=1), 12k steps; a refused
# projection destroys that host and tries up to 2 more 3090 hosts for the cell (3 in all); a cell that never admits is NOT RUN and the next cell still
# runs (FT-SELF without FT-CTRL: the §4.4 rows read UNREAD); the mirror every ~2k steps (from G-TIME); per-cell EXTRA_120 (FT-SELF ships the pool).
# ONE filler, cells in order, each on its own RTX 3090 (GPUS env), offer <= PMAX (default $0.25/h), never 580.x (offer_pick). Per cell: deploy
# (gen_deploy_120 also arms the gen8 drain watcher) -> HOSTPROBE OK -> queue line 1 = mem_check (G-S9; FT-SELF --mem_check_self at the 64 longest n 12
# records, K 16), line 2 = G-TIME (G-S10; FT-SELF alternates forced self K 16 / plain K 12) -> admission: proj = (gtime s/step x 12000 / 3600 + SETUP_H)
# x $/h <= the cell's cap AND admitted + BAD spend + proj <= $1.50. BAD / refused hosts: 4 -> held; BAD-host spend > $0.30 -> held.
#   nohup bash scratch/s215_fill.sh >> logs/fill_s215.out 2>&1 &
cd "$(dirname "$0")/.."
PMAX=${PMAX:-0.25}; CAPF=1.50; STEPS=12000; SETUP_H=0.4; BADN=${BAD0:-0}; BADMAX=4; HOSTMAX=3; BADLOG=logs/s215_bad_cost.log; ADMLOG=logs/s215_admitted.log
QS="${QS:-310:s215_ftself_s1:0.95:1.05 311:s215_ftctrl_s1:0.40:0.45}"
BASE="scratch/s214_args_ref_d3t.json boxes/runs_fsq/s213_d3t_s1/model.pt boxes/runs_fsq/s213_d3t_s1/calibration.json"; POOLF="boxes/runs_fsq/s215_pool/pool_r1.jsonl boxes/runs_fsq/s215_pool/pool_meta.json"
log() { echo "$(date +%T) fill_s215 $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
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
  echo "$(date '+%F %T') b$n offer $4 $5 ${mins}min \$$6/h ($8) $c" >> "$BADLOG"; echo "$5 # §215 refused ($8) $(date '+%F %T')" >> scratch/vast_exclude_ips.txt
  mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"
  [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
}
for cell in $QS; do
  IFS=: read -r n run CAPC GSTOP <<<"$cell"; tries=0
  if [ "$n" = 310 ]; then export EXTRA_120="$BASE $POOLF"; else export EXTRA_120="$BASE"; fi
  grep -q "^$n:$run " "$ADMLOG" 2>/dev/null && { log "b$n: $run already admitted - next"; continue; }
  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }
  while true; do
    [ -f scratch/fill_s215.stop ] && { log "stopped (scratch/fill_s215.stop)"; exit 0; }
    bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
    bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0)
    if python -c "import sys; sys.exit(0 if $bt > 0.30 else 1)"; then log "CAP: §215 BAD-host spend \$$bt > \$0.30 - held; tell Audit"; exit 3; fi
    [ "$tries" -ge "$HOSTMAX" ] && { log "b$n: $run NOT RUN - no admitting host in $HOSTMAX tries (G-S10); tell Audit / the user"; continue 2; }
    best=""; for gn in ${GPUS:-RTX_3090}; do
      r_=$(GPU_NAME=$gn python scratch/offer_pick.py - "$PMAX"); [ -z "$r_" ] && continue
      if [ -z "$best" ] || python -c "import sys; sys.exit(0 if float('$(echo $r_ | cut -d' ' -f3)') < float('$(echo $best | cut -d' ' -f3)') else 1)"; then best="$r_"; bgn=$gn; fi
    done
    read -r o ip dph <<<"$best"
    if [ -z "${o:-}" ]; then log "b$n: no ${GPUS:-RTX_3090} offer (non-580, <= \$$PMAX); retry in 10 min"; sleep 600; continue; fi
    log "b$n: deploying $run on offer $o ($ip, $bgn, \$$dph/h)"; t0=$(date +%s); tries=$((tries + 1)); bash scratch/gen_deploy_120.sh "$o $n"
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
      GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000 nohup python scratch/cost_guard.py "s215_b$n" "$gl" "$STEPS" "$n:$run:$proj" > "logs/s215_guard_b$n.out" 2>&1 &
      log "b$n: ckpt mirror (every $evh h) + cell guard (\$$GSTOP - rejected-host \$$cb = \$$gl, GUARD_STOP=1) started"
      if ! ps -ef | grep "cost_guard.py s215fam " | grep -v grep > /dev/null; then   # the PAIR \$1.50 guard with GUARD_STOP=1 (BAD hosts included)
        GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000 nohup python scratch/cost_guard.py s215fam 1.50 "$STEPS" 310:s215_ftself_s1:1.05 311:s215_ftctrl_s1:0.45 > logs/s215fam_guard.out 2>&1 &
        log "guard s215fam (\$1.50, GUARD_STOP=1) started"
      fi
      sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; break
    fi
    log "b$n: projection \$$proj (cell cap \$$CAPC; pair \$$fs + \$$proj vs \$$CAPF) NOT admitted on $h:$p (host $tries / $HOSTMAX) - destroying, next host"
    refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "projection"
  done
done
log "§215 filler done: $(cat $ADMLOG 2>/dev/null | tr '\n' ';')"
