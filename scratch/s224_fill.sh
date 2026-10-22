#!/usr/bin/env bash
# §224 FIXED-HOLD Stage-1 filler (FH_v1.md at 232e84ab §2 / §5 / §6; Audit approval 4e56c8ca, family cap $20; Theory's Stage-1 decisions 10-21).
# Copied from scratch/s216_fill.sh (G-D'd); changed:
#   - deploy via scratch/gen_deploy_120_s224.sh (arms auto_drain_gen8_s224.sh -> drain_gen_partial_s224.sh with the §224 done markers: the cells'
#     host evals vbank_eval acc,stall + vbank_diag khorizon land in the run dir; s224_ftctrl_s1_re = eval files only);
#   - cells QS "n:run:admission:guard:steps:far:seed"; per-cell EXTRA_120 (cell_extra: the base's calibration.json; b335 also the re-eval dir);
#   - projection = (sps x (steps + EQ) / 3600 + SETUP_H + EVAL_H x evals) x $/h, sps = G-TIME's forced shape (T_far 48: conservative), EQ = 2,400
#     G-FH-EQ steps on RD / FH hosts, evals = 1 (2 on b335: + the s224_ftctrl_s1_re evals); family guard s224fam at CAPF (Stage 1 cap) over QS;
#   - after ADMIT, scratch/s224_post_admit.sh runs in the background per cell (G-FH-EQ on RD / FH hosts, then the host args.json vs the frozen ref;
#     a failure HOLDs: instance destroyed, scratch/fill_s224.stop) so the filler moves to the next cell;
#   - start refusals: another filler running, uncommitted boxes/ or §224 script / frozen-ref edits (deploys ship the working tree), a frozen ref missing.
# ONE filler, cells in QS order, each on its own RTX 3090, offer <= PMAX, never 580.x. Per cell: deploy -> HOSTPROBE -> line 1 K7 mem_check -> line 2
# G-TIME -> admission: proj <= the cell's bound AND admitted + BAD + proj <= CAPF; then the off-host mirror (~2k steps), the cell guard
# (guard - the cell's BAD spend, GUARD_STOP=1), once s224fam, and the post-admit checks.
#   CAPF=6 QS="331:s224_fhft_s1:1.60:2.10:12000:1:1 ..." nohup bash scratch/s224_fill.sh >> logs/fill_s224.out 2>&1 &
cd "$(dirname "$0")/.."
PMAX=${PMAX:-0.25}; CAPF=${CAPF:-}; SETUP_H=0.4; EVAL_H=${EVAL_H:-1.3}; BADN=${BAD0:-0}; BADMAX=4; HOSTMAX=${HOSTMAX:-3}; BADLOG=logs/s224_bad_cost.log; ADMLOG=logs/s224_admitted.log
QS="${QS:-}"
cell_extra() {   # per-cell shipped files beside the queue's auto-shipped modules / tokens (init_from model.pt and the args ref are named in the queue)
  case "$1" in
    331|332) echo "boxes/runs_fsq/s213_d3t_s1/calibration.json" ;;
    333|334) echo "boxes/runs_fsq/s216_d3t_s2/calibration.json" ;;
    335) echo "boxes/runs_fsq/s216_d3t_s2/calibration.json boxes/runs_fsq/s224_ftctrl_s1_re/model.pt boxes/runs_fsq/s224_ftctrl_s1_re/args.json boxes/runs_fsq/s224_ftctrl_s1_re/calibration.json" ;;
  esac
}
nevals() { [ "$1" = 335 ] && echo 2 || echo 1; }
log() { echo "$(date +%T) fill_s224 $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
sq() { timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$2" "root@$1" "$3" 2>/dev/null; }
fam() { python -c "import sys; s = 0.0
for f in ('$ADMLOG', '$BADLOG'):
    try: s += sum(float(l.split()[-1]) for l in open(f) if l.strip() and not l.startswith('#'))
    except OSError: pass
print(round(s, 4))"; }
refuse() {   # host port iid offer ip dph t0 why
  for dp in $(ps -ef | grep "auto_drain_gen8_s224.sh $1 $2 gen_b$n $3" | grep -v grep | awk '{print $2}'); do [ "$dp" != "$$" ] && kill "$dp"; done
  echo "$1:$2" >> logs/drained_hosts.txt; log "b$n: $(bash scratch/vdestroy.sh "$3" | tail -1)"
  local mins c; mins=$(( ($(date +%s) - $7) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $6, 4))"); BADN=$((BADN + 1))
  echo "$(date '+%F %T') b$n offer $4 $5 ${mins}min \$$6/h ($8) $c" >> "$BADLOG"; echo "$5 # §224 refused ($8) $(date '+%F %T')" >> scratch/vast_exclude_ips.txt
  mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"
  [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
}
[ "${S224_FILL_LIB:-0}" = 1 ] && return 0

# ---------------------------------------------------------------- start refusals (nothing is created before these pass)
others=$(ps -ef | awk '$6 ~ /bash(\.exe)?$/ && $7 ~ /s2[0-9]+_fill[a-z0-9_]*\.sh$/ && $7 !~ /s224_fill\.sh$/')
[ -n "$others" ] && { echo "$(date '+%F %T') fill_s224 REFUSED: another filler is running (one filler at a time): $others"; exit 2; }
[ "$(ps -ef | awk -v me=$$ '$6 ~ /bash(\.exe)?$/ && $7 ~ /s224_fill\.sh$/ && $2 != me && $3 != me' | wc -l)" -gt 0 ] && { echo "$(date '+%F %T') fill_s224 REFUSED: a second s224_fill.sh is running"; exit 2; }
[ -n "$CAPF" ] && [ -n "$QS" ] || { echo "$(date '+%F %T') fill_s224 REFUSED: CAPF and QS must be given"; exit 2; }
for cell in $QS; do IFS=: read -r n_ r_ a_ g_ s_ f_ sd_ <<<"$cell"
  for x_ in scratch/s224_args_frozen_$r_.json; do git ls-files --error-unmatch "$x_" > /dev/null 2>&1 || { echo "$(date '+%F %T') fill_s224 REFUSED: $x_ is not committed"; exit 2; }; done; done
dirty=$( { git status --porcelain boxes/ | grep -v "^??" | grep -v "boxes/runs"; git status --porcelain scratch/s224_*.json scratch/s224_*.py scratch/s224_*.sh scratch/drain_gen_partial_s224.sh scratch/auto_drain_gen8_s224.sh scratch/gen_deploy_120_s224.sh | grep -v "^??"; } )
[ -n "$dirty" ] && { echo "$(date '+%F %T') fill_s224 REFUSED: uncommitted module / frozen-ref / script edits (deploys ship the working tree): $dirty"; exit 2; }
grep -q "def far_branch" boxes/vbank.py || { echo "$(date '+%F %T') fill_s224 REFUSED: boxes/vbank.py has no far branch (the §224 build is not committed)"; exit 2; }
log "start: QS $QS; family \$$CAPF; SETUP_H $SETUP_H; EVAL_H $EVAL_H; vbank sha $(sha256sum boxes/vbank.py | cut -c1-12); vbank_diag sha $(sha256sum boxes/vbank_diag.py | cut -c1-12); vbank_eval sha $(sha256sum boxes/vbank_eval.py | cut -c1-12)"

FAMC=""; for c_ in $QS; do IFS=: read -r n_ r_ a_ g_ s_ f_ sd_ <<<"$c_"; FAMC="$FAMC $n_:$r_:$g_:$s_"; done   # unstarted cells project at their guard
for cell in $QS; do
  IFS=: read -r n run CAPC GSTOP STEPS FAR SEED <<<"$cell"; tries=0; export EXTRA_120="$(cell_extra "$n")"; EQ=0; [ "$FAR" = 1 ] && EQ=2400; NE=$(nevals "$n")
  grep -q "^$n:$run " "$ADMLOG" 2>/dev/null && { log "b$n: $run already admitted - next"; continue; }
  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }
  for f_ in $EXTRA_120; do [ -s "$f_" ] || { log "b$n: shipped file $f_ missing - not starting"; exit 2; }; done
  while true; do
    [ -f scratch/fill_s224.stop ] && { log "stopped (scratch/fill_s224.stop)"; exit 0; }
    bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
    bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0)
    if python -c "import sys; sys.exit(0 if $bt > 0.50 else 1)"; then log "CAP: §224 BAD-host spend \$$bt > \$0.50 - held; tell Audit"; exit 3; fi
    [ "$tries" -ge "$HOSTMAX" ] && { log "b$n: $run NOT RUN - no admitting host in $HOSTMAX tries; tell Audit / Theory"; continue 2; }
    r_=$(GPU_NAME=RTX_3090 python scratch/offer_pick.py - "$PMAX"); o=""; ip=""; dph=""; read -r o ip dph <<<"$r_"
    if [ -z "${o:-}" ]; then log "b$n: no RTX_3090 offer (non-580, <= \$$PMAX); retry in 10 min"; sleep 600; continue; fi
    log "b$n: deploying $run on offer $o ($ip, RTX_3090, \$$dph/h)"; t0=$(date +%s); bash scratch/gen_deploy_120_s224.sh "$o $n"
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
    mc=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_mem/mem_check.json" | tr -d '\n ')
    gt=""; for i in $(seq 1 45); do gt=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_gtime/timing.json"); [ -n "$gt" ] && break; sleep 60; done
    sps=$(echo "$gt" | python -c "import json,sys; print(json.load(sys.stdin)['sec_per_step'])" 2>/dev/null)
    am=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_gtime/auto_micro.json" | tr -d '\n ')
    if [ -z "$sps" ]; then log "b$n: no G-TIME timing.json after 45 min on $h:$p ($(sq "$h" "$p" "tail -n 3 /workspace/Latent/logs/gen_${run}_gtime.log" | tr '\n' ' ' | cut -c1-300)) - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "no-gtime"; continue; fi
    proj=$(python -c "print(round(($sps * ($STEPS + $EQ) / 3600 + $SETUP_H + $EVAL_H * $NE) * $dph, 2))"); fs=$(fam)
    log "b$n: K7 mem_check $mc; G-TIME $sps s/step ($am) -> cell projection \$$proj ($STEPS + EQ $EQ steps, $NE eval set(s) x $EVAL_H h; cap \$$CAPC); family so far \$$fs (admitted + BAD) of \$$CAPF"
    if python -c "import sys; sys.exit(0 if $proj <= $CAPC and $fs + $proj <= $CAPF else 1)"; then
      echo "$n:$run $h:$p $iid \$$dph/h sps $sps $proj" >> "$ADMLOG"; log "b$n: ADMIT $run on $h:$p (instance $iid): \$$proj"
      evh=$(python -c "print(round(max(0.25, 2000 * $sps / 3600), 2))")       # the mirror every ~2k steps
      nohup python scratch/ckpt_mirror.py "$n:$run" --every_h "$evh" --optional log.jsonl,args.json,draws_log.json > "logs/ckpt_mirror_b$n.out" 2>&1 &
      cb=$(awk -v n="b$n" '!/^#/ && $3 == n {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0); gl=$(python -c "print(round($GSTOP - $cb, 4))")
      GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000,12000 nohup python scratch/cost_guard.py "s224_b$n" "$gl" "$STEPS" "$n:$run:$proj:$STEPS" > "logs/s224_guard_b$n.out" 2>&1 &
      log "b$n: ckpt mirror (every $evh h) + cell guard (\$$GSTOP - rejected-host \$$cb = \$$gl, GUARD_STOP=1) started"
      if ! ps -ef | grep "cost_guard.py s224fam " | grep -v grep > /dev/null; then   # the Stage-1 FAMILY guard, GUARD_STOP=1 (BAD hosts included)
        GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000,12000 nohup python scratch/cost_guard.py s224fam "$CAPF" 12000 $FAMC > logs/s224fam_guard.out 2>&1 &
        log "guard s224fam (\$$CAPF, GUARD_STOP=1;$FAMC) started"
      fi
      nohup bash scratch/s224_post_admit.sh "$n" "$run" "$h" "$p" "$iid" "$FAR" "$SEED" > "logs/s224_post_b$n.out" 2>&1 &
      log "b$n: post-admit checks started (G-FH-EQ: $FAR; args vs scratch/s224_args_frozen_$run.json)"
      sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; break
    fi
    log "b$n: projection \$$proj (cell cap \$$CAPC; family \$$fs + \$$proj vs \$$CAPF) NOT admitted on $h:$p (measured refusal $((tries + 1)) / $HOSTMAX) - destroying, next host"
    tries=$((tries + 1)); refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "projection"
  done
done
log "§224 filler done: $(cat $ADMLOG 2>/dev/null | tr '\n' ';')"
