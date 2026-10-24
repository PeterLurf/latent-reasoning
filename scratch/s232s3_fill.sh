#!/usr/bin/env bash
# §232 HOLD-C1 blind s3 filler (HC_v1 §1; Note 232-3; cap $4): ONE non-580 RTX 3090 running the chain in gen_q_b346 (D3-T base s232_d3t_s3 48k ->
# FT-CTRL s3 + HC s3 12k each -> the Stage-A battery + khorizon). A copy of scratch/s230_fill.sh (G-D'd; built by scratch/s232s3_build/mk_s232s3_fill.py,
# each edit asserted once); changed: its own ledgers / stop file / family guard s232s3fam / cell guard s232s3_b<n> (the base cell, marks to 47,000) /
# post-admit paths (scratch/s224_post_admit.sh: G-FH-EQ v2 on s232_d3t_s3_eqA / _eqB + the base cell's args), no shipped extras, the admission
# projection (G-TIME x (48k + EQ) + SETUP_H + EXTRA_H for the post-base lines) and, because the step guards watch the base only, the
# wall-clock guard scratch/s232_wall_guard.py over the WHOLE host (cap CAPF - outside, destroy at cap, per-line weights WEIGHTS).
#   CAPF=4 QS="346:s232_d3t_s3:3.30:4.00:48000:1:3" nohup bash scratch/s232s3_fill.sh >> logs/fill_s232s3.out 2>&1 &
cd "$(dirname "$0")/.."
PMAX=${PMAX:-0.25}; CAPF=${CAPF:-}; SETUP_H=0.4; EXTRA_H=${EXTRA_H:-7.4}; BADN=${BAD0:-0}; BADMAX=4; HOSTMAX=${HOSTMAX:-3}; BADLOG=logs/s232s3_bad_cost.log; ADMLOG=logs/s232s3_admitted.log
BADSPENDMAX=${BADSPENDMAX:-1.00}; VAST=${VAST:-C:/Users/peter/.local/vastai/vast.sh}   # Audit 10-22: BAD cap $1.00 (host-quality brake)
FAMGUARD_LOG=${FAMGUARD_LOG:-logs/s232s3fam_guard.out}; EST_FAR=${EST_FAR:-3.00}; EST_RE=${EST_RE:-0.00}; WALL_LOG=logs/s232s3_wall.out
WEIGHTS=${WEIGHTS:-0.1,0.1,0.15,0.15,5.0,1.4,2.1,0.75,0.1,0.15,0.75,0.1,0.15,0.75,0.1,0.15,0.7}   # est. hours per gen_q_b346 line (the wall guard's projection)   # Audit 10-22 (s224fam false stop): HIGH = estimates
QS="${QS:-}"
cell_extra() {   # the FT base's calibration.json (init_from's model.pt is named in the queue and auto-shipped), as §224 Stage 1's b331
  case "$1" in
  esac
}
nevals() { echo 1; }
log() { echo "$(date +%T) fill_s232s3 $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
sq() { timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$2" "root@$1" "$3" 2>/dev/null; }
inst_state() {   # Audit 10-22 (b333 vanished Vast-side): $1 = instance id -> 0 present, 1 absent, 2 unknown (CLI failure / unparsable)
  timeout 60 bash "$VAST" show instances --raw 2>/dev/null | python -c "import json, sys
try: ids = {int(x['id']) for x in json.load(sys.stdin)}
except Exception: sys.exit(2)
sys.exit(0 if int(sys.argv[1]) in ids else 1)" "$1"; }
vanished() {   # in a wait loop: $1 = instance id; counts consecutive ABSENT polls in VMISS; returns 0 once absent on 2 consecutive polls
  inst_state "$1"; local r=$?
  if [ $r = 1 ]; then VMISS=$((VMISS + 1)); elif [ $r = 0 ]; then VMISS=0; fi
  [ "$VMISS" -ge 2 ]; }
famcheck() {   # Audit 10-22: exit when the family guard has fired (ALERT / STOP in its log); $1 = where (loop top / after ADMIT)
  if grep -qE "ALERT|STOP" "$FAMGUARD_LOG" "$WALL_LOG" 2>/dev/null; then log "HOLD: s232s3 guard fired ($(grep -hE "ALERT|STOP" "$FAMGUARD_LOG" "$WALL_LOG" 2>/dev/null | tail -1 | cut -c1-160)) - filler exits at $1; tell Audit"; exit 3; fi
}
outside() { python -c "s = 0.0                                             # realised spend outside the guarded cells: realised: / gatehold: ledger lines + BAD
for l in open('$ADMLOG'):
    if l.strip() and not l.startswith('#') and l.startswith(('realised:', 'gatehold:')): s += float(l.split()[-1])
for l in open('$BADLOG'):
    if l.strip() and not l.startswith('#'): s += float(l.split()[-1])
print(round(s, 4))"; }
fam() { python -c "import sys; s = 0.0
for f in ('$ADMLOG', '$BADLOG'):
    try: s += sum(float(l.split()[-1]) for l in open(f) if l.strip() and not l.startswith('#'))
    except OSError: pass
print(round(s, 4))"; }
refuse() {   # host port iid offer ip dph t0 why
  for dp in $(ps -ef | grep "auto_drain_gen8_s224.sh $1 $2 gen_b$n $3" | grep -v grep | awk '{print $2}'); do [ "$dp" != "$$" ] && kill "$dp"; done
  echo "$1:$2" >> logs/drained_hosts.txt; log "b$n: $(bash scratch/vdestroy.sh "$3" | tail -1)"
  local mins c; mins=$(( ($(date +%s) - $7) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $6, 4))"); BADN=$((BADN + 1))
  echo "$(date '+%F %T') b$n offer $4 $5 ${mins}min \$$6/h ($8) $c" >> "$BADLOG"; echo "$5 # §232s3 refused ($8) $(date '+%F %T')" >> scratch/vast_exclude_ips.txt
  mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"
  [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
}
[ "${S224_FILL_LIB:-0}" = 1 ] && return 0

# ---------------------------------------------------------------- start refusals (nothing is created before these pass)
others=$(ps -ef | awk '$6 ~ /bash(\.exe)?$/ && $7 ~ /s2[0-9]+_fill[a-z0-9_]*\.sh$/ && $7 !~ /s232s3_fill\.sh$/')
[ -n "$others" ] && { echo "$(date '+%F %T') fill_s232s3 REFUSED: another filler is running (one filler at a time): $others"; exit 2; }
[ "$(ps -ef | awk -v me=$$ '$6 ~ /bash(\.exe)?$/ && $7 ~ /s232s3_fill\.sh$/ && $2 != me && $3 != me' | wc -l)" -gt 0 ] && { echo "$(date '+%F %T') fill_s232s3 REFUSED: a second s232s3_fill.sh is running"; exit 2; }
[ -n "$CAPF" ] && [ -n "$QS" ] || { echo "$(date '+%F %T') fill_s232s3 REFUSED: CAPF and QS must be given"; exit 2; }
for cell in $QS; do IFS=: read -r n_ r_ a_ g_ s_ f_ sd_ <<<"$cell"
  for x_ in scratch/s224_args_frozen_$r_.json; do git ls-files --error-unmatch "$x_" > /dev/null 2>&1 || { echo "$(date '+%F %T') fill_s232s3 REFUSED: $x_ is not committed"; exit 2; }; done; done
dirty=$( { git status --porcelain boxes/ | grep -v "^??" | grep -v "boxes/runs"; git status --porcelain scratch/s224_*.json scratch/s224_*.py scratch/s224_*.sh scratch/s232s3_*.sh scratch/s232_wall_guard.py scratch/gen_q_b346.txt scratch/drain_gen_partial_s224.sh scratch/auto_drain_gen8_s224.sh scratch/gen_deploy_120_s224.sh | grep -v "^??"; } )
grep -qE "ALERT|STOP" "$FAMGUARD_LOG" 2>/dev/null && { echo "$(date '+%F %T') fill_s232s3 REFUSED: $FAMGUARD_LOG holds an ALERT / STOP (rotate it after Audit's ruling)"; exit 2; }
[ -n "$dirty" ] && { echo "$(date '+%F %T') fill_s232s3 REFUSED: uncommitted module / frozen-ref / script edits (deploys ship the working tree): $dirty"; exit 2; }
{ grep -q "fh_lambda_fixed" boxes/vbank.py && grep -q "def part_c1hold" boxes/vbank_diag.py; } || { echo "$(date '+%F %T') fill_s232s3 REFUSED: the §230 / §232 builds are not committed"; exit 2; }
for f_ in "$ADMLOG" "$BADLOG"; do [ -f "$f_" ] || echo "# §232 s3 ledger (fill_s232s3); last field = \$" > "$f_"; done   # 10-24: outside() needs both (b343 started s230fam with an empty cap)
log "start: QS $QS; family \$$CAPF; SETUP_H $SETUP_H; EVAL_H $EVAL_H; vbank sha $(sha256sum boxes/vbank.py | cut -c1-12); vbank_diag sha $(sha256sum boxes/vbank_diag.py | cut -c1-12); vbank_eval sha $(sha256sum boxes/vbank_eval.py | cut -c1-12)"

FAMC=""; for c_ in $QS; do IFS=: read -r n_ r_ a_ g_ s_ f_ sd_ <<<"$c_"; e_=$(python -c "print(round($EST_FAR + $EST_RE * ($(nevals $n_) - 1), 2))"); FAMC="$FAMC $n_:$r_:$e_:$s_"; done   # Audit 10-22: unstarted cells project at their ESTIMATE (not the guard ceiling)
for cell in $QS; do
  IFS=: read -r n run CAPC GSTOP STEPS FAR SEED <<<"$cell"; tries=0; export EXTRA_120="$(cell_extra "$n")"; EQ=0; [ "$FAR" = 1 ] && EQ=4800;   # v2: 2 x 1,200 det-wrapped fp32 steps, projected at 2x the cell s/step
  NE=$(nevals "$n")
  grep -q "^$n:$run " "$ADMLOG" 2>/dev/null && { log "b$n: $run already admitted - next"; continue; }
  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }
  for f_ in $EXTRA_120; do [ -s "$f_" ] || { log "b$n: shipped file $f_ missing - not starting"; exit 2; }; done
  while true; do
    [ -f scratch/fill_s232s3.stop ] && { log "stopped (scratch/fill_s232s3.stop)"; exit 0; }
    famcheck "the loop top (b$n)"
    bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
    bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0)
    if python -c "import sys; sys.exit(0 if $bt > $BADSPENDMAX else 1)"; then log "CAP: §232 s3 BAD-host spend \$$bt > \$$BADSPENDMAX - held; tell Audit"; exit 3; fi
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
    pr=""; VMISS=0; VAN=""; for i in $(seq 1 40); do pr=$(sq "$h" "$p" "grep -h 'HOSTPROBE' /workspace/Latent/logs/queue_gen_b$n.out | tail -n 1"); echo "$pr" | grep -q "s/step" && break; pr=""
      vanished "$iid" && { VAN=1; break; }; sleep 60; done
    if [ -n "$VAN" ]; then log "b$n: VANISHED instance $iid ($ip) during HOSTPROBE (absent from show instances on 2 consecutive polls) - BAD, excluded"; echo "oid:$o # §232s3 vanished $(date '+%F %T')" >> scratch/vast_exclude_ips.txt; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "vanished"; continue; fi
    if [ -z "$pr" ] || echo "$pr" | grep -q HOSTPROBE_SLOW; then log "b$n: HOSTPROBE ${pr:-none after 40 min} on $h:$p - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "hostprobe"; continue; fi
    log "b$n: $pr on $h:$p"
    mc=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_mem/mem_check.json" | tr -d '\n ')
    gt=""; VMISS=0; VAN=""; for i in $(seq 1 45); do gt=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_gtime/timing.json"); [ -n "$gt" ] && break
      vanished "$iid" && { VAN=1; break; }; sleep 60; done
    if [ -n "$VAN" ]; then log "b$n: VANISHED instance $iid ($ip) during G-TIME (absent from show instances on 2 consecutive polls) - BAD, excluded"; echo "oid:$o # §232s3 vanished $(date '+%F %T')" >> scratch/vast_exclude_ips.txt; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "vanished"; continue; fi
    sps=$(echo "$gt" | python -c "import json,sys; print(json.load(sys.stdin)['sec_per_step'])" 2>/dev/null)
    am=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_gtime/auto_micro.json" | tr -d '\n ')
    if [ -z "$sps" ]; then log "b$n: no G-TIME timing.json after 45 min on $h:$p ($(sq "$h" "$p" "tail -n 3 /workspace/Latent/logs/gen_${run}_gtime.log" | tr '\n' ' ' | cut -c1-300)) - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "no-gtime"; continue; fi
    proj=$(python -c "print(round(($sps * ($STEPS + $EQ) / 3600 + $SETUP_H + $EXTRA_H) * $dph, 2))"); fs=$(fam)   # EXTRA_H: the FT cells + evals after the base
    log "b$n: K7 mem_check $mc; G-TIME $sps s/step ($am) -> cell projection \$$proj ($STEPS + EQ $EQ base steps + $EXTRA_H h post-base lines; cap \$$CAPC); family so far \$$fs (admitted + BAD) of \$$CAPF"
    if python -c "import sys; sys.exit(0 if $proj <= $CAPC and $fs + $proj <= $CAPF else 1)"; then
      echo "$n:$run $h:$p $iid \$$dph/h sps $sps $proj" >> "$ADMLOG"; log "b$n: ADMIT $run on $h:$p (instance $iid): \$$proj"
      evh=$(python -c "print(round(max(0.25, 2000 * $sps / 3600), 2))")       # the mirror every ~2k steps
      nohup python scratch/ckpt_mirror.py "$n:$run" --every_h "$evh" --optional log.jsonl,args.json,draws_log.json > "logs/ckpt_mirror_b$n.out" 2>&1 &
      cb=$(awk -v n="b$n" '!/^#/ && $3 == n {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0); gl=$(python -c "print(round($GSTOP - $cb, 4))")
      GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000,16000,24000,32000,40000,47000 nohup python scratch/cost_guard.py "s232s3_b$n" "$gl" "$STEPS" "$n:$run:$proj:$STEPS" > "logs/s232s3_guard_b$n.out" 2>&1 &
      log "b$n: ckpt mirror (every $evh h) + cell guard (\$$GSTOP - rejected-host \$$cb = \$$gl, GUARD_STOP=1) started"
      if ! ps -ef | grep "cost_guard.py s232s3fam " | grep -v grep > /dev/null; then   # the Stage-1 FAMILY guard, GUARD_STOP=1 (BAD hosts included)
        fcap=$(python -c "print(round($CAPF - $(outside), 4))")   # Audit 10-22: CAPF minus the realised spend outside the guarded cells
        GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000,16000,24000,32000,40000,47000 nohup python scratch/cost_guard.py s232s3fam "$fcap" 48000 $FAMC > "$FAMGUARD_LOG" 2>&1 &
        log "guard s232s3fam (\$$fcap = \$$CAPF - outside \$$(outside), GUARD_STOP=1;$FAMC) started"
      fi
      NL=$(grep -v "^#" scratch/gen_q_b$n.txt.norequeue | grep -c .); wc_=$(python -c "print(round($CAPF - $(outside), 4))")   # the whole-host wall-clock guard
      GUARD_STOP=1 GUARD_WEIGHTS="$WEIGHTS" nohup python scratch/s232_wall_guard.py "s232s3_wall_b$n" "$wc_" "$iid" "$h" "$p" "$dph" "$t0" "gen_b$n" "$NL" "$proj" > "$WALL_LOG" 2>&1 &
      log "b$n: wall-clock guard over the host (\$$wc_ = \$$CAPF - outside, destroy at cap, $NL lines, weights $WEIGHTS) started"
      STOPF=scratch/fill_s232s3.stop ALERTS=logs/s232s3_alerts.log PALOG=logs/s232s3_post_admit.log nohup bash scratch/s224_post_admit.sh "$n" "$run" "$h" "$p" "$iid" "$FAR" "$SEED" > "logs/s232s3_post_b$n.out" 2>&1 &
      log "b$n: post-admit checks started (G-FH-EQ: $FAR; args vs scratch/s224_args_frozen_$run.json)"
      sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; famcheck "after ADMIT (b$n)"; break
    fi
    log "b$n: projection \$$proj (cell cap \$$CAPC; family \$$fs + \$$proj vs \$$CAPF) NOT admitted on $h:$p (measured refusal $((tries + 1)) / $HOSTMAX) - destroying, next host"
    tries=$((tries + 1)); refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "projection"
  done
done
log "§232 s3 filler done: $(cat $ADMLOG 2>/dev/null | tr '\n' ';')"
