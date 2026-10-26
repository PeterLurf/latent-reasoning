#!/usr/bin/env bash
# §234 ANSWER-WRITE P-2 filler (AW_v1 §3 / §5; Note 234-2 user decision: run P-2; admit <= $1.85, guard $2.31; one non-580 RTX 3090). A mechanical copy of
# scratch/s233_fill.sh (G-D'd; sed: s233 -> s234 ledgers / stop file / family guard s234fam / cell guard s234_b<n> / post-admit / frozen-args paths, the build check
# on boxes/inject_aw.py, the module shas logged) + one addition: a whole-host wall-clock guard (scratch/s232_wall_guard.py, GUARD_STOP=1, cap = the cell guard) so
# the CD-AN cell and the evals after the LAT-AW training line are capped too. Queue gen_q_b347: LAT-AW aw_plaw2_s1 (F2 lines + the cell) then CD-AN aw_pcd2_s1, SAME host.
#   CAPF=29.81 QS="347:aw_plaw2_s1:1.85:2.31:12000:1.85" nohup bash scratch/s234_fill.sh >> logs/fill_s234.out 2>&1 &
cd "$(dirname "$0")/.."
WALLW=${WALLW:-0.1,0.1,6.0,0.5}; PMAX=${PMAX:-0.25}; CAPF=${CAPF:-}; SETUP_H=0.4; EVAL_H=${EVAL_H:-1.0}; BADN=${BAD0:-0}; BADMAX=4; HOSTMAX=${HOSTMAX:-3}; BADLOG=logs/s234_bad_cost.log; ADMLOG=logs/s234_admitted.log
BADSPENDMAX=${BADSPENDMAX:-1.00}; VAST=${VAST:-C:/Users/peter/.local/vastai/vast.sh}
FAMGUARD_LOG=${FAMGUARD_LOG:-logs/s234fam_guard.out}
QS="${QS:-}"
log() { echo "$(date +%T) fill_s234 $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
sq() { timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$2" "root@$1" "$3" 2>/dev/null; }
inst_state() {
  timeout 60 bash "$VAST" show instances --raw 2>/dev/null | python -c "import json, sys
try: ids = {int(x['id']) for x in json.load(sys.stdin)}
except Exception: sys.exit(2)
sys.exit(0 if int(sys.argv[1]) in ids else 1)" "$1"; }
vanished() {
  inst_state "$1"; local r=$?
  if [ $r = 1 ]; then VMISS=$((VMISS + 1)); elif [ $r = 0 ]; then VMISS=0; fi
  [ "$VMISS" -ge 2 ]; }
famcheck() {
  if grep -qE "ALERT|STOP" "$FAMGUARD_LOG" 2>/dev/null; then log "HOLD: s234fam fired ($(grep -E "ALERT|STOP" "$FAMGUARD_LOG" | tail -1 | cut -c1-160)) - filler exits at $1; tell Audit"; exit 3; fi
}
outside() { python -c "s = 0.0
for f, pre in (('$ADMLOG', ('realised:', 'gatehold:')), ('$BADLOG', None)):
    try:
        for l in open(f):
            if l.strip() and not l.startswith('#') and (pre is None or l.startswith(pre)): s += float(l.split()[-1])
    except OSError: pass
print(round(s, 4))"; }
fam() { python -c "import sys; s = 0.0
for f in ('$ADMLOG', '$BADLOG'):
    try: s += sum(float(l.split()[-1]) for l in open(f) if l.strip() and not l.startswith('#'))
    except OSError: pass
print(round(s, 4))"; }
marks() { python -c "s = $1; m = [x for x in (500, 1000, 2000, 4000, 8000, 16000, 24000, 32000, 40000) if x < s - 100]; print(','.join(str(x) for x in m + [s - 100]))"; }
refuse() {
  for dp in $(ps -ef | grep "auto_drain_gen8_s224.sh $1 $2 gen_b$n $3" | grep -v grep | awk '{print $2}'); do [ "$dp" != "$$" ] && kill "$dp"; done
  echo "$1:$2" >> logs/drained_hosts.txt; log "b$n: $(bash scratch/vdestroy.sh "$3" | tail -1)"
  local mins c; mins=$(( ($(date +%s) - $7) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $6, 4))"); BADN=$((BADN + 1))
  echo "$(date '+%F %T') b$n offer $4 $5 ${mins}min \$$6/h ($8) $c" >> "$BADLOG"; echo "$5 # §234 refused ($8) $(date '+%F %T')" >> scratch/vast_exclude_ips.txt
  mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"
  [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
}
[ "${S228_FILL_LIB:-0}" = 1 ] && return 0

# ---------------------------------------------------------------- start refusals (nothing is created before these pass)
others=$(ps -ef | awk '$6 ~ /bash(\.exe)?$/ && $7 ~ /s2[0-9]+_fill[a-z0-9_]*\.sh$/ && $7 !~ /s234_fill\.sh$/')
[ -n "$others" ] && { echo "$(date '+%F %T') fill_s234 REFUSED: another filler is running (one filler at a time): $others"; exit 2; }
[ "$(ps -ef | awk -v me=$$ '$6 ~ /bash(\.exe)?$/ && $7 ~ /s234_fill\.sh$/ && $2 != me && $3 != me' | wc -l)" -gt 0 ] && { echo "$(date '+%F %T') fill_s234 REFUSED: a second s234_fill.sh is running"; exit 2; }
[ -n "$CAPF" ] && [ -n "$QS" ] || { echo "$(date '+%F %T') fill_s234 REFUSED: CAPF and QS must be given"; exit 2; }
for cell in $QS; do IFS=: read -r n_ r_ a_ g_ s_ e_ <<<"$cell"
  [ -n "$e_" ] || { echo "$(date '+%F %T') fill_s234 REFUSED: QS cell $cell has no family estimate (6th field)"; exit 2; }
  for x_ in scratch/s234_args_frozen_$r_.json scratch/gen_q_b$n_.txt; do git ls-files --error-unmatch "$x_" > /dev/null 2>&1 || { echo "$(date '+%F %T') fill_s234 REFUSED: $x_ is not committed"; exit 2; }; done; done
dirty=$( { git status --porcelain boxes/ | grep -v "^??" | grep -v "boxes/runs"; git status --porcelain scratch/s234_*.json scratch/s234_*.sh scratch/s228_args_check.py scratch/gen_q_b34*.txt scratch/drain_gen_partial_s224.sh scratch/auto_drain_gen8_s224.sh scratch/gen_deploy_120_s224.sh scratch/s224_done_marker.sh | grep -v "^??"; } )
grep -qE "ALERT|STOP" "$FAMGUARD_LOG" 2>/dev/null && { echo "$(date '+%F %T') fill_s234 REFUSED: $FAMGUARD_LOG holds an ALERT / STOP (rotate it after Audit's ruling)"; exit 2; }
[ -n "$dirty" ] && { echo "$(date '+%F %T') fill_s234 REFUSED: uncommitted module / frozen-ref / script edits (deploys ship the working tree): $dirty"; exit 2; }
grep -q "def tok_aw" boxes/inject_aw.py || { echo "$(date "+%F %T") fill_s234 REFUSED: boxes/inject_aw.py has no tok_aw (the §234 build is not committed)"; exit 2; }
for f_ in "$ADMLOG" "$BADLOG"; do [ -f "$f_" ] || echo "# §234 ledger (fill_s234); last field = \$" > "$f_"; done
log "start: QS $QS; family \$$CAPF; SETUP_H $SETUP_H; EVAL_H $EVAL_H; inject_aw sha $(sha256sum boxes/inject_aw.py | cut -c1-12); inject_aw_eval sha $(sha256sum boxes/inject_aw_eval.py | cut -c1-12)"

FAMC=""; for c_ in $QS; do IFS=: read -r n_ r_ a_ g_ s_ e_ <<<"$c_"; FAMC="$FAMC $n_:$r_:$e_:$s_"; done   # unstarted cells project at their ESTIMATE
for cell in $QS; do
  IFS=: read -r n run CAPC GSTOP STEPS EST <<<"$cell"; tries=0; export EXTRA_120=""; NE=1; GM=$(marks "$STEPS")
  grep -q "^$n:$run " "$ADMLOG" 2>/dev/null && { log "b$n: $run already admitted - next"; continue; }
  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }
  while true; do
    [ -f scratch/fill_s234.stop ] && { log "stopped (scratch/fill_s234.stop)"; exit 0; }
    famcheck "the loop top (b$n)"
    bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
    bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0)
    if python -c "import sys; sys.exit(0 if $bt > $BADSPENDMAX else 1)"; then log "CAP: §234 BAD-host spend \$$bt > \$$BADSPENDMAX - held; tell Audit"; exit 3; fi
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
    if [ -n "$VAN" ]; then log "b$n: VANISHED instance $iid ($ip) during HOSTPROBE (absent from show instances on 2 consecutive polls) - BAD, excluded"; echo "oid:$o # §234 vanished $(date '+%F %T')" >> scratch/vast_exclude_ips.txt; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "vanished"; continue; fi
    if [ -z "$pr" ] || echo "$pr" | grep -q HOSTPROBE_SLOW; then log "b$n: HOSTPROBE ${pr:-none after 40 min} on $h:$p - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "hostprobe"; continue; fi
    log "b$n: $pr on $h:$p"
    mc=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_mem/mem_check.json" | tr -d '\n ')
    gt=""; VMISS=0; VAN=""; for i in $(seq 1 45); do gt=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_gtime/timing.json"); [ -n "$gt" ] && break
      vanished "$iid" && { VAN=1; break; }; sleep 60; done
    if [ -n "$VAN" ]; then log "b$n: VANISHED instance $iid ($ip) during G-TIME (absent from show instances on 2 consecutive polls) - BAD, excluded"; echo "oid:$o # §234 vanished $(date '+%F %T')" >> scratch/vast_exclude_ips.txt; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "vanished"; continue; fi
    sps=$(echo "$gt" | python -c "import json,sys; print(json.load(sys.stdin)['sec_per_step'])" 2>/dev/null)
    am=$(echo "$gt" | python -c "import json,sys; print('micro', json.load(sys.stdin)['micro'])" 2>/dev/null)
    if [ -z "$sps" ]; then log "b$n: no G-TIME timing.json after 45 min on $h:$p ($(sq "$h" "$p" "tail -n 3 /workspace/Latent/logs/gen_${run}_gtime.log" | tr '\n' ' ' | cut -c1-300)) - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "no-gtime"; continue; fi
    proj=$(python -c "print(round(($sps * $STEPS / 3600 + $SETUP_H + $EVAL_H * $NE) * $dph, 2))"); fs=$(fam)
    log "b$n: K7 mem_check $mc; G-TIME $sps s/step ($am) -> cell projection \$$proj ($STEPS steps, $NE x $EVAL_H h eval / tail; cap \$$CAPC); family so far \$$fs (admitted + BAD) of \$$CAPF"
    if python -c "import sys; sys.exit(0 if $proj <= $CAPC and $fs + $proj <= $CAPF else 1)"; then
      echo "$n:$run $h:$p $iid \$$dph/h sps $sps $proj" >> "$ADMLOG"; log "b$n: ADMIT $run on $h:$p (instance $iid): \$$proj"
      evh=$(python -c "print(round(max(0.25, 1000 * $sps / 3600), 2))")       # the mirror every ~1k steps (ckpt_every 1000)
      nohup python scratch/ckpt_mirror.py "$n:$run" --every_h "$evh" --optional log.jsonl,args.json > "logs/ckpt_mirror_b$n.out" 2>&1 &
      cb=$(awk -v n="b$n" '!/^#/ && $3 == n {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0); gl=$(python -c "print(round($GSTOP - $cb, 4))")
      GUARD_STOP=1 GUARD_MARKS=$GM nohup python scratch/cost_guard.py "s234_b$n" "$gl" "$STEPS" "$n:$run:$proj:$STEPS" > "logs/s234_guard_b$n.out" 2>&1 &
      log "b$n: ckpt mirror (every $evh h) + cell guard (\$$GSTOP - rejected-host \$$cb = \$$gl, GUARD_STOP=1, marks $GM) started"
      nl=$(grep -vc '^#' "scratch/gen_q_b$n.txt.norequeue")
      GUARD_STOP=1 GUARD_WEIGHTS="$WALLW" nohup python scratch/s232_wall_guard.py "s234_wall_b$n" "$gl" "$iid" "$h" "$p" "$dph" "$t0" "gen_b$n" "$nl" "$EST" > "logs/s234_wall_b$n.out" 2>&1 &
      log "b$n: wall guard over the host (\$$gl from the create, $nl lines, weights $WALLW, GUARD_STOP=1) started"
      if ! ps -ef | grep "cost_guard.py s234fam " | grep -v grep > /dev/null; then
        fcap=$(python -c "print(round($CAPF - $(outside), 4))")
        GUARD_STOP=1 GUARD_MARKS=$GM nohup python scratch/cost_guard.py s234fam "$fcap" "$STEPS" $FAMC > "$FAMGUARD_LOG" 2>&1 &
        log "guard s234fam (\$$fcap = \$$CAPF - outside \$$(outside), GUARD_STOP=1;$FAMC) started"
      fi
      nohup bash scratch/s234_post_admit.sh "$n" "$run" "$h" "$p" "$iid" > "logs/s234_post_b$n.out" 2>&1 &
      log "b$n: post-admit checks started (args + shas vs scratch/s234_args_frozen_$run.json)"
      sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; famcheck "after ADMIT (b$n)"; break
    fi
    log "b$n: projection \$$proj (cell cap \$$CAPC; family \$$fs + \$$proj vs \$$CAPF) NOT admitted on $h:$p (measured refusal $((tries + 1)) / $HOSTMAX) - destroying, next host"
    tries=$((tries + 1)); refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "projection"
  done
done
log "§234 filler done: $(grep -v '^#' $ADMLOG 2>/dev/null | tr '\n' ';')"
