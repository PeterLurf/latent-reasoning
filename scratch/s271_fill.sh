#!/usr/bin/env bash
# s271 template v3 (per-cell steps in the ledger / family guard) + v2 (Audit 11-01: BAD spend family-only, guard >= admit + 0.05, GPU class, dph_total) on the s258 filler template (= s257_fill.sh + Audit 11-01 guard fixes: WALLW count refusal, family guard over every cell, guard-health check after ADMIT).
# §251 ANSWER-EXTEND filler (AE_v1 §6; guard $4.5 over the design; non-580 RTX 3090s). A mechanical copy of scratch/s250_fill.sh (sed: s250 -> s271; with the
# fam() once-per-cell fix e0a987e4). One filler run per host, chained by scratch/s271_chain.sh.
#   WALLW=0.1,0.1,4.0 EVAL_H=0.5 CAPF=4.5 QS="376:aex_s1:<admit>:<guard>:16000:<est>" nohup bash scratch/s271_fill.sh >> logs/fill_s271.out 2>&1 &
#   360 lh_ctrl_s1:0.60:1.40, 361 lh_ctrl_s2:0.60:1.40)
cd "$(dirname "$0")/.."
# v6: WALLW / EVAL_H / GATES_H / SPS_EST / GPU come from the queue's # CAPS header (scratch/caps_check.py, per cell)
WALLW=${WALLW:-}; PMAX=${PMAX:-0.21}; CAPF=${CAPF:-}; SETUP_H=0.4; EVAL_H=${EVAL_H:-}; BADN=${BAD0:-0}; BADMAX=4; HOSTMAX=${HOSTMAX:-3}; BADLOG=logs/s271_bad_cost.log; ADMLOG=logs/s271_admitted.log; GPU=${GPU:-}   # s271 template v2 (3)
BADSPENDMAX=${BADSPENDMAX:-1.00}; VAST=${VAST:-C:/Users/peter/.local/vastai/vast.sh}
FAMGUARD_LOG=${FAMGUARD_LOG:-logs/s271fam_guard.out}
QS="${QS:-}"
log() { echo "$(date +%T) fill_s271 $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
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
  if grep -qE "ALERT|STOP" "$FAMGUARD_LOG" 2>/dev/null; then log "HOLD: s271fam fired ($(grep -E "ALERT|STOP" "$FAMGUARD_LOG" | tail -1 | cut -c1-160)) - filler exits at $1; tell Audit"; exit 3; fi
}
outside() { python -c "s = 0.0
for f, pre in (('$ADMLOG', ('realised:', 'gatehold:')), ('$BADLOG', None)):
    try:
        for l in open(f):
            if l.strip() and not l.startswith('#') and (pre is None or l.startswith(pre)): s += float(l.split()[-1])
    except OSError: pass
print(round(s, 4))"; }
fam() { python -c "import re; s = 0.0; real = {}; proj = {}   # fam_v5 + Audit 10-29 (b369 refused twice): a landed cell counts ONCE - its realised line if present, else its admit projection
try:
    for l in open('$ADMLOG'):
        if not l.strip() or l.startswith('#'): continue
        v = float(l.split()[-1]); m = re.match(r'realised:b(\d+)(:a\d+)?', l); c = re.match(r'(\d+):', l)
        if m and m.group(2): s += v   # v5 (Audit 11-03): a VOIDed attempt's realised spend adds to the family (it never replaces the relaunch's projection)
        elif m: real[m.group(1)] = real.get(m.group(1), 0.0) + v
        elif c: proj[c.group(1)] = v
        else: s += v
except OSError: pass
s += sum(real.values()) + sum(v for k, v in proj.items() if k not in real)
try: s += sum(float(l.split()[-1]) for l in open('$BADLOG') if l.strip() and not l.startswith('#'))
except OSError: pass
print(round(s, 4))"; }
marks() { python -c "s = $1; m = [x for x in (500, 1000, 2000, 4000, 8000, 16000, 24000, 32000, 40000) if x < s - 100]; print(','.join(str(x) for x in m + [s - 100]))"; }
refuse() {
  for dp in $(ps -ef | grep "auto_drain_gen9_s224.sh $1 $2 gen_b$n $3" | grep -v grep | awk '{print $2}'); do [ "$dp" != "$$" ] && kill "$dp"; done
  echo "$1:$2" >> logs/drained_hosts.txt; log "b$n: $(bash scratch/vdestroy.sh "$3" | tail -1)"
  local mins c; mins=$(( ($(date +%s) - $7) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $6, 4))"); BADN=$((BADN + 1))
  echo "$(date '+%F %T') b$n offer $4 $5 ${mins}min \$$6/h ($8) $c" >> "$BADLOG"; echo "$5 # §251 refused ($8) $(date '+%F %T')" >> scratch/vast_exclude_ips.txt
  mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"
  [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
}
[ "${S228_FILL_LIB:-0}" = 1 ] && return 0

# ---------------------------------------------------------------- start refusals (nothing is created before these pass)
others=$(ps -ef | awk '$6 ~ /bash(\.exe)?$/ && $7 ~ /s2[0-9]+_fill[a-z0-9_]*\.sh$/ && $7 !~ /s271_fill\.sh$/')
[ -n "$others" ] && { echo "$(date '+%F %T') fill_s271 REFUSED: another filler is running (one filler at a time): $others"; exit 2; }
[ "$(ps -ef | awk -v me=$$ '$6 ~ /bash(\.exe)?$/ && $7 ~ /s271_fill\.sh$/ && $2 != me && $3 != me' | wc -l)" -gt 0 ] && { echo "$(date '+%F %T') fill_s271 REFUSED: a second s271_fill.sh is running"; exit 2; }
[ -n "$CAPF" ] && [ -n "$QS" ] || { echo "$(date '+%F %T') fill_s271 REFUSED: CAPF and QS must be given"; exit 2; }
for cell in $QS; do IFS=: read -r n_ r_ a_ g_ s_ e_ <<<"$cell"
  [ -n "$e_" ] || { echo "$(date '+%F %T') fill_s271 REFUSED: QS cell $cell has no family estimate (6th field)"; exit 2; }
  for x_ in scratch/s271_args_frozen_$r_.json scratch/gen_q_b$n_.txt; do git ls-files --error-unmatch "$x_" > /dev/null 2>&1 || { echo "$(date '+%F %T') fill_s271 REFUSED: $x_ is not committed"; exit 2; }; done; done
dirty=$( { git status --porcelain boxes/ | grep -v "^??" | grep -v "boxes/runs"; git status --porcelain scratch/s271_*.json scratch/s271_*.sh scratch/s228_args_check.py scratch/gen_q_b34*.txt scratch/drain_gen_partial_s224.sh scratch/auto_drain_gen9_s224.sh scratch/gen_deploy_120_s267.sh scratch/s224_done_marker.sh | grep -v "^??"; } )
grep -qE "ALERT|STOP" "$FAMGUARD_LOG" 2>/dev/null && { echo "$(date '+%F %T') fill_s271 REFUSED: $FAMGUARD_LOG holds an ALERT / STOP (rotate it after Audit's ruling)"; exit 2; }
[ -n "$dirty" ] && { echo "$(date '+%F %T') fill_s271 REFUSED: uncommitted module / frozen-ref / script edits (deploys ship the working tree): $dirty"; exit 2; }
grep -q "def tok_aw" boxes/inject_aw.py || { echo "$(date "+%F %T") fill_s271 REFUSED: boxes/inject_aw.py has no tok_aw (the §251 build is not committed)"; exit 2; }
for f_ in "$ADMLOG" "$BADLOG"; do [ -f "$f_" ] || echo "# §251 ledger (fill_s271); last field = \$" > "$f_"; done
log "start: QS $QS; family \$$CAPF; SETUP_H $SETUP_H; EVAL_H $EVAL_H; inject_aw sha $(sha256sum boxes/inject_aw.py | cut -c1-12); inject_aw_eval sha $(sha256sum boxes/inject_aw_eval.py | cut -c1-12)"

FAMC=""; for c_ in $QS; do IFS=: read -r n_ r_ a_ g_ s_ e_ <<<"$c_"; FAMC="$FAMC $n_:$r_:$e_:$s_"; done   # unstarted cells project at their ESTIMATE
for cell in $QS; do
  IFS=: read -r n run CAPC GSTOP STEPS EST <<<"$cell"; tries=0; export EXTRA_120=""; NE=1; GM=$(marks "$STEPS")
  python -c "import sys; sys.exit(0 if $GSTOP >= $CAPC + 0.05 - 1e-9 else 1)" || { log "REFUSED b$n: guard \$$GSTOP < admit cap \$$CAPC + 0.05 (Audit 11-01 v2 (2))"; exit 2; }
  grep -q "^$n:$run " "$ADMLOG" 2>/dev/null && { log "b$n: $run already admitted - next"; continue; }
  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }
  cc_=$(python scratch/caps_check.py "scratch/gen_q_b$n.txt" "$STEPS" "$CAPC" "$PMAX" "$SETUP_H") || { log "REFUSED b$n: $cc_ (template v6 caps)"; exit 2; }
  eval "$cc_"; log "b$n: caps from the queue header: $cc_"   # v6: WALLW / EVAL_H / GATES_H / SPS_EST / GPU / FETCH_H / WALLW_G
  nlq=$(grep -vc '^#' "scratch/gen_q_b$n.txt"); nw=$(echo "$WALLW" | tr ',' '\n' | grep -c .)   # Audit 11-01 (1): one wall-guard weight per queue line
  [ "$nlq" = "$nw" ] || { log "REFUSED b$n: WALLW has $nw weights, the queue $nlq lines"; exit 2; }
  sc_=$(python scratch/check_queue_ships.py "scratch/gen_q_b$n.txt") || { log "REFUSED b$n: $sc_ (template v4, Audit 11-02)"; exit 2; }   # template v4
  while true; do
    [ -f scratch/fill_s271.stop ] && { log "stopped (scratch/fill_s271.stop)"; exit 0; }
    famcheck "the loop top (b$n)"
    bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
    bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0)
    if python -c "import sys; sys.exit(0 if $bt > $BADSPENDMAX else 1)"; then log "CAP: §251 BAD-host spend \$$bt > \$$BADSPENDMAX - held; tell Audit"; exit 3; fi
    [ "$tries" -ge "$HOSTMAX" ] && { log "b$n: $run NOT RUN - no admitting host in $HOSTMAX tries; tell Audit / Theory"; continue 2; }
    r_=$(GPU_NAME=$GPU python scratch/offer_pick.py - "$PMAX"); o=""; ip=""; dph=""; read -r o ip dph <<<"$r_"
    if [ -z "${o:-}" ]; then log "b$n: no $GPU offer (non-580, <= \$$PMAX); retry in 10 min"; sleep 600; continue; fi
    log "b$n: deploying $run on offer $o ($ip, $GPU, \$$dph/h)"; t0=$(date +%s); bash scratch/gen_deploy_120_s267.sh "$o $n"
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
    if [ -n "$VAN" ]; then log "b$n: VANISHED instance $iid ($ip) during HOSTPROBE (absent from show instances on 2 consecutive polls) - BAD, excluded"; echo "oid:$o # §251 vanished $(date '+%F %T')" >> scratch/vast_exclude_ips.txt; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "vanished"; continue; fi
    if [ -z "$pr" ] || echo "$pr" | grep -q HOSTPROBE_SLOW; then log "b$n: HOSTPROBE ${pr:-none after 40 min} on $h:$p - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "hostprobe"; continue; fi
    log "b$n: $pr on $h:$p"
    mc=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_mem/mem_check.json" | tr -d '\n ')
    gt=""; VMISS=0; VAN=""; for i in $(seq 1 45); do gt=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_gtime/timing.json"); [ -n "$gt" ] && break
      vanished "$iid" && { VAN=1; break; }; sleep 60; done
    if [ -n "$VAN" ]; then log "b$n: VANISHED instance $iid ($ip) during G-TIME (absent from show instances on 2 consecutive polls) - BAD, excluded"; echo "oid:$o # §251 vanished $(date '+%F %T')" >> scratch/vast_exclude_ips.txt; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "vanished"; continue; fi
    sps=$(echo "$gt" | python -c "import json,sys; print(json.load(sys.stdin)['sec_per_step'])" 2>/dev/null)
    am=$(echo "$gt" | python -c "import json,sys; print('micro', json.load(sys.stdin)['micro'])" 2>/dev/null)
    if [ -z "$sps" ]; then log "b$n: no G-TIME timing.json after 45 min on $h:$p ($(sq "$h" "$p" "tail -n 3 /workspace/Latent/logs/gen_${run}_gtime.log" | tr '\n' ' ' | cut -c1-300)) - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "no-gtime"; continue; fi
    da=$(bash "$VAST" show instance "$iid" --raw 2>/dev/null | python -c "import json,sys; print(json.load(sys.stdin).get('dph_total') or 0)" 2>/dev/null || echo 0)   # v2 (4)
    dph=$(python -c "print(round(max($dph, float('${da:-0}' or 0)), 4))")
    proj=$(python -c "print(round(($sps * $STEPS / 3600 + $SETUP_H + ${GATES_H:-0} + ${FETCH_H:-0} + $EVAL_H * $NE) * $dph, 2))"); fs=$(fam)
    log "b$n: K7 mem_check $mc; G-TIME $sps s/step ($am) -> cell projection \$$proj ($STEPS steps, $NE x $EVAL_H h eval / tail; cap \$$CAPC); family so far \$$fs (admitted + BAD) of \$$CAPF"
    if python -c "import sys; sys.exit(0 if $proj <= $CAPC and $fs + $proj <= $CAPF else 1)"; then
      echo "$n:$run $h:$p $iid \$$dph/h sps $sps steps $STEPS $proj" >> "$ADMLOG"   # v3: the cell's steps recorded (the family guard reads them); log "b$n: ADMIT $run on $h:$p (instance $iid): \$$proj"
      evh=$(python -c "print(round(max(0.25, 1000 * $sps / 3600), 2))")       # the mirror every ~1k steps (ckpt_every 1000)
      nohup python scratch/ckpt_mirror.py "$n:$run" --every_h "$evh" --optional log.jsonl,args.json > "logs/ckpt_mirror_b$n.out" 2>&1 &
      gl=$GSTOP                                                             # v2 (1): BAD-host spend is a family cost, never deducted from a cell
      python -c "import sys; sys.exit(0 if $gl >= $proj + 0.05 - 1e-9 else 1)" || { log "HOLD b$n: guard \$$gl < projection \$$proj + 0.05 - guards NOT started; tell Audit"; exit 3; }
      GUARD_STOP=1 GUARD_MARKS=$GM nohup python scratch/cost_guard.py "s271_b$n" "$gl" "$STEPS" "$n:$run:$proj:$STEPS" > "logs/s271_guard_b$n.out" 2>&1 &
      log "b$n: ckpt mirror (every $evh h) + cell guard (\$$gl = the approved cell guard; BAD spend is family, v2 (1); GUARD_STOP=1, marks $GM) started"
      nl=$(grep -vc '^#' "scratch/gen_q_b$n.txt.norequeue")
      GUARD_STOP=1 GUARD_WEIGHTS="$WALLW_G" nohup python scratch/wall_guard_v6.py "s271_wall_b$n" "$gl" "$iid" "$h" "$p" "$dph" "$t0" "gen_b$n" "$nl" "$EST" > "logs/s271_wall_b$n.out" 2>&1 &
      log "b$n: wall guard over the host (\$$gl from the create, $nl lines, weights $WALLW, GUARD_STOP=1) started"
      # Audit 11-01 (2): the family guard over EVERY cell (admitted: the ledger at their projection; unadmitted QS cells: their estimate), restarted at each ADMIT
      for fp in $(ps -ef | awk '$7 == "scratch/cost_guard.py" && $8 == "s271fam" {print $2}'); do kill "$fp"; done
      [ -f "$FAMGUARD_LOG" ] && mv "$FAMGUARD_LOG" "${FAMGUARD_LOG%.out}.part$(date +%H%M%S).out"
      ALLC=$(python scratch/ledger_steps.py "$ADMLOG") || { log "ALERT b$n: ledger_steps could not resolve a cell's steps - HOLD (family guard not restarted); tell Audit"; exit 3; }   # v3 (Audit)
      for c_ in $FAMC; do grep -q "^${c_%%:*}:" "$ADMLOG" || ALLC="$ALLC $c_"; done
      fcap=$(python -c "print(round($CAPF - $(outside), 4))")
      GUARD_STOP=1 GUARD_MARKS=$GM nohup python scratch/cost_guard.py s271fam "$fcap" "$STEPS" $ALLC > "$FAMGUARD_LOG" 2>&1 &
      log "guard s271fam (\$$fcap = \$$CAPF - outside \$$(outside), GUARD_STOP=1; $ALLC) (re)started"
      nohup bash scratch/s271_post_admit.sh "$n" "$run" "$h" "$p" "$iid" > "logs/s271_post_b$n.out" 2>&1 &
      log "b$n: post-admit checks started (args + shas vs scratch/s271_args_frozen_$run.json)"
      sleep 120                                                             # Audit 11-01 (3): every guard alive 120 s after ADMIT
      for gf in "logs/s271_guard_b$n.out" "logs/s271_wall_b$n.out" "$FAMGUARD_LOG"; do
        { grep -q "start:" "$gf" && ! grep -q "Traceback" "$gf"; } || { log "ALERT guard health: $gf has no start: line or a Traceback - HOLD; tell Audit"; exit 3; }
      done
      log "b$n: guard health OK (cell / wall / family started, no traceback)"
      sleep 300; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; famcheck "after ADMIT (b$n)"; break
    fi
    log "b$n: projection \$$proj (cell cap \$$CAPC; family \$$fs + \$$proj vs \$$CAPF) NOT admitted on $h:$p (measured refusal $((tries + 1)) / $HOSTMAX) - destroying, next host"
    tries=$((tries + 1)); refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "projection"
  done
done
log "§251 filler done: $(grep -v '^#' $ADMLOG 2>/dev/null | tr '\n' ';')"
