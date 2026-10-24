#!/usr/bin/env bash
# §232 HOLD-C1 Stage-A filler (HC_v1 §2, Note 232-1; Stage-A cap $2.5; ONE eval-only RTX 3090). Helpers (sq / inst_state / vanished / outside / fam /
# refuse) are scratch/s230_fill.sh's (G-D'd; sourced in library mode, its ledger / log paths overridden below). Differences from the trained-cell fillers:
# no K7 / G-TIME lines (no training): admission = HOSTPROBE ok and (SETUP_H + EVAL_H) x $/h <= the cell bound and fam + that <= CAPF; the guard is the
# wall-clock scratch/s232_wall_guard.py (cost_guard reads training steps), no ckpt mirror, no family guard (one cell; its guard <= CAPF); post-admit =
# scratch/s232_post_admit.sh (the shipped model.pt shas). Staged dirs boxes/runs_fsq/<model>_c1 (model.pt + args.json copies; committed run dirs are
# never written); their outputs drain through the §224 chain (s224_done_marker *_c1: eval_vb + eval_d3hist + eval_c1hold, model.pt not fetched back).
#   CAPF=2.5 QS="344:s232_stageA:1.60:2.50:4.1" nohup bash scratch/s232_fill.sh >> logs/fill_s232.out 2>&1 &
cd "$(dirname "$0")/.."
S224_FILL_LIB=1 . scratch/s230_fill.sh
SETUP_H=0.6; BADLOG=logs/s232_bad_cost.log; ADMLOG=logs/s232_admitted.log; FAMGUARD_LOG=logs/s232_guard_b344.out; QS="${QS:-}"; CAPF=${CAPF:-}
MODELS="s213_d3t_s1 s215_ftctrl_s1 s230_fhftfix_s1 s216_d3t_s2 s224_ftctrl_s2 s224_fhft_s2"
STAGED=$(for m_ in $MODELS; do echo "boxes/runs_fsq/${m_}_c1"; done | xargs)
log() { echo "$(date +%T) fill_s232 $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
famcheck() { if grep -qE "ALERT|STOP" "$FAMGUARD_LOG" 2>/dev/null; then log "HOLD: s232 guard fired ($(grep -E "ALERT|STOP" "$FAMGUARD_LOG" | tail -1 | cut -c1-160)) - filler exits at $1; tell Audit"; exit 3; fi; }

others=$(ps -ef | awk '$6 ~ /bash(\.exe)?$/ && $7 ~ /s2[0-9]+_fill[a-z0-9_]*\.sh$/ && $7 !~ /s232_fill\.sh$/')
[ -n "$others" ] && { echo "$(date '+%F %T') fill_s232 REFUSED: another filler is running (one filler at a time): $others"; exit 2; }
[ -n "$CAPF" ] && [ -n "$QS" ] || { echo "$(date '+%F %T') fill_s232 REFUSED: CAPF and QS must be given"; exit 2; }
git ls-files --error-unmatch scratch/gen_q_b344.txt > /dev/null 2>&1 || { echo "$(date '+%F %T') fill_s232 REFUSED: scratch/gen_q_b344.txt is not committed"; exit 2; }
dirty=$( { git status --porcelain boxes/ | grep -v "^??" | grep -v "boxes/runs"; git status --porcelain scratch/s232_*.sh scratch/s232_*.py scratch/gen_q_b344.txt scratch/s230_fill.sh scratch/drain_gen_partial_s224.sh scratch/auto_drain_gen8_s224.sh scratch/gen_deploy_120_s224.sh scratch/s224_done_marker.sh | grep -v "^??"; } )
[ -n "$dirty" ] && { echo "$(date '+%F %T') fill_s232 REFUSED: uncommitted module / script edits (deploys ship the working tree): $dirty"; exit 2; }
grep -q "def part_c1hold" boxes/vbank_diag.py || { echo "$(date '+%F %T') fill_s232 REFUSED: boxes/vbank_diag.py has no part_c1hold (the §232 build is not committed)"; exit 2; }
for d_ in $STAGED; do for f_ in model.pt args.json; do [ -s "$d_/$f_" ] || { echo "$(date '+%F %T') fill_s232 REFUSED: staged $d_/$f_ missing"; exit 2; }; done; done
for f_ in "$ADMLOG" "$BADLOG"; do [ -f "$f_" ] || echo "# §232 ledger (fill_s232); last field = \$" > "$f_"; done
log "start: QS $QS; family \$$CAPF; SETUP_H $SETUP_H; vbank_diag sha $(sha256sum boxes/vbank_diag.py | cut -c1-12); vbank_eval sha $(sha256sum boxes/vbank_eval.py | cut -c1-12); staged $STAGED"

for cell in $QS; do
  IFS=: read -r n run CAPC GSTOP EVALH <<<"$cell"; tries=0; export EXTRA_120="$(for d_ in $STAGED; do echo "$d_/model.pt $d_/args.json"; done | xargs)"
  NL=$(grep -v "^#" scratch/gen_q_b$n.txt | grep -c .)
  grep -q "^$n:$run " "$ADMLOG" 2>/dev/null && { log "b$n: $run already admitted - next"; continue; }
  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing - not starting"; exit 2; }
  while true; do
    [ -f scratch/fill_s232.stop ] && { log "stopped (scratch/fill_s232.stop)"; exit 0; }
    famcheck "the loop top (b$n)"
    bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
    bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0)
    if python -c "import sys; sys.exit(0 if $bt > $BADSPENDMAX else 1)"; then log "CAP: §232 BAD-host spend \$$bt > \$$BADSPENDMAX - held; tell Audit"; exit 3; fi
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
    if [ -n "$VAN" ]; then log "b$n: VANISHED instance $iid ($ip) during HOSTPROBE - BAD, excluded"; echo "oid:$o # §232 vanished $(date '+%F %T')" >> scratch/vast_exclude_ips.txt; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "vanished"; continue; fi
    if [ -z "$pr" ] || echo "$pr" | grep -q HOSTPROBE_SLOW; then log "b$n: HOSTPROBE ${pr:-none after 40 min} on $h:$p - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "hostprobe"; continue; fi
    log "b$n: $pr on $h:$p"
    proj=$(python -c "print(round(($SETUP_H + $EVALH) * $dph, 2))"); fs=$(fam)
    log "b$n: eval-only projection \$$proj (($SETUP_H + $EVALH h) x \$$dph/h; cap \$$CAPC); family so far \$$fs (admitted + BAD) of \$$CAPF"
    if python -c "import sys; sys.exit(0 if $proj <= $CAPC and $fs + $proj <= $CAPF else 1)"; then
      echo "$n:$run $h:$p $iid \$$dph/h evalonly $proj" >> "$ADMLOG"; log "b$n: ADMIT $run on $h:$p (instance $iid): \$$proj"
      cb=$(awk -v n="b$n" '!/^#/ && $3 == n {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0); gl=$(python -c "print(round($GSTOP - $cb, 4))")
      GUARD_STOP=1 nohup python scratch/s232_wall_guard.py "s232_b$n" "$gl" "$iid" "$h" "$p" "$dph" "$t0" "gen_b$n" "$NL" "$proj" > "$FAMGUARD_LOG" 2>&1 &
      log "b$n: wall-clock guard (\$$GSTOP - rejected-host \$$cb = \$$gl, GUARD_STOP=1, $NL queue lines) started"
      nohup bash scratch/s232_post_admit.sh "$n" "$h" "$p" "$iid" $STAGED > "logs/s232_post_b$n.out" 2>&1 &
      log "b$n: post-admit checks started (shipped model.pt shas, $(echo $STAGED | wc -w) staged dirs)"
      sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; famcheck "after ADMIT (b$n)"; break
    fi
    log "b$n: projection \$$proj (cell cap \$$CAPC; family \$$fs + \$$proj vs \$$CAPF) NOT admitted on $h:$p (refusal $((tries + 1)) / $HOSTMAX) - destroying, next host"
    tries=$((tries + 1)); refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "projection"
  done
done
log "§232 filler done: $(grep -v '^#' $ADMLOG 2>/dev/null | tr '\n' ';')"
