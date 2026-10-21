#!/usr/bin/env bash
# §222 PROBE-S cloud filler (Note 222-1: CERT's 4 groups + A1's §222 probe parts on ONE GPU class, RTX 3090, fp32; one group per host; Audit-approved
# under the user's $30 rule). Pattern of scratch/s219_fill.sh (G-D'd), adapted:
#   - cells QS "n:run:share" (Note 222-1: the cell's share of the $20 family cap; runs in QS order, each on its own RTX 3090); queue scratch/gen_q_b<n>.txt: line 1 G-TIME (--time_steps 60 of
#     the host's group -> <run>_gtime/timing.json), [A1 g0 only: line 2 the K2 guard + DIAG-ALIGN -> s222_a1_pre], then the group (--ckpt_dir = --out =
#     boxes/runs_fsq/<run>, --resume, full-state checkpoint every 2k), [A1 g0 only, last: the ACC cross-card check (Note 222-1 item 6) -> s222_acc3090];
#   - deploy via scratch/gen_deploy_120_s222.sh (arms auto_drain_gen8_s222.sh -> drain_gen_partial_s222.sh, §222 done markers); no HOSTPROBE line
#     (G-TIME on the real group shape is the speed test); hostcheck (VRAM pattern) always;
#   - G-TIME gate: timing.json must read the cell's group index, steps_timed 60, an RTX 3090, a positive sec_per_step and peak_gib <= 21.0;
#     projection = (sec_per_step x PSTEPS (20,000 = the worst case: every probe extends) / 3600 + SETUP_H (+ EXTRA_H for A1 g0's pre / ACC lines)) x $/h;
#     ADMIT iff proj <= the cell's share AND family (admitted + refused / BAD) + proj <= CAPF (Note 222-1); the cell guard = 1.3 x proj (Note 222-1);
#   - after admission: ckpt_mirror.py (every ~2k steps; ckpt_last.pt + ckpt_fingerprint.json + log.jsonl, sha-checked), the cell guard and the family
#     guard s222fam with scripts/cost_guard.py (f2e5253c: marks to the final step, recent-rate projection), GUARD_STOP=1;
#   - A1 cells (run s222_a1_*) wait until A1's own drain is logged (logs/auto_drain.log " gen_b316 ") and boxes/runs_fsq/lvb_sym_s1/model.pt exists;
#     the model is shipped with its args.json and its sha256 is logged here and re-read from the host's group-file meta after landing;
#   - start refusals: another filler running (one at a time), CAPF unset, m_rule.json / s222 scripts / boxes modules dirty or untracked, a queue missing.
#   - logs: logs/fill_s222.out (stdout), logs/s222_admitted.log, logs/s222_bad_cost.log; stop file scratch/fill_s222.stop.
# Unchanged from s219_fill: ONE filler; offer <= PMAX, never 580.x (offer_pick); HOSTMAX 3 measured refusals; BAD / non-start / missing-file hosts use
# no try but count toward the cell guard and the family; steps never reduced.
#   CAPF=20 QS="320:s222_cert_untrained_g0:2.857 321:s222_cert_untrained_g1:2.857 ..." nohup bash scratch/s222_fill.sh >> logs/fill_s222.out 2>&1 &
# Tests: S222_FILL_LIB=1 source scratch/s222_fill.sh (defines the helpers, returns before any check / deploy).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
PMAX=${PMAX:-0.25}; CAPF=${CAPF:-}; SETUP_H=${SETUP_H:-0.35}; EXTRA_H=${EXTRA_H:-0.6}; PSTEPS=20000; BADN=${BAD0:-0}; BADMAX=${BADMAX:-4}; HOSTMAX=${HOSTMAX:-3}
BADSPENDMAX=${BADSPENDMAX:-0.60}; BADLOG=${BADLOG:-logs/s222_bad_cost.log}; ADMLOG=${ADMLOG:-logs/s222_admitted.log}; MEMMAX=21.0; GT_STEPS=60
QS="${QS:-}"; A1=boxes/runs_fsq/lvb_sym_s1; REF=scratch/gate_s218/g_l1c_model; MR=scratch/gate_s222/m_rule.json
cell_group() { echo "$1" | sed -n 's/.*_g\([0-9]\+\)$/\1/p'; }     # run name ..._g<i> -> i
cell_extra() {   # per-cell files beyond auto_ship's queue tokens (auto_ship ships the queue's scratch module, its boxes closure and file tokens)
  case "$1" in
    s222_cert_untrained_*) echo "$MR" ;;
    s222_cert_ref_*) echo "$MR $REF/model.pt $REF/args.json" ;;
    s222_a1_*) echo "$MR scratch/s222_cert.py $A1/model.pt $A1/args.json" ;;
  esac
}
log() { echo "$(date +%T) fill_s222 $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
sq() { timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$2" "root@$1" "$3" 2>/dev/null; }
fam() { python -c "s = 0.0
for f in ('$ADMLOG', '$BADLOG'):
    try: s += sum(float(l.split()[-1]) for l in open(f) if l.strip() and not l.startswith('#'))
    except OSError: pass
print(round(s, 4))"; }
cellbad() { awk -v n="b$1" '!/^#/ && $3 == n {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0; }
gtimegate() {   # $1 = expected group; stdin = timing.json -> "OK <sec_per_step> <peak>" | "REFUSE <why>"
  python -c "import json, sys
try: d = json.load(sys.stdin)
except Exception: print('REFUSE unreadable timing.json'); sys.exit()
s = d.get('sec_per_step'); pk = d.get('peak_gib'); bad = []
if not isinstance(s, (int, float)) or isinstance(s, bool) or not s > 0: bad.append('sec_per_step %r' % (s,))
if not isinstance(pk, (int, float)) or isinstance(pk, bool) or pk > $MEMMAX: bad.append('peak_gib %r (<= $MEMMAX)' % (pk,))
for k, v in (('group', int(sys.argv[1])), ('steps_timed', $GT_STEPS)):
    if d.get(k) != v: bad.append('%s %r != %r' % (k, d.get(k), v))
if '3090' not in str(d.get('gpu')): bad.append('gpu %r (Note 222-1: RTX 3090 only)' % (d.get('gpu'),))
print(('REFUSE ' + '; '.join(bad)) if bad else 'OK %s %.3f' % (s, pk))" "$1"; }
projection() { python -c "print(round(($1 * $PSTEPS / 3600 + $SETUP_H + $2) * $3, 4))"; }   # sps extra_h dph -> $
admit() { python -c "import sys; p, cc, fs, cf = map(float, sys.argv[1:5]); sys.exit(0 if p <= cc + 1e-9 and fs + p <= cf + 1e-9 else 1)" "$@"; }   # proj share fam capf
extra_dedup() { local as_=" $2 " ex_="" f_; for f_ in $1; do case "$as_" in *" $f_ "*) ;; *) ex_="$ex_ $f_";; esac; done; echo "${ex_# }"; }
refuse() {   # host port iid offer ip dph t0 why
  for dp in $(ps -ef | grep "auto_drain_gen8_s222.sh $1 $2 gen_b$n $3" | grep -v grep | awk '{print $2}'); do [ "$dp" != "$$" ] && kill "$dp"; done
  echo "$1:$2" >> logs/drained_hosts.txt; log "b$n: $(bash scratch/vdestroy.sh "$3" | tail -1)"
  local mins c; mins=$(( ($(date +%s) - $7) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $6, 4))"); BADN=$((BADN + 1))
  echo "$(date '+%F %T') b$n offer $4 $5 ${mins}min \$$6/h ($8) $c" >> "$BADLOG"; echo "$5 # §222 refused ($8) $(date '+%F %T')" >> scratch/vast_exclude_ips.txt
  [ -f "scratch/gen_q_b$n.txt.norequeue" ] && mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"
  [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
  return 0
}
[ "${S222_FILL_LIB:-0}" = 1 ] && return 0

# ---------------------------------------------------------------- start refusals (nothing is created before these pass)
others=$(ps -ef | awk '$6 ~ /bash(\.exe)?$/ && $7 ~ /s2[0-9]+_fill[a-z0-9_]*\.sh$/ && $7 !~ /s222_fill\.sh$/')
[ -n "$others" ] && { echo "$(date '+%F %T') fill_s222 REFUSED: another filler is running (one filler at a time): $others"; exit 2; }
[ "$(ps -ef | awk -v me=$$ '$6 ~ /bash(\.exe)?$/ && $7 ~ /s222_fill\.sh$/ && $2 != me && $3 != me' | wc -l)" -gt 0 ] && { echo "$(date '+%F %T') fill_s222 REFUSED: a second s222_fill.sh is running"; exit 2; }
[ -n "$CAPF" ] && [ -n "$QS" ] || { echo "$(date '+%F %T') fill_s222 REFUSED: CAPF (Note 222-1 family cap) and QS must be given"; exit 2; }
for f_ in "$MR" scratch/s222_cert.py scratch/s222_a1.py "$REF/args.json"; do
  git ls-files --error-unmatch "$f_" > /dev/null 2>&1 || { echo "$(date '+%F %T') fill_s222 REFUSED: $f_ is not committed"; exit 2; }
done
dirty=$( { git status --porcelain boxes/ | grep -v "^??" | grep -v "boxes/runs"; git status --porcelain "$MR" scratch/s222_cert.py scratch/s222_a1.py scratch/s222_done_marker.sh scratch/drain_gen_partial_s222.sh scratch/auto_drain_gen8_s222.sh scratch/gen_deploy_120_s222.sh; } )
[ -n "$dirty" ] && { echo "$(date '+%F %T') fill_s222 REFUSED: uncommitted module / script / m_rule edits (deploys ship the working tree): $dirty"; exit 2; }
log "start: QS $QS; family \$$CAPF; PSTEPS $PSTEPS; SETUP_H $SETUP_H; HOSTMAX $HOSTMAX; m_rule sha $(sha256sum $MR | cut -c1-12); s222_cert sha $(sha256sum scratch/s222_cert.py | cut -c1-12); s222_a1 sha $(sha256sum scratch/s222_a1.py | cut -c1-12)"

for cell in $QS; do
  IFS=: read -r n run CAPC <<<"$cell"; tries=0; gi=$(cell_group "$run"); xh=0; [ "$run" = s222_a1_g0 ] && xh=$EXTRA_H; [ "$n" = 320 ] && xh=${ENV_H:-0.5}   # b320 also runs the Note 222-3 (b) envelope line
  grep -q "^$n:$run " "$ADMLOG" 2>/dev/null && { log "b$n: $run already admitted - next"; continue; }
  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }
  case "$run" in s222_a1_*)
    if ! { grep -q " gen_b316 " logs/auto_drain.log 2>/dev/null && [ -s "$A1/model.pt" ]; }; then
      log "b$n: waiting for A1's drain (logs/auto_drain.log ' gen_b316 ' + $A1/model.pt)"
      until grep -q " gen_b316 " logs/auto_drain.log 2>/dev/null && [ -s "$A1/model.pt" ]; do [ -f scratch/fill_s222.stop ] && { log "stopped (scratch/fill_s222.stop)"; exit 0; }; sleep 120; done
    fi
    log "b$n: A1 model.pt sha256 $(sha256sum $A1/model.pt | cut -c1-64) (drain: $(grep ' gen_b316 ' logs/auto_drain.log | tail -1 | cut -c1-160))" ;;
  esac
  ce_="$(cell_extra "$run")"; for f_ in $ce_; do [ -s "$f_" ] || { log "b$n: cell file $f_ missing - not starting"; exit 2; }; done
  export EXTRA_120="$(extra_dedup "$ce_" "$(python scratch/auto_ship.py scratch/gen_q_b$n.txt)")"; log "b$n: EXTRA_120 (beyond auto_ship): $EXTRA_120"
  while true; do
    [ -f scratch/fill_s222.stop ] && { log "stopped (scratch/fill_s222.stop)"; exit 0; }
    bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
    bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0)
    if python -c "import sys; sys.exit(0 if $bt > $BADSPENDMAX else 1)"; then log "CAP: §222 refused / BAD-host spend \$$bt > \$$BADSPENDMAX - held; tell Audit"; exit 3; fi
    [ "$tries" -ge "$HOSTMAX" ] && { log "b$n: $run NOT RUN - no admitting host in $HOSTMAX measured tries; tell Audit / Theory"; continue 2; }
    r_=$(GPU_NAME=RTX_3090 python scratch/offer_pick.py - "$PMAX"); o=""; ip=""; dph=""; read -r o ip dph <<<"$r_"
    if [ -z "${o:-}" ]; then log "b$n: no RTX_3090 offer (non-580, <= \$$PMAX); retry in 10 min"; sleep 600; continue; fi
    log "b$n: deploying $run on offer $o ($ip, RTX_3090, \$$dph/h)"; t0=$(date +%s); bash scratch/gen_deploy_120_s222.sh "$o $n"
    line=$(grep "gen host b$n offer $o:" logs/gen_deploy.log | tail -1)
    if ! echo "$line" | grep -q STARTED; then
      mins=$(( ($(date +%s) - t0) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $dph, 4))"); BADN=$((BADN + 1))
      echo "$(date '+%F %T') b$n offer $o $ip ${mins}min \$$dph/h $c" >> "$BADLOG"; log "b$n: did not start (BAD HOST $BADN / $BADMAX; billed ~\$$c)"
      [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
      continue
    fi
    mv "scratch/gen_q_b$n.txt" "scratch/gen_q_b$n.txt.norequeue"
    read -r h p iid <<<"$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="INSTANCE") print $(i+2)" "$(i+3)" "$(i+1)}')"
    gt=""; for i in $(seq 1 45); do gt=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_gtime/timing.json"); [ -n "$gt" ] && break; sleep 60; done
    if [ -z "$gt" ]; then log "b$n: no G-TIME timing.json after 45 min on $h:$p ($(sq "$h" "$p" "tail -n 3 /workspace/Latent/logs/gen_${run}_gtime.log; tail -n 2 /workspace/Latent/logs/queue_gen_b$n.out" | tr '\n' ' ' | cut -c1-300)) - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "no-gtime"; continue; fi
    gg=$(echo "$gt" | gtimegate "$gi"); log "b$n: G-TIME $(echo "$gt" | python -c "import json, sys; d = json.load(sys.stdin); print({k: d.get(k) for k in ('group', 'keys', 'Kmax', 'sec_per_step', 'peak_gib', 'gpu')}, d.get('provenance', {}).get('nvidia_smi'), d.get('provenance', {}).get('torch'))" 2>/dev/null | cut -c1-500) -> $gg"
    if ! echo "$gg" | grep -q "^OK "; then
      case "$gg" in *"group "*|*"steps_timed"*) log "ALERT b$n: G-TIME shape check failed ($gg) - a queue fault: refusing and HOLDING for Theory / Audit"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "gtime-shape"; exit 3;; esac
      log "b$n: G-TIME REFUSED on $h:$p ($gg; measured refusal $((tries + 1)) / $HOSTMAX) - destroying, next host"; tries=$((tries + 1)); refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "gtime"; continue
    fi
    sps=$(echo "$gg" | awk '{print $2}'); proj=$(projection "$sps" "$xh" "$dph"); fs=$(fam); cb=$(cellbad "$n")
    log "b$n: G-TIME $sps s/step -> projection \$$proj ($PSTEPS steps worst case; share \$$CAPC; this cell's refused / BAD so far \$$cb); family \$$fs + proj vs \$$CAPF"
    if admit "$proj" "$CAPC" "$fs" "$CAPF"; then
      img=$(bash C:/Users/peter/.local/vastai/vast.sh show instance "$iid" --raw 2>/dev/null | python -c "import json, sys; d = json.load(sys.stdin); print(d.get('image_uuid'), d.get('driver_version'))" 2>/dev/null)
      echo "$n:$run $h:$p $iid \$$dph/h sps $sps $proj" >> "$ADMLOG"; log "b$n: ADMIT $run on $h:$p (instance $iid; image / driver $img): \$$proj"
      evh=$(python -c "print(round(max(0.25, 2000 * $sps / 3600), 2))")
      nohup python scratch/ckpt_mirror.py "$n:$run" --every_h "$evh" --companions ckpt_fingerprint.json,log.jsonl >> "logs/ckpt_mirror_b$n.out" 2>&1 &
      gl=$(python -c "print(round(1.3 * $proj, 4))")                         # Note 222-1: per-host guard = 1.3 x the admitted projection
      GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,6000,8000,10000,12000 nohup python scratch/cost_guard.py "s222_b$n" "$gl" "$PSTEPS" "$n:$run:$proj:$PSTEPS" > "logs/s222_guard_b$n.out" 2>&1 &
      log "b$n: ckpt mirror (every $evh h ~ 2k steps; ckpt_last.pt + ckpt_fingerprint.json + log.jsonl -> scratch/ckpt_mirror/$run/) + cell guard \$$gl = 1.3 x \$$proj (GUARD_STOP=1) started"
      famc=""; for c_ in $QS; do IFS=: read -r n_ r_ a_ <<<"$c_"; famc="$famc $n_:$r_:$a_:$PSTEPS"; done   # unstarted cells project at their share
      if ! ps -ef | grep "cost_guard.py s222fam " | grep -v grep > /dev/null; then
        bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0); famcap=$(python -c "print(round($CAPF - $bt, 4))")
        GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,6000,8000,10000,12000 nohup python scratch/cost_guard.py s222fam "$famcap" "$PSTEPS" $famc > logs/s222fam_guard.out 2>&1 &
        log "guard s222fam (\$$CAPF - refused / BAD \$$bt = \$$famcap, GUARD_STOP=1;$famc) started"
      fi
      sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; break
    fi
    log "b$n: projection \$$proj NOT admitted on $h:$p (measured refusal $((tries + 1)) / $HOSTMAX) - destroying, next host"
    tries=$((tries + 1)); refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "projection"
  done
done
log "§222 filler done: $(cat $ADMLOG 2>/dev/null | tr '\n' ';')"
