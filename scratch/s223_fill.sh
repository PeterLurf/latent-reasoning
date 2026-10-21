#!/usr/bin/env bash
# §223 PORT-STATE cloud filler (PS2_v1 §3 / §8, Addenda A / B; Notes 223-1..223-3; family cap $20 under GUARD_STOP, Audit-approved under the user's
# $30 rule). Pattern of scratch/s222_fill.sh (G-D'd) and scratch/s219_fill.sh (args check), adapted:
#   - cells QS "n:run:share:tag" in QS order, each on its own RTX 3090 (DIRECT first: its acc fixes n* and the band before any latent cell);
#     queue scratch/gen_q_b<n>.txt: line 1 G-TIME (boxes.codi_ps, the cell's OWN flags, --d_stages 8 --time_steps 60 --auto_micro: the deepest
#     shape's step time and memory on this host's transformers - Audit 10-21 note 1) -> <run>_gtime/{timing.json, mem_check.json, args.json};
#     line 2 the cell (--auto_micro --resume --args_ref scratch/s223_args_frozen_<tag>.json; full-state ckpt every 2k); line 3 boxes.ps_eval (on-host
#     3090 fp32, Note 223-2 Q6) into the same run dir;
#   - deploy via scratch/gen_deploy_120_s223.sh (arms auto_drain_gen8_s223.sh -> drain_gen_partial_s223.sh, §223 done markers);
#   - G-TIME gate: timing.json (sec_per_step > 0, steps_timed 60, the cell's label, an RTX 3090) + mem_check.json (the chosen micro's peak <= 21.0 GiB);
#     projection = (sec_per_step x 24,000 / 3600 + SETUP_H + EVAL_H(cell)) x $/h; ADMIT iff proj <= the cell's share AND family + proj <= CAPF;
#   - after admission: the host cell's args.json through scratch/s223_args_check.py vs the frozen ref - BAD = refuse the host + HOLD (rule b: an
#     ARGS-DIFF voids every reading, so the cell must not run); ckpt_mirror.py (every ~2k steps; ckpt_last.pt + ckpt_fingerprint.json + log.jsonl,
#     optional args.json / stages.jsonl); the cell guard (1.3 x the admitted projection) and the family guard s223fam ($CAPF; unstarted cells at
#     their share) with cost_guard.py (f2e5253c: marks to the final step, recent-rate projection), GUARD_STOP=1; the host's transformers / torch /
#     driver logged at ADMIT.
#   - start refusals: another filler running, CAPF / QS unset, a frozen ref / module untracked or dirty (boxes/, the s223 scripts), a queue missing.
#   - logs: logs/fill_s223.out (stdout), logs/s223_admitted.log, logs/s223_bad_cost.log; stop file scratch/fill_s223.stop.
#   CAPF=20 QS="327:s223_direct_s1:5.0:direct" nohup bash scratch/s223_fill.sh >> logs/fill_s223.out 2>&1 &
# Tests: S223_FILL_LIB=1 source scratch/s223_fill.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
PMAX=${PMAX:-0.25}; CAPF=${CAPF:-}; SETUP_H=${SETUP_H:-0.35}; PSTEPS=24000; BADN=${BAD0:-0}; BADMAX=${BADMAX:-4}; HOSTMAX=${HOSTMAX:-3}
BADSPENDMAX=${BADSPENDMAX:-0.60}; BADLOG=${BADLOG:-logs/s223_bad_cost.log}; ADMLOG=${ADMLOG:-logs/s223_admitted.log}; MEMMAX=21.0; GT_STEPS=60
QS="${QS:-}"
eval_h() { case "$1" in s223_direct_*) echo "${EVAL_H_DIRECT:-0.25}";; *) echo "${EVAL_H_LATENT:-2.0}";; esac; }
cell_label() { case "$1" in s223_direct_*) echo DIRECT;; s223_ps_*) echo PS;; s223_m1_*) echo M1-only;; s223_m2_*) echo M2-only;; esac; }
log() { echo "$(date +%T) fill_s223 $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
sq() { timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$2" "root@$1" "$3" 2>/dev/null; }
fam() { python -c "s = 0.0
for f in ('$ADMLOG', '$BADLOG'):
    try: s += sum(float(l.split()[-1]) for l in open(f) if l.strip() and not l.startswith('#'))
    except OSError: pass
print(round(s, 4))"; }
cellbad() { awk -v n="b$1" '!/^#/ && $3 == n {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0; }
gtimegate() {   # $1 = expected cell label; stdin = '{"timing": ..., "mem": ...}' -> "OK <sec_per_step> <micro> <peak>" | "REFUSE <why>"
  python -c "import json, sys
try: d = json.load(sys.stdin); t = d['timing']; mc = d['mem']
except Exception as e: print('REFUSE unreadable timing / mem_check (%s)' % e); sys.exit()
s = t.get('sec_per_step'); bad = []
if not isinstance(s, (int, float)) or isinstance(s, bool) or not s > 0: bad.append('sec_per_step %r' % (s,))
if t.get('steps_timed') != $GT_STEPS: bad.append('steps_timed %r' % (t.get('steps_timed'),))
if t.get('cell') != sys.argv[1]: bad.append('cell %r != %r' % (t.get('cell'), sys.argv[1]))
if '3090' not in str(t.get('gpu')): bad.append('gpu %r (RTX 3090 only)' % (t.get('gpu'),))
chk = [c for c in mc.get('checks', []) if c.get('micro') == mc.get('micro')]; pk = chk[-1]['peak_gib'] if chk else None
if not isinstance(pk, (int, float)) or pk > $MEMMAX: bad.append('peak_gib %r at micro %r (<= $MEMMAX)' % (pk, mc.get('micro')))
print(('REFUSE ' + '; '.join(bad)) if bad else 'OK %s %s %.3f' % (s, mc.get('micro'), pk))" "$1"; }
projection() { python -c "print(round(($1 * $PSTEPS / 3600 + $SETUP_H + $2) * $3, 4))"; }   # sps eval_h dph -> $
admit() { python -c "import sys; p, cc, fs, cf = map(float, sys.argv[1:5]); sys.exit(0 if p <= cc + 1e-9 and fs + p <= cf + 1e-9 else 1)" "$@"; }
refuse() {   # host port iid offer ip dph t0 why
  for dp in $(ps -ef | grep "auto_drain_gen8_s223.sh $1 $2 gen_b$n $3" | grep -v grep | awk '{print $2}'); do [ "$dp" != "$$" ] && kill "$dp"; done
  echo "$1:$2" >> logs/drained_hosts.txt; log "b$n: $(bash scratch/vdestroy.sh "$3" | tail -1)"
  local mins c; mins=$(( ($(date +%s) - $7) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $6, 4))"); BADN=$((BADN + 1))
  echo "$(date '+%F %T') b$n offer $4 $5 ${mins}min \$$6/h ($8) $c" >> "$BADLOG"; echo "$5 # §223 refused ($8) $(date '+%F %T')" >> scratch/vast_exclude_ips.txt
  [ -f "scratch/gen_q_b$n.txt.norequeue" ] && mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"
  [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
  return 0
}
[ "${S223_FILL_LIB:-0}" = 1 ] && return 0

# ---------------------------------------------------------------- start refusals (nothing is created before these pass)
others=$(ps -ef | awk '$6 ~ /bash(\.exe)?$/ && $7 ~ /s2[0-9]+_fill[a-z0-9_]*\.sh$/ && $7 !~ /s223_fill\.sh$/')
[ -n "$others" ] && { echo "$(date '+%F %T') fill_s223 REFUSED: another filler is running (one filler at a time): $others"; exit 2; }
[ "$(ps -ef | awk -v me=$$ '$6 ~ /bash(\.exe)?$/ && $7 ~ /s223_fill\.sh$/ && $2 != me && $3 != me' | wc -l)" -gt 0 ] && { echo "$(date '+%F %T') fill_s223 REFUSED: a second s223_fill.sh is running"; exit 2; }
[ -n "$CAPF" ] && [ -n "$QS" ] || { echo "$(date '+%F %T') fill_s223 REFUSED: CAPF and QS must be given"; exit 2; }
for cell in $QS; do IFS=: read -r n_ r_ a_ tg_ <<<"$cell"; f_=scratch/s223_args_frozen_$tg_.json
  git ls-files --error-unmatch "$f_" > /dev/null 2>&1 || { echo "$(date '+%F %T') fill_s223 REFUSED: $f_ is not committed"; exit 2; }; done
for f_ in boxes/codi_ps.py boxes/ps_eval.py boxes/nested_words.py boxes/codi_port.py scratch/s223_args_check.py; do
  git ls-files --error-unmatch "$f_" > /dev/null 2>&1 || { echo "$(date '+%F %T') fill_s223 REFUSED: $f_ is not committed"; exit 2; }; done
dirty=$( { git status --porcelain boxes/ | grep -v "^??" | grep -v "boxes/runs"; git status --porcelain scratch/s223_*.json scratch/s223_args_check.py scratch/s223_done_marker.sh scratch/drain_gen_partial_s223.sh scratch/auto_drain_gen8_s223.sh scratch/gen_deploy_120_s223.sh; } )
[ -n "$dirty" ] && { echo "$(date '+%F %T') fill_s223 REFUSED: uncommitted module / frozen-ref / script edits (deploys ship the working tree): $dirty"; exit 2; }
log "start: QS $QS; family \$$CAPF; PSTEPS $PSTEPS; SETUP_H $SETUP_H; codi_ps sha $(sha256sum boxes/codi_ps.py | cut -c1-12); ps_eval sha $(sha256sum boxes/ps_eval.py | cut -c1-12); nested_words sha $(sha256sum boxes/nested_words.py | cut -c1-12)"

for cell in $QS; do
  IFS=: read -r n run CAPC TAG <<<"$cell"; tries=0; FROZEN=scratch/s223_args_frozen_$TAG.json; LBL=$(cell_label "$run"); xh=$(eval_h "$run")
  grep -q "^$n:$run " "$ADMLOG" 2>/dev/null && { log "b$n: $run already admitted - next"; continue; }
  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }
  log "b$n: $run ($LBL), share \$$CAPC, eval hours $xh, frozen $FROZEN sha $(sha256sum $FROZEN | cut -c1-12)"
  while true; do
    [ -f scratch/fill_s223.stop ] && { log "stopped (scratch/fill_s223.stop)"; exit 0; }
    bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
    bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0)
    if python -c "import sys; sys.exit(0 if $bt > $BADSPENDMAX else 1)"; then log "CAP: §223 refused / BAD-host spend \$$bt > \$$BADSPENDMAX - held; tell Audit"; exit 3; fi
    [ "$tries" -ge "$HOSTMAX" ] && { log "b$n: $run NOT RUN - no admitting host in $HOSTMAX measured tries; tell Audit / Theory"; continue 2; }
    r_=$(GPU_NAME=RTX_3090 python scratch/offer_pick.py - "$PMAX"); o=""; ip=""; dph=""; read -r o ip dph <<<"$r_"
    if [ -z "${o:-}" ]; then log "b$n: no RTX_3090 offer (non-580, <= \$$PMAX); retry in 10 min"; sleep 600; continue; fi
    log "b$n: deploying $run on offer $o ($ip, RTX_3090, \$$dph/h)"; t0=$(date +%s); bash scratch/gen_deploy_120_s223.sh "$o $n"
    line=$(grep "gen host b$n offer $o:" logs/gen_deploy.log | tail -1)
    if ! echo "$line" | grep -q STARTED; then
      mins=$(( ($(date +%s) - t0) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $dph, 4))"); BADN=$((BADN + 1))
      echo "$(date '+%F %T') b$n offer $o $ip ${mins}min \$$dph/h $c" >> "$BADLOG"; log "b$n: did not start (BAD HOST $BADN / $BADMAX; billed ~\$$c)"
      [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
      continue
    fi
    mv "scratch/gen_q_b$n.txt" "scratch/gen_q_b$n.txt.norequeue"
    read -r h p iid <<<"$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="INSTANCE") print $(i+2)" "$(i+3)" "$(i+1)}')"
    gt=""; for i in $(seq 1 45); do gt=$(sq "$h" "$p" "cd /workspace/Latent/boxes/runs_fsq/${run}_gtime && [ -s timing.json ] && python -c \"import json; print(json.dumps({'timing': json.load(open('timing.json')), 'mem': json.load(open('mem_check.json')), 'args': {k: json.load(open('args.json')).get(k) for k in ('transformers_version', 'torch_version', 'train_driver', 'train_gpu', 'cell')}}))\""); [ -n "$gt" ] && break; sleep 60; done
    if [ -z "$gt" ]; then log "b$n: no G-TIME timing.json after 45 min on $h:$p ($(sq "$h" "$p" "tail -n 3 /workspace/Latent/logs/gen_${run}_gtime.log; tail -n 2 /workspace/Latent/logs/queue_gen_b$n.out" | tr '\n' ' ' | cut -c1-300)) - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "no-gtime"; continue; fi
    gg=$(echo "$gt" | gtimegate "$LBL"); log "b$n: G-TIME $(echo "$gt" | cut -c1-600) -> $gg"
    if ! echo "$gg" | grep -q "^OK "; then
      case "$gg" in *"cell "*|*"steps_timed"*) log "ALERT b$n: G-TIME shape check failed ($gg) - a queue fault: refusing and HOLDING for Theory / Audit"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "gtime-shape"; exit 3;; esac
      log "b$n: G-TIME REFUSED on $h:$p ($gg; measured refusal $((tries + 1)) / $HOSTMAX) - destroying, next host"; tries=$((tries + 1)); refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "gtime"; continue
    fi
    sps=$(echo "$gg" | awk '{print $2}'); proj=$(projection "$sps" "$xh" "$dph"); fs=$(fam); cb=$(cellbad "$n")
    log "b$n: G-TIME $sps s/step (deepest shape) -> projection \$$proj ($PSTEPS steps + setup + eval $xh h; share \$$CAPC; this cell's refused / BAD so far \$$cb); family \$$fs + proj vs \$$CAPF"
    if admit "$proj" "$CAPC" "$fs" "$CAPF"; then
      echo "$n:$run $h:$p $iid \$$dph/h sps $sps $proj" >> "$ADMLOG"; log "b$n: ADMIT $run on $h:$p (instance $iid): \$$proj"
      ca=""; for i in $(seq 1 30); do ca=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/$run/args.json"); [ -n "$ca" ] && break; sleep 60; done
      ck=$(echo "$ca" | python scratch/s223_args_check.py - "$FROZEN"); log "b$n: host cell args.json: $ck"
      if ! echo "$ck" | grep -q "^ARGS-CHECK OK"; then log "ALERT b$n: the cell's host args.json differs from the frozen ref ($ck) - rule b would void every reading: refusing and HOLDING"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "args"; exit 3; fi
      evh=$(python -c "print(round(max(0.25, 2000 * $sps / 3600), 2))")
      nohup python scratch/ckpt_mirror.py "$n:$run" --every_h "$evh" --companions ckpt_fingerprint.json,log.jsonl --optional args.json,stages.jsonl >> "logs/ckpt_mirror_b$n.out" 2>&1 &
      gl=$(python -c "print(round(1.3 * $proj, 4))")
      GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000,12000,16000,20000 nohup python scratch/cost_guard.py "s223_b$n" "$gl" "$PSTEPS" "$n:$run:$proj:$PSTEPS" > "logs/s223_guard_b$n.out" 2>&1 &
      log "b$n: ckpt mirror (every $evh h ~ 2k steps) + cell guard \$$gl = 1.3 x \$$proj (GUARD_STOP=1) started"
      famc=""; for c_ in $QS; do IFS=: read -r n_ r_ a_ t_ <<<"$c_"; famc="$famc $n_:$r_:$a_:$PSTEPS"; done
      if ! ps -ef | grep "cost_guard.py s223fam " | grep -v grep > /dev/null; then
        bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0); famcap=$(python -c "print(round($CAPF - $bt, 4))")
        GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000,12000,16000,20000 nohup python scratch/cost_guard.py s223fam "$famcap" "$PSTEPS" $famc > logs/s223fam_guard.out 2>&1 &
        log "guard s223fam (\$$CAPF - refused / BAD \$$bt = \$$famcap, GUARD_STOP=1;$famc) started"
      fi
      sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; break
    fi
    log "b$n: projection \$$proj NOT admitted on $h:$p (measured refusal $((tries + 1)) / $HOSTMAX) - destroying, next host"
    tries=$((tries + 1)); refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "projection"
  done
done
log "§223 filler done: $(cat $ADMLOG 2>/dev/null | tr '\n' ';')"
