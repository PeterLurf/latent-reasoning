#!/usr/bin/env bash
# §219 TEXT-SELF filler (registered 2026-10-18 15:37 EDT; TS_v1.md at 146a25d0 §2.5 / §2.9 step 3 / §7 G-T5 G-T9 G-T10 G-T11 G-T14 / §8; Notes 219-1..219-5;
# family cap $3.00 under GUARD_STOP, Audit auto-approved under the user's $30 rule). Copied from scratch/s218_fill.sh; changed:
#   - cells QS "n:run:admission:guard:steps" (default 318 s219_t1_s1 $1.00 / $1.20, 12,000 steps); family s219fam $3.00 (FAMC: T1 at $1.20, T2 at $1.70);
#   - deploy via scratch/gen_deploy_120_s219.sh (arms auto_drain_gen8_s219.sh -> drain_gen_partial_s219.sh: §219 done markers, never eval.json); refuse()
#     kills that watcher;
#   - G-T9 (line 1): <run>_mem/mem_check_self.json must read pass true, a numeric peak <= 21.0 GiB, self_micro 2, bs 64, n_top 12, an RTX 3090; else the
#     host is REFUSED and that is a MEASURED refusal (one of HOSTMAX tries; the micro never changes); no file in 25 min -> refused, no try used (as no-gtime);
#   - G-T5 on the host smokes: <run>_mem / <run>_gtime args.json through scratch/s219_args_check.py (score_s219.args_diff smoke mode + the frozen args);
#     a BAD smoke args.json is a recipe fault, not a host fault: the host is refused and the filler HOLDS (exit 3) for Theory / Audit;
#   - G-T10 (line 2): <run>_gtime/timing.json (deep self on odd steps, Note 219-4 (3); deep_n 12, self_micro 2, 200 steps, RTX 3090) -> projection
#     (sec_per_step x steps / 3600 + SETUP_H) x $/h; ADMIT iff proj <= the cell's admission AND this cell's refused / BAD spend + proj <= its guard AND
#     family (admitted projections + every refused / BAD host) + proj <= $3.00; a projection refusal is a MEASURED refusal;
#   - after admission: ckpt_mirror.py every ~2k steps with --companions draws_log.json,log.jsonl,args.json,init_from.json (draws_log.json is a list
#     written atomically BEFORE each checkpoint; the mirror's pre-copy sha test rejects a ckpt newer than the staged draws_log, and an older ckpt is
#     covered since a resume keeps draws with step < step0); the cell guard and s219fam run scratch/cost_guard_s219.py (stock cost_guard.py cannot parse
#     textself_ft's log.jsonl: "loss_last" sits between "loss" and "sec"); the cell's host args.json through s219_args_check.py cell (ALERT if BAD; the
#     cell keeps running under its guard, as §218), and the trainer's own ARGS-DIFF line (vs s170_cot_s1/args.json) logged (G-T5 "printed in each launch log");
#   - start refusals: another filler running (one filler at a time), FROZEN_SHA unset or != sha256(scratch/s219_args_frozen_t1.json), the frozen ref
#     untracked / dirty, boxes/ modules dirty, s170_cot_s1/args.json or pool_meta.json dirty / untracked, model.pt sha != the V6 pin, pool_t1.jsonl sha !=
#     pool_meta.json's, or the queue's three lines not reproducing args that pass s219_args_check (CPU, the trainer's own main(), ~35 s);
#   - logs/fill_s219.out (stdout), logs/s219_admitted.log, logs/s219_bad_cost.log; stop file scratch/fill_s219.stop.
# Unchanged: ONE filler; cells in QS order each on its own RTX 3090; offer <= PMAX, never 580.x (offer_pick); HOSTPROBE; HOSTMAX 3 = 1 + 2 more
# MEASURED refusals; BAD / non-start / hostprobe / missing-file hosts use no try but count toward the cell guard and the family; steps never reduced.
#   FROZEN_SHA=<Theory's sha256 of scratch/s219_args_frozen_t1.json> nohup bash scratch/s219_fill.sh >> logs/fill_s219.out 2>&1 &
# Tests: S219_FILL_LIB=1 source scratch/s219_fill.sh (defines the helpers, returns before any check / deploy).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
PMAX=${PMAX:-0.25}; CAPF=3.00; SETUP_H=${SETUP_H:-0.35}; BADN=${BAD0:-0}; BADMAX=${BADMAX:-4}; HOSTMAX=${HOSTMAX:-3}; BADSPENDMAX=${BADSPENDMAX:-0.30}
BADLOG=${BADLOG:-logs/s219_bad_cost.log}; ADMLOG=${ADMLOG:-logs/s219_admitted.log}; FROZEN=scratch/s219_args_frozen_t1.json; MEMMAX=21.0
QS="${QS:-318:s219_t1_s1:1.00:1.20:12000}"
FAMC="318:s219_t1_s1:1.20:12000 319:s219_t2_s1:1.70:12000"   # unstarted cells project at their guard
BASE_SHA=36f4659b99f08a65202b303f4af74ef20ac65f1e60d2a25f3e84aafbfb2e3392   # TS_v1 V6 (s170_cot_s1/model.pt)
cell_extra() {   # per-cell shipped files beside the queue's modules (auto_ship also ships existing file tokens of the queue): frozen ref, init model, base args, pool
  case "$1" in
    318) echo "scratch/s219_args_frozen_t1.json boxes/runs_fsq/s170_cot_s1/model.pt boxes/runs_fsq/s170_cot_s1/args.json boxes/runs_fsq/s219_pool_t1_s1/pool_t1.jsonl boxes/runs_fsq/s219_pool_t1_s1/pool_meta.json" ;;
  esac
}
log() { echo "$(date +%T) fill_s219 $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
sq() { timeout 40 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$2" "root@$1" "$3" 2>/dev/null; }
fam() { python -c "s = 0.0
for f in ('$ADMLOG', '$BADLOG'):
    try: s += sum(float(l.split()[-1]) for l in open(f) if l.strip() and not l.startswith('#'))
    except OSError: pass
print(round(s, 4))"; }
cellbad() { awk -v n="b$1" '!/^#/ && $3 == n {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0; }   # this cell's refused / BAD spend
memgate() {   # stdin = mem_check_self.json -> "OK <peak>" | "REFUSE <why>" (G-T9)
  python -c "import json, sys
try: d = json.load(sys.stdin)
except Exception as e: print('REFUSE unreadable mem_check_self.json'); sys.exit()
pk = d.get('peak_gib'); bad = []
if not isinstance(pk, (int, float)) or isinstance(pk, bool): bad.append('peak %r' % (pk,))
elif pk > $MEMMAX: bad.append('peak %.3f GiB > $MEMMAX' % pk)
if d.get('pass') is not True: bad.append('pass %r' % (d.get('pass'),))
for k, v in (('self_micro', 2), ('bs', 64), ('n_top', 12), ('steps', 2)):
    if d.get(k) != v: bad.append('%s %r != %r' % (k, d.get(k), v))
if '3090' not in str(d.get('gpu')): bad.append('gpu %r (G-T14: RTX 3090 only)' % (d.get('gpu'),))
print(('REFUSE ' + '; '.join(bad)) if bad else 'OK %.3f' % pk)"; }
gtimegate() {   # stdin = timing.json -> "OK <sec_per_step> <self> <plain> micro <m>" | "REFUSE <why>" (G-T10 shape checks; the price test is admit())
  python -c "import json, sys
try: d = json.load(sys.stdin)
except Exception: print('REFUSE unreadable timing.json'); sys.exit()
s = d.get('sec_per_step'); bad = []
if not isinstance(s, (int, float)) or isinstance(s, bool) or not s > 0: bad.append('sec_per_step %r' % (s,))
for k, v in (('deep_n', 12), ('self_micro', 2), ('steps_timed', 200), ('p_self', 0.5), ('arm', 'textcot')):
    if d.get(k) != v: bad.append('%s %r != %r' % (k, d.get(k), v))
if '3090' not in str(d.get('gpu')): bad.append('gpu %r (G-T14: RTX 3090 only)' % (d.get('gpu'),))
print(('REFUSE ' + '; '.join(bad)) if bad else 'OK %s %s %s micro %s' % (s, d.get('sec_per_step_self'), d.get('sec_per_step_plain'), d.get('micro')))"; }
projection() { python -c "print(round(($1 * $2 / 3600 + $SETUP_H) * $3, 4))"; }   # sps steps dph -> $
admit() {   # proj capc gstop cellbad fam capf -> exit 0 iff proj <= capc and cellbad + proj <= gstop and fam + proj <= capf
  python -c "import sys; p, cc, gs, cb, fs, cf = map(float, sys.argv[1:7]); sys.exit(0 if p <= cc + 1e-9 and cb + p <= gs + 1e-9 and fs + p <= cf + 1e-9 else 1)" "$@"; }
extra_dedup() { local as_=" $2 " ex_="" f_; for f_ in $1; do case "$as_" in *" $f_ "*) ;; *) ex_="$ex_ $f_";; esac; done; echo "${ex_# }"; }   # cell files minus auto_ship's list
argcheck() { python -B scratch/s219_args_check.py "$1" - --frozen "$FROZEN"; }   # role; stdin = a host args.json -> one ARGS-CHECK line, rc 0 OK
refuse() {   # host port iid offer ip dph t0 why
  for dp in $(ps -ef | grep "auto_drain_gen8_s219.sh $1 $2 gen_b$n $3" | grep -v grep | awk '{print $2}'); do [ "$dp" != "$$" ] && kill "$dp"; done
  echo "$1:$2" >> logs/drained_hosts.txt; log "b$n: $(bash scratch/vdestroy.sh "$3" | tail -1)"
  local mins c; mins=$(( ($(date +%s) - $7) / 60 + 1 )); c=$(python -c "print(round($mins / 60 * $6, 4))"); BADN=$((BADN + 1))
  echo "$(date '+%F %T') b$n offer $4 $5 ${mins}min \$$6/h ($8) $c" >> "$BADLOG"; echo "$5 # §219 refused ($8) $(date '+%F %T')" >> scratch/vast_exclude_ips.txt
  [ -f "scratch/gen_q_b$n.txt.norequeue" ] && mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"
  [ "$BADN" -ge "$BADMAX" ] && { log "CAP: $BADMAX BAD / refused hosts - held; tell Audit / Theory"; exit 3; }
  return 0
}
[ "${S219_FILL_LIB:-0}" = 1 ] && return 0

# ---------------------------------------------------------------- start refusals (nothing is created before these pass)
others=$(ps -ef | awk '$6 ~ /bash(\.exe)?$/ && $7 ~ /s2[0-9]+_fill[a-z0-9_]*\.sh$/ && $7 !~ /s219_fill\.sh$/')   # a filler = "bash scratch/s2NN_fill*.sh" (not a shell whose -c text names one)
[ -n "$others" ] && { echo "$(date '+%F %T') fill_s219 REFUSED: another filler is running (one filler at a time): $others"; exit 2; }
[ "$(ps -ef | awk -v me=$$ '$6 ~ /bash(\.exe)?$/ && $7 ~ /s219_fill\.sh$/ && $2 != me && $3 != me' | wc -l)" -gt 0 ] && { echo "$(date '+%F %T') fill_s219 REFUSED: a second s219_fill.sh is running"; exit 2; }
[ -n "${FROZEN_SHA:-}" ] || { echo "$(date '+%F %T') fill_s219 REFUSED: FROZEN_SHA (Theory's sha256 of $FROZEN) not given"; exit 2; }
[ -s "$FROZEN" ] || { echo "$(date '+%F %T') fill_s219 REFUSED: $FROZEN absent (the .draft is not the frozen ref)"; exit 2; }
[ "$(sha256sum "$FROZEN" | cut -c1-64)" = "$FROZEN_SHA" ] || { echo "$(date '+%F %T') fill_s219 REFUSED: sha256($FROZEN) != FROZEN_SHA"; exit 2; }
for f_ in "$FROZEN" boxes/runs_fsq/s170_cot_s1/args.json boxes/runs_fsq/s219_pool_t1_s1/pool_meta.json; do
  git ls-files --error-unmatch "$f_" > /dev/null 2>&1 || { echo "$(date '+%F %T') fill_s219 REFUSED: $f_ is not committed"; exit 2; }
done
dirty=$( { git status --porcelain boxes/ | grep -v "^??" | grep -v "boxes/runs"; git status --porcelain "$FROZEN" boxes/runs_fsq/s170_cot_s1/args.json boxes/runs_fsq/s219_pool_t1_s1/pool_meta.json; } )   # deploys ship the working tree
[ -n "$dirty" ] && { echo "$(date '+%F %T') fill_s219 REFUSED: uncommitted module / ref / input edits: $dirty"; exit 2; }
ms=$(sha256sum boxes/runs_fsq/s170_cot_s1/model.pt | cut -c1-64); [ "$ms" = "$BASE_SHA" ] || { echo "$(date '+%F %T') fill_s219 REFUSED: s170_cot_s1/model.pt sha $ms != V6 $BASE_SHA"; exit 2; }
ps_=$(sha256sum boxes/runs_fsq/s219_pool_t1_s1/pool_t1.jsonl 2>/dev/null | cut -c1-64); pm_=$(python -c "import json; print(json.load(open('boxes/runs_fsq/s219_pool_t1_s1/pool_meta.json'))['sha256'])")
[ -n "$ps_" ] && [ "$ps_" = "$pm_" ] || { echo "$(date '+%F %T') fill_s219 REFUSED: pool_t1.jsonl sha '${ps_:-missing}' != pool_meta $pm_"; exit 2; }
log "start: QS $QS; frozen $FROZEN sha ${FROZEN_SHA:0:12}; model V6 ok; pool sha ${ps_:0:12} ok; SETUP_H $SETUP_H; HOSTMAX $HOSTMAX; family \$$CAPF"

for cell in $QS; do
  IFS=: read -r n run CAPC GSTOP STEPS <<<"$cell"; tries=0
  grep -q "^$n:$run " "$ADMLOG" 2>/dev/null && { log "b$n: $run already admitted - next"; continue; }
  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }
  ce_="$(cell_extra "$n")"; for f_ in $ce_; do [ -s "$f_" ] || { log "b$n: cell file $f_ missing - not starting"; exit 2; }; done
  export EXTRA_120="$(extra_dedup "$ce_" "$(python scratch/auto_ship.py scratch/gen_q_b$n.txt)")"   # vdeploy's tar does not dedup: a file auto_ship already ships is not listed twice
  log "b$n: EXTRA_120 (beyond auto_ship's queue tokens): $EXTRA_120"
  for L_ in 1:mem 2:gtime 3:cell; do   # the queue's own lines -> the trainer's args.json on CPU -> the same check the hosts get (G-T5 before any spend)
    ln_=${L_%%:*}; ro_=${L_##*:}; sc_=scratch/_tmp/s219_fill_selfcheck_b${n}_L$ln_.json
    python -B scratch/s219_args_draft.py --queue scratch/gen_q_b$n.txt --line "$ln_" --out_file "$sc_" --tmp "scratch/_tmp/s219_fill_tree_L$ln_" > "logs/s219_fill_selfcheck_L$ln_.log" 2>&1 \
      || { log "b$n: self-check line $ln_ could not produce args.json (logs/s219_fill_selfcheck_L$ln_.log) - not starting"; exit 2; }
    ck_=$(argcheck "$ro_" < "$sc_"); log "b$n: self-check line $ln_: $ck_"
    echo "$ck_" | grep -q "^ARGS-CHECK $ro_ OK" || { log "b$n: queue line $ln_ does not reproduce the frozen recipe - not starting"; exit 2; }
  done
  while true; do
    [ -f scratch/fill_s219.stop ] && { log "stopped (scratch/fill_s219.stop)"; exit 0; }
    bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
    bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0)
    if python -c "import sys; sys.exit(0 if $bt > $BADSPENDMAX else 1)"; then log "CAP: §219 refused / BAD-host spend \$$bt > \$$BADSPENDMAX - held; tell Audit"; exit 3; fi
    [ "$tries" -ge "$HOSTMAX" ] && { log "b$n: $run NOT RUN - no admitting host in $HOSTMAX measured tries (TS_v1 §8: readings UNREAD, TEXT-CENSORED (not admitted)); tell Audit / the user"; continue 2; }
    best=""; for gn in ${GPUS:-RTX_3090}; do
      r_=$(GPU_NAME=$gn python scratch/offer_pick.py - "$PMAX"); [ -z "$r_" ] && continue
      if [ -z "$best" ] || python -c "import sys; sys.exit(0 if float('$(echo $r_ | cut -d' ' -f3)') < float('$(echo $best | cut -d' ' -f3)') else 1)"; then best="$r_"; bgn=$gn; fi
    done
    o=""; ip=""; dph=""; read -r o ip dph <<<"$best"
    if [ -z "${o:-}" ]; then log "b$n: no ${GPUS:-RTX_3090} offer (non-580, <= \$$PMAX); retry in 10 min"; sleep 600; continue; fi
    log "b$n: deploying $run on offer $o ($ip, $bgn, \$$dph/h)"; t0=$(date +%s); bash scratch/gen_deploy_120_s219.sh "$o $n"
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
    # ---- G-T9 (line 1): the deep self shape at the fixed self micro 2, two optimiser steps, peak incl. AdamW <= 21.0 GiB
    mj=""; for i in $(seq 1 25); do mj=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_mem/mem_check_self.json"); [ -n "$mj" ] && break; sleep 60; done
    if [ -z "$mj" ]; then log "b$n: no G-T9 mem_check_self.json after 25 min on $h:$p ($(sq "$h" "$p" "tail -n 3 /workspace/Latent/logs/gen_${run}_mem.log" | tr '\n' ' ' | cut -c1-300)) - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "no-memcheck"; continue; fi
    mg=$(echo "$mj" | memgate); log "b$n: G-T9 $(echo "$mj" | tr -d '\n ' | cut -c1-400) -> $mg"
    if ! echo "$mg" | grep -q "^OK "; then
      log "b$n: G-T9 REFUSED on $h:$p ($mg; measured refusal $((tries + 1)) / $HOSTMAX; the micro never changes) - destroying, next host"
      tries=$((tries + 1)); refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "G-T9"; continue
    fi
    ma=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_mem/args.json" | argcheck mem); log "b$n: G-T5 mem smoke: $ma"
    if ! echo "$ma" | grep -q "^ARGS-CHECK mem OK"; then log "ALERT b$n: G-T5 on the G-T9 smoke's args.json failed - a recipe fault: refusing the host and HOLDING for Theory / Audit"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "G-T5-mem"; exit 3; fi
    # ---- G-T10 (line 2): forced G-TIME (odd steps deep self, even plain) -> admission
    gt=""; for i in $(seq 1 45); do gt=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_gtime/timing.json"); [ -n "$gt" ] && break; sleep 60; done
    if [ -z "$gt" ]; then log "b$n: no G-TIME timing.json after 45 min on $h:$p ($(sq "$h" "$p" "tail -n 3 /workspace/Latent/logs/gen_${run}_gtime.log" | tr '\n' ' ' | cut -c1-300)) - refusing"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "no-gtime"; continue; fi
    gg=$(echo "$gt" | gtimegate); log "b$n: G-T10 $(echo "$gt" | tr -d '\n' | cut -c1-500) -> $gg"
    if ! echo "$gg" | grep -q "^OK "; then log "ALERT b$n: G-T10 timing.json shape check failed ($gg) - refusing the host and HOLDING for Theory / Audit"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "G-T10-shape"; exit 3; fi
    ga=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/${run}_gtime/args.json" | argcheck gtime); log "b$n: G-T5 gtime smoke: $ga"
    if ! echo "$ga" | grep -q "^ARGS-CHECK gtime OK"; then log "ALERT b$n: G-T5 on the G-TIME smoke's args.json failed - a recipe fault: refusing the host and HOLDING for Theory / Audit"; refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "G-T5-gtime"; exit 3; fi
    sps=$(echo "$gg" | awk '{print $2}'); proj=$(projection "$sps" "$STEPS" "$dph"); fs=$(fam); cb=$(cellbad "$n")
    log "b$n: G-TIME $sps s/step -> cell projection \$$proj (admission \$$CAPC; this cell's refused / BAD \$$cb + proj vs guard \$$GSTOP); family so far \$$fs (admitted + refused / BAD) + proj vs \$$CAPF"
    if admit "$proj" "$CAPC" "$GSTOP" "$cb" "$fs" "$CAPF"; then
      echo "$n:$run $h:$p $iid \$$dph/h sps $sps $proj" >> "$ADMLOG"; log "b$n: ADMIT $run on $h:$p (instance $iid): \$$proj"
      evh=$(python -c "print(round(max(0.25, 2000 * $sps / 3600), 2))")       # the mirror every ~2k steps (G-T11; user rule 2026-10-15)
      nohup python scratch/ckpt_mirror.py "$n:$run" --every_h "$evh" --companions draws_log.json,log.jsonl,args.json,init_from.json --optional args_diff.json,mem_check.json >> "logs/ckpt_mirror_b$n.out" 2>&1 &
      gl=$(python -c "print(round($GSTOP - $cb, 4))")
      GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,6000,8000,10000 nohup python scratch/cost_guard_s219.py "s219_b$n" "$gl" "$STEPS" "$n:$run:$proj:$STEPS" > "logs/s219_guard_b$n.out" 2>&1 &
      log "b$n: ckpt mirror (every $evh h ~ 2k steps; ckpt_last.pt + draws_log.json + log.jsonl + args.json + init_from.json -> scratch/ckpt_mirror/$run/, sha-checked) + cell guard (\$$GSTOP - refused / BAD \$$cb = \$$gl, GUARD_STOP=1, cost_guard_s219) started"
      ca=""; for i in $(seq 1 20); do ca=$(sq "$h" "$p" "cat /workspace/Latent/boxes/runs_fsq/$run/args.json"); [ -n "$ca" ] && break; sleep 60; done
      ck=$(echo "$ca" | argcheck cell); log "b$n: host cell args.json: $ck"
      echo "$ck" | grep -q "^ARGS-CHECK cell OK" || log "ALERT b$n: the cell's host args.json fails score_s219.args_diff or the frozen ref ($ck) - Audit / Theory review (the cell keeps running under its guard)"
      ad=$(sq "$h" "$p" "grep -h '^ARGS-DIFF' /workspace/Latent/logs/gen_$run.log | tail -n 1")
      log "b$n: trainer ARGS-DIFF vs s170_cot_s1/args.json keys: $(echo "${ad#ARGS-DIFF }" | python -c "import json, sys
try: print(' '.join(sorted(json.load(sys.stdin))))
except Exception: print('UNREADABLE / none')" 2>/dev/null)"
      if ! ps -ef | grep "cost_guard_s219.py s219fam " | grep -v grep > /dev/null; then   # the FAMILY \$3.00 guard, GUARD_STOP=1, minus refused / BAD spend it cannot see
        bt=$(awk '!/^#/ {s += $NF} END {printf "%.4f", s + 0}' "$BADLOG" 2>/dev/null || echo 0); famcap=$(python -c "print(round($CAPF - $bt, 4))")
        GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,6000,8000,10000 nohup python scratch/cost_guard_s219.py s219fam "$famcap" 12000 $FAMC > logs/s219fam_guard.out 2>&1 &
        log "guard s219fam (\$$CAPF - refused / BAD \$$bt = \$$famcap, GUARD_STOP=1; $FAMC) started"
      fi
      sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; break
    fi
    log "b$n: projection \$$proj NOT admitted on $h:$p (admission \$$CAPC; guard \$$GSTOP - \$$cb; family \$$fs + proj vs \$$CAPF; measured refusal $((tries + 1)) / $HOSTMAX) - destroying, next host"
    tries=$((tries + 1)); refuse "$h" "$p" "$iid" "$o" "$ip" "$dph" "$t0" "projection"
  done
done
log "§219 filler done: $(cat $ADMLOG 2>/dev/null | tr '\n' ';')"
