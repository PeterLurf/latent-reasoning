s = open("scratch/s230_fill.sh", encoding="utf-8").read()
i = s.index('cd "$(dirname "$0")/.."')
hdr = ("#!/usr/bin/env bash\n"
       "# §232 HOLD-C1 blind s3 filler (HC_v1 §1; Note 232-3; cap $4): ONE non-580 RTX 3090 running the chain in gen_q_b346 (D3-T base s232_d3t_s3 48k ->\n"
       "# FT-CTRL s3 + HC s3 12k each -> the Stage-A battery + khorizon). A copy of scratch/s230_fill.sh (G-D'd; built by scratch/s232s3_build/mk_s232s3_fill.py,\n"
       "# each edit asserted once); changed: its own ledgers / stop file / family guard s232s3fam / cell guard s232s3_b<n> (the base cell, marks to 47,000) /\n"
       "# post-admit paths (scratch/s224_post_admit.sh: G-FH-EQ v2 on s232_d3t_s3_eqA / _eqB + the base cell's args), no shipped extras, the admission\n"
       "# projection (G-TIME x (48k + EQ) + SETUP_H + EXTRA_H for the post-base lines) and, because the step guards watch the base only, the\n"
       "# wall-clock guard scratch/s232_wall_guard.py over the WHOLE host (cap CAPF - outside, destroy at cap, per-line weights WEIGHTS).\n"
       "#   CAPF=4 QS=\"346:s232_d3t_s3:3.30:4.00:48000:1:3\" nohup bash scratch/s232s3_fill.sh >> logs/fill_s232s3.out 2>&1 &\n")
s = hdr + s[i:]
E = [
 ('EVAL_H=${EVAL_H:-1.3}; BADN=${BAD0:-0}; BADMAX=4; HOSTMAX=${HOSTMAX:-3}; BADLOG=logs/s230_bad_cost.log; ADMLOG=logs/s230_admitted.log',
  'EXTRA_H=${EXTRA_H:-7.4}; BADN=${BAD0:-0}; BADMAX=4; HOSTMAX=${HOSTMAX:-3}; BADLOG=logs/s232s3_bad_cost.log; ADMLOG=logs/s232s3_admitted.log'),
 ('FAMGUARD_LOG=${FAMGUARD_LOG:-logs/s230fam_guard.out}; EST_FAR=${EST_FAR:-1.20}; EST_RE=${EST_RE:-0.00}',
  'FAMGUARD_LOG=${FAMGUARD_LOG:-logs/s232s3fam_guard.out}; EST_FAR=${EST_FAR:-3.00}; EST_RE=${EST_RE:-0.00}; WALL_LOG=logs/s232s3_wall.out\n'
  'WEIGHTS=${WEIGHTS:-0.1,0.1,0.15,0.15,5.0,1.4,2.1,0.75,0.1,0.15,0.75,0.1,0.15,0.75,0.1,0.15,0.7}   # est. hours per gen_q_b346 line (the wall guard\'s projection)'),
 ('    343) echo "boxes/runs_fsq/s213_d3t_s1/calibration.json" ;;\n', ''),
 ('log() { echo "$(date +%T) fill_s230 $*"', 'log() { echo "$(date +%T) fill_s232s3 $*"'),
 ('  if grep -qE "ALERT|STOP" "$FAMGUARD_LOG" 2>/dev/null; then log "HOLD: s230fam fired ($(grep -E "ALERT|STOP" "$FAMGUARD_LOG" | tail -1 | cut -c1-160)) - filler exits at $1; tell Audit"; exit 3; fi',
  '  if grep -qE "ALERT|STOP" "$FAMGUARD_LOG" "$WALL_LOG" 2>/dev/null; then log "HOLD: s232s3 guard fired ($(grep -hE "ALERT|STOP" "$FAMGUARD_LOG" "$WALL_LOG" 2>/dev/null | tail -1 | cut -c1-160)) - filler exits at $1; tell Audit"; exit 3; fi'),
 ("$7 !~ /s230_fill\\.sh$/')", "$7 !~ /s232s3_fill\\.sh$/')"),
 ("$7 ~ /s230_fill\\.sh$/ && $2 != me", "$7 ~ /s232s3_fill\\.sh$/ && $2 != me"),
 ('a second s230_fill.sh is running', 'a second s232s3_fill.sh is running'),
 ('scratch/s230_*.sh scratch/gen_q_b343.txt', 'scratch/s232s3_*.sh scratch/s232_wall_guard.py scratch/gen_q_b346.txt'),
 ('''grep -q "fh_lambda_fixed" boxes/vbank.py || { echo "$(date '+%F %T') fill_s230 REFUSED: boxes/vbank.py has no --fh_lambda_fixed (the §230 build is not committed)"; exit 2; }''',
  '''{ grep -q "fh_lambda_fixed" boxes/vbank.py && grep -q "def part_c1hold" boxes/vbank_diag.py; } || { echo "$(date '+%F %T') fill_s232s3 REFUSED: the §230 / §232 builds are not committed"; exit 2; }'''),
 ('# §230 ledger (fill_s230)', '# §232 s3 ledger (fill_s232s3)'),
 ('[ -f scratch/fill_s230.stop ] && { log "stopped (scratch/fill_s230.stop)"', '[ -f scratch/fill_s232s3.stop ] && { log "stopped (scratch/fill_s232s3.stop)"'),
 ('log "CAP: §230 BAD-host spend', 'log "CAP: §232 s3 BAD-host spend'),
 ('    proj=$(python -c "print(round(($sps * ($STEPS + $EQ) / 3600 + $SETUP_H + $EVAL_H * $NE) * $dph, 2))"); fs=$(fam)',
  '    proj=$(python -c "print(round(($sps * ($STEPS + $EQ) / 3600 + $SETUP_H + $EXTRA_H) * $dph, 2))"); fs=$(fam)   # EXTRA_H: the FT cells + evals after the base'),
 ('($STEPS + EQ $EQ steps, $NE eval set(s) x $EVAL_H h; cap', '($STEPS + EQ $EQ base steps + $EXTRA_H h post-base lines; cap'),
 ('GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000,11900 nohup python scratch/cost_guard.py "s230_b$n" "$gl" "$STEPS" "$n:$run:$proj:$STEPS" > "logs/s230_guard_b$n.out" 2>&1 &',
  'GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000,16000,24000,32000,40000,47000 nohup python scratch/cost_guard.py "s232s3_b$n" "$gl" "$STEPS" "$n:$run:$proj:$STEPS" > "logs/s232s3_guard_b$n.out" 2>&1 &'),
 ('if ! ps -ef | grep "cost_guard.py s230fam " | grep -v grep > /dev/null; then', 'if ! ps -ef | grep "cost_guard.py s232s3fam " | grep -v grep > /dev/null; then'),
 ('GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000,11900 nohup python scratch/cost_guard.py s230fam "$fcap" 12000 $FAMC > "$FAMGUARD_LOG" 2>&1 &\n        log "guard s230fam (',
  'GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000,16000,24000,32000,40000,47000 nohup python scratch/cost_guard.py s232s3fam "$fcap" 48000 $FAMC > "$FAMGUARD_LOG" 2>&1 &\n        log "guard s232s3fam ('),
 ('      STOPF=scratch/fill_s230.stop ALERTS=logs/s230_alerts.log PALOG=logs/s230_post_admit.log nohup bash scratch/s224_post_admit.sh "$n" "$run" "$h" "$p" "$iid" "$FAR" "$SEED" > "logs/s230_post_b$n.out" 2>&1 &',
  '      NL=$(grep -v "^#" scratch/gen_q_b$n.txt.norequeue | grep -c .); wc_=$(python -c "print(round($CAPF - $(outside), 4))")   # the whole-host wall-clock guard\n'
  '      GUARD_STOP=1 GUARD_WEIGHTS="$WEIGHTS" nohup python scratch/s232_wall_guard.py "s232s3_wall_b$n" "$wc_" "$iid" "$h" "$p" "$dph" "$t0" "gen_b$n" "$NL" "$proj" > "$WALL_LOG" 2>&1 &\n'
  '      log "b$n: wall-clock guard over the host (\\$$wc_ = \\$$CAPF - outside, destroy at cap, $NL lines, weights $WEIGHTS) started"\n'
  '      STOPF=scratch/fill_s232s3.stop ALERTS=logs/s232s3_alerts.log PALOG=logs/s232s3_post_admit.log nohup bash scratch/s224_post_admit.sh "$n" "$run" "$h" "$p" "$iid" "$FAR" "$SEED" > "logs/s232s3_post_b$n.out" 2>&1 &'),
 ('log "§230 filler done:', 'log "§232 s3 filler done:'),
]
ALL = [('fill_s230 REFUSED', 'fill_s232s3 REFUSED'), ('# §230 vanished', '# §232s3 vanished'), ('# §230 refused', '# §232s3 refused')]
for o, n in E:
    assert s.count(o) == 1, ("miss", s.count(o), o[:80]); s = s.replace(o, n)
for o, n in ALL:
    assert s.count(o) >= 1, ("miss-all", o); s = s.replace(o, n)
open("scratch/s232s3_fill.sh", "w", encoding="utf-8", newline="\n").write(s); print("written")
