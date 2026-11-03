"""Filler template v6 (Audit 11-03 LAUNCH_POSTMORTEM items 1 / 3 / 5; v5 = s270): family <fam> = the s270 family with s270 -> <fam> (the shared gen10 deploy
gen_deploy_120_s267.sh kept) and:
  (5) the caps come from the queue's '# CAPS' header via scratch/caps_check.py (env WALLW / EVAL_H / GATES_H / SPS_EST / GPU may only repeat it; a mismatch, a missing
      header, a weight-count mismatch, a HOSTPROBE mismatch or PMAX > admit / hours REFUSES, rc 2) - this replaces v5's PMAX line, whose inline '# v5' comment had
      swallowed its sys.exit (a no-op check in s270);
  (1) FETCH_H (the header's fetch_gb / FETCH_GBPH) in the admit projection; the wall guard is scratch/wall_guard_v6.py (DRAINING mode) with GUARD_WEIGHTS = WALLW_G
      (the last line's weight + FETCH_H);
  (3) the gate-line micro check lives in scratch/check_queue_ships.py (already run at every cell start, template v4).
Every edit asserted once; outputs bash -n checked; + scratch/v6_build/fixture_filler_v6.sh.
    python -B scratch/v6_build/mk_family_v6.py <fam e.g. s271>"""
import subprocess
import sys

FAM = sys.argv[1]; assert FAM.startswith("s") and FAM != "s270"
for f in ("fill", "post_admit", "chain", "watch"):
    s = open("scratch/s270_%s.sh" % f, encoding="utf-8").read().replace("s270", FAM)
    if f == "fill":
        assert s.count("gen_deploy_120_s267.sh") == 2                        # the s270 filler already calls the shared gen10 deploy
        E = [
            ('WALLW=${WALLW:-0.1,0.1,6.0,0.5}; PMAX=${PMAX:-0.21}; CAPF=${CAPF:-}; SETUP_H=0.4; EVAL_H=${EVAL_H:-1.0};',
             'WALLW=${WALLW:-}; PMAX=${PMAX:-0.21}; CAPF=${CAPF:-}; SETUP_H=0.4; EVAL_H=${EVAL_H:-};'),   # no inline comment: the line chains more assignments
            ('\ncd "$(dirname "$0")/.."\n', '\ncd "$(dirname "$0")/.."\n# v6: WALLW / EVAL_H / GATES_H / SPS_EST / GPU come from the queue\'s # CAPS header (scratch/caps_check.py, per cell)\n'),
            ('GPU=${GPU:-RTX_3090}', 'GPU=${GPU:-}'),
            ('  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }\n',
             '  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }\n'
             '  cc_=$(python scratch/caps_check.py "scratch/gen_q_b$n.txt" "$STEPS" "$CAPC" "$PMAX" "$SETUP_H") || { log "REFUSED b$n: $cc_ (template v6 caps)"; exit 2; }\n'
             '  eval "$cc_"; log "b$n: caps from the queue header: $cc_"   # v6: WALLW / EVAL_H / GATES_H / SPS_EST / GPU / FETCH_H / WALLW_G\n'),
            ('proj=$(python -c "print(round(($sps * $STEPS / 3600 + $SETUP_H + ${GATES_H:-0} + $EVAL_H * $NE) * $dph, 2))")',
             'proj=$(python -c "print(round(($sps * $STEPS / 3600 + $SETUP_H + ${GATES_H:-0} + ${FETCH_H:-0} + $EVAL_H * $NE) * $dph, 2))")'),
            ('GUARD_STOP=1 GUARD_WEIGHTS="$WALLW" nohup python scratch/s232_wall_guard.py', 'GUARD_STOP=1 GUARD_WEIGHTS="$WALLW_G" nohup python scratch/wall_guard_v6.py')]
        for o, n in E:
            assert s.count(o) == 1, (o[:70], s.count(o)); s = s.replace(o, n)
        lines = s.split("\n"); v5 = [i for i, l in enumerate(lines) if "# v5: + GATES_H" in l and "python -c" in l]; assert len(v5) == 1, v5
        del lines[v5[0]]; s = "\n".join(lines)                              # v5's no-op PMAX line (caps_check does the check now)
        k = [i for i, l in enumerate(s.split("\n")) if "Audit 11-03 (b412: a $0.243/h offer" in l]
        if k:
            lines = s.split("\n"); del lines[k[0]]; s = "\n".join(lines)
        assert "s232_wall_guard" not in s and "caps_check.py" in s
    assert "gen_deploy_120_%s" % FAM not in s
    open("scratch/%s_%s.sh" % (FAM, f), "w", encoding="utf-8", newline="\n").write(s); assert subprocess.run(["bash", "-n", "scratch/%s_%s.sh" % (FAM, f)]).returncode == 0, f
print("%s family written (template v6: caps from the queue, FETCH_H, wall_guard_v6); bash -n OK" % FAM)
