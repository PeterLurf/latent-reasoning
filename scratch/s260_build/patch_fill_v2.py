"""s260 template v2 (Audit 11-01 standing rules; applied to scratch/s260_fill.sh only while NO s260 filler runs; each edit asserted once):
  (1) refused / BAD-host spend is a FAMILY cost: the cell and wall guards start at the approved cell guard GSTOP (no rejected-host deduction); the family guard keeps BAD spend.
  (2) REFUSE a cell unless GSTOP >= CAPC + 0.05 (so the guard >= the admit projection + 0.05 for every admit); re-asserted at admit (HOLD, exit 3, if ever violated).
  (3) GPU=${GPU:-RTX_3090} (the offer class; §258 S0-S3 on RTX_A6000 with PMAX from env).
  (4) the $/h used for the projection and every guard = max(offer dph, the instance's dph_total from show instance) (dph_total ran 3-17% above the offer on 11-01).
    python -B scratch/s260_build/patch_fill_v2.py"""
import subprocess

p = "scratch/s260_fill.sh"; s = open(p, encoding="utf-8").read(); assert "s260 template v2" not in s, "already applied"
E = [
    ('BADLOG=logs/s260_bad_cost.log; ADMLOG=logs/s260_admitted.log', 'BADLOG=logs/s260_bad_cost.log; ADMLOG=logs/s260_admitted.log; GPU=${GPU:-RTX_3090}   # s260 template v2 (3)'),
    ('  IFS=: read -r n run CAPC GSTOP STEPS EST <<<"$cell"; tries=0; export EXTRA_120=""; NE=1; GM=$(marks "$STEPS")\n',
     '  IFS=: read -r n run CAPC GSTOP STEPS EST <<<"$cell"; tries=0; export EXTRA_120=""; NE=1; GM=$(marks "$STEPS")\n'
     '  python -c "import sys; sys.exit(0 if $GSTOP >= $CAPC + 0.05 - 1e-9 else 1)" || { log "REFUSED b$n: guard \\$$GSTOP < admit cap \\$$CAPC + 0.05 (Audit 11-01 v2 (2))"; exit 2; }\n'),
    ('    r_=$(GPU_NAME=RTX_3090 python scratch/offer_pick.py - "$PMAX")', '    r_=$(GPU_NAME=$GPU python scratch/offer_pick.py - "$PMAX")'),
    ('log "b$n: no RTX_3090 offer (non-580, <= \\$$PMAX); retry in 10 min"', 'log "b$n: no $GPU offer (non-580, <= \\$$PMAX); retry in 10 min"'),
    ('log "b$n: deploying $run on offer $o ($ip, RTX_3090, \\$$dph/h)"', 'log "b$n: deploying $run on offer $o ($ip, $GPU, \\$$dph/h)"'),
    ('    proj=$(python -c "print(round(($sps * $STEPS / 3600 + $SETUP_H + $EVAL_H * $NE) * $dph, 2))"); fs=$(fam)',
     '    da=$(bash "$VAST" show instance "$iid" --raw 2>/dev/null | python -c "import json,sys; print(json.load(sys.stdin).get(\'dph_total\') or 0)" 2>/dev/null || echo 0)   # v2 (4)\n'
     '    dph=$(python -c "print(round(max($dph, float(\'${da:-0}\' or 0)), 4))")\n'
     '    proj=$(python -c "print(round(($sps * $STEPS / 3600 + $SETUP_H + $EVAL_H * $NE) * $dph, 2))"); fs=$(fam)'),
    ('      cb=$(awk -v n="b$n" \'!/^#/ && $3 == n {s += $NF} END {printf "%.4f", s + 0}\' "$BADLOG" 2>/dev/null || echo 0); gl=$(python -c "print(round($GSTOP - $cb, 4))")',
     '      gl=$GSTOP                                                             # v2 (1): BAD-host spend is a family cost, never deducted from a cell\n'
     '      python -c "import sys; sys.exit(0 if $gl >= $proj + 0.05 - 1e-9 else 1)" || { log "HOLD b$n: guard \\$$gl < projection \\$$proj + 0.05 - guards NOT started; tell Audit"; exit 3; }'),
    ('log "b$n: ckpt mirror (every $evh h) + cell guard (\\$$GSTOP - rejected-host \\$$cb = \\$$gl, GUARD_STOP=1, marks $GM) started"',
     'log "b$n: ckpt mirror (every $evh h) + cell guard (\\$$gl = the approved cell guard; BAD spend is family, v2 (1); GUARD_STOP=1, marks $GM) started"'),
]
for o, n in E:
    assert s.count(o) == 1, ("s260 v2 edit", s.count(o), o[:80]); s = s.replace(o, n)
s = s.replace("# s260 filler template", "# s260 template v2 (Audit 11-01: BAD spend family-only, guard >= admit + 0.05, GPU class, dph_total) on the s258 filler template", 1)
open(p, "w", encoding="utf-8", newline="\n").write(s); assert subprocess.run(["bash", "-n", p]).returncode == 0; print("s260 v2 applied, bash -n OK")
