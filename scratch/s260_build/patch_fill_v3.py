"""s260 template v3 (11-01 22:47: the family guard got this filler's STEPS for EVERY ledger cell - b396 16000 ran as 100): each admit line records its cell's steps
('steps S' before the final $ field; parsers read the LAST field, unchanged), and ALLC = scratch/ledger_steps.py (each cell's 'steps' field, else its own run line's
--steps / --stop_at in its queue file; unresolvable -> ALERT + HOLD, never a guess; Audit 11-01). Applied to scratch/s260_fill.sh only while NO s260 filler runs; each edit asserted once.
    python -B scratch/s260_build/patch_fill_v3.py"""
import subprocess

p = "scratch/s260_fill.sh"; s = open(p, encoding="utf-8").read(); assert "s260 template v3" not in s, "already applied"
E = [
    ('      echo "$n:$run $h:$p $iid \\$$dph/h sps $sps $proj" >> "$ADMLOG"',
     '      echo "$n:$run $h:$p $iid \\$$dph/h sps $sps steps $STEPS $proj" >> "$ADMLOG"   # v3: the cell\'s steps recorded (the family guard reads them)'),
    ('      ALLC=$(awk -v S="$STEPS" \'!/^#/ && /^[0-9]+:/ {split($1, a, ":"); print a[1] ":" a[2] ":" $NF ":" S}\' "$ADMLOG" | tr \'\\n\' \' \')',
     '      ALLC=$(python scratch/ledger_steps.py "$ADMLOG") || { log "ALERT b$n: ledger_steps could not resolve a cell\'s steps - HOLD (family guard not restarted); tell Audit"; exit 3; }   # v3 (Audit)'),
]
for o, n in E:
    assert s.count(o) == 1, ("s260 v3 edit", s.count(o), o[:80]); s = s.replace(o, n)
s = s.replace("# s260 template v2 (", "# s260 template v3 (per-cell steps in the ledger / family guard) + v2 (", 1)
open(p, "w", encoding="utf-8", newline="\n").write(s); assert subprocess.run(["bash", "-n", p]).returncode == 0; print("s260 v3 applied, bash -n OK")
