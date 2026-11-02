"""Note 257-2 (score_hx L1): part_halthead tests TEST2 d 1..7 (was 1..6; the fit, the head and d 1..6 unchanged - one more d appended). Edits the eval parts source (each
anchor asserted once, inside part_halthead only), then regenerates boxes/inject_aw_eval.py.
    python -B scratch/s259_build/edit_eval_s259.py"""
import subprocess
import sys

q = "scratch/s234_build/inject_aw_eval_parts.py"; t = open(q, encoding="utf-8").read(); i = t.index("def part_halthead("); j = t.index("\ndef ", i + 10); body = t[i:j]
E = [("    TEST2-DF d 1..6: halt at the first K'", "    TEST2-DF d 1..7 (Note 257-2; was 1..6): halt at the first K'"),
     ("    T = tokd(sets, a, range(1, 7))\n    for d in range(1, 7):\n", "    T = tokd(sets, a, range(1, 8))                                       # Note 257-2: d 1..7\n    for d in range(1, 8):\n")]
for o, n in E:
    assert body.count(o) == 1, ("Note 257-2 halthead edit", body.count(o), o[:60]); body = body.replace(o, n)
open(q, "w", encoding="utf-8", newline="\n").write(t[:i] + body + t[j:])
assert subprocess.call([sys.executable, "-B", "scratch/s234_build/mk_inject_aw_eval.py"]) == 0; print("Note 257-2 halthead d 1..7 applied")
