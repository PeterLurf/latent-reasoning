"""Note 260-3: append parts_s266.py (part_stab; existing parts untouched) and register 'stab' in mk_inject_aw_eval.py's PARTS / AW_ONLY / dispatcher (each anchor
asserted once); regenerate boxes/inject_aw_eval.py.
    python -B scratch/s266_build/edit_eval_s266.py"""
import subprocess
import sys

p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read(); assert '"stab"' not in s, "already applied"
E = [('"halt_inc", "halthead2", "stab_fresh")\\nAW_ONLY = ("test_perm", "insert_ho", "ws_deep", "halt_inc", "halthead2", "stab_fresh", ',
      '"halt_inc", "halthead2", "stab_fresh", "stab")\\nAW_ONLY = ("test_perm", "insert_ho", "ws_deep", "halt_inc", "halthead2", "stab_fresh", "stab", '),
     ('"stab_fresh": lambda: part_stab_fresh(m, g, sets, a, dev, bs)}[p_]()',
      '"stab_fresh": lambda: part_stab_fresh(m, g, sets, a, dev, bs),\\n              "stab": lambda: part_stab(m, g, sets, a, dev, bs)}[p_]()')]
for o, n in E:
    assert s.count(o) == 1, ("STAB eval edit", s.count(o), o[:80]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s)
q = "scratch/s234_build/inject_aw_eval_parts.py"; t = open(q, encoding="utf-8").read(); assert "def part_stab(" not in t
open(q, "w", encoding="utf-8", newline="\n").write(t + open("scratch/s266_build/parts_s266.py", encoding="utf-8").read())
assert subprocess.call([sys.executable, "-B", "scratch/s234_build/mk_inject_aw_eval.py"]) == 0; print("STAB eval part applied")
