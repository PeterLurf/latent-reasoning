"""Note 259-11 C5-STABLE FRESH: append parts_s264.py (fit_sigs / part_stab_fresh; existing parts untouched) and register 'stab_fresh' in mk_inject_aw_eval.py's PARTS /
AW_ONLY / dispatcher (each anchor asserted once); regenerate boxes/inject_aw_eval.py.
    python -B scratch/s264_build/edit_eval_s264.py"""
import subprocess
import sys

p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read(); assert "stab_fresh" not in s, "already applied"
E = [('"halt_inc", "halthead2")\\nAW_ONLY = ("test_perm", "insert_ho", "ws_deep", "halt_inc", "halthead2", ',
      '"halt_inc", "halthead2", "stab_fresh")\\nAW_ONLY = ("test_perm", "insert_ho", "ws_deep", "halt_inc", "halthead2", "stab_fresh", '),
     ('"halthead2": lambda: part_halthead2(m, g, sets, a, dev, bs)}[p_]()',
      '"halthead2": lambda: part_halthead2(m, g, sets, a, dev, bs),\\n              "stab_fresh": lambda: part_stab_fresh(m, g, sets, a, dev, bs)}[p_]()')]
for o, n in E:
    assert s.count(o) == 1, ("C5-STABLE FRESH eval edit", s.count(o), o[:80]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s)
q = "scratch/s234_build/inject_aw_eval_parts.py"; t = open(q, encoding="utf-8").read(); assert "def part_stab_fresh" not in t
open(q, "w", encoding="utf-8", newline="\n").write(t + open("scratch/s264_build/parts_s264.py", encoding="utf-8").read())
assert subprocess.call([sys.executable, "-B", "scratch/s234_build/mk_inject_aw_eval.py"]) == 0; print("C5-STABLE FRESH eval part applied")
