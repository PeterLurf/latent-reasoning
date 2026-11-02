"""Note 259-5 C5-BUN: append parts_s263.py (hh_dump_inc / part_halt_inc / part_halthead2; existing parts untouched) to the eval parts source and register 'halt_inc' /
'halthead2' in mk_inject_aw_eval.py's PARTS / AW_ONLY / dispatcher (each anchor asserted once), then regenerate boxes/inject_aw_eval.py.
    python -B scratch/s263_build/edit_eval_s263.py"""
import subprocess
import sys

p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read(); assert "halt_inc" not in s, "already applied"
E = [('"test_perm", "insert_ho", "ws_deep")\\nAW_ONLY = ("test_perm", "insert_ho", "ws_deep", ',
      '"test_perm", "insert_ho", "ws_deep", "halt_inc", "halthead2")\\nAW_ONLY = ("test_perm", "insert_ho", "ws_deep", "halt_inc", "halthead2", '),
     ('"ws_deep": lambda: part_ws_deep(m, g, sets, a, dev, bs)}[p_]()',
      '"ws_deep": lambda: part_ws_deep(m, g, sets, a, dev, bs),\\n'
      '              "halt_inc": lambda: part_halt_inc(m, g, sets, a, dev, bs), "halthead2": lambda: part_halthead2(m, g, sets, a, dev, bs)}[p_]()')]
for o, n in E:
    assert s.count(o) == 1, ("C5-BUN eval edit", s.count(o), o[:80]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s)
q = "scratch/s234_build/inject_aw_eval_parts.py"; t = open(q, encoding="utf-8").read(); assert "def part_halt_inc" not in t
open(q, "w", encoding="utf-8", newline="\n").write(t + open("scratch/s263_build/parts_s263.py", encoding="utf-8").read())
assert subprocess.call([sys.executable, "-B", "scratch/s234_build/mk_inject_aw_eval.py"]) == 0; print("C5-BUN eval parts applied")
