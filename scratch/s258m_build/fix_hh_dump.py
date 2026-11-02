"""§258 S0 fix (b398 step0 gate: TypeError in hh_dump): edits_base.py's hh_dump edit put its '# §258 S0 ...' comment mid-line, commenting out the chained
'; P = np.zeros(...)' (P then resolved to the module-level int). The comment moves to the end of the line (asserted once); regenerates boxes/inject_aw_eval.py.
    python -B scratch/s258m_build/fix_hh_dump.py"""
import subprocess
import sys

p = "scratch/s234_build/inject_aw_eval_parts.py"; s = open(p, encoding="utf-8").read()
o = ("    N = len(T); H = np.zeros((N, K + 1, m.g.D), dtype=np.float32)   # §258 S0: the model's width (was 768); P = np.zeros((N, K + 1), dtype=np.int64)")
n = ("    N = len(T); H = np.zeros((N, K + 1, m.g.D), dtype=np.float32); P = np.zeros((N, K + 1), dtype=np.int64)   # §258 S0: the model's width (was 768)")
assert s.count(o) == 1, s.count(o); s = s.replace(o, n); open(p, "w", encoding="utf-8", newline="\n").write(s)
assert subprocess.call([sys.executable, "-B", "scratch/s234_build/mk_inject_aw_eval.py"]) == 0; print("hh_dump fixed")
