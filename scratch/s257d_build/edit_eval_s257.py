"""§257 DISTRACT-CHAIN: in the eval parts source rename §256's part_insert_ho -> part_insert_ho_s256 (body unchanged) and append parts_s257.py (the new part_insert_ho =
part_insert_ho_s256 + the §257 held-out families); the dispatcher name 'insert_ho' unchanged. Then regenerate boxes/inject_aw_eval.py.
    python -B scratch/s257d_build/edit_eval_s257.py"""
import subprocess
import sys

q = "scratch/s234_build/inject_aw_eval_parts.py"; t = open(q, encoding="utf-8").read(); assert "def part_insert_ho_s256" not in t, "parts_s257 already applied"
o = "def part_insert_ho(m, g, sets, a, dev, bs):\n"; assert t.count(o) == 1, ("§257 rename", t.count(o))
t = t.replace(o, "def part_insert_ho_s256(m, g, sets, a, dev, bs):\n") + open("scratch/s257d_build/parts_s257.py", encoding="utf-8").read()
open(q, "w", encoding="utf-8", newline="\n").write(t)
assert subprocess.call([sys.executable, "-B", "scratch/s234_build/mk_inject_aw_eval.py"]) == 0; print("§257 eval edits applied")
