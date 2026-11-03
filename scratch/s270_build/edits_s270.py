"""§263 ORDER-FREE: E270E (inject_aw_eval run(), applied by mk_inject_aw_eval.py after E269E): the parts late_premise / name_swap / shuffle_graded (boxes/of_parts.py)
in PARTS, AW_ONLY and the dispatcher; every other part byte-identical. Regenerates boxes/inject_aw_eval.py.
    python -B scratch/s270_build/edits_s270.py"""
import subprocess
import sys

NEW = ("late_premise", "name_swap", "shuffle_graded")
E270E = [
    ('"stab_fresh", "stab", "nk_pilot")\nAW_ONLY = (', '"stab_fresh", "stab", "nk_pilot", %s)\nAW_ONLY = (' % ", ".join('"%s"' % p for p in NEW)),
    ('"c7", "halt", "halthead", "frames", "nk_pilot")\n', '"c7", "halt", "halthead", "frames", "nk_pilot", %s)\n' % ", ".join('"%s"' % p for p in NEW)),
    ('.part_nk_pilot(m, g, sets, a, dev, bs)}[p_]()   # §262 NK_v1 §4',
     '.part_nk_pilot(m, g, sets, a, dev, bs),   # §262 NK_v1 §4\n'
     + "".join('              "%s": lambda: __import__("boxes.of_parts", fromlist=["part_%s"]).part_%s(m, g, sets, a, dev, bs),\n' % (p, p, p) for p in NEW[:-1])
     + '              "%s": lambda: __import__("boxes.of_parts", fromlist=["part_%s"]).part_%s(m, g, sets, a, dev, bs)}[p_]()   # §263 ORDER-FREE' % (NEW[-1], NEW[-1], NEW[-1]))]
open("scratch/s270_build/edits_s270e.py", "w", encoding="utf-8", newline="\n").write('"""§263 ORDER-FREE: inject_aw_eval run() edits (applied by mk_inject_aw_eval.py after E269E)."""\nE270E = %r\n' % (E270E,))
me = "scratch/s234_build/mk_inject_aw_eval.py"; t = open(me, encoding="utf-8").read(); b_ = 'parts = open("scratch/s234_build/inject_aw_eval_parts.py", encoding="utf-8").read()\n'
assert t.count(b_) == 1 and "E270E" not in t
open(me, "w", encoding="utf-8", newline="\n").write(t.replace(b_, 'for o, n in __import__("runpy").run_path("scratch/s270_build/edits_s270e.py")["E270E"]:   # §263 ORDER-FREE parts\n'
                                                                 '    assert body.count(o) == 1, ("§263 eval edit not applied exactly once", body.count(o), o[:90]); body = body.replace(o, n)\n' + b_, 1))
assert subprocess.call([sys.executable, "-B", "scratch/s234_build/mk_inject_aw_eval.py"]) == 0
print("§263 eval edits applied")
