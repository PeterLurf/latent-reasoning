"""§243 DEPTH-CURRIC: append wslong_part.py to the eval parts source; add ws_long to mk_inject_aw_eval.py's PARTS / AW_ONLY / dispatcher (each asserted once)."""
pp = "scratch/s234_build/inject_aw_eval_parts.py"; s = open(pp, encoding="utf-8").read(); add = open("scratch/s243_build/wslong_part.py", encoding="utf-8").read()
assert "def part_ws_long" not in s, "already applied"; open(pp, "w", encoding="utf-8", newline="\n").write(s.rstrip("\n") + "\n" + add)
p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read()
E = [
    ('"lenstep", "normctr")\\nAW_ONLY = ("probe", "ws", "opclass", "lenstep", "normctr")\')',
     '"lenstep", "normctr", "ws_long")\\nAW_ONLY = ("probe", "ws", "opclass", "lenstep", "normctr", "ws_long")\')'),
    ('''"normctr": lambda: part_normctr(m, g, sets, a, dev, bs)}[p_]()'),   # §241 LEN-STEP, §242 NORM-COUNTER''',
     '''"normctr": lambda: part_normctr(m, g, sets, a, dev, bs),\\n'
 '              "ws_long": lambda: part_ws_long(m, g, sets, a, dev, bs)}[p_]()'),   # §241 LEN-STEP, §242 NORM-COUNTER, §243 WS-LONG'''),
]
for o, n in E:
    assert s.count(o) == 1, (s.count(o), o[:70]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("s243 eval edits applied", len(E))
