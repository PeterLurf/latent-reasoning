"""§247 READ-HALT: append rh_parts.py to the eval parts source; add c7 / halt to mk_inject_aw_eval.py's PARTS / AW_ONLY / dispatcher (each asserted once)."""
pp = "scratch/s234_build/inject_aw_eval_parts.py"; s = open(pp, encoding="utf-8").read(); add = open("scratch/s247_build/rh_parts.py", encoding="utf-8").read()
assert "def part_c7" not in s, "already applied"; open(pp, "w", encoding="utf-8", newline="\n").write(s.rstrip("\n") + "\n" + add)
p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read()
E = [
    ('"lenstep", "normctr", "ws_long")\\nAW_ONLY = ("probe", "ws", "opclass", "lenstep", "normctr", "ws_long")\')',
     '"lenstep", "normctr", "ws_long", "c7", "halt")\\nAW_ONLY = ("probe", "ws", "opclass", "lenstep", "normctr", "ws_long", "c7", "halt")\')'),
    (''''              "ws_long": lambda: part_ws_long(m, g, sets, a, dev, bs)}[p_]()'),   # §241 LEN-STEP, §242 NORM-COUNTER, §243 WS-LONG''',
     ''''              "ws_long": lambda: part_ws_long(m, g, sets, a, dev, bs), "c7": lambda: part_c7(m, g, sets, a, dev, bs),\\n'
 '              "halt": lambda: part_halt(m, g, sets, a, dev, bs)}[p_]()'),   # §241 LEN-STEP, §242 NORM-COUNTER, §243 WS-LONG, §247 READ-HALT'''),
]
for o, n in E:
    assert s.count(o) == 1, (s.count(o), o[:70]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("s247 eval edits applied", len(E))
