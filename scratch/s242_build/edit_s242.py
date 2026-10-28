"""§242 NORM-COUNTER: append normctr_part.py to the eval parts source; add normctr to mk_inject_aw_eval.py's PARTS / AW_ONLY / dispatcher (each asserted once)."""
pp = "scratch/s234_build/inject_aw_eval_parts.py"; s = open(pp, encoding="utf-8").read(); add = open("scratch/s242_build/normctr_part.py", encoding="utf-8").read()
assert "def part_normctr" not in s, "already applied"; open(pp, "w", encoding="utf-8", newline="\n").write(s.rstrip("\n") + "\n" + add)
p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read()
E = [
    ('"opclass", "swap_prod", "lenstep")\\nAW_ONLY = ("probe", "ws", "opclass", "lenstep")\')',
     '"opclass", "swap_prod", "lenstep", "normctr")\\nAW_ONLY = ("probe", "ws", "opclass", "lenstep", "normctr")\')'),
    (''' '              "lenstep": lambda: part_lenstep(m, g, sets, a, dev, bs)}[p_]()'),   # §241 LEN-STEP''',
     ''' '              "lenstep": lambda: part_lenstep(m, g, sets, a, dev, bs), "normctr": lambda: part_normctr(m, g, sets, a, dev, bs)}[p_]()'),   # §241 LEN-STEP, §242 NORM-COUNTER'''),
]
for o, n in E:
    assert s.count(o) == 1, (s.count(o), o[:70]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("s242 edits applied", len(E))
