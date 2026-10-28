"""§241 LEN-STEP: add the lenstep part to mk_inject_aw_eval.py's PARTS / AW_ONLY / dispatcher edits (each asserted once)."""
p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read()
E = [
    ('"opclass", "swap_prod")\\nAW_ONLY = ("probe", "ws", "opclass")\')',
     '"opclass", "swap_prod", "lenstep")\\nAW_ONLY = ("probe", "ws", "opclass", "lenstep")\')'),
    (''' '              "opclass": lambda: part_opclass(m, g, sets, a, dev, bs), "swap_prod": lambda: part_swap_prod(m, g, sets, a, dev, bs)}[p_]()'),''',
     ''' '              "opclass": lambda: part_opclass(m, g, sets, a, dev, bs), "swap_prod": lambda: part_swap_prod(m, g, sets, a, dev, bs),\\n'
 '              "lenstep": lambda: part_lenstep(m, g, sets, a, dev, bs)}[p_]()'),   # §241 LEN-STEP'''),
]
for o, n in E:
    assert s.count(o) == 1, (s.count(o), o[:70]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("mk edited", len(E))
