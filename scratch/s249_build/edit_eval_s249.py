"""§249 FRAME-TRANSFER: append ft_parts.py to the eval parts source (once); add transfer to mk_inject_aw_eval.py's PARTS and dispatcher (NOT AW_ONLY: CD-AN runs it too;
the driver calls the part directly, since run() keeps only acc for direct runs)."""
pp = "scratch/s234_build/inject_aw_eval_parts.py"; s = open(pp, encoding="utf-8").read(); add = open("scratch/s249_build/ft_parts.py", encoding="utf-8").read()
if "def part_transfer" not in s:
    open(pp, "w", encoding="utf-8", newline="\n").write(s.rstrip("\n") + "\n" + add)
p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read()
E = [
    ('"ws_long", "c7", "halt", "halthead")\\nAW_ONLY', '"ws_long", "c7", "halt", "halthead", "transfer")\\nAW_ONLY'),
    ('''"halthead": lambda: part_halthead(m, g, sets, a, dev, bs)}[p_]()'),   # §241-§243, §247 READ-HALT, §248 HALT-HEAD''',
     '''"halthead": lambda: part_halthead(m, g, sets, a, dev, bs),\\n'
 '              "transfer": lambda: part_transfer(m, g, sets, a, dev, bs)}[p_]()'),   # §241-§243, §247 READ-HALT, §248 HALT-HEAD, §249 FRAME-TRANSFER'''),
]
for o, n in E:
    assert s.count(o) == 1, (s.count(o), o[:70]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("s249 eval edits applied", len(E))
