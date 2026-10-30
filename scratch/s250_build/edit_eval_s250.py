"""§250 FRAME-DIV: append fd_parts.py to the eval parts source (once); add frames to mk_inject_aw_eval.py's PARTS / AW_ONLY / dispatcher (each asserted once)."""
pp = "scratch/s234_build/inject_aw_eval_parts.py"; s = open(pp, encoding="utf-8").read(); add = open("scratch/s250_build/fd_parts.py", encoding="utf-8").read()
if "def part_frames" not in s:
    open(pp, "w", encoding="utf-8", newline="\n").write(s.rstrip("\n") + "\n" + add)
p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read()
E = [
    ('"halthead", "transfer")\\nAW_ONLY = ("probe", "ws", "opclass", "lenstep", "normctr", "ws_long", "c7", "halt", "halthead")',
     '"halthead", "transfer", "frames")\\nAW_ONLY = ("probe", "ws", "opclass", "lenstep", "normctr", "ws_long", "c7", "halt", "halthead", "frames")'),
    ('''"transfer": lambda: part_transfer(m, g, sets, a, dev, bs)}[p_]()'),''',
     '''"transfer": lambda: part_transfer(m, g, sets, a, dev, bs), "frames": lambda: part_frames(m, g, sets, a, dev, bs)}[p_]()'),'''),
]
for o, n in E:
    assert s.count(o) == 1, (s.count(o), o[:70]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("s250 eval edits applied", len(E))
