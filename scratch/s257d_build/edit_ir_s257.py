"""§257 DISTRACT-CHAIN (DC2_v1): edits to boxes/insert_render.py (each asserted once) + the appended §257 code (ir_s257.py).
  §256 held-out metas gain c / v / j / op (+ op_idx, the NO* sentence indices) - meta only: no rng draw and no text changes (the §256 set shas re-asserted by gate_s257 equiv256).
    python -B scratch/s257d_build/edit_ir_s257.py"""
p = "boxes/insert_render.py"; s = open(p, encoding="utf-8").read(); assert "def distract_mix" not in s, "already applied"
EDITS = [
    ("        seq = seq[:s1] + giv + seq[s1:]; seq = _put(seq, [(\"NO0\", txt)], *GAP[pos]); meta.update(frame=f)\n",
     "        seq = seq[:s1] + giv + seq[s1:]; seq = _put(seq, [(\"NO0\", txt)], *GAP[pos])\n"
     "        meta.update(frame=f, op=op, v=vy, c=None if op == \"add_v\" else c, **({\"v2\": vz} if op == \"add_v\" else {}))   # §257: meta only\n"),
    ("\"after\", \"S%d\" % j); meta.update(frame=f, j=j)\n",
     "\"after\", \"S%d\" % j); meta.update(frame=f, j=j, op=\"add_c\", c=c, v=int(it[\"values\"][j]))\n"),
    ("            seq = _put(seq, g, \"after\", \"S4\")\n        meta.update(frame=f)\n",
     "            seq = _put(seq, g, \"after\", \"S4\")\n        meta.update(frame=f, op=\"add_c\", c=c, v=vy)\n"),
    ("        seq = _put(seq, ops, \"after\", \"S4\"); meta.update(frames=[f1, f2])\n",
     "        seq = _put(seq, ops, \"after\", \"S4\"); meta.update(frames=[f1, f2], op=\"add_c\", j=4, c=[c1, c2], v=int(it[\"values\"][4]))\n"),
    ("        seq = _put(seq, ops, *GAP[pos]); meta.update(frames=[f1, f2])\n",
     "        seq = _put(seq, ops, *GAP[pos]); meta.update(frames=[f1, f2], op=\"add_c\", c=[c1, c2], v=vy)\n"),
    ("        meta.update(pairs=m3)\n", "        meta.update(pairs=m3, op=\"add_c\")\n"),
    ("    assert OW.solve(o[\"text\"]) == it[\"answer\"], (\"held-out insert: solve != answer\", tag, it[\"sig\"]); meta[\"labels\"] = lab; return o\n",
     "    assert OW.solve(o[\"text\"]) == it[\"answer\"], (\"held-out insert: solve != answer\", tag, it[\"sig\"]); meta[\"labels\"] = lab\n"
     "    meta[\"op_idx\"] = [lab.index(x) for x in sorted(y for y in lab if y.startswith(\"NO\"))]; return o   # §257: the insert ops' sentence indices\n"),
]
for o, n in EDITS:
    assert s.count(o) == 1, ("§257 insert_render edit", s.count(o), o[:80]); s = s.replace(o, n)
s += open("scratch/s257d_build/ir_s257.py", encoding="utf-8").read()
open(p, "w", encoding="utf-8", newline="\n").write(s); print("§257 insert_render edits applied")
