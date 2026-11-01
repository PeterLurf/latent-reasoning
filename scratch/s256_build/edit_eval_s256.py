"""§256 DISTRACT: append parts_s256.py to the eval parts source (once); edit mk_inject_aw_eval.py (each asserted once): insert_ho / ws_deep in PARTS / AW_ONLY / the dispatcher.
    python -B scratch/s256_build/edit_eval_s256.py"""
p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read()
EDITS = [
    ('"frames", "test_perm")\\nAW_ONLY = ("test_perm", ', '"frames", "test_perm", "insert_ho", "ws_deep")\\nAW_ONLY = ("test_perm", "insert_ho", "ws_deep", '),
    ('"test_perm": lambda: part_test_perm(m, g, sets, a, dev, bs)}[p_]()',
     '"test_perm": lambda: part_test_perm(m, g, sets, a, dev, bs),\\n'
     '              "insert_ho": lambda: part_insert_ho(m, g, sets, a, dev, bs), "ws_deep": lambda: part_ws_deep(m, g, sets, a, dev, bs)}[p_]()'),
]
for o, n in EDITS:
    assert s.count(o) == 1, ("§256 eval edit", s.count(o), o[:80]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s)
q = "scratch/s234_build/inject_aw_eval_parts.py"; t = open(q, encoding="utf-8").read(); assert "def part_insert_ho" not in t, "parts_s256 already appended"
open(q, "w", encoding="utf-8", newline="\n").write(t + open("scratch/s256_build/parts_s256.py", encoding="utf-8").read()); print("§256 eval edits applied")
