"""§238 DAG-FREE (DF_v1 §1, b14bb1c8): boxes/ow_words.py mul_s_p = 2 ("DF"): no forced mul_s (no ts draw); per step t >= 3 a mul_s with probability q_t
(0.179 t 3, 0.151 t 4, 0.147 t >= 5), partner j ~ U over the valid j in 1..t-2 (S_j >= 2, S_{t-1} >= 2), else (no valid partner, or the coin says no) a regular op
as REG; steps 1-2 REG. 0 / 1 unchanged (bitwise; the DF coin is drawn only when mul_s_p == 2). Each edit asserted once."""
p = "boxes/ow_words.py"; s = open(p, encoding="utf-8").read()
E = [
('''    mul_s_p (§236 REG-CHAIN, RC_v1 §1) in {0, 1}: 0 = no program has the forced mul_s (no ts draw at any d); 1 = unchanged. mul_s_p3 must be 1 when mul_s_p = 0.
''', '''    mul_s_p (§236 REG-CHAIN, RC_v1 §1) in {0, 1}: 0 = no program has the forced mul_s (no ts draw at any d); 1 = unchanged. mul_s_p3 must be 1 when mul_s_p = 0.
    mul_s_p 2 (§238 DAG-FREE, DF_v1 §1): no forced mul_s; per step t >= 3 a mul_s with probability DF_Q[t] (partner j ~ U over the valid j in 1..t-2), else -
    or when no partner is valid - a regular op as REG; mul_s_p3 must be 1.
'''),
('''    assert mul_s_p in (0, 1) and (mul_s_p == 1 or mul_s_p3 == 1.0), ("mul_s_p", mul_s_p, mul_s_p3)''',
 '''    assert mul_s_p in (0, 1, 2) and (mul_s_p == 1 or mul_s_p3 == 1.0), ("mul_s_p", mul_s_p, mul_s_p3)'''),
('''        ts = int(rng.integers(3, d + 1)) if (dag and d >= 3 and not no_ts and mul_s_p) else None''',
 '''        ts = int(rng.integers(3, d + 1)) if (dag and d >= 3 and not no_ts and mul_s_p == 1) else None   # §238: no forced step under 0 / 2'''),
('''            for _ in range(50):
                op = "mul_v" if d == 1 else reg[int(rng.integers(0, len(reg)))]''',
 '''            if mul_s_p == 2 and dag and t >= 3 and rng.random() < DF_Q[min(t, 5)]:   # §238 DF: the per-step mul_s coin (t >= 3)
                cand = [j for j in range(1, t - 1) if s[j] >= 2 and s[t - 1] >= 2]
                if cand:
                    j = cand[int(rng.integers(0, len(cand)))]; ops.append("mul_s"); bs.append(s[j]); js.append(j); raw.append(_NW_APPLY("mul_s", s[t - 1], s[j])); s.append(apply("mul_s", s[t - 1], s[j])); continue
            for _ in range(50):                                             # (§238: no valid partner -> this regular-op draw, as REG)
                op = "mul_v" if d == 1 else reg[int(rng.integers(0, len(reg)))]'''),
('''OPS_CHAIN = tuple(op for op in OPS_REG if op != "add_s")                  # §226 dag=False: every step reads only S_{t-1} (+ a constant / given)
''', '''OPS_CHAIN = tuple(op for op in OPS_REG if op != "add_s")                  # §226 dag=False: every step reads only S_{t-1} (+ a constant / given)
DF_Q = {3: 0.179, 4: 0.151, 5: 0.147}                                       # §238 DF_v1 §1: P(propose mul_s | t) (t >= 5 -> 5); realised ~0.125 after fallbacks
'''),
('''        self.mul_s_p = int(mul_s_p)                                         # §236: 1 = unchanged; 0 = REG (no forced mul_s)''',
 '''        self.mul_s_p = int(mul_s_p); assert self.mul_s_p in (0, 1, 2), ("mul_s_p", mul_s_p)   # §236: 0 = REG; §238: 2 = DF; 1 = unchanged'''),
]
for o, n in E:
    assert s.count(o) == 1, ("edit not applied exactly once", s.count(o), o[:80]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("ow_words:", len(E), "edits")
