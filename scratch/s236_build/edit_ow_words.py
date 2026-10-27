"""§236 REG-CHAIN (RC_v1 §1): boxes/ow_words.py gains mul_s_p in {0, 1} (default 1 = current, bitwise): at 0 no program has the forced mul_s (no ts draw).
Threaded through _draw_program / make / Stream / test_set / pair_candidates. Each edit asserted to apply once."""
p = "boxes/ow_words.py"; s = open(p, encoding="utf-8").read()
E = [
('''def _draw_program(rng, d, dag=True, mul_s_p3=1.0):''', '''def _draw_program(rng, d, dag=True, mul_s_p3=1.0, mul_s_p=1):'''),
('''    coin is drawn ONCE per program, before the restart loop, and only when mul_s_p3 < 1 (1.0 = no extra rng draw = the stream unchanged).
''', '''    coin is drawn ONCE per program, before the restart loop, and only when mul_s_p3 < 1 (1.0 = no extra rng draw = the stream unchanged).
    mul_s_p (§236 REG-CHAIN, RC_v1 §1) in {0, 1}: 0 = no program has the forced mul_s (no ts draw at any d); 1 = unchanged. mul_s_p3 must be 1 when mul_s_p = 0.
'''),
('''    restarts = 0; reg = OPS_REG if dag else OPS_CHAIN
    no_ts =''', '''    assert mul_s_p in (0, 1) and (mul_s_p == 1 or mul_s_p3 == 1.0), ("mul_s_p", mul_s_p, mul_s_p3)
    restarts = 0; reg = OPS_REG if dag else OPS_CHAIN
    no_ts ='''),
('''        ts = int(rng.integers(3, d + 1)) if (dag and d >= 3 and not no_ts) else None''',
 '''        ts = int(rng.integers(3, d + 1)) if (dag and d >= 3 and not no_ts and mul_s_p) else None'''),
('''def make(rng, d, frames=FRAME_TRAIN, stats=None, order_mode="forward", dag=True, mul_s_p3=1.0):''',
 '''def make(rng, d, frames=FRAME_TRAIN, stats=None, order_mode="forward", dag=True, mul_s_p3=1.0, mul_s_p=1):'''),
('''    ops, bs, js, s, ts, rs, raw = _draw_program(rng, d, dag, mul_s_p3)''', '''    ops, bs, js, s, ts, rs, raw = _draw_program(rng, d, dag, mul_s_p3, mul_s_p)'''),
('''    def __init__(self, seed, excl=(), frames=FRAME_TRAIN, order_mode="forward", dag=True, mul_s_p3=1.0):''',
 '''    def __init__(self, seed, excl=(), frames=FRAME_TRAIN, order_mode="forward", dag=True, mul_s_p3=1.0, mul_s_p=1):'''),
('''        self.mul_s_p3 = float(mul_s_p3)                                     # §235: 1.0 = unchanged (no coin, no extra stats keys)''',
 '''        self.mul_s_p3 = float(mul_s_p3)                                     # §235: 1.0 = unchanged (no coin, no extra stats keys)
        self.mul_s_p = int(mul_s_p)                                         # §236: 1 = unchanged; 0 = REG (no forced mul_s)'''),
('''            it = make(self.rng, d, self.frames, self.stats, self.order_mode, self.dag, self.mul_s_p3)''',
 '''            it = make(self.rng, d, self.frames, self.stats, self.order_mode, self.dag, self.mul_s_p3, self.mul_s_p)'''),
('''def test_set(per_d=1000, ds=range(1, 13), seed=SEED_TEST, order_mode="forward", dag=True):
    s = Stream(seed, frames=FRAME_ALL, order_mode=order_mode, dag=dag); return''',
 '''def test_set(per_d=1000, ds=range(1, 13), seed=SEED_TEST, order_mode="forward", dag=True, mul_s_p=1):
    s = Stream(seed, frames=FRAME_ALL, order_mode=order_mode, dag=dag, mul_s_p=mul_s_p); return'''),
('''def pair_candidates(per_d=3000, ds=range(1, 13), seed=SEED_PAIRS, order_mode="forward", dag=True):
    r = np.random.default_rng(seed); s = Stream(seed + 1, frames=FRAME_ALL, order_mode=order_mode, dag=dag); out = {}''',
 '''def pair_candidates(per_d=3000, ds=range(1, 13), seed=SEED_PAIRS, order_mode="forward", dag=True, mul_s_p=1):
    r = np.random.default_rng(seed); s = Stream(seed + 1, frames=FRAME_ALL, order_mode=order_mode, dag=dag, mul_s_p=mul_s_p); out = {}'''),
]
for o, n in E:
    assert s.count(o) == 1, ("edit not applied exactly once", s.count(o), o[:80]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("ow_words:", len(E), "edits")
