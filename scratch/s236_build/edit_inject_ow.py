"""§236 REG-CHAIN (RC_v1 §1): boxes/inject_ow.py exports the REG set seeds and build_sets(mul_s_p=1) (default = current, bitwise); at 0 every set (TEST2, TRANSFER,
pairs / floors, and the legacy exclusion sets TE6 / PC6 / ED7) is drawn under the REG law, TEST2 / TRANSFER / pairs / floors from NEW seeds. Each edit asserted once."""
p = "boxes/inject_ow.py"; s = open(p, encoding="utf-8").read()
E = [
('''SEED_CAL, SEED_MEMCHECK = 22996, 22997                                      # g_add calibration / memory-check items (unused elsewhere, grep 10-23)
''', '''SEED_CAL, SEED_MEMCHECK = 22996, 22997                                      # g_add calibration / memory-check items (unused elsewhere, grep 10-23)
SEED_TEST2_REG, SEED_TRANSFER_REG, SEED_PAIRS2_REG, SEED_FLOOR2_REG = 23800, 23801, 23802, 23804   # §236 REG sets (+1 of PAIRS2_REG = its stream; unused, grep 10-27)
'''),
('''def build_sets():
    """the §228 evaluation sets''', '''def build_sets(mul_s_p=1):
    """§236: mul_s_p 0 = every set under the REG law (no forced mul_s) with the REG seeds for TEST2 / TRANSFER / pairs / floors (default 1 = unchanged).
    the §228 evaluation sets'''),
('''    s = OW.Stream(SEED_TEST2, frames=OW.FRAME_TRAIN, **OWKW); test2 = {d: [s.draw(d) for _ in range(TEST_N)] for d in TEST_DS}
    s = OW.Stream(SEED_TRANSFER, frames=(6, 7), **OWKW); transfer = {d: [s.draw(d) for _ in range(TEST_N)] for d in TEST_DS}
    r = np.random.default_rng(SEED_PAIRS2); s = OW.Stream(SEED_PAIRS2 + 1, frames=OW.FRAME_TRAIN, **OWKW); rf = np.random.default_rng(SEED_FLOOR2)''',
 '''    assert mul_s_p in (0, 1); kw = {} if mul_s_p else {"mul_s_p": 0}
    sT, sX, sP, sF = (SEED_TEST2, SEED_TRANSFER, SEED_PAIRS2, SEED_FLOOR2) if mul_s_p else (SEED_TEST2_REG, SEED_TRANSFER_REG, SEED_PAIRS2_REG, SEED_FLOOR2_REG)
    s = OW.Stream(sT, frames=OW.FRAME_TRAIN, **OWKW, **kw); test2 = {d: [s.draw(d) for _ in range(TEST_N)] for d in TEST_DS}
    s = OW.Stream(sX, frames=(6, 7), **OWKW, **kw); transfer = {d: [s.draw(d) for _ in range(TEST_N)] for d in TEST_DS}
    r = np.random.default_rng(sP); s = OW.Stream(sP + 1, frames=OW.FRAME_TRAIN, **OWKW, **kw); rf = np.random.default_rng(sF)'''),
('''    TE6 = OW.test_set(**OWKW); PC6 = OW.pair_candidates(**OWKW); ex0 = {it["sig"] for d, its in TE6.items() for it in its}   # §226 / §227 (codi_ps's order)
    es = OW.Stream(22950, ex0, frames=OW.FRAME_TRAIN, **OWKW); ED7 = [es.draw(d) for d in range(1, 9) for _ in range(200)]''',
 '''    TE6 = OW.test_set(**OWKW, **kw); PC6 = OW.pair_candidates(**OWKW, **kw); ex0 = {it["sig"] for d, its in TE6.items() for it in its}   # §226 / §227 (codi_ps's order)
    es = OW.Stream(22950, ex0, frames=OW.FRAME_TRAIN, **OWKW, **kw); ED7 = [es.draw(d) for d in range(1, 9) for _ in range(200)]'''),
('''    return {"test2": test2, "transfer": transfer, "pairs": pairs, "floors": floors, "pair_stats": pst, "excl": excl, "shas": shas}''',
 '''    if not mul_s_p:
        shas["set_law"] = "reg"                                             # §236: the REG sets' shas carry the law tag (absent = the default law)
    return {"test2": test2, "transfer": transfer, "pairs": pairs, "floors": floors, "pair_stats": pst, "excl": excl, "shas": shas}'''),
]
for o, n in E:
    assert s.count(o) == 1, ("edit not applied exactly once", s.count(o), o[:80]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("inject_ow:", len(E), "edits")
