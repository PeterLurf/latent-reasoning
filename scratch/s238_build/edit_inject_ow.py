"""§238 DAG-FREE (DF_v1 §1-2): boxes/inject_ow.py build_sets(mul_s_p) takes 2 (DF): every set under the DF law, TEST2 / TRANSFER / pairs / floors from NEW DF seeds,
shas["set_law"] = "df"; exports SEED_TEST2_DF and SEED_OPCLASS_DF (the OPCLASS-DF set: OW.Stream(SEED_OPCLASS_DF, frames=FRAME_TRAIN, **OWKW, mul_s_p=2), 3,000 at d 4,
no EXCL - score_df regenerates it exactly so). 0 / 1 unchanged. Each edit asserted once."""
p = "boxes/inject_ow.py"; s = open(p, encoding="utf-8").read()
E = [
('''SEED_TEST2_REG, SEED_TRANSFER_REG, SEED_PAIRS2_REG, SEED_FLOOR2_REG = 23800, 23801, 23802, 23804   # §236 REG sets (+1 of PAIRS2_REG = its stream; unused, grep 10-27)
''', '''SEED_TEST2_REG, SEED_TRANSFER_REG, SEED_PAIRS2_REG, SEED_FLOOR2_REG = 23800, 23801, 23802, 23804   # §236 REG sets (+1 of PAIRS2_REG = its stream; unused, grep 10-27)
SEED_TEST2_DF, SEED_TRANSFER_DF, SEED_PAIRS2_DF, SEED_FLOOR2_DF = 24010, 24011, 24012, 24014       # §238 DF sets (+1 of PAIRS2_DF = its stream; unused, grep 10-27)
SEED_OPCLASS_DF, OPCLASS_N, OPCLASS_D = 24015, 3000, 4                       # §238 OPCLASS-DF: Stream(SEED_OPCLASS_DF, frames=FRAME_TRAIN, **OWKW, mul_s_p=2), 3,000 at d 4, no EXCL
'''),
('''    assert mul_s_p in (0, 1); kw = {} if mul_s_p else {"mul_s_p": 0}
    sT, sX, sP, sF = (SEED_TEST2, SEED_TRANSFER, SEED_PAIRS2, SEED_FLOOR2) if mul_s_p else (SEED_TEST2_REG, SEED_TRANSFER_REG, SEED_PAIRS2_REG, SEED_FLOOR2_REG)''',
 '''    assert mul_s_p in (0, 1, 2), ("build_sets: mul_s_p", mul_s_p); kw = {} if mul_s_p == 1 else {"mul_s_p": mul_s_p}
    sT, sX, sP, sF = {1: (SEED_TEST2, SEED_TRANSFER, SEED_PAIRS2, SEED_FLOOR2), 0: (SEED_TEST2_REG, SEED_TRANSFER_REG, SEED_PAIRS2_REG, SEED_FLOOR2_REG),
                      2: (SEED_TEST2_DF, SEED_TRANSFER_DF, SEED_PAIRS2_DF, SEED_FLOOR2_DF)}[mul_s_p]'''),
('''    if not mul_s_p:
        shas["set_law"] = "reg"                                             # §236: the REG sets' shas carry the law tag (absent = the default law)''',
 '''    if mul_s_p != 1:
        shas["set_law"] = {0: "reg", 2: "df"}[mul_s_p]                       # §236 / §238: the law tag (absent = the default law)'''),
]
for o, n in E:
    assert s.count(o) == 1, ("edit not applied exactly once", s.count(o), o[:80]); s = s.replace(o, n)
g = 'if __name__ == "__main__":'; assert s.count(g) == 1; gi = s.index(g)   # opclass_set() goes above the main guard
s = s[:gi] + '''def opclass_set():
    """§238 OPCLASS-DF (DF_v1 §2): 3,000 d 4 items from OW.Stream(SEED_OPCLASS_DF, frames=FRAME_TRAIN, **OWKW, mul_s_p=2), no EXCL (score_df regenerates it so)."""
    s = OW.Stream(SEED_OPCLASS_DF, frames=OW.FRAME_TRAIN, **OWKW, mul_s_p=2); return [s.draw(OPCLASS_D) for _ in range(OPCLASS_N)]


''' + s[gi:]
open(p, "w", encoding="utf-8", newline="\n").write(s); print("inject_ow:", len(E), "edits + opclass_set()")
