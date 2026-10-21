"""apply G-T13 items 4 / 5 (and Note 219-4 (3)) to the scratch draft of boxes/textself_ft.py (never boxes/ itself)."""
p = "D:/Latent/scratch/_tmp/s219_fix/textself_ft.py"; s = open(p, encoding="utf-8").read()
NL = chr(92) + "n"                                                          # the two characters backslash-n as they appear in the source
R = [
    ("""  A7 --time_steps (G-T10) alternates self (odd steps; the next pool records in pool order, as vbank) and plain (even) and writes timing.json only""",
     """  A7 --time_steps (G-T10) alternates self (odd steps) and plain (even) and writes timing.json only. Note 219-4 (3): the forced self batch is the 64
     LONGEST records of the round's deepest n (T1 n 12, T2 n 16) at the round's fixed self micro, every odd step (deep_self, the same batch as G-T9),
     so the projection is the worst self shape and never an under-projection from pool order. Timing only:"""),
    ("""def coin_step(rng_s, p_self, bs):""",
     """def atomic_json(obj, path, **kw):
    \"\"\"json to path + '.tmp', then os.replace: a crash leaves the previous complete file (G-T13 item 4).\"\"\"
    with open(path + ".tmp", "w", newline="NLX") as fh:
        json.dump(obj, fh, **kw)
    os.replace(path + ".tmp", path)


def deep_self(pool, bs):
    \"\"\"the deepest self shape (G-T9 and, Note 219-4 (3), G-T10): the bs longest records of the pool's deepest n. -> (records, n_top).\"\"\"
    n_top = max(r_["n"] for r_ in pool)                                     # once (O(N); b310's lesson)
    return sorted([r_ for r_ in pool if r_["n"] == n_top], key=lambda r_: -len(r_["syms"]))[:bs], n_top


def coin_step(rng_s, p_self, bs):""".replace("NLX", NL)),
    ("""        n_top = max(r_["n"] for r_ in pool); deep = sorted([r_ for r_ in pool if r_["n"] == n_top], key=lambda r_: -len(r_["syms"]))[:a.bs]""",
     """        deep, n_top = deep_self(pool, a.bs)"""),
    ("""            draws_log = [d for d in json.load(open(os.path.join(a.out, "draws_log.json"))) if d["step"] < step0]
        log.write(""",
     """            draws_log = [d for d in json.load(open(os.path.join(a.out, "draws_log.json"))) if d["step"] < step0]
        if a.draws_log_steps and not a.time_steps:                          # G-T13 item 4: draws_log is written BEFORE each checkpoint, so it covers 1..step0-1
            assert [d["step"] for d in draws_log] == list(range(1, min(step0 - 1, a.draws_log_steps) + 1)), "§219: draws_log.json does not cover the resumed range"
        log.write("""),
    ("""    def draw_step(step):""",
     """    if a.time_steps and pool is not None:                                   # Note 219-4 (3): G-T10's forced self batch = the deepest self shape
        deep_recs, deep_n = deep_self(pool, a.bs); deep_its = [TS.self_item(r_) for r_ in deep_recs]
        deep_T = max(len(r_["syms"]) + 2 + TS.c_struct(r_["syms"]) for r_ in deep_recs)

    def draw_step(step):"""),
    ("""        if a.time_steps:                                                    # A7: forced alternation (G-TIME), U still drawn as vbank
            coin = step % 2 == 1; U = [int(rng_s.integers(0, 5)) for _ in range(a.bs)] if coin else None
        else:""",
     """        if a.time_steps:                                                    # A7 / Note 219-4 (3): forced alternation; odd steps = the deepest self shape
            coin = step % 2 == 1
            if coin:
                U = [int(rng_s.integers(0, 5)) for _ in range(a.bs)]            # (drawn as vbank's G-TIME; no cursor: the deep batch every odd step)
                return deep_its, True, None
            U = None
        else:"""),
    ("""            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({""",
     """            cp = os.path.join(a.out, "ckpt_last.pt")
            if draws_log:                                                   # G-T13 item 4: draws_log FIRST (atomic), then the checkpoint (tmp + os.replace), so
                atomic_json(draws_log, os.path.join(a.out, "draws_log.json"))   # every checkpoint on disk is covered by the draws_log on disk
            torch.save({"""),
    ("""                        **({"rng_s": rng_s.bit_generator.state, "self_state": dict(selfst)} if rng_s is not None else {})}, cp)
            if draws_log:
                json.dump(draws_log, open(os.path.join(a.out, "draws_log.json"), "w", newline="NLX"))
    if draws_log:
        json.dump(draws_log, open(os.path.join(a.out, "draws_log.json"), "w", newline="NLX"))""".replace("NLX", NL),
     """                        **({"rng_s": rng_s.bit_generator.state, "self_state": dict(selfst)} if rng_s is not None else {})}, cp + ".tmp")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            os.replace(cp + ".tmp", cp)
    if draws_log:
        atomic_json(draws_log, os.path.join(a.out, "draws_log.json"))"""),
    ("""              "forced": "odd steps self (pool order), even steps plain; sec_per_step = p_self-weighted mix" if pool is not None else "plain"}""",
     """              "forced": ("odd steps self = the %d longest records of the deepest n %d (T_max %d; Note 219-4 (3)), even steps plain; sec_per_step = p_self-weighted "
                         "mix" % (len(deep_its), deep_n, deep_T)) if pool is not None else "plain", **({"deep_n": deep_n, "deep_T_max": deep_T} if pool is not None else {})}"""),
]
for a_, b_ in R:
    assert s.count(a_) == 1, a_[:80]
    s = s.replace(a_, b_)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("ok", len(R))
