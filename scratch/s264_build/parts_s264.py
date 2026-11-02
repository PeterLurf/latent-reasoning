

SEED_SF = 25310                                                             # Note 259-11 C5-STABLE FRESH: the fresh REG test draw (unused elsewhere, grep 11-02)
SF_N, SF_DS, SF_KMAX = 1000, tuple(range(1, 9)), 14


def fit_sigs(a, sets):
    """the sigs of the halthead / halthead2 fit set (Stream(SEED_HH_FIT + 1), d ~ rng(SEED_HH_FIT), HH_N) and the halt_inc fit set (Stream(SEED_HI_FIT + 1), d ~
    rng(SEED_HI_FIT), HI_N), re-drawn exactly as part_halthead / part_halthead2 / part_halt_inc draw them."""
    kw = dict(frames=OW.FRAME_TRAIN, mul_s_p=law(a), **IO.OWKW); out = set()
    for seed, n in ((SEED_HH_FIT, HH_N), (SEED_HI_FIT, HI_N)):
        ds = np.random.default_rng(seed).integers(1, 5, size=n); s_ = OW.Stream(seed + 1, excl=sets["excl"], **kw); out |= {s_.draw(int(d))["sig"] for d in ds}
    return out


@torch.no_grad()
def part_stab_fresh(m, g, sets, a, dev, bs):
    """Note 259-11 C5-STABLE FRESH (score_c5c --fresh): a NEW REG test draw, Stream(SEED_SF + 1, excl = the eval sets incl. TEST2 (the training exclusion) | the
    halthead / halthead2 / halt_inc fit sigs), SF_N items per d 1..8, no duplicates within the draw; disjointness asserted; per d the set sha. Per item the decoder's pred
    at K' 0..14 (hh_dump_inc) + ans + sig. No halting decisions (the scorer recomputes the rule)."""
    kw = dict(frames=OW.FRAME_TRAIN, mul_s_p=law(a), **IO.OWKW); fs = fit_sigs(a, sets); t2 = {it["sig"] for d in sets["test2"] for it in sets["test2"][d]}
    ex = set(sets["excl"]) | fs | t2; s_ = OW.Stream(SEED_SF + 1, excl=ex, **kw); raw = {}; seen = set()
    for d in SF_DS:
        its = []; tries = 0
        while len(its) < SF_N:
            tries += 1; assert tries <= 200 * SF_N, ("C5-STABLE FRESH: d %d item space exhausted" % d, len(its))
            it = s_.draw(d)
            if it["sig"] not in seen:
                seen.add(it["sig"]); its.append(it)
        raw[d] = its
    assert not (seen & t2) and not (seen & fs) and not (seen & set(sets["excl"])), "C5-STABLE FRESH: a fresh item overlaps TEST2 / a fit set / the training exclusion"
    out = {"set_shas": {str(d): OW.set_sha(raw[d]) for d in SF_DS}, "excl_n": len(ex), "fit_sigs_n": len(fs)}
    for d in SF_DS:
        T = [AW.tok_aw(it, a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all")) for it in raw[d]]
        _, P, _ = hh_dump_inc(m, T, dev, bs, SF_KMAX)
        out[str(d)] = {"n": len(T), "items": {"pred": P.tolist(), "ans": [it["ans"] for it in T], "sig": [it["raw"]["sig"] for it in T]}}
    out["def"] = ("Note 259-11 C5-STABLE FRESH: a new REG draw (Stream(25311), d 1..8 x 1,000, sigs disjoint from TEST2, the halthead / halthead2 / halt_inc fit sets and "
                  "the training exclusion = the eval sets; no within-draw duplicates); per item the decoder's pred at K' 0..14 (seq_std passes, S from zero), ans, sig; fp32")
    return out
