"""§238 DAG-FREE (DF_v1 §1-2): the eval parts take the run's law 0 / 1 / 2 (law() RAISES on anything else); the eval-own streams (swap_k d 2, swap_kd d 3..d_max,
probe) use per-law seeds (DF = new); REG's d 3 t 2 extra stays REG-only; new parts opclass (WS-format readout on OPCLASS-DF) and swap_prod (PROD-SWAP);
law_eval_sigs() (= reg_eval_sigs, kept as an alias) covers DF (+ OPCLASS-DF + PROD-SWAP pairs) for the training EXCL. Edits the generator input; each asserted once."""
p = "scratch/s234_build/inject_aw_eval_parts.py"; s = open(p, encoding="utf-8").read()
E = [
('''SEED_SWAP_REC_REG, SEED_SWAP_DON_REG, SEED_PROBE_REG = 23810, 23811, 23812  # §236 REG (mul_s_p 0) eval-own streams: new seeds (unused, grep 10-27)


def reg_kw(a):
    """§236: the OW law kwargs of the run (args mul_s_p 0 -> REG; absent / 1 -> {} = unchanged)."""
    return {"mul_s_p": 0} if int(a.get("mul_s_p", 1)) == 0 else {}
''', '''SEED_SWAP_REC_REG, SEED_SWAP_DON_REG, SEED_PROBE_REG = 23810, 23811, 23812  # §236 REG (mul_s_p 0) eval-own streams: new seeds (unused, grep 10-27)
SEED_SWAP_REC_DF, SEED_SWAP_DON_DF, SEED_PROBE_DF = 24020, 24021, 24022     # §238 DF (mul_s_p 2) eval-own streams: new seeds (unused, grep 10-27)
SEED_PROD_REC_DF, SEED_PROD_DON_DF = 24050, 24051                          # §238 PROD-SWAP (DF d 3, step 3 = mul_s): recipients / donors' redraw rng


def law(a):
    """§236 / §238: the run's OW law (args mul_s_p; absent = 1). RAISES on anything but 0 / 1 / 2 (DF_v1 §1: no silent fallback)."""
    v = a.get("mul_s_p", 1)
    if v not in (0, 1, 2):
        raise ValueError("unknown mul_s_p law %r (0 REG / 1 default / 2 DF)" % (v,))
    return int(v)


def reg_kw(a):
    """the OW law kwargs of the run: {} for the default law, else {"mul_s_p": law} (§236 REG 0, §238 DF 2)."""
    v = law(a); return {} if v == 1 else {"mul_s_p": v}
'''),
('''    sr, sd = (SEED_SWAP_REC_REG, SEED_SWAP_DON_REG) if reg_kw(a) else (SEED_SWAP_REC, SEED_SWAP_DON)   # §236 REG: new seeds + the REG law''',
 '''    sr, sd = {1: (SEED_SWAP_REC, SEED_SWAP_DON), 0: (SEED_SWAP_REC_REG, SEED_SWAP_DON_REG), 2: (SEED_SWAP_REC_DF, SEED_SWAP_DON_DF)}[law(a)]   # per-law seeds'''),
('''    s = OW.Stream(SEED_PROBE_REG if reg_kw(a) else SEED_PROBE, sets["excl"], frames=OW.FRAME_TRAIN, **IO.OWKW, **reg_kw(a)); raw = [s.draw(2) for _ in range(PROBE_N)]''',
 '''    s = OW.Stream({1: SEED_PROBE, 0: SEED_PROBE_REG, 2: SEED_PROBE_DF}[law(a)], sets["excl"], frames=OW.FRAME_TRAIN, **IO.OWKW, **reg_kw(a)); raw = [s.draw(2) for _ in range(PROBE_N)]'''),
('''SEED_SWAPD2_REC_REG, SEED_SWAPD2_DON_REG = 23840, 23850                    # §236 REG d 3 at t 2 (by_d.3.swap_mid_t2, descriptive): (+ d), new seeds
''', '''SEED_SWAPD2_REC_REG, SEED_SWAPD2_DON_REG = 23840, 23850                    # §236 REG d 3 at t 2 (by_d.3.swap_mid_t2, descriptive): (+ d), new seeds
SEED_SWAPD_REC_DF, SEED_SWAPD_DON_DF = 24030, 24040                        # §238 DF: (+ d), new seeds
'''),
('''    sr, sd = seeds if seeds is not None else ((SEED_SWAPD_REC_REG + d, SEED_SWAPD_DON_REG + d) if reg_kw(a) else (SEED_SWAPD_REC + d, SEED_SWAPD_DON + d))''',
 '''    sr, sd = seeds if seeds is not None else {1: (SEED_SWAPD_REC + d, SEED_SWAPD_DON + d), 0: (SEED_SWAPD_REC_REG + d, SEED_SWAPD_DON_REG + d),
                                              2: (SEED_SWAPD_REC_DF + d, SEED_SWAPD_DON_DF + d)}[law(a)]'''),
('''        if d == 3 and reg_kw(a):                                            # §236 (RC_v1 §1): REG d 3 also at t 2 (SWAP-MID only, descriptive)''',
 '''        if d == 3 and law(a) == 0:                                          # §236 (RC_v1 §1): REG d 3 also at t 2 (SWAP-MID only, descriptive; REG only)'''),
('''def reg_eval_sigs(a, excl):
    """§236: {set name: sigs (recipients + donors / items)} of the eval module's OWN REG streams for a run with args a (d_max) - swap_k d 2, swap_kd d 3..d_max
    (+ d 3 at t 2), probe (drawn excluding excl = build_sets(0)["excl"]); the training EXCL and the disjointness gate use it. REG only."""
    assert reg_kw(a), "reg_eval_sigs: REG runs only"
''', '''def law_eval_sigs(a, excl):
    """§236 / §238: {set name: sigs (recipients + donors / items)} of the eval module's OWN streams under the run's non-default law (args a: d_max, mul_s_p) -
    swap_k d 2, swap_kd d 3..d_max (+ REG: d 3 at t 2), probe (drawn excluding excl = build_sets(law)["excl"]); DF also OPCLASS-DF and the PROD-SWAP pairs.
    The training EXCL and the disjointness gates use it."""
    assert law(a) != 1, "law_eval_sigs: non-default laws only"
'''),
('''    if int(a["d_max"]) >= 3:
        _, _, kp, _, _ = swap_pairs_d(a, 3, 2, (SEED_SWAPD2_REC_REG + 3, SEED_SWAPD2_DON_REG + 3)); out["swap_kd_d3_t2"] = {x["sig"] for p_ in kp for x in p_[:2]}
    s = OW.Stream(SEED_PROBE_REG, excl, frames=OW.FRAME_TRAIN, **IO.OWKW, **reg_kw(a)); out["probe"] = {s.draw(2)["sig"] for _ in range(PROBE_N)}
    return out
''', '''    if int(a["d_max"]) >= 3 and law(a) == 0:
        _, _, kp, _, _ = swap_pairs_d(a, 3, 2, (SEED_SWAPD2_REC_REG + 3, SEED_SWAPD2_DON_REG + 3)); out["swap_kd_d3_t2"] = {x["sig"] for p_ in kp for x in p_[:2]}
    s = OW.Stream({0: SEED_PROBE_REG, 2: SEED_PROBE_DF}[law(a)], excl, frames=OW.FRAME_TRAIN, **IO.OWKW, **reg_kw(a)); out["probe"] = {s.draw(2)["sig"] for _ in range(PROBE_N)}
    if law(a) == 2:                                                         # §238: the DF-only eval sets
        out["opclass"] = {it["sig"] for it in IO.opclass_set()}
        _, _, kp, _ = prod_pairs(a); out["swap_prod"] = {x["sig"] for p_ in kp for x in p_[:2]}
    return out


reg_eval_sigs = law_eval_sigs                                               # §236 name (inject_aw's reg_excl calls it)


def prod_pairs(a):
    """§238 PROD-SWAP pairs (DF_v1 §2): d 3 recipients from OW.Stream(24050, frames 0-5, topo, dag, mul_s_p 2) conditioned on step 3 = mul_s (rejection; its partner is
    j = 1), donors OW.redraw(rng 24051) (same skeleton, so step 3 = mul_s on j = 1 too); kept iff own != donor final and cb not in {own, donor final}, cb = cf_b(2) =
    donor S_2 x recipient S_1; the first 1,000 (3,000 tries); cues equal (asserted). -> tokenised recipients, donors, kept [(rec, don, cb)], tried."""
    assert law(a) == 2, "PROD-SWAP: DF runs only"
    r2 = np.random.default_rng(SEED_PROD_DON_DF); s2 = OW.Stream(SEED_PROD_REC_DF, frames=OW.FRAME_TRAIN, **IO.OWKW, mul_s_p=2); kept = []; tried = 0
    while len(kept) < SWAP_CAP and tried < SWAP_TRIES:
        rec = s2.draw(3)
        if rec["ops"][2] != "mul_s":
            continue
        don = OW.redraw(r2, rec)
        if don is None or don["sig"] == rec["sig"]:
            continue
        tried += 1; own, dfin, cb = rec["answer"], don["answer"], IO.cf(rec, don, 2, "b")
        assert don["ops"][2] == "mul_s" and rec["src_j"][2] == don["src_j"][2] == 1 and cb == OW.apply("mul_s", don["values"][2], rec["values"][1]), "PROD-SWAP law"
        if own != dfin and cb not in (own, dfin):
            kept.append((rec, don, cb))
    for rec, don, _ in kept:
        assert all(ST.cue_str(rec, c) == ST.cue_str(don, c) for c in (1, 2, 3)), "PROD-SWAP: the cues must be equal (same skeleton)"
    tk = lambda it: AW.tok_aw(it, a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all"))  # noqa: E731
    return [tk(p_[0]) for p_ in kept], [tk(p_[1]) for p_ in kept], kept, tried


@torch.no_grad()
def part_swap_prod(m, g, sets, a, dev, bs):
    """§238 PROD-SWAP (DF_v1 §2): the recipient's memory after pass 2 := the donor's S_2, then passes 3..K' on the recipient's own inputs (pass 3 = its own cue);
    answers at K' = 3 (= d; the flat fields) and think_k (by_k). follow_dfin = both factors from memory; follow_cb = S_2 from memory, the partner S_1 recomputed from
    the text; follow_U = follow_dfin + follow_cb (disjoint by the kept rule); floor = the share in {dfin, cb} with S_2 := the donors' mean S_2 (MEAN at pass 2)."""
    K = a["think_k"]; R, Dn, kept, tried = prod_pairs(a); own = np.array([p_[0]["answer"] for p_ in kept]); dfin = np.array([p_[1]["answer"] for p_ in kept])
    cb = np.array([p_[2] for p_ in kept]); ks = sorted({3, K}); S2 = grab_states(m, Dn, dev, bs, (2,))[2]; m2 = S2.mean(0)
    I, _ = passes(m, R, dev, bs, set(ks))
    M, _ = passes(m, R, dev, bs, set(ks), edit=lambda t_, S, sl: S2[sl].to(dev, S.dtype) if t_ == 2 else None)
    F, _ = passes(m, R, dev, bs, set(ks), edit=lambda t_, S, sl: m2.to(dev, S.dtype)[None].expand(S.shape[0], *m2.shape).contiguous() if t_ == 2 else None)
    inU = lambda p: (p == dfin) | (p == cb)  # noqa: E731
    by_k = {str(k): {"follow_dfin": float((M[k] == dfin).mean()), "follow_cb": float((M[k] == cb).mean()), "follow_U": float(inU(M[k]).mean()),
                     "floor": float(inU(F[k]).mean()), "stay_own": float((M[k] == own).mean()), "intact_acc": float((I[k] == own).mean())} for k in ks}
    return {"n_pairs": len(kept), "tried": tried, **by_k["3"], "k_flat": 3, "by_k": by_k,
            "items": {"rec_sig": [p_[0]["sig"] for p_ in kept], "don_sig": [p_[1]["sig"] for p_ in kept], "own": own.tolist(), "dfin": dfin.tolist(), "cb": cb.tolist(),
                      "intact": {str(k): I[k].tolist() for k in ks}, "swap": {str(k): M[k].tolist() for k in ks}, "mean": {str(k): F[k].tolist() for k in ks}},
            "def": ("PROD-SWAP (§238 DF_v1 §2): DF d 3 pairs with step 3 = mul_s (S_3 = S_2 x S_1), recipients Stream(24050) / donors redraw rng 24051, kept iff own != dfin and "
                    "cb not in {own, dfin} (cb = donor S_2 x recipient S_1), cap 1,000; S after pass 2 := donor's; flat fields at K' = 3, by_k also K' = think_k; "
                    "floor: S after pass 2 := the donors' mean S_2")}


@torch.no_grad()
def part_opclass(m, g, sets, a, dev, bs):
    """§238 OPCLASS-DF (DF_v1 §2): the WS-READOUT (trunk argmax at ':' of pass t vs v_min(t, 4)) on the 3,000 d 4 items of inject_ow.opclass_set(), in the ws item
    format: items {sig, ans, pred_by_pass {"1".."8"}, v}; by_pass acc. For the pass-4 per-class readings (the scorer splits mul_s by j)."""
    K = a["think_k"]; its = [AW.tok_aw(it, a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all")) for it in IO.opclass_set()]
    r_, pv = AW.ws_readout(m, {4: its}, dev, K, bs, per_item=True)
    return {"by_pass": r_["4"], "items": items_rec(its, {"pred_by_pass": pv["4"], "v": [[int(it["raw"]["values"][min(t, 4)]) for t in range(1, K + 1)] for it in its]}),
            "n": len(its), "def": "OPCLASS-DF (§238): Stream(24015, frames 0-5, topo, dag, mul_s_p 2) x 3,000 at d 4 (no EXCL; inject_ow.opclass_set), WS-READOUT per pass, fp32"}
'''),
]
for o, n in E:
    assert s.count(o) == 1, ("edit not applied exactly once", s.count(o), o[:80]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("eval parts:", len(E), "edits")
p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read()
E2 = [
("""\"swap_k\", \"swap_1\", \"probe\", \"ws\", \"lp5h\", \"swap_kd\")\\nAW_ONLY = (\"probe\", \"ws\")'),""",
 """\"swap_k\", \"swap_1\", \"probe\", \"ws\", \"lp5h\", \"swap_kd\", \"opclass\", \"swap_prod\")\\nAW_ONLY = (\"probe\", \"ws\", \"opclass\")'),"""),
(""""lp5h": lambda: part_lp5h(m, g, sets, a, dev, bs), "swap_kd": lambda: part_swap_kd(m, g, sets, a, dev, bs)}[p_]()'),""",
 """"lp5h": lambda: part_lp5h(m, g, sets, a, dev, bs), "swap_kd": lambda: part_swap_kd(m, g, sets, a, dev, bs),\\n'
 '              "opclass": lambda: part_opclass(m, g, sets, a, dev, bs), "swap_prod": lambda: part_swap_prod(m, g, sets, a, dev, bs)}[p_]()'),"""),
("""        sets = IO.build_sets(mul_s_p=int(a.get("mul_s_p", 1)))   # §236: the run\\'s law'),""",
 """        sets = IO.build_sets(mul_s_p=law(a))   # §236 / §238: the run\\'s law (law() raises on unknown values)'),"""),
]
for o, n in E2:
    assert s.count(o) == 1, ("mk edit not applied exactly once", s.count(o), o[:80]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("mk_inject_aw_eval:", len(E2), "edits")
