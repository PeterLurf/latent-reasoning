"""§236 REG-CHAIN (RC_v1 §1): the eval parts' own streams (swap_k d 2 pairs, swap_kd d 3..d_max pairs, probe) under the REG law with NEW seeds when the run's
args carry mul_s_p 0 (default 1 = unchanged); swap_kd at d 3 under REG also runs t 2 (by_d.3.swap_mid_t2, descriptive); reg_eval_sigs() = every REG eval-own set's
sigs (the training EXCL and the disjointness gate). Edits scratch/s234_build/inject_aw_eval_parts.py (the generator input); each asserted once."""
p = "scratch/s234_build/inject_aw_eval_parts.py"; s = open(p, encoding="utf-8").read()
E = [
('''SEED_PROBE = 23471; PROBE_N = 3000; P = 11                                  # aw_gate's G-AW0 item stream (d 2, EXCL), 60 / 20 / 20 by sig (rng 0)
''', '''SEED_PROBE = 23471; PROBE_N = 3000; P = 11                                  # aw_gate's G-AW0 item stream (d 2, EXCL), 60 / 20 / 20 by sig (rng 0)
SEED_SWAP_REC_REG, SEED_SWAP_DON_REG, SEED_PROBE_REG = 23810, 23811, 23812  # §236 REG (mul_s_p 0) eval-own streams: new seeds (unused, grep 10-27)


def reg_kw(a):
    """§236: the OW law kwargs of the run (args mul_s_p 0 -> REG; absent / 1 -> {} = unchanged)."""
    return {"mul_s_p": 0} if int(a.get("mul_s_p", 1)) == 0 else {}
'''),
('''    r2 = np.random.default_rng(SEED_SWAP_DON); s2 = OW.Stream(SEED_SWAP_REC, frames=OW.FRAME_TRAIN, **IO.OWKW); kept = []; tried = 0
    while len(kept) < SWAP_CAP and tried < SWAP_TRIES:
        rec = s2.draw(2)''',
 '''    sr, sd = (SEED_SWAP_REC_REG, SEED_SWAP_DON_REG) if reg_kw(a) else (SEED_SWAP_REC, SEED_SWAP_DON)   # §236 REG: new seeds + the REG law
    r2 = np.random.default_rng(sd); s2 = OW.Stream(sr, frames=OW.FRAME_TRAIN, **IO.OWKW, **reg_kw(a)); kept = []; tried = 0
    while len(kept) < SWAP_CAP and tried < SWAP_TRIES:
        rec = s2.draw(2)'''),
('''    s = OW.Stream(SEED_PROBE, sets["excl"], frames=OW.FRAME_TRAIN, **IO.OWKW); raw = [s.draw(2) for _ in range(PROBE_N)]''',
 '''    s = OW.Stream(SEED_PROBE_REG if reg_kw(a) else SEED_PROBE, sets["excl"], frames=OW.FRAME_TRAIN, **IO.OWKW, **reg_kw(a)); raw = [s.draw(2) for _ in range(PROBE_N)]'''),
('''SEED_SWAPD_REC, SEED_SWAPD_DON = 23260, 23270                              # §234 P-4 (Note 234-5 b): recipients Stream(23260 + d), donors redraw rng (23270 + d), d 3 / 4
''', '''SEED_SWAPD_REC, SEED_SWAPD_DON = 23260, 23270                              # §234 P-4 (Note 234-5 b): recipients Stream(23260 + d), donors redraw rng (23270 + d), d 3 / 4
SEED_SWAPD_REC_REG, SEED_SWAPD_DON_REG = 23820, 23830                      # §236 REG: (+ d), new seeds
SEED_SWAPD2_REC_REG, SEED_SWAPD2_DON_REG = 23840, 23850                    # §236 REG d 3 at t 2 (by_d.3.swap_mid_t2, descriptive): (+ d), new seeds
'''),
('''def swap_pairs_d(a, d):''', '''def swap_pairs_d(a, d, t=None, seeds=None):'''),
('''    t = swap_t(d); r2 = np.random.default_rng(SEED_SWAPD_DON + d); s2 = OW.Stream(SEED_SWAPD_REC + d, frames=OW.FRAME_TRAIN, **IO.OWKW); kept = []; tried = 0''',
 '''    t = swap_t(d) if t is None else t                                      # §236: t / seeds overridable (REG d 3 at t 2); default = P-4's
    sr, sd = seeds if seeds is not None else ((SEED_SWAPD_REC_REG + d, SEED_SWAPD_DON_REG + d) if reg_kw(a) else (SEED_SWAPD_REC + d, SEED_SWAPD_DON + d))
    r2 = np.random.default_rng(sd); s2 = OW.Stream(sr, frames=OW.FRAME_TRAIN, **IO.OWKW, **reg_kw(a)); kept = []; tried = 0'''),
('''        out["items"][str(d)] = {"rec_sig": [p_[0]["sig"] for p_ in kept], "don_sig": [p_[1]["sig"] for p_ in kept], "own": own.tolist(), "dfin": dfin.tolist(), "ca": ca.tolist(),
                                "cb": cb.tolist(), "intact": {str(k): I[k].tolist() for k in ks}, "swap_k": sw.tolist(), "mean_k": mn.tolist(),
                                "swap_mid": {str(k): M_[k].tolist() for k in ks}, "mean_mid": {str(k): F_[k].tolist() for k in ks}}
''', '''        out["items"][str(d)] = {"rec_sig": [p_[0]["sig"] for p_ in kept], "don_sig": [p_[1]["sig"] for p_ in kept], "own": own.tolist(), "dfin": dfin.tolist(), "ca": ca.tolist(),
                                "cb": cb.tolist(), "intact": {str(k): I[k].tolist() for k in ks}, "swap_k": sw.tolist(), "mean_k": mn.tolist(),
                                "swap_mid": {str(k): M_[k].tolist() for k in ks}, "mean_mid": {str(k): F_[k].tolist() for k in ks}}
        if d == 3 and reg_kw(a):                                            # §236 (RC_v1 §1): REG d 3 also at t 2 (SWAP-MID only, descriptive)
            R2, Dn2, kp2, tr2, t2 = swap_pairs_d(a, 3, 2, (SEED_SWAPD2_REC_REG + 3, SEED_SWAPD2_DON_REG + 3))
            out["by_d"]["3"].update({"n_pairs_t2": len(kp2), "tried_t2": tr2})
            if kp2:
                own2 = np.array([p_[0]["answer"] for p_ in kp2]); ca2 = np.array([p_[2] for p_ in kp2]); cb2 = np.array([p_[3] for p_ in kp2])
                S2_ = grab_states(m, Dn2, dev, bs, (t2,))[t2]; mt2 = S2_.mean(0)
                M2, _ = passes(m, R2, dev, bs, set(ks), edit=lambda t_, S, sl: S2_[sl].to(dev, S.dtype) if t_ == t2 else None)
                F2, _ = passes(m, R2, dev, bs, set(ks), edit=lambda t_, S, sl: mt2.to(dev, S.dtype)[None].expand(S.shape[0], *mt2.shape).contiguous() if t_ == t2 else None)
                inM2 = lambda p: (p == ca2) | (p == cb2)  # noqa: E731
                out["by_d"]["3"]["swap_mid_t2"] = {str(k): {"follow_U": float(inM2(M2[k]).mean()), "floor": float(inM2(F2[k]).mean()), "follow_ca": float((M2[k] == ca2).mean()),
                                                            "follow_cb": float((M2[k] == cb2).mean()), "stay_own": float((M2[k] == own2).mean())} for k in ks}
                out["items"]["3"]["t2"] = {"rec_sig": [p_[0]["sig"] for p_ in kp2], "don_sig": [p_[1]["sig"] for p_ in kp2], "own": own2.tolist(),
                                           "dfin": [p_[1]["answer"] for p_ in kp2], "ca": ca2.tolist(), "cb": cb2.tolist(),
                                           "swap_mid": {str(k): M2[k].tolist() for k in ks}, "mean_mid": {str(k): F2[k].tolist() for k in ks}}
'''),
('''                  "later passes, U = {cf_a(t), cf_b(t)}, floor = MEAN at t (the donors' mean S_t); answers at K' = d and think_k")
    return out
''', '''                  "later passes, U = {cf_a(t), cf_b(t)}, floor = MEAN at t (the donors' mean S_t); answers at K' = d and think_k. §236 REG (args mul_s_p 0): every pair stream "
                  "under the REG law from new seeds (recipients 23820 + d, donors 23830 + d); d 3 also at t 2 (by_d.3.swap_mid_t2 / n_pairs_t2, items.3.t2; seeds 23843 / 23853)")
    return out


def reg_eval_sigs(a, excl):
    """§236: {set name: sigs (recipients + donors / items)} of the eval module's OWN REG streams for a run with args a (d_max) - swap_k d 2, swap_kd d 3..d_max
    (+ d 3 at t 2), probe (drawn excluding excl = build_sets(0)["excl"]); the training EXCL and the disjointness gate use it. REG only."""
    assert reg_kw(a), "reg_eval_sigs: REG runs only"
    out = {}; _, _, kp, _ = swap_pairs(a); out["swap_k_d2"] = {x["sig"] for p_ in kp for x in p_[:2]}
    for d in range(3, int(a["d_max"]) + 1):
        _, _, kp, _, _ = swap_pairs_d(a, d); out["swap_kd_d%d" % d] = {x["sig"] for p_ in kp for x in p_[:2]}
    if int(a["d_max"]) >= 3:
        _, _, kp, _, _ = swap_pairs_d(a, 3, 2, (SEED_SWAPD2_REC_REG + 3, SEED_SWAPD2_DON_REG + 3)); out["swap_kd_d3_t2"] = {x["sig"] for p_ in kp for x in p_[:2]}
    s = OW.Stream(SEED_PROBE_REG, excl, frames=OW.FRAME_TRAIN, **IO.OWKW, **reg_kw(a)); out["probe"] = {s.draw(2)["sig"] for _ in range(PROBE_N)}
    return out
'''),
]
for o, n in E:
    assert s.count(o) == 1, ("edit not applied exactly once", s.count(o), o[:80]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("eval parts:", len(E), "edits")
p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read()
o = "('        assert a.get(k_) in (None, v_), (\"§233: set sha mismatch vs args.json\", k_)', '        assert a.get(k_) in (None, v_), (\"§234: set sha mismatch vs args.json\", k_)'),"
n = (o + "\n('    if sets is None:\\n        sets = IO.build_sets()', '    if sets is None:\\n        sets = IO.build_sets(mul_s_p=int(a.get(\"mul_s_p\", 1)))   # §236: the run\\'s law'),")
assert s.count(o) == 1; s = s.replace(o, n); open(p, "w", encoding="utf-8", newline="\n").write(s); print("mk_inject_aw_eval: +1 edit")
