

S256_SHAS = "scratch/s256_ho_shas.json"                                    # §256 DS_v1 §5: set shas committed from a clean (twice-identical) local build


@torch.no_grad()
def part_insert_ho(m, g, sets, a, dev, bs):
    """§256 HELD-OUT INSERT (DS_v1 §3; boxes.insert_render.held_out_sets on REG TEST2 d 4, built HERE; their shas asserted == scratch/s256_ho_shas.json). Per family: acc at
    K' = think_k (acc8) pooled over its positions + by_pos (H-OP pooled over sub_c / mul_c / add_v, + by_op); indist = the Note 255-3 INSERT conditions' acc8. Per item."""
    import boxes.insert_render as IR
    assert law(a) == 0 and a["d_max"] >= 4, "insert_ho: REG, d_max >= 4"
    K = a["think_k"]; raw = sets["test2"][4]; HO, metas = IR.held_out_sets(raw); sh = IR.set_shas(HO); ref = json.load(open(S256_SHAS))
    assert sh == ref["held_out"], "insert_ho: held-out set shas != the committed clean build"
    out = {"fam": {}, "indist": {}, "items": {}, "set_shas": sh}

    def acc_of(its):
        T = tokd({"test2": {4: its}}, a, (4,))[4]; A, _ = passes(m, T, dev, bs, {K}); y = np.array([it["ans"] for it in T]); return (A[K] == y), A[K]
    res = {}
    for fam, d in HO.items():
        for pos, its in d.items():
            ok, pr = acc_of(its); res[(fam, pos)] = ok; out["items"]["%s:%s" % (fam, pos)] = {"sig": [it["sig"] for it in its], "pred": pr.tolist(), "meta": metas[fam][pos]}
    fams = {}
    for (fam, pos), ok in res.items():
        top = "H-OP" if fam.startswith("H-OP-") else fam; fams.setdefault(top, {}).setdefault(pos, []).append((fam, ok))
    for top, d in fams.items():
        by_pos = {p: float(np.concatenate([ok for _, ok in L]).mean()) for p, L in d.items()}; allok = np.concatenate([ok for L in d.values() for _, ok in L])
        out["fam"][top] = {"acc8": float(allok.mean()), "by_pos": by_pos}
        if top == "H-OP":
            out["fam"][top]["by_op"] = {f[len("H-OP-"):]: {"acc8": float(np.concatenate([res[(f, p)] for p in IR.HO_FAMS[f]]).mean()),
                                                           "by_pos": {p: float(res[(f, p)].mean()) for p in IR.HO_FAMS[f]}} for f in IR.HO_FAMS if f.startswith("H-OP-")}
    ins, _ = IR.insert_eval_sets(raw)
    for c in IR.CONDS:
        ok, pr = acc_of(ins[c]); out["indist"][c] = {"acc8": float(ok.mean())}; out["items"]["indist:" + c] = {"pred": pr.tolist()}
    out["def"] = ("§256 HELD-OUT INSERT (DS_v1 §3): families H-OP (sub_c / mul_c / add_v; before / inside / after), H-CHAINSRC (X = S_j + c after S_j, j 1-3, no new given), "
                  "H-GIVPOS (the op after S2; its given at the text start / right before the op / after S4), H-EXTEND (X1 = S4 + c, X2 = X1 + c after S4), H-LIVE (Y; X1 = Y + c; "
                  "X2 = X1 + c at before / inside / after), K3 (3 add_c pairs, random gaps); frames 0-5; acc8 = acc at K' think_k; indist = Note 255-3's conditions")
    return out


@torch.no_grad()
def part_ws_deep(m, g, sets, a, dev, bs):
    """§256 WS-DEEP (DS_v1 §3 SELECTION-CREDIT): WS per pass 1..think_k on TEST2 d 5-8 (1,000 each; set shas asserted == scratch/s256_ho_shas.json 'ws_deep'); per item the
    pass argmax and v_min(t, d)."""
    K = a["think_k"]; ds = (5, 6, 7, 8); ref = json.load(open(S256_SHAS))
    assert {str(d): OW.set_sha(sets["test2"][d]) for d in ds} == ref["ws_deep"], "ws_deep: TEST2 d 5-8 shas != the committed build"
    T = tokd(sets, a, ds); r_, pv = AW.ws_readout(m, T, dev, K, bs, per_item=True)
    return {"by_d": r_, "items": {str(d): items_rec(T[d], {"ws_pred": pv[str(d)], "v": [[int(it["raw"]["values"][min(t, d)]) for t in range(1, K + 1)] for it in T[d]]})
                                  for d in T},
            "def": "§256 WS-DEEP: the trunk's argmax over the 11 number tokens at ':' of pass t vs v_min(t, d), TEST2 d 5-8, fp32"}
