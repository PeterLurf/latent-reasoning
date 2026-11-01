

S257_SHAS = "scratch/s257_ho_shas.json"                                    # §257 DC2_v1 §3: set shas committed from a clean (twice-identical) local build


@torch.no_grad()
def part_insert_ho(m, g, sets, a, dev, bs):
    """§257 (DC2_v1 §3): §256's insert_ho (part_insert_ho_s256, unchanged; its shas asserted) + the §257 held-out families (boxes.insert_render.held_out2_sets on REG TEST2
    d 4, built HERE; shas asserted == scratch/s257_ho_shas.json): H-CHAINSRC-OP ("X has c fewer than S_j" / "X has c times as many as S_j" right after S_j, j 1-3; acc8 pooled
    over sub_c / mul_c, by_pos = j, + by_op) and H-CHAINSRC-LIVE (X1 = S_j + c right after S_j, X2 = X1 + c' right after it; descriptive). Per item: sig / pred / meta."""
    import boxes.insert_render as IR
    out = part_insert_ho_s256(m, g, sets, a, dev, bs); K = a["think_k"]; HO, metas = IR.held_out2_sets(sets["test2"][4]); sh = IR.set_shas(HO)
    assert sh == json.load(open(S257_SHAS))["held_out2"], "insert_ho: §257 held-out set shas != the committed clean build"
    res = {}
    for fam, d in HO.items():
        for pos, its in d.items():
            T = tokd({"test2": {4: its}}, a, (4,))[4]; A, _ = passes(m, T, dev, bs, {K}); res[(fam, pos)] = A[K] == np.array([it["ans"] for it in T])
            out["items"]["%s:%s" % (fam, pos)] = {"sig": [it["sig"] for it in its], "pred": A[K].tolist(), "meta": metas[fam][pos]}
    for top in ("H-CHAINSRC-OP", "H-CHAINSRC-LIVE"):
        fs = [f for f in IR.HO2_FAMS if f == top or f.startswith(top + "-")]; ps = IR.HO2_FAMS[fs[0]]
        out["fam"][top] = {"acc8": float(np.concatenate([res[(f, p)] for f in fs for p in ps]).mean()),
                           "by_pos": {p: float(np.concatenate([res[(f, p)] for f in fs]).mean()) for p in ps}}
        if len(fs) > 1:
            out["fam"][top]["by_op"] = {f[len(top) + 1:]: {"acc8": float(np.concatenate([res[(f, p)] for p in ps]).mean()), "by_pos": {p: float(res[(f, p)].mean()) for p in ps}}
                                        for f in fs}
    out["set_shas2"] = sh
    out["def_s257"] = ("§257 DC2_v1 §3: H-CHAINSRC-OP (sub_c / mul_c chain-reading insert right after S_j, no new given; by_pos = j 1-3; pooled over the two ops), "
                       "H-CHAINSRC-LIVE (X1 = S_j + c right after S_j, X2 = X1 + c' right after it; by_pos = j); frames 0-5; acc8 = acc at K' think_k")
    return out
