

@torch.no_grad()
def part_test_perm(m, g, sets, a, dev, bs):
    """§252 TEST-PERM (NS_v1 §2-§3; the 622fb717 renderer = boxes.perm_render, seeds 25150 + d): TEST2 d 1..min(4, d_max) re-rendered with the op sentences in a
    dependency-violating order (>= 1 inversion, d >= 2; d 1 unchanged); acc at K' {d - 1, d, think_k}, WS per pass 1..think_k; LP5-D4-style MEAN-AT-2 on TEST-PERM d 4
    (S after pass 2 := the TEST-PERM d 4 mean of S_2, the other passes on their own inputs; answers at K' {4, think_k}, + the intact answers). Per item."""
    import boxes.perm_render as PR
    K = a["think_k"]; ds = list(range(1, min(4, a["d_max"]) + 1)); P_, info = PR.test_perm_sets({d: sets["test2"][d] for d in ds}, ds)
    T = tokd({"test2": P_}, a, ds); out = {"info": info, "by_d": {}, "items": {}}
    for d in ds:
        its = T[d]; y = np.array([it["ans"] for it in its]); ks = sorted({d - 1, d, K})
        A, _ = passes(m, its, dev, bs, set(ks)); r_, pv = AW.ws_readout(m, {d: its}, dev, K, bs, per_item=True)
        out["by_d"][str(d)] = {"n": len(its), "acc_kprime": {str(k): float((A[k] == y).mean()) for k in ks}, "ws": r_[str(d)]}
        out["items"][str(d)] = items_rec(its, {"ans_kprime": {str(k): A[k].tolist() for k in ks}, "ws_pred": pv[str(d)], "order": [it["raw"]["order"] for it in its]})
    if 4 in ds:
        t = 2; its = T[4]; acc_ = {"sum": None}; y = np.array([it["ans"] for it in its]); ks = sorted({4, K})

        def grab(t_, S, sl):
            if t_ == t:
                s_ = S.float().sum(0); acc_["sum"] = s_ if acc_["sum"] is None else acc_["sum"] + s_
        passes(m, its, dev, bs, {t}, on_state=grab); mean_t = acc_["sum"] / len(its)
        I, _ = passes(m, its, dev, bs, set(ks))
        M, _ = passes(m, its, dev, bs, set(ks), edit=lambda t_, S, sl: mean_t.to(S.dtype)[None].expand(S.shape[0], *S.shape[1:]).contiguous() if t_ == t else None)
        out["mean_at_2_d4"] = {"intact": {str(k): float((I[k] == y).mean()) for k in ks}, "mean_at_t": {str(k): float((M[k] == y).mean()) for k in ks},
                               "mean_state_norm": float(mean_t.norm())}
        out["items"]["4"]["mean_at_2"] = {"intact": {str(k): I[k].tolist() for k in ks}, "mean_at_t": {str(k): M[k].tolist() for k in ks}}
    out["def"] = ("§252 TEST-PERM: TEST2 d 1-4 with the op sentences S_1..S_d reassigned over their slots by default_rng(25150 + d) (>= 1 inversion d >= 2; OW.solve "
                  "== answer asserted; boxes.perm_render = 622fb717's renderer); acc_kprime {d-1, d, K}, WS per pass; mean_at_2_d4 = LP5-D4 style on TEST-PERM d 4")
    return out
