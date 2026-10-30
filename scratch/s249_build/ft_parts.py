

FT_DS, FT_CUE_DS = tuple(range(1, 7)), (2, 3, 4)                          # §249 FRAME-TRANSFER (FT_v1 §2): TRANSFER of the run's law, d 1..6; cue-only d 2..4


def ft_by_frame(out, T, ws):
    """Note 249-1 (3): items draw one frame per op (6 or 7), so d > 1 items can mix them. by_frame '6' / '7': acc over items whose every op uses that frame (n), and
    (LAT) WS at pass t over items whose op at step min(t, d) uses that frame (n_t); 'mixed': acc over the rest."""
    bf = {"6": {}, "7": {}, "mixed": {}}
    for d in FT_DS:
        it_ = out["items"][str(d)]; fr = [it["raw"]["frames"] for it in T[d]]; y = np.array(it_["ans"]); pr = np.array(it_["pred"])
        for f in (6, 7):
            mk = np.array([all(x == f for x in q) for q in fr]); rec = {"n": int(mk.sum()), "acc": float((pr[mk] == y[mk]).mean()) if mk.any() else None}
            if ws:
                wsr, wsn = {}, {}
                for t in range(1, len(it_["ws_pred"]) + 1):
                    mt = np.array([q[min(t, d) - 1] == f for q in fr]); v = np.array([int(it["raw"]["values"][min(t, d)]) for it in T[d]])
                    p_ = np.array(it_["ws_pred"][str(t)]); wsr[str(t)] = float((p_[mt] == v[mt]).mean()) if mt.any() else None; wsn[str(t)] = int(mt.sum())
                rec["ws"], rec["ws_n"] = wsr, wsn
            bf[str(f)][str(d)] = rec
        mx = np.array([len(set(q)) > 1 for q in fr]); bf["mixed"][str(d)] = {"n": int(mx.sum()), "acc": float((pr[mx] == y[mx]).mean()) if mx.any() else None}
    return bf


@torch.no_grad()
def part_transfer(m, g, sets, a, dev, bs):
    """§249 FRAME-TRANSFER (FT_v1 §2): build_sets(law)['transfer'] (frames 6-7, never trained), d 1..6, all items. LAT (m not None): acc at K' = think_k, WS per pass (AW.ws_readout,
    write_src answer), cue_only d 2..4 (part_cue_only on the transfer items: think passes on the cue only, decoder on the full transfer text) as {acc, majority}. CD (m None):
    acc only (the direct path of answers_by_kprime). The transfer set's sha is recomputed and asserted == build_sets' transfer_sha (FT_v1 §2 gate)."""
    sha = OW.set_sha([it for d in IO.TEST_DS for it in sets["transfer"][d]]); assert sha == sets["shas"]["transfer_sha"], ("§249: transfer sha", sha, sets["shas"]["transfer_sha"])
    st = dict(sets); st["test2"] = sets["transfer"]; T = tokd(st, a, FT_DS); out = {"transfer_sha": sha, "by_d": {}, "items": {}}
    for d in FT_DS:
        y = np.array([it["ans"] for it in T[d]])
        if m is None:
            pr = answers_by_kprime(m, g, T[d], 0, dev, bs)[:, 0]; out["by_d"][str(d)] = {"n": len(y), "acc": float((pr == y).mean()), "k": 0}
            out["items"][str(d)] = items_rec(T[d], {"pred": pr.tolist(), "frames": [list(it["raw"]["frames"]) for it in T[d]]}); continue
        K = a["think_k"]; A, _ = passes(m, T[d], dev, bs, {K}); rec = {"n": len(y), "acc": float((A[K] == y).mean()), "k": K}; extra = {"pred": A[K].tolist()}
        if a.get("write_src") == "answer":
            r_, pv = AW.ws_readout(m, {d: T[d]}, dev, K, bs, per_item=True); rec["ws"] = r_[str(d)]; extra["ws_pred"] = pv[str(d)]
        out["by_d"][str(d)] = rec; out["items"][str(d)] = items_rec(T[d], dict(extra, frames=[list(it["raw"]["frames"]) for it in T[d]]))
    out["by_frame"] = ft_by_frame(out, T, m is not None and a.get("write_src") == "answer")   # Note 249-1 (3)
    if m is not None:
        co = part_cue_only(m, g, st, a, dev, bs, list(FT_CUE_DS))
        out["cue_only"] = {d: {"acc": v["answer_acc"], "majority": v["majority"]} for d, v in co["by_d"].items()}; out["cue_only_items"] = co["items"]
    out["def"] = ("§249 FRAME-TRANSFER (FT_v1 §2): build_sets(law)['transfer'] (frames 6-7), d 1..6, all items; LAT acc at K' = think_k + WS per pass + cue_only d 2-4 "
                  "(cue-only passes, full-text decoder); CD acc (direct); fp32")
    return out
