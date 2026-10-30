

@torch.no_grad()
def part_frames(m, g, sets, a, dev, bs):
    """§250 FRAME-DIV Stage 0 (FD_v1 §3): build_sets(law)['framesets'] (frames 8-15, every op in that frame, d 1-4, 1,000 each): per frame and d, acc at K' = think_k, WS per pass
    (AW.ws_readout), the wrap share (items whose mod-11 chain wraps: OW's wrap_any) and acc on the wrap / no-wrap subsets (FD_v1 §1 mod-11 caveat, descriptive);
    per item {sig, pred, ws_pred, wrap_any, ans}. -> {frame: {d: {...}}} + the frameset shas."""
    K = a["think_k"]; out = {"frameset_shas": {str(f): sets["shas"]["frameset%d_sha" % f] for f in IO.FRAMESET_FRAMES}}
    for f in IO.FRAMESET_FRAMES:
        T = tokd({"test2": sets["framesets"][f]}, a, IO.FRAMESET_DS); rec = {}
        assert all(set(it["raw"]["frames"]) == {f} for d in T for it in T[d]), ("§250: frame set", f)
        for d in IO.FRAMESET_DS:
            y = np.array([it["ans"] for it in T[d]]); A, _ = passes(m, T[d], dev, bs, {K}); ok = A[K] == y
            r_, pv = AW.ws_readout(m, {d: T[d]}, dev, K, bs, per_item=True); wa = np.array([bool(it["raw"].get("wrap_any")) for it in T[d]])
            rec[str(d)] = {"n": len(y), "acc": float(ok.mean()), "ws": r_[str(d)], "wrap_any_share": float(wa.mean()),
                           "acc_nowrap": float(ok[~wa].mean()) if (~wa).any() else None, "acc_wrap": float(ok[wa].mean()) if wa.any() else None,
                           "items": items_rec(T[d], {"pred": A[K].tolist(), "ws_pred": pv[str(d)], "wrap_any": wa.astype(int).tolist()})}
        out[str(f)] = rec
    out["def"] = ("§250 FRAME-DIV Stage 0 (FD_v1 §3): per-frame test sets (frames 8-11 DIV-train, 12-15 held out; every op in that frame; build_sets law; Stream(25000 + f)), "
                  "d 1-4 x 1,000; acc at K' = think_k, WS per pass, wrap share + wrap / no-wrap acc; fp32")
    return out
