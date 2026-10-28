

NC_ARMS = (                                                                 # §242 NORM-COUNTER (NC_v1 §2 + Note 242-1): (arm, d, edit) ; edit = (pass k, ref, heads)
    ("intact_d4", 4, None), ("intact_d6", 6, None),
    ("identity_d6", 6, (4, "one", "all")),                                  # 242-1 (1): scale exactly 1.0 through the same path (must equal intact_d6 bitwise)
    ("down_d6", 6, (4, 2, "all")), ("down_add_d6", 6, (4, 2, "add")),       # S_4 -> ||S_2|| (all heads) ; 242-1 (2): the additive heads only, to their own ||S_2||
    ("up_d4", 4, (2, 4, "all")), ("harm_d4", 4, (3, 1, "all")))             # S_2 -> ||S_4|| ; S_3 -> ||S_1||


def nc_norms(S, mem):
    """per item ||S|| over all heads, the additive heads [0, h_add) and the delta heads [h_add, h_add + h_del), fp32."""
    f = S.float(); return f.flatten(1).norm(dim=1), f[:, :mem.h_add].flatten(1).norm(dim=1), f[:, mem.h_add:mem.h_add + mem.h_del].flatten(1).norm(dim=1)


@torch.no_grad()
def nc_run(m, T, dev, bs, K, edit=None, ref=None):
    """passes 1..K as passes() / ws_readout (seq_std inputs, one padding, one lower cache per distinct input, S from zero, pass_aw); per pass t: the WS argmax at
    ':' and the intact norms (all / add / del); edit (k, r, heads): after pass k, S[heads] <- S[heads] * s with s = ref[heads][r] / ||S_k[heads]|| per item
    (r "one": s = 1.0 exactly). -> {ans (N,), ws {t: (N,)}, norms {all/add/del: {t: (N,)}}, scale (N,), post_relerr (max | ||S'_k[heads]|| / target - 1 |)}."""
    mem = m.mem; num = torch.tensor(NUM_IDS, device=dev); N = len(T); ans = np.full(N, -1); wsp = {t: np.full(N, -1) for t in range(1, K + 1)}
    nrm = {h: {t: np.zeros(N, dtype=np.float32) for t in range(1, K + 1)} for h in ("all", "add", "del")}; sc = np.full(N, np.nan, dtype=np.float32); rel = 0.0
    Td = n_distinct(T)
    for i in range(0, N, bs):
        ch = T[i:i + bs]; sl = slice(i, i + len(ch)); xs = []; m0 = None
        for t in range(1, Td + 1):
            ids_, mk = pad([seq_std(it, t) for it in ch]); assert m0 is None or (mk == m0).all(), "one padding across passes"; m0 = mk
            xs.append(torch.tensor(ids_, device=dev))
        msk = torch.tensor(m0, device=dev); hls = [m.lower(x) for x in xs]; S = mem.zero_state(len(ch), dev)
        assert S.shape[1] == mem.h_add + mem.h_del, ("§242: no den block expected", S.shape)
        for t in range(1, K + 1):
            j = min(t, Td) - 1; S, h, pos = m.pass_aw(S, xs[j], msk, hls[j])
            wsp[t][sl] = m.colon_logits(h, pos)[:, num].argmax(-1).cpu().numpy()
            na, nd_add, nd_del = nc_norms(S, mem)
            for k_, v_ in (("all", na), ("add", nd_add), ("del", nd_del)):
                nrm[k_][t][sl] = v_.cpu().numpy()
            if edit is not None and t == edit[0]:
                k, r, heads = edit; cur = {"all": na, "add": nd_add}[heads]
                s_ = torch.ones_like(cur) if r == "one" else torch.tensor(ref[heads][r][sl], device=dev) / cur
                sc[sl] = s_.cpu().numpy(); hs = slice(0, S.shape[1]) if heads == "all" else slice(0, mem.h_add)
                S = S.clone(); S[:, hs] = S[:, hs] * s_.to(S.dtype).view(-1, 1, 1, 1)
                if r != "one":
                    post = S[:, hs].float().flatten(1).norm(dim=1); rel = max(rel, float((post / torch.tensor(ref[heads][r][sl], device=dev) - 1).abs().max()))
        ans[sl] = pred_of(m.dec_logits(S, ch, dev))
    return {"ans": ans, "ws": wsp, "norms": nrm, "scale": sc, "post_relerr": rel}


@torch.no_grad()
def part_normctr(m, g, sets, a, dev, bs):
    """§242 NORM-COUNTER (NC_v1 + Note 242-1): TEST2 of the run's law, d 4 / d 6 (all 1,000), K' 8 (= think_k); arms NC_ARMS; refs = the same items' intact norms
    (same call, same batches). -> {arm: {d, n, acc, ws {t}, mean_scale, post_relerr, items {sig, ans, pred, ws_pred {t}, scale}}, head_norms {d: {t: {add, del, all}}},
    identity_equal_items}."""
    K = a["think_k"]; assert K == 8, ("§242: K' 8 = think_k", K)
    T = tokd(sets, a, [4, 6]); intact = {}; out = {"head_norms": {}}
    for arm, d, ed in NC_ARMS:
        r_ = nc_run(m, T[d], dev, bs, K, ed, intact[d]["norms"] if ed is not None else None)
        if ed is None:
            intact[d] = r_; out["head_norms"][str(d)] = {str(t): {h: float(r_["norms"][h][t].mean()) for h in ("add", "del", "all")} for t in range(1, K + 1)}
        y = np.array([it["ans"] for it in T[d]]); v = {t: np.array([int(it["raw"]["values"][min(t, d)]) for it in T[d]]) for t in range(1, K + 1)}
        out[arm] = {"d": d, "n": len(T[d]), "acc": float((r_["ans"] == y).mean()), "ws": {str(t): float((r_["ws"][t] == v[t]).mean()) for t in range(1, K + 1)},
                    "mean_scale": float(np.nanmean(r_["scale"])) if ed is not None else None, "edit": list(ed) if ed is not None else None,
                    "post_relerr": r_["post_relerr"] if ed is not None else None,
                    "items": {"sig": [it["sig"] for it in T[d]], "ans": y.tolist(), "pred": r_["ans"].tolist(), "ws_pred": {str(t): r_["ws"][t].tolist() for t in range(1, K + 1)},
                              "scale": [None if np.isnan(x) else float(x) for x in r_["scale"]]}}
    out["identity_equal_items"] = bool(out["identity_d6"]["items"]["pred"] == out["intact_d6"]["items"]["pred"]
                                       and out["identity_d6"]["items"]["ws_pred"] == out["intact_d6"]["items"]["ws_pred"])
    out["def"] = ("NORM-COUNTER (§242 NC_v1 + Note 242-1): TEST2 of the run's law d 4 / d 6, K' 8, fp32; rescale S <- S * n_ref / ||S|| (all heads; down_add_d6: the 6 additive "
                  "heads to their own ||S_2||) after pass k with n_ref the same item's intact norm; identity_d6 = scale 1.0 on the same path; ws = WS-READOUT per pass")
    return out
