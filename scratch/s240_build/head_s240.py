

def state_n(S):
    """per-item Frobenius norm of a DeltaMem state (all heads), fp32."""
    return S.float().flatten(1).norm(dim=1)


def hold_step(m, keep, far_items, T_far, K, BS, inc_on, lam, ac, dev):
    """§240 HOLD-INC (LH_v1 §2 + Note 240-1), called after the plain backward of every micro-chunk (so the plain grads are complete and gn_plain is read first).
    keep = [(states S_0..S_K detached, xs, mask, ds)] per chunk (one d per step). Stats (no_grad, every call): comp / hold = the mean ||S_t - S_{t-1}|| / ||S_{d+2}||
    at t <= d / t >= d + 3; s8 / sd2 = the mean ||S_K|| / ||S_{d+2}|| (the norm guard). Penalty (inc_on): for each penalised pass t in H = {d+3..K}, its OWN forward
    S_pen = write(sg S_{t-1}, x_{S_d cue}) with the lower blocks recomputed (hl_pen, not the CE cache) -> a graph disjoint from the CE graph; r_t = ||S_pen - sg S_{t-1}||^2
    / ||sg S_{d+2}||^2; far pass on the first len(far_items) items of the batch: a no-grad prefix to T_far - 1, then one grad pass at T_far from the detached S_{T_far - 1}.
    L_inc = lam * mean over the batch of the per-item mean over its terms (H, + T_far for the far items); backward()-ed here, chunk by chunk. -> the stats dict."""
    st = {k: [0.0, 0] for k in ("comp", "hold", "s8", "sd2", "inc", "far")}; off = 0
    for Ssv, xs, msk, ds in keep:
        Ssv = [s_.detach() for s_ in Ssv]; n = Ssv[0].shape[0]; d = ds[0]; assert all(x == d for x in ds), "one d per step"; H = list(range(d + 3, K + 1))
        with torch.no_grad():
            den = state_n(Ssv[d + 2]).clamp_min(1e-12)
            for t in range(1, K + 1):
                key = "comp" if t <= d else ("hold" if t >= d + 3 else None)
                if key:
                    st[key][0] += float((state_n(Ssv[t] - Ssv[t - 1]) / den).sum()); st[key][1] += n
            st["s8"][0] += float(state_n(Ssv[K]).sum()); st["s8"][1] += n; st["sd2"][0] += float(den.sum()); st["sd2"][1] += n
        if inc_on and H:
            w = torch.tensor([1.0 / (len(H) + (1 if off + i < len(far_items) else 0)) for i in range(n)], device=dev); tot = 0.0
            with ac():
                hlp = m.lower(xs[-1])                                       # the penalty's own lower blocks (the hold passes read the S_d cue = xs[-1])
                for t in H:
                    Sp = m.write(Ssv[t - 1], xs[-1], msk, hlp); tot = tot + (Sp.float() - Ssv[t - 1].float()).flatten(1).pow(2).sum(1) / den ** 2
            (lam * (w * tot).sum() / BS).backward()
            st["inc"][0] += float((tot / len(H)).detach().sum()); st["inc"][1] += n
        off += n
    if inc_on and len(far_items):
        d = far_items[0]["d"]; H = list(range(d + 3, K + 1)); xs, msk = cue_inputs(m, far_items, dev); S = m.mem.zero_state(len(far_items), dev); den = None
        with torch.no_grad(), ac():
            hls = [m.lower(x_) for x_ in xs]
            for t in range(1, T_far):
                j = min(t, len(xs)) - 1; S = m.write(S, xs[j], msk, hls[j])
                if t == d + 2:
                    den = state_n(S).clamp_min(1e-12)
        assert den is not None and T_far - 1 >= d + 2
        S = S.detach()
        with ac():
            Sp = m.write(S, xs[-1], msk, m.lower(xs[-1]))
        r = (Sp.float() - S.float()).flatten(1).pow(2).sum(1) / den ** 2
        (lam * (r / (len(H) + 1)).sum() / BS).backward(); st["far"][0] += float(r.detach().sum()); st["far"][1] += len(far_items)
    return st


def hold_acc(hb, st):
    for k_, v_ in st.items():
        b_ = hb.setdefault(k_, [0.0, 0]); b_[0] += v_[0]; b_[1] += v_[1]


def hold_mean(hb):
    return {k_: (v_[0] / v_[1] if v_[1] else None) for k_, v_ in hb.items()}
