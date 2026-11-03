"""THEORY §264 SCORECARD-CHECK eval parts (eval only; registered in inject_aw_eval's run() by scratch/s271_build/edits_s271.py; the run's own tokenization, fp32, no_grad):
  cue_clock  REG TEST2 (all items), K' 0..KM (24), via inject_aw_eval.passes(seqs_of_t, Td = KM, on_state): the pass-t input = the run's own cue input (seq_std) with the
             cue's entity token (CUE_NAME_OFF = 6 before ':', the original asserted first) swapped per schedule:
               STD d 3-8  S_min(t, d) (the run's cue; the swap is the identity);   ER d 3-6  S_min(t, d - 2);
               LF  d 3-6  S_t for t <= d, then per pass a name absent from the item, drawn without replacement from the 40 names minus the item's
                          (default_rng((SEED_LF, d, i)).permutation; cycled if exhausted);
               CY  d 3-6  S_((t - 1) mod d) + 1.
             -> items[sch][d] = {ans [N], pred [N][KM + 1], rel_inc [N][KM - 1] (t 2..KM: ||S_t - S_t-1|| / ||S_t-1||, Frobenius over every state block), norm [N][KM]
             (t 1..KM: ||S_t||), the same split by head family (rel_inc_add / rel_inc_del / norm_add / norm_del: blocks [0, h_add) / [h_add, h_add + h_del)), sig};
             STD d 6-8 also logits [N][13][11] (the decoder's 11-way logits over ' 0'..' 10' at K' 0..12, 4 dp).
  op_count   REG TEST2 d 4 + k in {0, 2, 4, 6} extra inserts (insert_render.distract_mix with n = k forced, types s256 / chain / givop, default_rng((SEED_OC, ...)) via its
             seed / step / k arguments: seed = SEED_OC, step = k, k = item index; answer unchanged, asserted) -> by_k {k: {acc (K' = think_k), ws by pass (inject_aw.ws_readout),
             n_ops (op sentences = d + k)}} + items.
Seeds 26400 / 26410 unused elsewhere in boxes/*.py and scratch/*.py (grep 2026-11-03)."""
import numpy as np
import torch

KM, CUE_NAME_OFF, SEED_LF, SEED_OC, SEED_INS = 24, 6, 26400, 26410, 26420
SCH = {"STD": (3, 4, 5, 6, 7, 8), "ER": (3, 4, 5, 6), "LF": (3, 4, 5, 6), "CY": (3, 4, 5, 6), "STDI": (3, 4, 5, 6), "LFI": (3, 4, 5, 6)}   # Note 264-3: + STDI / LFI


def sched_name(sch, raw, t, extra):
    d = raw["d"]; sp = raw["names"]["spine"]
    if sch in ("STD", "STDI"):
        return sp[min(t, d)]
    if sch == "ER":
        return sp[min(t, d - 2)]
    if sch in ("LF", "LFI"):                                               # LFI: extra = the two inserts' entities X_1, X_2 (cycled)
        return sp[t] if t <= d else extra[(t - d - 1) % len(extra)]
    if sch == "CY":
        return sp[((t - 1) % d) + 1]
    raise ValueError(sch)


@torch.no_grad()
def part_cue_clock(m, g, sets, a, dev, bs):
    import boxes.inject_aw_eval as EV
    import boxes.nested_words as NW
    from boxes.inject_ow import ENC
    from boxes.nested170 import NUM_IDS
    tok = {n: ENC.encode(" " + n)[0] for n in NW.NAMES}; assert all(len(ENC.encode(" " + n)) == 1 for n in NW.NAMES)
    ha, hd = m.mem.h_add, m.mem.h_del; out = {"KM": KM, "items": {}, "acc_by_kprime": {}}
    for sch, ds in SCH.items():
        out["items"][sch] = {}; out["acc_by_kprime"][sch] = {}
        for d in ds:
            raw_d = sets["test2"][d]
            if sch in ("STDI", "LFI"):                                     # Note 264-3: the same items + 2 s256 inserts (seed 26420; answers unchanged, asserted)
                import boxes.insert_render as IR
                raw_d = [IR.distract_mix(it, SEED_INS, d, i, types=("s256",), n=2)[0] for i, it in enumerate(sets["test2"][d])]
                assert all(len(o.get("distract_meta", [])) == 2 and o["answer"] == it["answer"] for o, it in zip(raw_d, sets["test2"][d])), "cue_clock: STDI / LFI inserts"
            T = EV.tokd({"test2": {d: raw_d}}, a, (d,))[d]; N = len(T); extra = {}
            if sch == "LFI":
                for it in T:
                    extra[id(it)] = [mt["X"] for mt in it["raw"]["distract_meta"]]
            if sch == "LF":
                for i, it in enumerate(T):
                    nm = it["raw"]["names"]; used = set(nm["spine"]) | set(nm["givens"]) | set(nm["distractors"]); pool = sorted(n for n in NW.NAMES if n not in used)
                    extra[id(it)] = [pool[j] for j in np.random.default_rng((SEED_LF, d, i)).permutation(len(pool))]

            def seqs(it, t, sch=sch):
                q = list(EV.seq_std(it, t)); p_ = len(q) - 1 - CUE_NAME_OFF; raw = it["raw"]
                assert q[p_] == tok[raw["names"]["spine"][min(t, raw["d"])]], ("cue_clock: the cue's entity token", sch, t)
                q[p_] = tok[sched_name(sch, raw, t, extra.get(id(it)))]; return q
            rec = {k: [[None] * KM for _ in range(N)] for k in ("norm", "norm_add", "norm_del")}
            rec.update({k: [[None] * (KM - 1) for _ in range(N)] for k in ("rel_inc", "rel_inc_add", "rel_inc_del")}); prev = {}
            lg = [[None] * 13 for _ in range(N)] if (sch == "STD" and d >= 6) else None

            def on_state(t, S, sl, T=T, rec=rec, prev=prev, lg=lg):
                f = S.float(); parts = {"": f, "_add": f[:, :ha], "_del": f[:, ha:ha + hd]}
                for sfx, x in parts.items():
                    nr = x.flatten(1).norm(dim=1)
                    for r_, i in enumerate(range(sl.start, sl.stop)):
                        rec["norm" + sfx][i][t - 1] = float(nr[r_])
                    if t >= 2:
                        dn = (x - prev[(sl.start, sfx)]).flatten(1).norm(dim=1) / prev[(sl.start, sfx)].flatten(1).norm(dim=1).clamp_min(1e-30)
                        for r_, i in enumerate(range(sl.start, sl.stop)):
                            rec["rel_inc" + sfx][i][t - 2] = float(dn[r_])
                    prev[(sl.start, sfx)] = x.clone()
                if lg is not None and t <= 12:
                    ch = T[sl]
                    if t == 1:
                        l0 = m.dec_logits(m.mem.zero_state(len(ch), dev), ch, dev)[:, NUM_IDS]
                        for r_, i in enumerate(range(sl.start, sl.stop)):
                            lg[i][0] = [round(float(v), 4) for v in l0[r_]]
                    lt = m.dec_logits(S, ch, dev)[:, NUM_IDS]
                    for r_, i in enumerate(range(sl.start, sl.stop)):
                        lg[i][t] = [round(float(v), 4) for v in lt[r_]]
            A, _ = EV.passes(m, T, dev, bs, set(range(KM + 1)), seqs_of_t=seqs, Td=KM, on_state=on_state); y = [it["ans"] for it in T]
            item = {"ans": y, "pred": [[int(A[k][i]) for k in range(KM + 1)] for i in range(N)], "sig": [it["sig"] for it in T], **rec}
            if lg is not None:
                item["logits"] = lg
            out["items"][sch][str(d)] = item; out["acc_by_kprime"][sch][str(d)] = {str(k): float((A[k] == np.array(y)).mean()) for k in range(KM + 1)}
    out["def"] = ("§264 CUE-CLOCK: REG TEST2, K' 0..24; pass-t input = the run's cue input with the entity token (6 before ':') set per schedule: STD S_min(t, d) d 3-8, "
                  "ER S_min(t, d-2) d 3-6, LF S_t (t <= d) then a fresh absent name per pass (default_rng((26400, d, i)), cycled) d 3-6, CY S_((t-1) mod d)+1 d 3-6; STDI / LFI (Note 264-3) = STD / LF on the same items + 2 s256 inserts (seed 26420; LFI's post-d cue = the inserts' X_1, X_2 cycled) d 3-6; "
                  "rel_inc / norm Frobenius over every state block (+ the additive / delta split); STD d 6-8 11-way decoder logits K' 0..12 (4 dp)")
    return out


@torch.no_grad()
def part_op_count(m, g, sets, a, dev, bs):
    import boxes.inject_aw as AW
    import boxes.inject_aw_eval as EV
    import boxes.insert_render as IR
    K = a["think_k"]; raw = sets["test2"][4]; out = {"by_k": {}, "items": {}}
    for k in (0, 2, 4, 6):
        its = []
        for i, it in enumerate(raw):
            o, n_ = IR.distract_mix(it, SEED_OC, k, i, n=k) if k else (it, 0); assert n_ == k and o["answer"] == it["answer"], ("op_count", k, n_)
            its.append(o)
        T = EV.tokd({"test2": {4: its}}, a, (4,))[4]
        for p_, q_ in zip(T, EV.tokd({"test2": {4: raw}}, a, (4,))[4]):
            assert p_["ans"] == q_["ans"] and len(p_["think_cues"]) == len(q_["think_cues"]), "op_count: answers / cues unchanged"
        A, _ = EV.passes(m, T, dev, bs, {K}); y = np.array([it["ans"] for it in T]); ws, pv = AW.ws_readout(m, {4: T}, dev, K, bs, per_item=True)
        out["by_k"][str(k)] = {"acc": float((A[K] == y).mean()), "ws": ws["4"], "n": len(T), "n_ops": 4 + k}
        out["items"][str(k)] = {"sig": [it["sig"] for it in T], "pred": A[K].tolist(), "ans": y.tolist(), "ws_pred": pv["4"]}
    out["def"] = ("§264 OP-COUNT: REG TEST2 d 4 + k in {0, 2, 4, 6} forced inserts (insert_render.distract_mix n = k, types s256 / chain / givop, seed 26410, step = k, "
                  "item index = k; answer unchanged); acc at K' = think_k, WS per pass")
    return out


SEED_AF, AF_N = 26430, 500                                                  # 26430: off §264's insert seed 26420 (Note 265-2)
AF_CONDS = {"d4k0": (4, 0, (2, 3, 4)), "d4k6": (4, 6, (4,)), "d8": (8, 0, (4, 7))}


def _spans(o, think):
    """per-sentence token spans [start, end) in the pass input (EOT at 0): the sentences of o['text'] (labels = insert_labels or order) re-tokenised one by one and
    asserted to tile enc(' ' + text) exactly."""
    import re
    from boxes.inject_ow import ENC
    sents = re.findall(r"[^.?]*[.?]", o["text"]); labs = list(o.get("insert_labels") or o["order"]); assert len(sents) == len(labs) + 1, ("attn_fade: sentences", len(sents), len(labs))
    toks = []; spans = {}; pos = 1
    for i, s in enumerate(sents):
        tk = ENC.encode(s if s.startswith(" ") else " " + s); toks += tk
        if i < len(labs):
            spans[labs[i]] = (pos, pos + len(tk))
        pos += len(tk)
    assert [think[0]] + toks == think, "attn_fade: the per-sentence tokens must tile the text"
    return spans


@torch.no_grad()
def part_attn_fade(m, g, sets, a, dev, bs):
    """§265 D1a ATTENTION-FADING (read-only): per condition (d4k0 TEST2 d 4, passes 2-4; d4k6 TEST2 d 4 + 6 inserts (distract_mix n = 6, seed 26420), pass 4; d8 TEST2 d 8,
    passes 4 / 7; the first 500 items), at the pass-t ':' query of every layer x head: the fp32 softmax attention mass on S_t's op-sentence token span (q / k recomputed
    from the block input exactly as deltamem._block does; the block output itself is untouched). Read-only asserted: the decoder logits after the condition's last pass are
    bitwise equal with and without the capture. + ws_ok_d8_t7: per d 8 item 1 / 0 (pass-7 WS ok) when passes 1-6 WS are all ok, else None."""
    import torch.nn.functional as F
    import boxes.deltamem as DM
    import boxes.inject_aw as AW
    import boxes.inject_aw_eval as EV
    import boxes.insert_render as IR
    from boxes.nested170 import NUM_IDS
    num = torch.tensor(NUM_IDS, device=dev); orig = DM._block; out = {"conds": {}, "n": AF_N, "L": g.L, "H": g.H}; CAP = {}

    def blk(g_, i, h, hd):
        if CAP.get("on"):
            B, T_, D = h.shape; pre = "h.%d." % i; x = F.layer_norm(h.float(), (D,), g_.w(pre + "ln_1.weight").float(), g_.w(pre + "ln_1.bias").float())
            q, k, _ = (x @ g_.w(pre + "attn.c_attn.weight").float() + g_.w(pre + "attn.c_attn.bias").float()).split(D, -1)
            ar = torch.arange(B, device=h.device); pos = CAP["pos"]; qh = q[ar, pos].view(B, g_.H, hd); kh = k.view(B, T_, g_.H, hd)
            sc = torch.einsum("bhd,bthd->bht", qh, kh) / hd ** 0.5; tt = torch.arange(T_, device=h.device)
            sc = sc.masked_fill(tt[None, None, :] > pos[:, None, None], float("-inf")); pr = sc.softmax(-1)
            CAP["mass"][i] = (pr * CAP["span"][:, None, :]).sum(-1).cpu()          # (B, H)
            CAP["opm"][i] = (pr * CAP["ops"][:, None, :]).sum(-1).cpu()            # Note 265-2: the mass on every op-sentence token (S_1..S_d + inserted ops)
        return orig(g_, i, h, hd)
    DM._block = blk                                                         # the capture wrapper (restored in finally)
    try:
        for cond, (d, k, tset) in AF_CONDS.items():
            raw = sets["test2"][d][:AF_N]; its = []
            for j, it in enumerate(raw):
                if k:                                                       # Note 265-2: the k inserts (new given + op pairs) all placed AFTER S_d
                    g_, o_, mt_ = IR.draw_pairs(it, k, np.random.default_rng((SEED_AF, d, j))); o = IR.assemble(it, g_, [(("after", "S%d" % d), x_) for x_ in o_], mt_)
                else:
                    o = it
                assert o["answer"] == it["answer"]; its.append(o)
            T = EV.tokd({"test2": {d: its}}, a, (d,))[d]; res = {str(t): [None] * len(T) for t in tset}; shr = {str(t): [None] * len(T) for t in tset}
            wsok = [None] * len(T) if cond == "d8" else None; tmax = max(tset)
            for b0 in range(0, len(T), bs):
                ch = T[b0:b0 + bs]; xs, msk = AW.cue_inputs(m, ch, dev); B = len(ch); pos = msk.long().sum(1) - 1; S = m.mem.zero_state(B, dev); ok_ = []
                sp = [_spans(it["raw"], it["think"]) for it in ch]
                for t in range(1, max(tmax, 7 if cond == "d8" else 0) + 1):
                    x = xs[min(t, len(xs)) - 1]
                    if t in tset:
                        span = torch.zeros(B, x.shape[1], device=dev); ops = torch.zeros(B, x.shape[1], device=dev)
                        for r_, s_ in enumerate(sp):
                            a_, e_ = s_["S%d" % t]; span[r_, a_:e_] = 1.0
                            for lab, (a2, e2) in s_.items():
                                if (lab[0] == "S" and lab[1:].isdigit() and int(lab[1:]) >= 1) or lab.startswith("NO"):
                                    ops[r_, a2:e2] = 1.0
                        CAP.update({"on": True, "pos": pos, "span": span, "ops": ops, "mass": {}, "opm": {}})
                    try:
                        S, h, p_ = m.pass_aw(S, x, msk, m.lower(x))
                    finally:
                        CAP["on"] = False
                    if t in tset:
                        M = torch.stack([CAP["mass"][i] for i in range(g.L)], 1); O = torch.stack([CAP["opm"][i] for i in range(g.L)], 1)   # (B, L, H)
                        Sh = M / O.clamp_min(1e-12)                         # Note 265-2: the target SHARE = mass on S_t / mass on all op-sentence tokens
                        for r_ in range(B):
                            res[str(t)][b0 + r_] = [[round(float(v), 5) for v in row] for row in M[r_]]; shr[str(t)][b0 + r_] = [[round(float(v), 5) for v in row] for row in Sh[r_]]
                    if cond == "d8":
                        pw = m.colon_logits(h, p_)[:, num].argmax(-1).tolist(); ok_.append([int(pw[r_] == int(it["raw"]["values"][min(t, d)])) for r_, it in enumerate(ch)])
                if cond == "d8":
                    for r_ in range(B):
                        wsok[b0 + r_] = ok_[6][r_] if all(ok_[tt][r_] for tt in range(6)) else None
                if b0 == 0:                                                 # read-only check: the decoder logits after pass tmax, capture on vs off
                    S0 = m.mem.zero_state(B, dev); S1 = m.mem.zero_state(B, dev)
                    for t in range(1, tmax + 1):
                        x = xs[min(t, len(xs)) - 1]; S0 = m.write(S0, x, msk, m.lower(x))
                        CAP.update({"on": True, "pos": pos, "span": torch.zeros(B, x.shape[1], device=dev), "ops": torch.zeros(B, x.shape[1], device=dev), "mass": {}, "opm": {}})
                        try:
                            S1 = m.write(S1, x, msk, m.lower(x))
                        finally:
                            CAP["on"] = False
                    assert torch.equal(m.dec_logits(S0, ch, dev), m.dec_logits(S1, ch, dev)), "attn_fade: the capture changed the decoder logits"
            out["conds"][cond] = shr; out.setdefault("conds_mass", {})[cond] = res    # conds = the target share (Note 265-2); conds_mass = the raw mass
            if wsok is not None:
                out["ws_ok_d8_t7"] = wsok
    finally:
        DM._block = orig
    out["def"] = ("§265 D1a ATTN-FADE: conds = the TARGET SHARE (Note 265-2) = fp32 softmax mass of the pass-t ':' query on S_t's op-sentence tokens / its mass on every op-sentence token (S_1..S_d + inserted ops), layer x head [L][H]; conds_mass = the raw mass; d4k0 (TEST2 d 4, t 2-4), d4k6 "
                  "(d 4 + 6 inserts, seed 26420, t 4), d8 (TEST2 d 8, t 4 / 7); first 500 items; read-only asserted; ws_ok_d8_t7 = pass-7 WS ok given passes 1-6 ok")
    return out
