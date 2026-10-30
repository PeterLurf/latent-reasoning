

RH_CELLS = (("d3t1", 3, 1), ("d3t2", 3, 2), ("d4t2", 4, 2), ("d4t3", 4, 3))   # §247 READ-HALT C7 cells (RH_v1 §1): (name, d, t); K' = d
SEED_RH_EDIT = 24700                                                        # §247 x' draws: default_rng(24700 + 10 d + t) per cell (unused, grep 10-29)
RH_TAU, RH_KMAX = 0.90, 8                                                   # §247 C5 halt rule (RH_v1 §2): first K' in 0..8 with max p >= 0.90, else 8


def rh_input(it, t, aug):
    """pass t's input (tok_aw, cue step, write_src answer) with aug inserted right after the problem text; aug '' gives the original tokens (asserted by the caller)."""
    raw = it["raw"]; return [EOT] + IO.ENC.encode(" " + raw["text"] + aug + " " + ST.cue_str(raw, t)) + IO.SFX


def rh_sent(it, t, x):
    """the step sentence for S_t (the R_step format): ' <S_t> has <x> <unit>.'"""
    raw = it["raw"]; return " " + OW.GIVEN.format(A=raw["names"]["spine"][t], v=int(x), u=raw["unit"])


def rh_answer(raw, t, x):
    """the generator's answer with S_t := x: later steps recomputed with the item's ops / operands, a state back-reference (op *_s) re-read from the edited chain."""
    s = [int(v) for v in raw["values"]]; s[t] = int(x)
    for k in range(t + 1, raw["d"] + 1):
        op, j, b = raw["ops"][k - 1], raw["src_j"][k - 1], raw["operands"][k - 1]
        s[k] = OW.apply(op, s[k - 1], s[j] if op.endswith("_s") else b)
    return s[raw["d"]]


@torch.no_grad()
def rh_run(m, T, dev, bs, K, inp, probs=False):
    """passes 1..K with per-pass inputs inp(i, it, t) -> token list (one padding per pass; S from zero; pass_aw writes; lower recomputed per pass input), the WS argmax at
    ':' of every pass, the decoder answer at K' = K (probs: the decoder's 11-number softmax at every K' 0..K). -> (ans (N,), ws {t: (N,)}, P (N, K + 1, 11) or None)."""
    num = torch.tensor(NUM_IDS, device=dev); N = len(T); ans = np.full(N, -1); ws = {t: np.full(N, -1) for t in range(1, K + 1)}; P = np.zeros((N, K + 1, 11), dtype=np.float32) if probs else None
    for i in range(0, N, bs):
        ch = T[i:i + bs]; sl = slice(i, i + len(ch)); S = m.mem.zero_state(len(ch), dev)
        if probs:
            P[sl, 0] = torch.softmax(m.dec_logits(S, ch, dev)[:, NUM_IDS].float(), -1).cpu().numpy()
        for t in range(1, K + 1):
            ids_, mk = pad([inp(i + r, it, t) for r, it in enumerate(ch)]); x = torch.tensor(ids_, device=dev); msk = torch.tensor(mk, device=dev)
            S, h, pos = m.pass_aw(S, x, msk, m.lower(x)); ws[t][sl] = m.colon_logits(h, pos)[:, num].argmax(-1).cpu().numpy()
            if probs:
                P[sl, t] = torch.softmax(m.dec_logits(S, ch, dev)[:, NUM_IDS].float(), -1).cpu().numpy()
        ans[sl] = pred_of(m.dec_logits(S, ch, dev))
    return ans, ws, P


@torch.no_grad()
def part_c7(m, g, sets, a, dev, bs):
    """§247 C7 (RH_v1 §1): at pass t the write comes from the same cue c_t over the text + ' <S_t> has <x> <unit>.'; other passes and the decoder on the original inputs; K' = d.
    rt: x = x_hat_t (pass t's ':' argmax); edit: x' uniform over the other 10 values (seeded), a' = rh_answer(S_t := x'); ctrl_aug_all: x_hat_t's sentence at EVERY pass;
    oracle_rt: x = gold v_t. Agreements vs the intact answer; per item {sig, x_hat, x_edit, a_edit, ans_intact, ans_rt, ans_edit, ans_ctrl, ans_oracle, ans}."""
    T = tokd(sets, a, [3, 4]); out = {}; intact = {}
    for d in (3, 4):
        assert all(rh_input(it, t, "") == it["think_cues"][t - 1] for it in T[d] for t in range(1, d + 1)), "§247: the empty augmentation must give the original tokens"
        for it in T[d]:
            assert rh_answer(it["raw"], 1, it["raw"]["values"][1]) == it["ans"], "§247: rh_answer must reproduce the generator's answer"
        intact[d] = rh_run(m, T[d], dev, bs, d, lambda i, it, t: it["think_cues"][t - 1])
    for name, d, t in RH_CELLS:
        its = T[d]; y = np.array([it["ans"] for it in its]); ai, wi, _ = intact[d]; xh = wi[t]
        rng = np.random.default_rng(SEED_RH_EDIT + 10 * d + t); xe = (xh + 1 + rng.integers(0, 10, size=len(its))) % 11
        ae = np.array([rh_answer(it["raw"], t, xe[i]) for i, it in enumerate(its)]); vg = np.array([int(it["raw"]["values"][t]) for it in its])

        def at_t(xv):
            return lambda i, it, tt: rh_input(it, tt, rh_sent(it, t, xv[i])) if tt == t else it["think_cues"][tt - 1]
        a_rt = rh_run(m, its, dev, bs, d, at_t(xh))[0]; a_ed = rh_run(m, its, dev, bs, d, at_t(xe))[0]; a_or = rh_run(m, its, dev, bs, d, at_t(vg))[0]
        a_ct = rh_run(m, its, dev, bs, d, lambda i, it, tt: rh_input(it, tt, rh_sent(it, t, xh[i])))[0]
        out[name] = {"d": d, "t": t, "n": len(its), "intact_acc": float((ai == y).mean()), "rt_agree": float((a_rt == ai).mean()), "edit_follow": float((a_ed == ae).mean()),
                     "floor": float((ai == ae).mean()), "ctrl_aug_all": float((a_ct == ai).mean()), "oracle_rt": float((a_or == ai).mean()), "x_hat_acc": float((xh == vg).mean()),
                     "items": {"sig": [it["sig"] for it in its], "x_hat": xh.tolist(), "x_edit": xe.tolist(), "a_edit": ae.tolist(), "ans_intact": ai.tolist(), "ans_rt": a_rt.tolist(),
                               "ans_edit": a_ed.tolist(), "ans_ctrl": a_ct.tolist(), "ans_oracle": a_or.tolist(), "ans": y.tolist(), "v_gold": vg.tolist()}}
    out["def"] = ("§247 C7 (RH_v1 §1): pass-t write from the same cue over text + ' <S_t> has <x> <unit>.', other passes / decoder original, K' = d; rt x = x_hat_t, edit x' uniform != x_hat_t "
                  "(seed 24700 + 10 d + t), a' = the generator's answer with S_t := x'; ctrl_aug_all = x_hat_t's sentence at every pass; oracle_rt x = gold v_t; agreements vs intact")
    return out


@torch.no_grad()
def part_halt(m, g, sets, a, dev, bs):
    """§247 C5 (RH_v1 §2): TEST2 d 1..6, decode after every K' 0..8; maxprob = the max of the decoder's softmax over the 11 number tokens; K'_halt = the first K' with maxprob
    >= 0.90, else 8. Per d {acc_k8, acc_halt, mean_khalt, khalt_hist}; per item {sig, maxprob [K' 0..8], pred [K' 0..8], khalt, ans}."""
    T = tokd(sets, a, range(1, 7)); out = {}
    for d in range(1, 7):
        y = np.array([it["ans"] for it in T[d]]); _, _, P = rh_run(m, T[d], dev, bs, RH_KMAX, lambda i, it, t: seq_std(it, t), probs=True)
        mp = P.max(-1); pr = P.argmax(-1); hit = mp >= RH_TAU; kh = np.where(hit.any(1), hit.argmax(1), RH_KMAX); ah = pr[np.arange(len(y)), kh]
        out[str(d)] = {"n": len(y), "acc_k8": float((pr[:, RH_KMAX] == y).mean()), "acc_halt": float((ah == y).mean()), "mean_khalt": float(kh.mean()),
                       "khalt_hist": {str(k): int((kh == k).sum()) for k in range(RH_KMAX + 1)},
                       "never_halt_n": int((~hit.any(1)).sum()),               # Note 247-1: max-prob never >= 0.90 by K' 8 (counted at K' 8 with the K' 8 answer)
                       "items": {"sig": [it["sig"] for it in T[d]], "maxprob": mp.round(6).tolist(), "pred": pr.tolist(), "khalt": kh.tolist(), "ans": y.tolist()}}
    out["def"] = "§247 C5 (RH_v1 §2): readout halting, tau 0.90 fixed a priori; maxprob over the 11 number tokens of the decoder softmax after each K' 0..8; TEST2 d 1..6, fp32"
    return out
