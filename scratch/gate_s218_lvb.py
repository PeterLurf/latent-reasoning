"""§218 LANG-VB model gates (LANG_v1 §7 at 83f2dea4) for boxes/langvb.py; each writes scratch/gate_s218/<gate>.json.
  l1a  G-L1(a): all-position slots, K = 1, bank masked: the coda's slot rows == GPT-2's final hidden (ln_f) at every prelude position, fp64 <= 1e-9 rel
       (the decoder rows follow the slots and are causally invisible to them).
  l1b  G-L1(b): LangSLVA with S = all positions on nested inputs, D3-T's weights (s213_d3t_s1/model.pt; LangSLVA's 2-class R_P left out) == D3-T's own
       SLVA rollout (vbank.VBankN.rollout with SLVA's gather): every stored version V_0..V_K and the coda's slot rows after pass K, bitwise (fp32, CPU).
    python scratch/gate_s218_lvb.py l1a --device cpu"""
import argparse
import json
import os
import sys

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import torch  # noqa: E402

import boxes.langvb as LV  # noqa: E402
import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.slva as SL  # noqa: E402
from boxes import s5  # noqa: E402

OUT = "scratch/gate_s218"


def all_slots(pad):
    B, T = pad.shape; si = torch.arange(T)[None].expand(B, T).clone(); return si, pad.clone()


def g_l1a(dev):
    g = s5.load_gpt2(dev).double(); m = LV.LangSLVA(g).to(dev).double(); m.c_anneal.fill_(1e4)
    items = [it for it in N.test_set(1000) if it["n"] == 5][:4]; x, pad, lens = VB.batch_x(items, dev); si, sp = all_slots(pad)
    dec = torch.tensor([LV.dec_ids("alpha")] * len(items), device=dev)
    m.bank_off = True; r = m.rollout_ls(x, pad, si, sp, 1, coda_rows={1: torch.arange(len(items))}, dec=dec, lens=lens)
    h = r["coda"][1][1][:, :x.shape[1]]
    ref, _ = g(g.embed(x)); rel = float(((h - ref).abs() * pad[..., None]).max() / ref.abs().max())
    return {"rel_max": rel, "pass": rel <= 1e-9, "items": len(items)}


def g_l1b(dev):
    torch.manual_seed(0); g1 = s5.load_gpt2(dev); g2 = s5.load_gpt2(dev)
    sd = torch.load("boxes/runs_fsq/s213_d3t_s1/model.pt", map_location=dev, weights_only=True)
    ref = SL.SLVA(g1).to(dev); ref.load_state_dict(sd, strict=True); ref.eval()
    m = LV.LangSLVA(g2).to(dev); mk = m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("rec.")}, strict=False); m.eval()
    assert sorted(mk.missing_keys) == ["rec.bias", "rec.weight"] and not mk.unexpected_keys, mk
    for mm in (ref, m):
        mm.c_anneal.fill_(0.0)                                               # the trained regime (c = 0 from step 8000)
    items = [it for it in N.test_set(1000) if it["n"] == 6][:8]; x, pad, lens = VB.batch_x(items, dev); si, sp = all_slots(pad); K = 8
    dec = torch.tensor([LV.dec_ids("alpha")] * len(items), device=dev)
    with torch.no_grad():
        ra = ref.rollout(x, pad, K, coda_rows={K: torch.arange(len(items))})
        rb = m.rollout_ls(x, pad, si, sp, K, coda_rows={K: torch.arange(len(items))}, dec=dec, lens=lens)
    eqV = [bool(torch.equal(a_ * pad[..., None], b_ * pad[..., None])) for a_, b_ in zip(ra["S"], rb["S"])]
    ca = ra["coda"][K][1]; cb = rb["coda"][K][1][:, :x.shape[1]]
    eq_coda = bool(torch.equal(ca * pad[..., None], cb * pad[..., None])); md = float(((ca - cb).abs() * pad[..., None]).max())
    return {"versions_equal": eqV, "coda_slot_rows_equal": eq_coda, "coda_max_abs_diff": md, "K": K, "items": len(items), "pass": all(eqV) and eq_coda}


def gnorm(m):
    return float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in m.parameters() if p.grad is not None)))


def g_l2(dev):
    """G-L2 on A1 batches (SYM v1 D 3-4, Stream 21130, 4 items, K = D + 2, c 0 so the SLVA input is live): each term's grad norm ALONE (R1 = lvb_loss
    at rec_w 0; R2-changed / R2-unchanged = the two halves of R2 recomputed from the kept rollout with lvb_loss's targets and weights); dL/d bank K/V
    (versions 0 and >= 1, core and coda layers); dL_ans/dV_1 with the SLVA input stop-gradiented (sg_input); decorators of langvb by ast."""
    import ast
    import torch.nn.functional as F
    torch.manual_seed(0); m = LV.LangSLVA(s5.load_gpt2(dev)).to(dev); m.c_anneal.fill_(0.0); m.train(); ac = lambda: torch.autocast("cuda", enabled=False)  # noqa: E731
    s = LV.PS().Stream(21130); items = [LV.lvb_item(s.draw(D)) for D in (3, 4, 3, 4)]; Ks = [it["D"] + 2 for it in items]; out = {}
    m.zero_grad(set_to_none=True); L, st, _ = LV.lvb_loss(m, items, Ks, dev, ac, rec_w=0.0); L.backward(); out["R1"] = gnorm(m)
    m.zero_grad(set_to_none=True); Lfull, _, r = LV.lvb_loss(m, items, Ks, dev, ac, rec_w=1.0, keep=True)
    x, pad, lens, si, sp, dR, dP = LV.lvb_batch(items, dev); Kb = max(Ks); Kt = torch.tensor(Ks, device=dev); E = LV.reach_targets(items, Kb, si.shape[1], dev)
    half = {"chg": [], "unc": []}
    for t in range(1, Kb + 1):                                              # lvb_loss's R2 terms, split
        Et, Ep = E[:, t], E[:, t - 1]; valid = sp & (t <= Kt)[:, None] & (Et >= 0)
        ce = F.cross_entropy(m.rec_logits(r["V"][t]).transpose(1, 2), Et.clamp_min(0), reduction="none").float(); w = 1.0 + (Et == LV.IN).float()
        for k_, msk in (("chg", (Et != Ep) & valid), ("unc", (Et == Ep) & valid)):
            if bool(msk.any()):
                half[k_].append((w * ce * msk.float()).sum() / (w * msk.float()).sum())
    kv_ins = {"core_v0": [r["kv"][i][0][0] for i in LV.CORE], "core_v1plus": [r["kv"][i][s_][0] for i in LV.CORE for s_ in range(1, Kb)],
              "coda_v1plus": [r["kv"][i][s_][1] for i in LV.CODA for s_ in range(1, Kb)]}
    for k_, v in kv_ins.items():                                            # the full loss (R1 + R2): the coda's bank reaches only R1
        gs = torch.autograd.grad(Lfull, v, retain_graph=True, allow_unused=True)
        out["dL_dbank_" + k_] = float(sum(g_.float().norm() for g_ in gs if g_ is not None)); out["dL_dbank_%s_unused" % k_] = sum(g_ is None for g_ in gs)
    for k_ in ("chg", "unc"):
        m.zero_grad(set_to_none=True); torch.stack(half[k_]).mean().backward(retain_graph=True); out["R2_" + k_] = gnorm(m); out["R2_%s_terms" % k_] = len(half[k_])
    del r, half, Lfull
    with torch.enable_grad():                                               # dL_ans / dV_1 with the SLVA input stop-gradiented
        rows = torch.arange(len(items), device=dev); Kc = 3
        r = m.rollout_ls(x, pad, si, sp, Kc, coda_rows={Kc: rows}, dec=dR, lens=lens, sg_input=True)
        La = -m.dec_logprobs(r["coda"][Kc][1], sp, dR).mean(); gV1 = torch.autograd.grad(La, r["V"][1], allow_unused=True)[0]
        out["dLans_dV1_sg"] = None if gV1 is None else float(gV1.norm())
    deco = {}
    for n_ in ast.parse(open("boxes/langvb.py", encoding="utf-8").read()).body:
        if isinstance(n_, ast.FunctionDef) and n_.decorator_list:
            deco[n_.name] = [ast.unparse(d) for d in n_.decorator_list]
        if isinstance(n_, ast.ClassDef):
            for f_ in n_.body:
                if isinstance(f_, ast.FunctionDef) and f_.decorator_list:
                    deco[n_.name + "." + f_.name] = [ast.unparse(d) for d in f_.decorator_list]
    out["decorators"] = deco; out["decorators_ok"] = deco == {"cand_scores": ["torch.no_grad()"], "LangSLVA.calibrate": ["torch.no_grad()"]}
    keys = ["R1", "R2_chg", "R2_unc", "dL_dbank_core_v0", "dL_dbank_core_v1plus", "dL_dbank_coda_v1plus", "dLans_dV1_sg"]
    out["pass"] = all(out[k_] is not None and out[k_] > 0 for k_ in keys) and out["decorators_ok"]; out["items"] = [(it["D"], len(it["ids"])) for it in items]
    out["pending"] = "A2 self batches (SOUND-ONLY R2) once boxes/langvb_label.self_item exists"
    return out


def g_l7b(dev, n=2000, bs=8, cal_n=16, fp64=False):
    """G-L7(b') (Note 218-1, 8f2a2fcb) at init (pretrained GPT-2, calibrated LangSLVA on cal_n items, c 1e4 = step 1's regime) on n plain items of
    A1's first-stage law (D ~ U{3..4}, K = min(D + U{0..4}, 12); Stream 21140 / rng 21141, test + GATE signatures excluded):
      (i) accuracy at t = D (gold string's summed log-prob > the other's) 0.50 +- 0.03;
      (ii) mean plain_cand_ce (lvb_loss's window-mean per item, the CurGate statistic) >= ln 2 - 0.02;
      (iii) per item, lvb_loss's cand_ce == mean over its window t in [D, K] of softplus(-g_t), g_t = sum log p("### o_R.") - sum log p("### o_P."),
            recomputed independently (its own rollout; logits = h @ wte^T and the row alignment re-derived here, not dec_logprobs / cand_scores),
            max |diff| <= 1e-5. The gap histogram (g at t = D) is reported."""
    import numpy as np_
    torch.manual_seed(0); m = LV.LangSLVA(s5.load_gpt2(dev)).to(dev); m.eval(); m.c_anneal.fill_(1e4); ac = lambda: torch.autocast("cuda", enabled=False)  # noqa: E731
    if fp64:                                                                # (iii) as a code check without fp32 reduction-order noise (the model's
        m = m.double()                                                      # dec_logprobs casts logits to fp32; the recompute mirrors that cast)
    EX = LV.PS().exclusion_sigs(); s = LV.PS().Stream(21140, EX); rng = np_.random.default_rng(21141)
    sc = LV.PS().Stream(21110, EX); r_c = np_.random.default_rng(21110); cal_its = [LV.lvb_item(sc.draw(int(r_c.integers(3, 9)))) for _ in range(cal_n)]
    x_, p_, l_, si_, sp_, dR_, _ = LV.lvb_batch(cal_its, dev)
    with torch.no_grad():
        cal = m.calibrate(x_, p_, si_, sp_, dR_, l_)
    del x_, p_, l_, si_, sp_, dR_
    W = m.g.w("wte.weight")

    def lp_sum(h, sp, dec):                                                 # independent: row before each decoder token; the last real slot for token 0
        S = sp.shape[1]; out = []
        for b in range(h.shape[0]):
            last = int(sp[b].nonzero().max()); rows = [last] + [S + k for k in range(dec.shape[1] - 1)]
            lg = h[b, rows] @ W.T; lp = torch.log_softmax(lg if lg.dtype == torch.float64 else lg.float(), -1)   # the model's dtype end to end
            out.append(float(sum(lp[k, int(dec[b, k])] for k in range(dec.shape[1]))))
        return out
    rows_out = []
    for i in range(0, n, bs):
        its = [LV.lvb_item(s.draw(int(D))) for D in rng.integers(3, 5, size=bs)]; Ks = [min(it["D"] + int(rng.integers(0, 5)), 12) for it in its]
        with torch.no_grad():
            _, st, _ = LV.lvb_loss(m, its, Ks, dev, ac, keep=True)
            x, pad, lens, si, sp, dR, dP = LV.lvb_batch(its, dev); Kb = max(Ks)
            win = {t: torch.tensor([b for b, it in enumerate(its) if it["D"] <= t <= Ks[b]], device=dev) for t in range(1, Kb + 1)}
            win = {t: r_ for t, r_ in win.items() if len(r_)}
            r = m.rollout_ls(x, pad, si, sp, Kb, coda_rows=win, dec=dR, lens=lens); g_by = {b: {} for b in range(len(its))}
            for t, (rw, hR) in r["coda"].items():
                hP = m.coda_ls(t, r["V"][t][rw], {i_: [(k[rw], v[rw]) for k, v in r["kv"][i_]] for i_ in LV.CODA}, sp[rw], dP[rw], lens[rw])
                for q, (a_, b_) in enumerate(zip(lp_sum(hR, sp[rw], dR[rw]), lp_sum(hP, sp[rw], dP[rw]))):
                    g_by[int(rw[q])][t] = a_ - b_
            del r
        for b, it in enumerate(its):
            gs = g_by[b]; ref = float(np_.mean([np_.logaddexp(0.0, -g_) for g_ in gs.values()]))
            rows_out.append({"D": it["D"], "K": Ks[b], "cand_ce": st["cand_ce_items"][b], "cand_ce_indep": ref, "g_tD": gs[it["D"]]})
    cc = np_.array([r_["cand_ce"] for r_ in rows_out]); ci = np_.array([r_["cand_ce_indep"] for r_ in rows_out]); g = np_.array([r_["g_tD"] for r_ in rows_out])
    acc = float((g > 0).mean()); mean = float(cc.mean()); mx = float(np_.abs(cc - ci).max())
    hist = np_.histogram(g, bins=[-1e9, -8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8, 1e9])[0].tolist()
    res = {"i_acc_tD": acc, "i_pass": abs(acc - 0.5) <= 0.03, "ii_mean_plain_cand_ce": mean, "ii_pass": bool(mean >= np_.log(2) - 0.02),
           "iii_max_abs_diff": mx, "iii_pass": mx <= 1e-5, "gap_hist_edges": [-8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8], "gap_hist": hist,
           "gap_abs_mean": float(np_.abs(g).mean()), "gap_abs_median": float(np_.median(np_.abs(g))), "n": len(rows_out), "bs": bs, "cal_n": cal_n,
           "cal_max_m99": cal["max_m99"], "items": rows_out, "rule": "Note 218-1 G-L7(b'): (i) 0.50 +- 0.03, (ii) >= ln 2 - 0.02, (iii) <= 1e-5"}
    res["pass"] = res["i_pass"] and res["ii_pass"] and res["iii_pass"]
    return res


def g_l4b_hook(dev, n=64):
    """G-L4b (hook half): LangSLVA.v0_switch on the first n CF pairs of prosqa_sym.cf_select over the test items D 4..8 (cells in sorted order), K' = D:
    switch at t = 0 == x''s own run and switch at t = D == x's own run, bitwise (fp32): every stored version V_1..V_D and the coda after pass D.
    Untrained LangSLVA (pretrained GPT-2, step-0 heads) with c = 0, so SLVA's s = 0 key / value (switched) is read with weight."""
    import boxes.langvb_eval as LE
    torch.manual_seed(0); m = LV.LangSLVA(s5.load_gpt2(dev)).to(dev); m.eval(); m.c_anneal.fill_(0.0)
    TE = LV.PS().test_set(1000, range(4, 9)); raw = [it for D in range(4, 9) for it in TE[D]]; sel = LV.PS().cf_select(raw)
    take = [list(sel[key]) for key in sorted(sel)]; picked = []            # n pairs spread over the cells: round-robin by cell
    while len(picked) < n and any(take):
        for lst in take:
            if lst and len(picked) < n:
                picked.append(lst.pop(0))
    res = {"n": len(picked), "t0_equal": 0, "tD_equal": 0, "fail": []}
    with torch.no_grad():
        for i, r_ in picked:
            it = raw[i]; D = it["D"]; R = [LE.ev(it)]; Dn = [LE.ev(r_["donor"])]
            bR = LV.lvb_batch(R, dev); bD = LV.lvb_batch(Dn, dev); rows = torch.arange(1, device=dev)
            full = {}
            for nm, b in (("x", bR), ("xp", bD)):                         # the coda reads the RECIPIENT's decoder string in every run (the switched run's)
                x, pad, lens, si, sp, dR, dP = b; rr = m.rollout_ls(x, pad, si, sp, D, coda_rows={D: rows}, dec=bR[5], lens=lens); full[nm] = (rr["S"], rr["coda"][D][1])
            H6sw = m.prelude(bD[0]); x, pad, lens, si, sp, dR, dP = bR
            for t, ref, key in ((0, "xp", "t0_equal"), (D, "x", "tD_equal")):
                m.v0_switch = (t, H6sw, None)
                try:
                    rs = m.rollout_ls(x, pad, si, sp, D, coda_rows={D: rows}, dec=dR, lens=lens)
                finally:
                    m.v0_switch = None
                Sref, cref = full[ref]
                eq = all(torch.equal(a_, b_) for a_, b_ in zip(rs["S"][1:], Sref[1:])) and torch.equal(rs["coda"][D][1], cref)
                if eq:
                    res[key] += 1
                else:
                    res["fail"].append({"i": i, "D": D, "j": r_["j"], "t": t})
    res["pass"] = res["t0_equal"] == res["n"] and res["tD_equal"] == res["n"]
    res["note"] = "the coda is question-free; at t = 0 every version-0 read is x''s (core bank entry 0, SLVA s = 0 key / value, the RMS reference)"
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("gate"); ap.add_argument("--device", default="cpu"); ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    r = {"l1a": g_l1a, "l1b": g_l1b, "l2": g_l2, "l7b": lambda d: g_l7b(d, a.n), "l7b_fp64": lambda d: g_l7b(d, a.n, fp64=True), "l4b_hook": lambda d: g_l4b_hook(d, 64)}[a.gate](a.device); r["device"] = a.device
    json.dump(r, open(os.path.join(OUT, "g_%s.json" % a.gate), "w", newline="\n"), indent=1); print(a.gate, "PASS" if r["pass"] else "FAIL", json.dumps(r)[:600], flush=True)


if __name__ == "__main__":
    main()
