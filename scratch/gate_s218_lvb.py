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
    out["decorators"] = deco; out["decorators_ok"] = deco == {"cand_scores": ["torch.no_grad()"]}
    keys = ["R1", "R2_chg", "R2_unc", "dL_dbank_core_v0", "dL_dbank_core_v1plus", "dL_dbank_coda_v1plus", "dLans_dV1_sg"]
    out["pass"] = all(out[k_] is not None and out[k_] > 0 for k_ in keys) and out["decorators_ok"]; out["items"] = [(it["D"], len(it["ids"])) for it in items]
    out["pending"] = "A2 self batches (SOUND-ONLY R2) once boxes/langvb_label.self_item exists"
    return out


def g_l7b(dev, n=2000, bs=8, cal_n=16):
    """G-L7(b): plain_cand_ce at init (pretrained GPT-2, calibrated LangSLVA on cal_n items, c 1e4 = step 1's regime) on n plain items of A1's
    first-stage law (D ~ U{3..4}, K = min(D + U{0..4}, 12); Stream 21140 / rng 21141, test + GATE signatures excluded): the registered statistic
    (lvb_loss's batch plain_cand_ce, item-weighted) -> rule 0.693 +- 0.05. Diagnostics per item at t = D (the window start): cand_ce and
    gap = sum log p("### o_R.") - sum log p("### o_P.")."""
    import numpy as np_
    torch.manual_seed(0); m = LV.LangSLVA(s5.load_gpt2(dev)).to(dev); m.eval(); m.c_anneal.fill_(1e4); ac = lambda: torch.autocast("cuda", enabled=False)  # noqa: E731
    EX = LV.PS().exclusion_sigs(); s = LV.PS().Stream(21140, EX); rng = np_.random.default_rng(21141)
    sc = LV.PS().Stream(21110, EX); r_c = np_.random.default_rng(21110); cal_its = [LV.lvb_item(sc.draw(int(r_c.integers(3, 9)))) for _ in range(cal_n)]
    x_, p_, l_, si_, sp_, dR_, _ = LV.lvb_batch(cal_its, dev)
    with torch.no_grad():
        cal = m.calibrate(x_, p_, si_, sp_, dR_, l_)
    del x_, p_, l_, si_, sp_, dR_
    bm, items = [], []
    for i in range(0, n, bs):
        its = [LV.lvb_item(s.draw(int(D))) for D in rng.integers(3, 5, size=bs)]; Ks = [min(it["D"] + int(rng.integers(0, 5)), 12) for it in its]
        with torch.no_grad():
            _, st, _ = LV.lvb_loss(m, its, Ks, dev, ac)
        bm.append(st["plain_cand_ce"])
        for D in (3, 4):
            sub = [it for it in its if it["D"] == D]
            if sub:
                sR, sP = LV.cand_scores(m, sub, D, dev, ac); cc = -torch.log_softmax(torch.stack([sR, sP], 1), 1)[:, 0]
                items += [{"D": D, "cand_ce_tD": float(c_), "gap": float(g_)} for c_, g_ in zip(cc, sR - sP)]
    mean = float(np_.mean(bm)); gaps = np_.array([x["gap"] for x in items])
    return {"plain_cand_ce_init": mean, "rule": "0.693 +- 0.05", "pass": abs(mean - 0.693) <= 0.05, "batch_means": bm, "n": n, "bs": bs, "cal_n": cal_n,
            "cal_max_m99": cal["max_m99"], "gap_abs_mean": float(np_.abs(gaps).mean()), "gap_abs_median": float(np_.median(np_.abs(gaps))),
            "share_gap_abs_gt_1": float((np_.abs(gaps) > 1).mean()), "acc_tD": float((gaps > 0).mean()), "items": items}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("gate"); ap.add_argument("--device", default="cpu"); ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    r = {"l1a": g_l1a, "l1b": g_l1b, "l2": g_l2, "l7b": lambda d: g_l7b(d, a.n)}[a.gate](a.device); r["device"] = a.device
    json.dump(r, open(os.path.join(OUT, "g_%s.json" % a.gate), "w", newline="\n"), indent=1); print(a.gate, "PASS" if r["pass"] else "FAIL", json.dumps(r)[:600], flush=True)


if __name__ == "__main__":
    main()
