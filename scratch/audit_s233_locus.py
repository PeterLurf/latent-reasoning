"""Audit $0 DESCRIPTIVE diagnostic: where is v_1 lost in the §233 STEP-TOLD model boxes/runs_fsq/st_plst2_s1?
Items: N d=2 items, OW.Stream(23301, excl=build_sets excl, frames=FRAME_TRAIN, topo+dag) (TEST2's generator/frames). Model fp32, no_grad, pass 1 with its S_1 cue.
Feature points (pass 1, S from zero): (i) hw = trunk top residual (after block 11, pre-ln_f; what write consumes) at the cue's last token, before the write;
(ii) embedding+wpe at that token (control); (iii) S after pass 1 (flat); (iv) decoder-side read of S for the aux prompt [EOT, Step, 1, :]: the read contribution
(read_fixed output - input) and the final hidden at the prompt's last position (what aux_logits predicts v_1 from). (v) positive control v_0 from (i) at S_0's sentence end.
Probe: sig-split 60/20/20, standardised multinomial logistic regression (LBFGS), L2 wd in {1e-4,1e-3,1e-2} chosen on val, test acc; chance = test majority share, 1/11.
Usage: python -B scratch/audit_s233_locus.py [device=cuda:1] [N=3000]"""
import json
import os
import re
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
import boxes.inject_ow as IO  # noqa: E402
import boxes.inject_st as ST  # noqa: E402
import boxes.inject_st_eval as SE  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes.deltas5 import fwd_read  # noqa: E402
from boxes.nested170 import pad  # noqa: E402

RUN = os.path.join(ROOT, "boxes/runs_fsq/st_plst2_s1"); OUT = os.path.join(ROOT, "scratch/audit_s233_locus/locus.json")
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000; P = 11
if dev == "cpu":
    torch.set_num_threads(3)


def s0_end_pos(it):
    t = it["text"]; n0 = it["names"]["spine"][0]; v0 = int(it["values"][0])
    ms = [m for m in re.finditer(r"\b%s has (\d+) \w+\." % re.escape(n0), t) if int(m.group(1)) == v0]
    assert len(ms) == 1, ("S_0 sentence", it["sig"], len(ms)); e = ms[0].end()
    pre = IO.ENC.encode(" " + t[:e]); full = IO.ENC.encode(" " + t)
    assert full[:len(pre)] == pre, "prefix tokenisation"; return len(pre)          # +1 (EOT) -1 (last tok) -> index len(pre)


def fit_probe(X, y, tr, va, te, dv):
    mu = X[tr].mean(0, keepdim=True); sd = X[tr].std(0, keepdim=True) + 1e-6; Z = ((X - mu) / sd).to(dv); Y = torch.tensor(y, device=dv); best = None
    for wd in (1e-4, 1e-3, 1e-2):
        W = torch.zeros(Z.shape[1], P, device=dv, requires_grad=True); b = torch.zeros(P, device=dv, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=200, history_size=20, line_search_fn="strong_wolfe")

        def clo():
            opt.zero_grad(); l = torch.nn.functional.cross_entropy(Z[tr] @ W + b, Y[tr]) + 0.5 * wd * (W ** 2).sum(); l.backward(); return l
        with torch.enable_grad():
            opt.step(clo)
        with torch.no_grad():
            pr = (Z @ W + b).argmax(-1); va_acc = float((pr[va] == Y[va]).float().mean())
            if best is None or va_acc > best[0]:
                best = (va_acc, wd, pr[te].cpu().numpy(), float((pr[tr] == Y[tr]).float().mean()))
    ytest = y[te]
    return {"test_acc": float((best[2] == ytest).mean()), "val_acc": best[0], "train_acc": best[3], "wd": best[1], "n_test": int(len(te)),
            "chance_majority": float(np.bincount(ytest, minlength=P).max() / len(ytest)), "chance_uniform": 1 / P, "test_pred": best[2].tolist()}


@torch.no_grad()
def main():
    t0 = time.time(); sets = IO.build_sets(); a, m, g = SE.load(RUN, dev); m.eval()
    st = OW.Stream(23301, sets["excl"], frames=OW.FRAME_TRAIN, **IO.OWKW); raw = [st.draw(2) for _ in range(N)]
    T = [ST.tok_st(it, "value", "step") for it in raw]; sigs = [it["sig"] for it in raw]
    F = {k: [] for k in ("i_hw_cue", "ii_embed_cue", "i_h6_cue", "iii_S1", "iv_read_aux1", "iv_hid_aux1", "v_hw_s0end")}
    y1 = np.array([int(it["values"][1]) for it in raw]); y0 = np.array([int(it["values"][0]) for it in raw])
    pre = m.aux_prompt(1); xa = torch.tensor([pre], device=dev); aux_pos = len(pre) - 1; ok = []
    for i in range(0, N, 32):
        ch = T[i:i + 32]; B = len(ch); ids_, mk = pad([it["think_cues"][0] for it in ch]); x = torch.tensor(ids_, device=dev); msk = torch.tensor(mk, device=dev)
        pos = torch.tensor([len(it["think_cues"][0]) - 1 for it in ch], device=dev); ar = torch.arange(B, device=dev)
        S0 = m.mem.zero_state(B, dev); cap = {}; hl = m.lower(x)
        _, S1 = DM.gpt_forward_mem_r(g, g.embed(x), m.mem, S0, cap=cap, mask=msk, h_low=hl)
        S1b = m.write(S0, x, msk, hl); assert torch.allclose(S1, S1b, atol=1e-5, rtol=1e-4), "write path mismatch"
        emb = g.embed(x) + DM.wpe_slice(g, x.shape[1])[None]
        F["i_hw_cue"].append(cap["hw"][ar, pos].float().cpu()); F["i_h6_cue"].append(cap["h6"][ar, pos].float().cpu()); F["ii_embed_cue"].append(emb[ar, pos].float().cpu())
        F["iii_S1"].append(S1.flatten(1).float().cpu())
        xe = xa.expand(B, -1); h = DM.gpt_lower(g, g.embed(xe), m.mem.layer); hr = DM.read_fixed(m.mem, h, S1); F["iv_read_aux1"].append((hr - h)[:, aux_pos].float().cpu())
        F["iv_hid_aux1"].append(fwd_read(g, g.embed(xe), m.mem, S1)[:, aux_pos].float().cpu())
        p0 = torch.tensor([s0_end_pos(raw[i + j]) for j in range(B)], device=dev); F["v_hw_s0end"].append(cap["hw"][ar, p0].float().cpu())
        lg, yy = m.aux_logits(S1, ch, 1, dev); pr = lg.argmax(-1); vi = torch.tensor([it["vidx"][0] for it in ch], device=dev)
        ok += (pr[ar, vi] == yy[ar, vi]).long().cpu().tolist()
    X = {k: torch.cat(v) for k, v in F.items()}
    n_ans = min(N, 1000); A, _ = SE.passes(m, T[:n_ans], dev, 50, {2}); ans_acc = float((A[2] == np.array([it["ans"] for it in T[:n_ans]])).mean())
    rng = np.random.default_rng(0); us = sorted(set(sigs)); perm = rng.permutation(len(us)); rank = {us[j]: r for r, j in enumerate(perm)}; r = np.array([rank[s] for s in sigs]) / len(us)
    tr, va, te = [np.where(c)[0] for c in (r < 0.6, (r >= 0.6) & (r < 0.8), r >= 0.8)]
    res = {}
    for name, tgt, y in (("i_hw_cue", "v1", y1), ("i_h6_cue", "v1", y1), ("ii_embed_cue", "v1", y1), ("iii_S1", "v1", y1), ("iv_read_aux1", "v1", y1), ("iv_hid_aux1", "v1", y1),
                         ("v_hw_s0end", "v0", y0), ("iii_S1", "v0", y0)):
        t1 = time.time(); r_ = fit_probe(X[name], y, tr, va, te, dev); r_["feature_dim"] = int(X[name].shape[1]); r_["sec"] = round(time.time() - t1); res[name + "|" + tgt] = r_
        print(name, tgt, "test %.3f val %.3f train %.3f wd %g chance %.3f dim %d" % (r_["test_acc"], r_["val_acc"], r_["train_acc"], r_["wd"], r_["chance_majority"], r_["feature_dim"]), flush=True)
    out = {"run": "st_plst2_s1", "n_items": N, "seed": 23301, "split_sizes": [len(tr), len(va), len(te)], "test_item_index": te.tolist(), "test_y_v1": y1[te].tolist(), "test_y_v0": y0[te].tolist(),
           "aux_vacc_pass1_on_these_items": float(np.mean(ok)), "answer_acc_Kp2_first_items": ans_acc, "n_answer_items": n_ans, "probes": res, "device": dev, "sec": round(time.time() - t0),
           "notes": "i_hw_cue = trunk top residual (blocks 0-11, pre-ln_f) at the cue's last token from the pass-1 write path (S=0 so the read adds 0); ii = wte+wpe at that token; "
                    "i_h6_cue = residual entering block 6 (pre-read) at that token (extra); iv_* = aux prompt [EOT,Step,1,:] last position after the read on S_1; v = hw at S_0's sentence end"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w", newline="\n").write(json.dumps(out)); print("WROTE", OUT, out["sec"], "s", "aux_vacc", out["aux_vacc_pass1_on_these_items"], "ans", ans_acc)


main()
