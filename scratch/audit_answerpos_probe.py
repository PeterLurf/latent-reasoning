"""Audit $0 DESCRIPTIVE diagnostic: is v_1 linearly decodable at the answer position of the §233 P1-ASK model boxes/runs_fsq/s233_askname_s1?
Model: one-pass DIRECT GPT-2; dec = [EOT] + enc(' ' + text + ' ' + QUESTION(u, X)) + enc(' The answer is:') (st_ask_name_local.tok_askname, class (i): X = S_1, target v_1).
Items: N d=2 items, OW.Stream(23301, excl, frames=FRAME_TRAIN, **OWKW) (as audit_s233_locus.py). Model fp32, no_grad.
Feature points: answer position (last input token ':') = top residual after block 11 pre-ln_f, same after ln_f, layer 0 (wte+wpe) control;
and the question's last name token (S_1's name in the question): pre-ln_f, after ln_f, layer 0 control.
Probe: sig-split 60/20/20, standardised multinomial logistic regression (LBFGS), L2 wd in {1e-4,1e-3,1e-2} chosen on val, test acc; chance = test majority share, 1/11.
Usage: python -B scratch/audit_answerpos_probe.py [device=cuda:1] [N=3000]"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
import boxes.inject_ow as IO  # noqa: E402
import boxes.inject_st_eval as SE  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import EOT, pad  # noqa: E402

RUN = os.path.join(ROOT, "boxes/runs_fsq/s233_askname_s1"); OUT = os.path.join(ROOT, "scratch/audit_answerpos_probe/probe.json")
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000; P = 11
if dev == "cpu":
    torch.set_num_threads(3)


def fit_probe(X, y, tr, va, te, dv):
    mu = X[tr].mean(0, keepdim=True); sd = X[tr].std(0, keepdim=True) + 1e-6; Z = ((X - mu) / sd).to(dv); Y = torch.tensor(y, device=dv); best = None
    for wd in (1e-4, 1e-3, 1e-2):
        W = torch.zeros(Z.shape[1], P, device=dv, requires_grad=True); b = torch.zeros(P, device=dv, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=200, history_size=20, line_search_fn="strong_wolfe")

        def clo():
            opt.zero_grad(); l = F.cross_entropy(Z[tr] @ W + b, Y[tr]) + 0.5 * wd * (W ** 2).sum(); l.backward(); return l
        with torch.enable_grad():
            opt.step(clo)
        with torch.no_grad():
            pr = (Z @ W + b).argmax(-1); va_acc = float((pr[va] == Y[va]).float().mean())
            if best is None or va_acc > best[0]:
                best = (va_acc, wd, pr[te].cpu().numpy(), float((pr[tr] == Y[tr]).float().mean()))
    ytest = y[te]
    return {"test_acc": float((best[2] == ytest).mean()), "val_acc": best[0], "train_acc": best[3], "wd": best[1], "n_test": int(len(te)),
            "chance_majority": float(np.bincount(ytest, minlength=P).max() / len(ytest)), "chance_uniform": 1 / P, "test_pred": best[2].tolist()}


def trunk(g, x):
    """GPT2KV.forward's body without cache: -> (pre-ln_f residual (B,T,D), after ln_f). x: token ids (B,T), right-padded (causal => pads harmless)."""
    B, T = x.shape; D = g.D; hd = D // g.H; h = g.embed(x) + g.w("wpe.weight")[:T][None]
    for i in range(g.L):
        pre = f"h.{i}."
        a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
        q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
        q, k, v = (t.view(B, T, g.H, hd).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
        a = F.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
        h = h + F.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh") @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")
    return h, F.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias"))


@torch.no_grad()
def main():
    t0 = time.time(); sets = IO.build_sets(); a, m, g = SE.load(RUN, dev); g.eval(); assert m is None
    st = OW.Stream(23301, sets["excl"], frames=OW.FRAME_TRAIN, **IO.OWKW); raw = [st.draw(2) for _ in range(N)]
    decs, qpos, keep = [], [], []
    for i, it in enumerate(raw):
        X = it["names"]["spine"][1]; base = " " + it["text"] + " " + OW.QUESTION.format(u=it["unit"], A=X)
        full = IO.ENC.encode(base); pre = IO.ENC.encode(base[:base.rindex(" have?")]); tail = IO.ENC.encode(" have?")
        if full[:len(pre)] != pre or full[len(pre):len(pre) + len(tail)] != tail:
            continue
        keep.append(i); decs.append([EOT] + full + IO.SFX); qpos.append(len(pre))          # [EOT] shifts +1, last name token = index len(pre)
    raw = [raw[i] for i in keep]; n = len(raw); sigs = [it["sig"] for it in raw]; y1 = np.array([int(it["values"][1]) for it in raw])
    print("kept", n, "of", N, flush=True)
    F_ = {k: [] for k in ("ans_pre_lnf", "ans_post_lnf", "ans_layer0", "qname_pre_lnf", "qname_post_lnf", "qname_layer0")}; pred = []; pred_r = []
    ids = [IO.ENC.encode(" %d" % v)[0] for v in range(P)]
    for i in range(0, n, 32):
        ch = decs[i:i + 32]; B = len(ch); ids_, _ = pad(ch); x = torch.tensor(ids_, device=dev); ar = torch.arange(B, device=dev)
        ap = torch.tensor([len(q) - 1 for q in ch], device=dev); qp = torch.tensor(qpos[i:i + B], device=dev)
        hp, hf = trunk(g, x); l0 = g.embed(x) + g.w("wpe.weight")[:x.shape[1]][None]
        for nm, pos in (("ans", ap), ("qname", qp)):
            F_[nm + "_pre_lnf"].append(hp[ar, pos].float().cpu()); F_[nm + "_post_lnf"].append(hf[ar, pos].float().cpu()); F_[nm + "_layer0"].append(l0[ar, pos].float().cpu())
        lg = g.logits(hf[ar, ap]).float(); pred.append(lg.argmax(-1).cpu()); pred_r.append(lg[:, ids].argmax(-1).cpu())
        if i == 0:                                                                          # cross-check with the training-path forward
            h2, _ = g(g.embed(x), None); assert torch.allclose(h2[ar, ap], hf[ar, ap], atol=1e-3, rtol=1e-3), "trunk mismatch"
    pred = torch.cat(pred).numpy(); pred_r = torch.cat(pred_r).numpy(); tgt_tok = np.array([ids[v] for v in y1])
    acc_full = float((pred == tgt_tok).mean()); acc_restr = float((pred_r == y1).mean()); print("model argmax acc v_1: full-vocab %.4f restricted-to-0..10 %.4f" % (acc_full, acc_restr), flush=True)
    X = {k: torch.cat(v) for k, v in F_.items()}
    rng = np.random.default_rng(0); us = sorted(set(sigs)); perm = rng.permutation(len(us)); rank = {us[j]: r for r, j in enumerate(perm)}; r = np.array([rank[s] for s in sigs]) / len(us)
    tr, va, te = [np.where(c)[0] for c in (r < 0.6, (r >= 0.6) & (r < 0.8), r >= 0.8)]
    res = {}
    for name in X:
        t1 = time.time(); r_ = fit_probe(X[name], y1, tr, va, te, dev); r_["feature_dim"] = int(X[name].shape[1]); r_["sec"] = round(time.time() - t1); res[name + "|v1"] = r_
        print(name, "v1 test %.3f val %.3f train %.3f wd %g chance %.3f dim %d" % (r_["test_acc"], r_["val_acc"], r_["train_acc"], r_["wd"], r_["chance_majority"], r_["feature_dim"]), flush=True)
    out = {"run": "s233_askname_s1", "n_items": n, "n_drawn": N, "seed": 23301, "split_sizes": [len(tr), len(va), len(te)], "test_item_index": te.tolist(), "test_y_v1": y1[te].tolist(),
           "model_argmax_acc_v1_fullvocab": acc_full, "model_argmax_acc_v1_restricted": acc_restr, "model_acc_on_test_split": float((pred_r[te] == y1[te]).mean()), "probes": res, "device": dev,
           "sec": round(time.time() - t0),
           "notes": "class-(i) input of st_ask_name_local.tok_askname: dec=[EOT]+enc(' '+text+' How many <u> does <S_1> have?')+enc(' The answer is:'); ans_* at the last token ':'; qname_* at the last token "
                    "of S_1's name in the question; pre_lnf = residual after block 11; layer0 = wte+wpe"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w", newline="\n").write(json.dumps(out)); print("WROTE", OUT, out["sec"], "s")


main()
