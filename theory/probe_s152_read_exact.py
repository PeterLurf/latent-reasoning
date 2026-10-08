"""THEORY note 152-5 follow-up (descriptive; Audit): is the column sum exact in what the model actually READS?
For §152's models, after the 4 segments, the read-only reader pass [EOT, " =", d1 .. d7] (teacher-forced with the true S_4 digits, LSB first) is re-run op for
op (multrow.fwd_mem / mem_read), capturing at each digit's prediction position (position k predicts digit k; position 1 = " ="):
  read   the branch's read output, fade * W_o [g_add read_add || g_del read_del] (768-d), the query-dependent quantity
  L6in   the residual entering block 6 (after the read is added), then L7 .. L12 the residual after blocks 6 .. 11
Targets per column k = 1..8: cs_k (the carry-save column sum, no carries) and the accumulator c^_k = cs_k + carry-in (Bai's c^). Primal ridge (768-d, standardised;
lambda by 3-fold CV) fit on 4,000 train items, scored on the 1,000 test items: exact = round(pred) == target. Beside it: the model's own teacher-forced digit accuracy
at that position. Reading (Audit): exact at `read` but the digit fails = read-time resolution; imprecise already at `read` = storage or read precision.
CPU only (torch threads <= 3; run at Idle priority). Writes theory/results/read_exact_s152.json.
    python theory/probe_s152_read_exact.py s152_a_4x4_s1 s152_ctrl_4x4_s1
"""
import os
os.environ["BOXES_NB"] = "5"                                                # boxes.core must load with BOXES_NB=5
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
import boxes.multrow as MR  # noqa: E402
from boxes.multcol import load_bench, load_bench_head  # noqa: E402

N_FIT = 4000
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
LAYERS = ["read", "L6in", "L7", "L8", "L9", "L10", "L11", "L12"]


def targets(A, B):
    cs = np.zeros((len(A), 8), np.int64)
    for j in range(4):
        for i in range(4):
            cs[:, i + j] += A[:, i] * B[:, j]
    acc = np.zeros_like(cs); carry = np.zeros(len(A), np.int64)
    for k in range(8):
        acc[:, k] = cs[:, k] + carry; carry = acc[:, k] // 10
    return cs, acc


@torch.no_grad()
def captures(m, A, B, bs=100):
    """-> {layer: (N, 8, 768) float32} at positions 1..8, and the teacher-forced digit predictions (N, 8)."""
    g, mem = m.g, m.mem; N = len(A); out = {L: np.zeros((N, 8, 768), np.float32) for L in LAYERS}; pred = np.zeros((N, 8), np.int64)
    S4 = MR.sj_values(A, B)[:, 3]; ans = np.array(MR.RowFam().answer_tokens(S4))
    lut = {int(t): d for d, t in enumerate(MR.DTOK)}
    for s in range(0, N, bs):
        sl = slice(s, min(N, s + bs)); Ab, Bb = A[sl], B[sl]; nb = len(Ab)
        MR.set_ops(m, Ab, Bb, "cpu"); S = m.start(nb, "cpu")
        for t in range(4):
            S = m.write(S, t)
        ids = torch.cat([torch.tensor([[MR.EOT, MR.EQTOK]]).expand(nb, 2), torch.tensor(ans[sl][:, :7])], 1)
        x = g.embed(ids); Bn, T, D = x.shape; hd = D // g.H; h = x + g.w("wpe.weight")[:T][None]
        for i in range(g.L):
            if i == mem.layer:
                h2 = MR.mem_read(mem, h, S); out["read"][sl] = (h2 - h)[:, 1:9].float().numpy(); h = h2; out["L6in"][sl] = h[:, 1:9].float().numpy()
            pre = f"h.{i}."
            a = Fn.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
            q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
            q, k, v = (t_.view(Bn, T, g.H, hd).transpose(1, 2) for t_ in (q, k, v))
            o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h + o.transpose(1, 2).reshape(Bn, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
            a = Fn.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
            mm = Fn.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh")
            h = h + mm @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")
            if i >= mem.layer:
                out["L%d" % (i + 1)][sl] = h[:, 1:9].float().numpy()
        lg = g.logits(Fn.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias"))[:, 1:9]).float()
        pred[sl] = np.vectorize(lambda t_: lut.get(int(t_), -1))(lg.argmax(-1).numpy())
    return out, pred


def ridge_fit_pred(Xf, yf, Xs):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Zf = (Xf - mu) / sd; Zs = (Xs - mu) / sd; d = Zf.shape[1]
    def fit(Za, ya, lam):
        ym = ya.mean(); W = np.linalg.solve(Za.T @ Za + lam * len(Za) * np.eye(d), Za.T @ (ya - ym)); return W, ym
    def r2(y_, p_):
        return 1 - ((y_ - p_) ** 2).sum() / max(((y_ - y_.mean()) ** 2).sum(), 1e-12)
    fold = np.arange(len(Zf)) % 3
    def cv(l_):
        sc = []
        for k in range(3):
            W, ym = fit(Zf[fold != k], yf[fold != k], l_); sc.append(r2(yf[fold == k], Zf[fold == k] @ W + ym))
        return np.mean(sc)
    best = max(LAMS, key=cv); W, ym = fit(Zf, yf, best)
    return Zs @ W + ym, best


def main():
    torch.set_num_threads(3); t0 = time.time()
    teA, teB = load_bench(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt"))[:2]
    trA, trB = load_bench_head(os.path.join(ROOT, "bench", "data", "mult_train", "4x4_train.txt"), N_FIT)
    cs_tr, ac_tr = targets(trA, trB); cs_te, ac_te = targets(teA, teB)
    res = {"def": __doc__.split("\n")[1], "n_fit_train": len(trA), "n_score_test": len(teA), "lams": list(LAMS)}
    for d in sys.argv[1:]:
        run = os.path.join(ROOT, "boxes", "runs_fsq", d); a = json.load(open(os.path.join(run, "args.json"))); F = MR.F
        if int(a["d_q"]) != F.D_Q or int(a["levels"]) != F.LEV:
            F.D_Q, F.LEV = int(a["d_q"]), int(a["levels"]); F.HALF = (F.LEV - 1) / 2; F.SHIFT = math.atanh(F.OFF / F.HALF)
        m = MR.MultRowDelta(s5.load_gpt2("cpu"), MR.RowFam(), int(a.get("mem_layer", 6)))
        m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location="cpu")); m.eval(); m.p_hard = 1.0
        Ctr, _ = captures(m, trA, trB); Cte, pr_te = captures(m, teA, teB)
        dig_te = (ac_te % 10)
        out = {"digit_acc_teacher_forced": {str(k + 1): float((pr_te[:, k] == dig_te[:, k]).mean()) for k in range(8)}}
        for L in LAYERS:
            out[L] = {}
            for k in range(8):
                rec = {}
                for nm, Ytr, Yte in (("cs", cs_tr, cs_te), ("acc", ac_tr, ac_te)):
                    if Ytr[:, k].std() == 0:
                        continue
                    p_, lam = ridge_fit_pred(Ctr[L][:, k].astype(np.float64), Ytr[:, k].astype(np.float64), Cte[L][:, k].astype(np.float64))
                    rec[nm] = {"exact": float((np.rint(p_) == Yte[:, k]).mean()),
                               "r2": float(1 - ((Yte[:, k] - p_) ** 2).sum() / max(((Yte[:, k] - Yte[:, k].mean()) ** 2).sum(), 1e-12)), "lam": lam}
                out[L][str(k + 1)] = rec
            print(d, L, {k: (v.get("acc") or {}).get("exact") for k, v in out[L].items()}, round(time.time() - t0), flush=True)
        print(d, "digit acc (teacher-forced)", out["digit_acc_teacher_forced"], flush=True)
        res[d] = out; del Ctr, Cte, m
    res["sec"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "read_exact_s152.json"), "w", newline=chr(10)), indent=1)


if __name__ == "__main__":
    main()
