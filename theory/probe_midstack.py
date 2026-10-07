"""THEORY §146 MID-PROBE (pre-registered cb1ee1cf; descriptive, $0, CPU at Idle priority, 3 threads).
Is the digit-pair accumulator readable mid-stack where the top layer loses it? On <run>/model.pt (multpair, §140's model), the model's own hard
chains over the 1,000 test items; at every pair step, the residual stream at the segment's LAST position after each block l = 0..12 (pre ln_f;
mirrors core.GPT2KV.forward, checked against the model's own h after ln_f). Per layer, 140-3's method exactly: one ridge fit over all pair steps
(lambda by 5-fold CV on the fit half; fit on half the problems, scored on the other half) of the TRUE accumulator; held-out R^2 within the bins
< 100, 100-200, > 200. Row "h_lnf" = the top state after ln_f (140-3's h; a check).
Decision (§146, on s140): MID-STACK SIGNAL = some l <= 9 with R^2 >= 0.5 in the 100-200 bin; NO SIGNAL otherwise.
    python theory/probe_midstack.py boxes/runs_fsq/s140_pair_4x4_s1      -> theory/results/probe_midstack_<run>.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("MKL_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("BOXES_NB", "5")
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch"))
import boxes.fsq120 as F  # noqa: E402
import boxes.multcol as MC  # noqa: E402
import boxes.multpair as MP  # noqa: E402
from boxes import s5  # noqa: E402
from probe137_percol import ridge_cv_r2  # noqa: E402


def idle_priority():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)   # IDLE_PRIORITY_CLASS
        except Exception as e:                                               # noqa: BLE001
            print("idle priority not set:", e)


def layer_states(g, x):
    """Last-position residual stream after each block (pre ln_f), l = 0 (embeddings + positions) .. L; mirrors core.GPT2KV.forward (no cache)."""
    B, T, D = x.shape; hd = D // g.H; h = x + g.w("wpe.weight")[:T][None]; out = [h[:, -1]]
    for i in range(g.L):
        pre = f"h.{i}."
        a = Fn.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
        q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
        q, k, v = (t.view(B, T, g.H, hd).transpose(1, 2) for t in (q, k, v))
        o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
        a = Fn.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
        h = h + Fn.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh") @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")
        out.append(h[:, -1])
    return out


def seg_input(m, prev, t):
    """MultPair.segment's input embeddings, exactly."""
    g = m.g; B = prev.shape[0]; dev = prev.device; eot = g.embed(torch.full((B, 1), F.EOT, device=dev)); i, j = (int(v) for v in m.pairs[t])
    if i < 0:
        tail = torch.full((B, 1), MP.EQTOK, device=dev)
    else:
        flag = MP.NEWTOK if m.newcol[t] else MP.SAMETOK
        tail = torch.cat([m.adig[:, i:i + 1], m.bdig[:, j:j + 1], torch.full((B, 1), flag, device=dev)], 1)
    return torch.cat([eot, prev.to(eot.dtype), g.embed(tail)], 1)


def bin_r2(Xf, yf, Xt, yt):
    """140-3's method: lambda by CV (ridge_cv_r2), one fit, held-out R^2 / RMSE within magnitude bins."""
    r2_all, lam = ridge_cv_r2(Xf, yf, Xt, yt)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    Z, Zt = np.c_[(Xf - mu) / sd, np.ones(len(Xf))], np.c_[(Xt - mu) / sd, np.ones(len(Xt))]
    w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ yf); pr = Zt @ w
    out = {"r2_all": round(r2_all, 4), "lam": lam}
    for lo, hi, key in ((0, 100, "lt100"), (100, 200, "100_200"), (200, 10 ** 9, "gt200")):
        mk = (yt >= lo) & (yt < hi)
        if mk.sum() > 1:
            out[key] = {"r2": round(float(1 - ((yt[mk] - pr[mk]) ** 2).sum() / max(((yt[mk] - yt[mk].mean()) ** 2).sum(), 1e-12)), 4),
                        "rmse": round(float(np.sqrt(((yt[mk] - pr[mk]) ** 2).mean())), 3), "n": int(mk.sum())}
    return out


def main():
    idle_priority(); torch.set_num_threads(3); t0 = time.time()
    d = sys.argv[1].rstrip("/"); dev = "cpu"; a = json.load(open(os.path.join(d, "args.json")))
    F.D_Q, F.LEV = int(a.get("d_q", 8)), int(a.get("levels", 8)); F.HALF = (F.LEV - 1) / 2; F.SHIFT = math.atanh(F.OFF / F.HALF)
    m = MP.MultPair(s5.load_gpt2(dev), MP.PairFam()).to(dev); m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev)); m.eval(); m.p_hard = 1.0
    A, B = MC.load_bench(os.path.join(ROOT, "bench", "data", "mult_%s_test.txt" % a["task"]))
    if len(sys.argv) > 2:                                                   # smoke only
        A, B = A[:int(sys.argv[2])], B[:int(sys.argv[2])]
    N = len(A); half = N // 2; acc, _, _, pairs, col, emit = MP.accumulators(A, B); T = len(pairs); L1 = m.g.L + 1
    LS = np.zeros((N, T, L1 + 1, 768), np.float16); bs = 250; maxdiff = 0.0
    with torch.no_grad():
        for s_ in range(0, N, bs):
            sl = slice(s_, s_ + bs); Bn = len(A[sl]); MP.set_ops(m, A[sl], B[sl], dev); prev = m.start(Bn, dev)
            for t in range(T):
                ls = layer_states(m.g, seg_input(m, prev, t))
                h = m.segment(prev, torch.full((Bn,), t, device=dev))
                top = Fn.layer_norm(ls[-1], (768,), m.g.w("ln_f.weight"), m.g.w("ln_f.bias"))
                maxdiff = max(maxdiff, float((top - h).abs().max()))
                LS[sl, t, :L1] = torch.stack(ls, 1).numpy().astype(np.float16); LS[sl, t, L1] = h.numpy().astype(np.float16)
                qs, q, zb = m.quant(h); prev = m.embed_code(qs)[:, None]
    assert maxdiff < 1e-2, "layer mirror differs from the model's forward: %g" % maxdiff
    pos = np.zeros(T, int)
    for t in range(T):
        pos[t] = 0 if pairs[t][0] < 0 else (1 if t == 0 or col[t] != col[t - 1] else pos[t - 1] + 1)
    tp = np.where(pos > 0)[0]; yf, yt = acc[:half, tp].reshape(-1).astype(float), acc[half:, tp].reshape(-1).astype(float)
    res = {"run": os.path.basename(d), "n": N, "fit": half, "pair_steps": int(len(tp)), "mirror_maxdiff": maxdiff, "layers": {}}
    for li in range(L1 + 1):
        key = "h_lnf" if li == L1 else str(li)
        X = LS[:, tp, li].astype(np.float32)
        res["layers"][key] = bin_r2(X[:half].reshape(len(yf), -1), yf, X[half:].reshape(len(yt), -1), yt)
        print(key, json.dumps(res["layers"][key]), flush=True)
    mids = {k: v.get("100_200", {}).get("r2") for k, v in res["layers"].items() if k != "h_lnf" and int(k) <= 9}
    best = max(mids.items(), key=lambda kv: -1e9 if kv[1] is None else kv[1])
    res["decision"] = {"best_layer_le9_100_200": best[0], "r2": best[1],
                       "verdict": "MID-STACK SIGNAL" if (best[1] is not None and best[1] >= 0.5) else "NO SIGNAL"}
    res["sec"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "probe_midstack_%s.json" % os.path.basename(d)), "w", newline=chr(10)), indent=1)
    print("DECISION (§146):", json.dumps(res["decision"]), flush=True)


if __name__ == "__main__":
    main()
