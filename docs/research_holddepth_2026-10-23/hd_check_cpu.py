"""HOLD-DEPTH $0 Stage 0, check B (HD_draft.md §2): CPU fp32 rollouts of committed model.pt files, 2 torch threads, Idle priority, no GPU, eval-only.
Writes only hd_check_cpu.<run>.json next to this script (LF). Nothing is trained; model weights are never written.
    python -B docs/research_holddepth_2026-10-23/hd_check_cpu.py boxes/runs_fsq/s224_rd_s1 [--validate] [--per 64]
--validate: ONLY the known-answer checks (n 8 items vs eval_khorizon.json items.intact at K' 24; idle-classifier fit / held-out accuracy);
            reads no n >= 9 item. Run this first on every model (HD_draft §5 K1-K3).
Items: the §170 test set (N.test_set(1000)), the first --per (default 128) test-order items of n 9, 10, 11 (deep), chunks of 8 (one n per chunk), one intact
rollout to K' 24 with the coda read after EVERY pass. Fit set for the idle classifier: the first 16 items of each n 3..8 (rollout to K' 12).
Per deep item (descriptive fields; HD_draft §2 names the registered aggregates):
  ans[t]       argmax over NUM_IDS of the coda after pass t, t 0..24 ('ans_ok' = ans == y)
  t_ans        first t such that ans is correct at every t' in [t, 24] (None if never)
  exact[t]     R(V_t) argmax == E_t on every real slot (E = vbank.recode_targets, HOLD past n);  ahead[t]: == E_{t+1} on every real slot (and E_t != E_{t+1})
  alive_fin    non-VOID real slots of R(V_{n+2}) minus 2 (EOT, ' ='); finished = alive_fin == 1; S = finished and that slot's class == VAL0 + y
  act[t]       state activity: mean over real slots of 1 - cos(V_t, V_{t-1}), t 1..n+2
  idle[t]      idle-classifier score of V_t (ridge on the real-slot mean of V_t; fit on n 3..8: t <= n - 1 -> 0, t in [n + 2, 12] -> 1)
  height       height of the expression tree (vbank_eval.height)"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
torch.set_num_threads(2)
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if hasattr(psutil, "IDLE_PRIORITY_CLASS") else 19)
except Exception:                                                          # noqa: BLE001
    pass
import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--validate", action="store_true"); ap.add_argument("--per", type=int, default=128)
ap.add_argument("--deep_ns", default="9,10,11", help="code-path smoke only: e.g. 8 (in-range, writes .smoke json)")
a = ap.parse_args()
K = 24; BS = 8; dev = "cpu"; DEEP = tuple(int(v) for v in a.deep_ns.split(",")); SMOKE = DEEP != (9, 10, 11); FIT = tuple(range(3, 9)); PER_FIT = 16
t0 = time.time(); m, args = VE.load(os.path.join(ROOT, a.run), dev); TE = N.test_set(1000); num = torch.tensor(N.NUM_IDS)


def cos(u, v):
    return (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp_min(1e-12)


def items_of(n, k):
    idx = [(n - 1) * 1000 + i for i in range(k)]; assert all(TE[j]["n"] == n for j in idx); return idx


def roll(its, Kr):
    x, pad, lens = VB.batch_x(its, dev); B, T = x.shape; rows = torch.arange(B)
    r = m.rollout(x, pad, Kr, coda_rows={t: rows for t in range(0, Kr + 1)})
    ans = torch.stack([m.answer_logits(r["coda"][t][1], lens)[:, num].argmax(-1) for t in range(Kr + 1)], 1)   # (B, Kr + 1)
    return r, pad, ans


def pooled(V, pad):
    p = pad[..., None].double(); return (V.double() * p).sum(1) / p.sum(1)


with torch.no_grad():
    # ---- idle classifier fit (n 3..8; even item index = fit, odd = held out)
    Xf, yf, hf = [], [], []
    for n in FIT:
        idx = items_of(n, PER_FIT)
        for c0 in range(0, PER_FIT, BS):
            its = [TE[j] for j in idx[c0:c0 + BS]]; r, pad, _ = roll(its, 12)
            for t in range(1, 13):
                if n <= t <= n + 1:
                    continue
                f = pooled(r["V"][t], pad); lab = 0.0 if t <= n - 1 else 1.0
                for b in range(len(its)):
                    Xf.append(f[b]); yf.append(lab); hf.append((c0 + b) % 2)
            del r
    X = torch.stack(Xf); y_ = torch.tensor(yf, dtype=torch.float64); h_ = torch.tensor(hf)
    mu = X[h_ == 0].mean(0); sd = X[h_ == 0].std(0).clamp_min(1e-8); Z = (X - mu) / sd
    Zf = torch.cat([Z[h_ == 0], torch.ones(int((h_ == 0).sum()), 1, dtype=torch.float64)], 1)
    Rg = 10.0 * torch.eye(Zf.shape[1], dtype=torch.float64); Rg[-1, -1] = 0
    w = torch.linalg.solve(Zf.T @ Zf + Rg, Zf.T @ y_[h_ == 0])
    score = lambda F_: (torch.cat([(F_ - mu) / sd, torch.ones(len(F_), 1, dtype=torch.float64)], 1) @ w)   # noqa: E731
    pz = score(X); idle_fit = {"heldout_acc": float(((pz[h_ == 1] > 0.5).double() == y_[h_ == 1]).double().mean()),
                               "fit_acc": float(((pz[h_ == 0] > 0.5).double() == y_[h_ == 0]).double().mean()), "n_rows": len(yf), "ridge": 10.0}
    print("idle classifier", idle_fit, "sec", round(time.time() - t0), flush=True)
    out = {"run": a.run, "K": K, "card": "cpu", "dtype": "fp32", "threads": 2, "idle_fit": idle_fit, "def": __doc__}
    # ---- known-answer check: n 8 items vs eval_khorizon items.intact at K' 24
    kh = json.load(open(os.path.join(ROOT, a.run, "eval_khorizon.json")))["items"]["intact"]["8"]["24"]
    eq = []
    for c0 in range(0, 16, BS):
        its = [TE[j] for j in items_of(8, 16)[c0:c0 + BS]]; _, _, ans = roll(its, 24)
        eq += [int(int(p) == int(q)) for p, q in zip(ans[:, 24].tolist(), kh[c0:c0 + BS])]
    out["cpu_vs_khorizon_n8_k24_share_equal"] = float(np.mean(eq)); print("K1 n8 K'24 cpu==khorizon share", out["cpu_vs_khorizon_n8_k24_share_equal"], flush=True)
    if not a.validate:
        out["per"] = a.per; out["deep"] = {}
        for n in DEEP:
            recs = []
            idx = items_of(n, a.per)
            for c0 in range(0, a.per, BS):
                its = [TE[j] for j in idx[c0:c0 + BS]]; r, pad, ans = roll(its, K); B = len(its)
                E = [VB.recode_targets(it["root"], K + 1) for it in its]
                dec = [m.rec_logits(r["V"][t]).argmax(-1) for t in range(K + 1)]
                idl = [score(pooled(r["V"][t], pad)) for t in range(K + 1)]
                for b, it in enumerate(its):
                    L = len(E[b][0]); pb = pad[b, :L]; assert int(pb.sum()) == L, "real slots != E length"
                    y = int(it["t"]); ok = (ans[b] == y).tolist()
                    t_ans = next((t for t in range(K + 1) if all(ok[t:])), None)
                    ex = [bool((dec[t][b, :L] == torch.tensor(E[b][t])).all()) for t in range(K + 1)]
                    ah = [bool((dec[t][b, :L] == torch.tensor(E[b][t + 1])).all()) and E[b][t] != E[b][t + 1] for t in range(K + 1)]
                    df = dec[n + 2][b, :L].tolist(); live = [j for j in range(1, L - 1) if df[j] != VB.VOID]
                    fin = len(live) == 1; S = fin and df[live[0]] == VB.VAL0 + y
                    act = [None] + [float(((1 - cos(r["V"][t][b, :L], r["V"][t - 1][b, :L])).mean())) for t in range(1, n + 3)]
                    recs.append({"y": y, "ans": ans[b].tolist(), "t_ans": t_ans, "exact": ex, "ahead": ah, "alive_fin": len(live), "finished": fin, "S": S,
                                 "act": act, "idle": [float(idl[t][b]) for t in range(K + 1)], "height": VE.height(it["root"])})
                del r
                print("deep n", n, c0, "sec", round(time.time() - t0), flush=True)
            out["deep"][str(n)] = recs
    out["sec"] = round(time.time() - t0)
    nm = "hd_check_cpu.%s%s.json" % (os.path.basename(a.run.rstrip("/\\")), ".validate" if a.validate else (".smoke" if SMOKE else ""))
    json.dump(out, open(os.path.join(HERE, nm), "w", newline="\n"))
    print("DONE", nm, "sec", out["sec"])
