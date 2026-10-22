"""REVIEW_FH $0 diagnostic (red team, not registered): does a PERFECT idle fixed point hold the answer? CPU fp32, 2 threads, idle priority, eval-only.
Items as fh_check_cpu.py (N.test_set(1000), n in NS, the first PER test-order items per n, chunks of BS, same n per chunk). One rollout per arm to K.
Arms (all phase-oracle: the idle phase starts after pass n + 2, using the true n):
  intact : the plain rollout; coda on the computed V_t (check: answers at 24/32/48/64 == eval_khorizon items.intact).
  freeze : stored entry S_t := V_{n+2} for t > n + 2 (bank + carry); the coda after pass t reads S_t (= V_{n+2}) as its current version over bank
           versions 1..t-1 (copies of V_{n+2} after n + 2). This is the state an ideal FH model would sit in (r = 0 exactly from n + 3 on).
  damp   : S_t := S_{t-1} + DAMP * (V_t - S_{t-1}) for t > n + 2 (Guo/Wu/Du 2609.16665 quarter step, inference-time, idle phase only); coda on S_t.
  freeze_win : freeze + the coda / core banks keep V_0 and versions >= t - 1 - WIN only (ver_keep): removes the long, aging copy tail.
Readout per arm and t in TS: argmax answer over NUM_IDS and margin (correct minus best other logit). Writes rv_oracle_freeze.<run>.json (LF)."""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
torch.set_num_threads(2)
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:                                                          # noqa: BLE001
    pass
import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402

run = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 128; NS = (1, 4, 8); PER = int(sys.argv[3]) if len(sys.argv) > 3 else 16
BS = 8; DAMP = 0.25; WIN = 8; ARMS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["intact", "freeze", "damp", "freeze_win"]
TS = [t for t in (12, 16, 24, 32, 48, 64, 96, 128) if t <= K]
t0 = time.time(); m, args = VE.load(os.path.join(ROOT, run), "cpu"); TE = N.test_set(1000); num = torch.tensor(N.NUM_IDS)
kh = json.load(open(os.path.join(ROOT, run, "eval_khorizon.json")))["items"]
res = {a: {str(n): {"ans": {str(t): [] for t in TS}, "margin": {str(t): [] for t in TS}} for n in NS} for a in ARMS}; Y = {}
with torch.no_grad():
    for n in NS:
        idx = [(n - 1) * 1000 + i for i in range(PER)]; assert all(TE[j]["n"] == n for j in idx); Y[str(n)] = [TE[j]["t"] for j in idx]
        for c0 in range(0, PER, BS):
            its = [TE[j] for j in idx[c0:c0 + BS]]; x, pad, lens = VB.batch_x(its, "cpu"); B = x.shape[0]; rows = torch.arange(B)
            y = torch.tensor([it["t"] for it in its])
            for arm in ARMS:
                S = {}

                def edit(t, Vt, arm=arm, S=S):
                    if arm == "intact" or t <= n + 2:
                        S[t] = Vt; return None
                    if arm in ("freeze", "freeze_win"):
                        S[t] = S[n + 2]
                    elif arm == "damp":
                        S[t] = S[t - 1] + DAMP * (Vt - S[t - 1])
                    return S[t]
                m.ver_keep = (lambda t, nver: (torch.arange(nver) == 0) | (torch.arange(nver) >= t - 1 - WIN) | (torch.arange(nver) <= n + 2)) if arm == "freeze_win" else None
                try:
                    if arm == "intact":
                        r_ = m.rollout(x, pad, K, coda_rows={t: rows for t in TS}); H = {t: r_["coda"][t][1] for t in TS}
                    else:
                        r_ = m.rollout(x, pad, K, edit=edit); H = {t: m.coda(t, r_["S"][t], {i: r_["kv"][i] for i in VB.CODA}, pad) for t in TS}
                finally:
                    m.ver_keep = None
                for t in TS:
                    lg = m.answer_logits(H[t], lens)[:, num]; oth = lg.clone(); oth[rows, y] = -1e30
                    res[arm][str(n)]["ans"][str(t)] += lg.argmax(-1).tolist(); res[arm][str(n)]["margin"][str(t)] += (lg[rows, y] - oth.max(-1).values).tolist()
                del r_, H
                print(run, "n", n, "chunk", c0, arm, "sec", round(time.time() - t0), flush=True)
out = {"run": run, "K": K, "ns": list(NS), "per": PER, "damp": DAMP, "win": WIN, "ts": TS, "y": Y, "def": __doc__, "acc": {}, "margin_mean": {}}
for a in ARMS:
    out["acc"][a] = {str(t): float(np.mean([int(p == q) for n in NS for p, q in zip(res[a][str(n)]["ans"][str(t)], Y[str(n)])])) for t in TS}
    out["margin_mean"][a] = {str(t): float(np.mean([v for n in NS for v in res[a][str(n)]["margin"][str(t)]])) for t in TS}
    print(a, "acc", out["acc"][a], "margin", {k: round(v, 2) for k, v in out["margin_mean"][a].items()}, flush=True)
if "intact" in ARMS:
    out["intact_vs_khorizon_share_equal"] = {k: float(np.mean([int(p == q) for n in NS for p, q in zip(res["intact"][str(n)]["ans"][k], kh["intact"][str(n)][k][:PER])]))
                                             for k in ("24", "32", "48", "64") if k in res["intact"]["1"]["ans"]}
    print("intact vs khorizon", out["intact_vs_khorizon_share_equal"])
out["items"] = res
json.dump(out, open(os.path.join(HERE, "rv_oracle_freeze.%s.json" % os.path.basename(run)), "w", newline="\n"), indent=0)
print("DONE sec", round(time.time() - t0))
