"""Audit $0 DESCRIPTIVE diagnostic (§234 seed-2 d=3 chain): aw_plaw4_s1 vs _s2, d = 3, N = 2000, seed 23401. Same measurements as audit_d4_chain.py (no oracle):
per pass t = 1..3: ':' argmax vs v_t / v_{t-1} / v_{t+1} / v_0; probe v_t from u = mem.ln_w(h) at ':'; v_t from S_t flat; v_{t-1} from read_vec at ':' (t >= 2);
pass 2 only: v_2 from the pre-ln_f residual h at ':'. fp32, no_grad. Usage: python -B scratch/audit_s2_chain.py [device=cuda:1] [N=2000]"""
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
import boxes.inject_aw as AW  # noqa: E402
import boxes.inject_aw_eval as EV  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import NUM_IDS  # noqa: E402

RUNS = ["aw_plaw4_s1", "aw_plaw4_s2"]; OUT = os.path.join(ROOT, "scratch/audit_s2_chain/diag.json")
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000; P = 11; SEED = 23401; BS = 64; D = 3
if dev == "cpu":
    torch.set_num_threads(3)


def fit_probe(X, y, tr, va, te):
    X = X.float(); mu = X[tr].mean(0, keepdim=True); sd = X[tr].std(0, keepdim=True) + 1e-6; Z = ((X - mu) / sd).to(dev); Y = torch.tensor(y, device=dev); best = None
    for wd in (1e-4, 1e-3, 1e-2):
        W = torch.zeros(Z.shape[1], P, device=dev, requires_grad=True); b = torch.zeros(P, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=200, history_size=20, line_search_fn="strong_wolfe")

        def clo():
            opt.zero_grad(); l = F.cross_entropy(Z[tr] @ W + b, Y[tr]) + 0.5 * wd * (W ** 2).sum(); l.backward(); return l
        with torch.enable_grad():
            opt.step(clo)
        with torch.no_grad():
            pr = (Z @ W + b).argmax(-1); va_acc = float((pr[va] == Y[va]).float().mean())
            if best is None or va_acc > best[0]:
                best = (va_acc, wd, pr[te].cpu().numpy(), float((pr[tr] == Y[tr]).float().mean()))
    yt = y[te]
    del Z
    return {"test_acc": float((best[2] == yt).mean()), "val_acc": best[0], "train_acc": best[3], "wd": best[1], "n_test": int(len(te)),
            "chance_majority": float(np.bincount(yt, minlength=P).max() / len(yt)), "chance_uniform": 1 / P, "feature_dim": int(X.shape[1]), "test_pred": best[2].tolist()}


def run_d(d, m, sets):
    t0 = time.time(); mem = m.mem; num = torch.tensor(NUM_IDS, device=dev)
    st = OW.Stream(SEED, sets["excl"], frames=OW.FRAME_TRAIN, **IO.OWKW); raw = [st.draw(d) for _ in range(N)]
    T = [AW.tok_aw(it, "value", "step", "answer") for it in raw]; sigs = [it["sig"] for it in raw]; V = np.array([[int(v) for v in it["values"]] for it in raw])
    n = N; arg = np.zeros((n, d), int); U = {}; RD = {}; SF = {}; HR = []
    for t in range(1, d + 1):
        U[t] = []; RD[t] = []; SF[t] = []
    for i in range(0, n, BS):
        ch = T[i:i + BS]; B = len(ch); ar = torch.arange(B, device=dev); xs, msk = AW.cue_inputs(m, ch, dev); hls = [m.lower(x) for x in xs]
        S = mem.zero_state(B, dev); Sl = [S]
        for t in range(1, d + 1):
            Sp = Sl[-1]; S, h, pos = m.pass_aw(Sp, xs[t - 1], msk, hls[t - 1]); Sl.append(S)
            arg[i:i + B, t - 1] = m.colon_logits(h, pos)[:, num].argmax(-1).cpu().numpy()
            h6p = hls[t - 1][ar, pos].float(); U[t].append(mem.ln_w(h[ar, pos].float()).cpu()); SF[t].append(S.flatten(1).float().cpu())
            RD[t].append(EV.read_vec(mem, h6p, Sp).cpu())
            if t == 2:
                HR.append(h[ar, pos].float().cpu())
        print("batch", i, "/", n, flush=True)
    us = sorted(set(sigs)); perm = np.random.default_rng(0).permutation(len(us)); rank = {us[j]: r for r, j in enumerate(perm)}; r = np.array([rank[s_] for s_ in sigs]) / len(us)
    tr, va, te = [np.where(c)[0] for c in (r < 0.6, (r >= 0.6) & (r < 0.8), r >= 0.8)]
    res = {"d": d, "n": n, "seed": SEED, "split": [len(tr), len(va), len(te)]}; pa = {}
    for t in range(1, d + 1):
        a = arg[:, t - 1]; vt = V[:, t]; vm = V[:, t - 1]; vp = V[:, t + 1] if t < d else None
        pa[str(t)] = {"eq_vt": float((a == vt).mean()), "eq_vtm1": float((a == vm).mean()), "eq_vtp1": float((a == vp).mean()) if vp is not None else None,
                      "eq_v0": float((a == V[:, 0]).mean()), "majority_vt": float(np.bincount(vt, minlength=P).max() / n)}
    res["argmax"] = pa; res["argmax_per_item"] = arg.tolist(); res["v_per_item"] = V.tolist(); res["test_idx"] = te.tolist(); pr = {}
    for t in range(1, d + 1):
        row = {"u|v_t": fit_probe(torch.cat(U[t]), V[:, t], tr, va, te), "S_t|v_t": fit_probe(torch.cat(SF[t]), V[:, t], tr, va, te)}
        if t >= 2:
            row["read_at_colon|v_t-1"] = fit_probe(torch.cat(RD[t]), V[:, t - 1], tr, va, te)
        if t == 2:
            row["h_preLNf|v_t"] = fit_probe(torch.cat(HR), V[:, t], tr, va, te)
        pr[str(t)] = row; print("t", t, {k: round(v["test_acc"], 3) for k, v in row.items()}, flush=True)
    res["probes"] = pr; res["sec"] = round(time.time() - t0); return res


@torch.no_grad()
def main():
    t0 = time.time(); sets = IO.build_sets(); out = {"device": dev, "dtype": "fp32", "d": D, "models": {}}
    for nm in RUNS:
        a, m, g = EV.load(os.path.join(ROOT, "boxes/runs_fsq", nm), dev); m.eval(); assert float(m.mem.gamma) == 1.0
        out["models"][nm] = run_d(D, m, sets); del m
    out["sec"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w", newline=chr(10)).write(json.dumps(out)); print("WROTE", OUT, out["sec"], "s")


main()
