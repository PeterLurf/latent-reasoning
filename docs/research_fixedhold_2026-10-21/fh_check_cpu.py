"""FIXED-HOLD $0 check B (draft FH_draft.md §1): CPU fp32 rollout of committed model.pt files, 2 torch threads, no GPU, eval-only, writes only
fh_check_cpu.<run>.json next to this script (LF). Run: python -B docs/research_fixedhold_2026-10-21/fh_check_cpu.py boxes/runs_fsq/s213_d3t_s1
Items: §170 test set (N.test_set(1000)), n in {1, 4, 8}, the first 16 test-order items of each n, chunks of 8 (same n per chunk), one intact rollout
to K' 64 with the coda read after EVERY pass t (coda_rows all t). Per pass t (descriptive):
  ans_t / margin_t: argmax answer over NUM_IDS and correct-minus-best-other logit at the ' =' slot (CPU-vs-3070 check: ans at K' 24/32/48/64 vs
      eval_khorizon.json items.intact, share equal);
  step_ang_t: 1 - cos(V_t, V_{t-1}) at the value slot (the E_n slot whose class is a value) and mean over real slots (scale-free, the perturbation rule);
  drift_ang_t: 1 - cos(V_t, V_{n+2}) at the value slot (cumulative rotation since the in-range hold);
  rms_t: rms(V_t) / rms(V_0) at the value slot (norm growth);
  rec_ok_t: argmax R(V_t) == E_n class at the value slot (does the state itself still decode the held value);
  newest_t: SLVA newest-version mass at the value slot (head mean, m.slva_log);
  clock_t: ridge (lambda 1e4, unpenalised intercept) of t on [U_t, 1] over in-range idle passes t in [n + 1, 12] at every real slot (all chunks pooled),
      then the mean prediction per t up to 64 (213-27's AGE-PROJ clock, extrapolated): linear continuation -> clock_t ~ t; saturation -> plateau."""
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

run = sys.argv[1]; K = 64; NS = (1, 4, 8); PER = 16; BS = 8; dev = "cpu"
t0 = time.time(); m, args = VE.load(os.path.join(ROOT, run), dev); TE = N.test_set(1000)
num = torch.tensor(N.NUM_IDS); kh = json.load(open(os.path.join(ROOT, run, "eval_khorizon.json")))["items"]
rec = {n: {k: [] for k in ("ans", "margin", "step_ang_v", "step_ang_all", "drift_ang_v", "rms_v", "rec_ok_v", "newest_v")} for n in NS}
Ufit = []; agree = {n: {} for n in NS}; Uch = {}


def cos(a, b):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)


with torch.no_grad():
    for n in NS:
        idx = [(n - 1) * 1000 + i for i in range(PER)]; assert all(TE[j]["n"] == n for j in idx)
        for c0 in range(0, PER, BS):
            its = [TE[j] for j in idx[c0:c0 + BS]]; x, pad, lens = VB.batch_x(its, dev); B, T = x.shape; rows = torch.arange(B)
            vsl = []; vcl = []
            for it in its:
                e = np.array(VB.recode_targets(it["root"], n))[n]; js = [j for j, c in enumerate(e) if c >= VB.VAL0]; assert len(js) == 1, e
                vsl.append(js[0]); vcl.append(int(e[js[0]]))
            vsl = torch.tensor(vsl); vcl = torch.tensor(vcl); Us = {}
            m.input_edit = lambda t, U: (Us.__setitem__(t, U.detach().clone()), U)[1]
            m.slva_log = []
            r_ = m.rollout(x, pad, K, coda_rows={t: rows for t in range(0, K + 1)})
            m.input_edit = None; nw = {t: w for t, w in m.slva_log}; m.slva_log = None
            y = torch.tensor([it["t"] for it in its])
            V = r_["V"]; Vv = [v[rows, vsl] for v in V]
            per = {k: [] for k in rec[n]}
            for t in range(0, K + 1):
                lg = m.answer_logits(r_["coda"][t][1], lens)[:, num]; a_ = lg.argmax(-1)
                oth = lg.clone(); oth[rows, y] = -1e30
                per["ans"].append(a_.tolist()); per["margin"].append((lg[rows, y] - oth.max(-1).values).tolist())
                if t >= 1:
                    per["step_ang_v"].append((1 - cos(Vv[t], Vv[t - 1])).tolist())
                    sa = (1 - cos(V[t], V[t - 1])); per["step_ang_all"].append(((sa * pad).sum(1) / pad.sum(1)).tolist())
                    per["newest_v"].append(nw[t][rows, vsl].tolist() if t in nw else [None] * B)
                else:
                    per["step_ang_v"].append([None] * B); per["step_ang_all"].append([None] * B); per["newest_v"].append([None] * B)
                per["drift_ang_v"].append((1 - cos(Vv[t], Vv[n + 2])).tolist())
                per["rms_v"].append((Vv[t].pow(2).mean(-1).sqrt() / Vv[0].pow(2).mean(-1).sqrt()).tolist())
                per["rec_ok_v"].append((m.rec_logits(Vv[t]).argmax(-1) == vcl).tolist())
            for k in rec[n]:
                rec[n][k].append(per[k])                                   # [chunk][t][item]
            for t in range(n + 1, 13):
                Ufit.append((t, Us[t][pad].double()))
            Uch.setdefault(n, []).append({t: Us[t][pad].double() for t in Us})
            for k in ("24", "32", "48", "64"):
                ref = kh["intact"][str(n)][k][c0:c0 + BS]; agree[n].setdefault(k, []).extend([int(a == b) for a, b in zip(per["ans"][int(k)], ref)])
            del r_
            print("chunk n", n, c0, "sec", round(time.time() - t0), flush=True)
    # clock ridge fit (fp64 normal equations)
    D = Ufit[0][1].shape[1]; XtX = torch.zeros(D + 1, D + 1, dtype=torch.float64); Xty = torch.zeros(D + 1, dtype=torch.float64)
    for t, U in Ufit:
        X = torch.cat([U, torch.ones(len(U), 1, dtype=torch.float64)], 1); XtX += X.T @ X; Xty += X.T @ torch.full((len(U),), float(t), dtype=torch.float64)
    R = 1e4 * torch.eye(D + 1, dtype=torch.float64); R[-1, -1] = 0; w = torch.linalg.solve(XtX + R, Xty)
    out = {"run": run, "K": K, "items_per_n": PER, "ns": list(NS), "card": "cpu", "dtype": "fp32", "threads": 2, "def": __doc__}
    for n in NS:
        o = {}
        for k in ("ans", "margin", "step_ang_v", "step_ang_all", "drift_ang_v", "rms_v", "rec_ok_v", "newest_v"):
            o[k] = [sum((ch[t] for ch in rec[n][k]), []) for t in range(K + 1)]   # [t][item]
        o["y"] = [TE[(n - 1) * 1000 + i]["t"] for i in range(PER)]
        clk = {}
        for t in range(1, K + 1):
            vals = torch.cat([ch[t] for ch in Uch[n]]); clk[t] = float((torch.cat([vals, torch.ones(len(vals), 1, dtype=torch.float64)], 1) @ w).mean())
        o["clock_mean_pred"] = clk
        o["cpu_vs_3070_share_equal"] = {k: float(np.mean(v)) for k, v in agree[n].items()}
        out[str(n)] = o
        summ = lambda k, t: float(np.mean([v for v in o[k][t] if v is not None]))  # noqa: E731
        acc = lambda t: float(np.mean([a == b for a, b in zip(o["ans"][t], o["y"])]))  # noqa: E731
        print("n", n, "cpu==3070", o["cpu_vs_3070_share_equal"])
        for t in (n, n + 2, 12, 16, 20, 24, 32, 40, 48, 56, 64):
            print("  t %2d acc %.3f margin %.2f step_v %.2e step_all %.2e drift_v %.3f rms_v %.2f rec_ok %.2f newest %.3f clock %.1f" % (
                t, acc(t), summ("margin", t), summ("step_ang_v", t), summ("step_ang_all", t), summ("drift_ang_v", t), summ("rms_v", t), summ("rec_ok_v", t), summ("newest_v", t), clk[t]))
    json.dump(out, open(os.path.join(HERE, "fh_check_cpu.%s.json" % os.path.basename(run)), "w", newline="\n"), indent=0)
    print("DONE sec", round(time.time() - t0))
