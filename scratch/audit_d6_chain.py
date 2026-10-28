"""Audit $0 DESCRIPTIVE diagnostic (lh_fh d6 chain locus): boxes/runs_fsq/lh_fh_s1 and _s2 (InjectAW, DF law mul_s_p 2), d = 6, K = 8 passes (passes 7, 8 reuse cue 6), N = 2000.
Items: OW.Stream(24601, build_sets(2)['excl'], frames=FRAME_TRAIN, mul_s_p=2, **OWKW).draw(6) (TEST2-DF's generator / law, fresh seed, training-excluded), tok_aw(step cue, answer write).
Protocol = audit_d4_chain.py / audit_s2_chain.py: ':' argmax over the 11 number tokens; probes (sig split 60/20/20 rng 0, standardised logistic LBFGS, wd on val, test acc).
Per pass t = 1..6: argmax vs v_t / v_{t-1} / v_{t+1} / v_{t-2} / v_0 + wrong-token histogram; probes v_t from u = mem.ln_w(h) at ':', v_t from S_t, v_{t-1} from S_{t-1} (t >= 2),
v_{t-1} from read_vec at ':' of pass t (t >= 2; checked == read_fixed capture). Pass-5 argmax and probes split by the step-5 op (ops[4] + src_j[4]). fp32, no_grad.
Usage: python -B scratch/audit_d6_chain.py [device=cuda:1] [N=2000]"""
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

RUNS = ["lh_fh_s1", "lh_fh_s2"]; OUT = os.path.join(ROOT, "scratch/audit_d6_chain/diag.json")
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000; P = 11; SEED = 24601; BS = 64; D = 6; K = 8
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


def law_ok(a):
    return EV.law(a) == 2


def run_d(d, m, sets):
    t0 = time.time(); mem = m.mem; num = torch.tensor(NUM_IDS, device=dev)
    st = OW.Stream(SEED, sets["excl"], frames=OW.FRAME_TRAIN, mul_s_p=2, **IO.OWKW); raw = [st.draw(d) for _ in range(N)]
    T = [AW.tok_aw(it, "value", "step", "answer") for it in raw]; sigs = [it["sig"] for it in raw]; V = np.array([[int(v) for v in it["values"]] for it in raw])
    n = N; arg = np.zeros((n, K), int); U = {}; RD = {}; SF = {}; chk = 0.0; num_t = torch.tensor(NUM_IDS, device=dev)
    for t in range(1, K + 1):
        U[t] = []; RD[t] = []; SF[t] = []
    for i in range(0, n, BS):
        ch = T[i:i + BS]; B = len(ch); ar = torch.arange(B, device=dev); xs, msk = AW.cue_inputs(m, ch, dev); xs = xs + [xs[-1]] * (K - len(xs)); hls = [m.lower(x) for x in xs]
        S = mem.zero_state(B, dev); Sl = [S]
        for t in range(1, K + 1):
            Sp = Sl[-1]; S, h, pos = m.pass_aw(Sp, xs[t - 1], msk, hls[t - 1]); Sl.append(S)
            arg[i:i + B, t - 1] = m.colon_logits(h, pos)[:, num].argmax(-1).cpu().numpy()
            h6p = hls[t - 1][ar, pos].float(); U[t].append(mem.ln_w(h[ar, pos].float()).cpu()); SF[t].append(S.flatten(1).float().cpu())
            RD[t].append(EV.read_vec(mem, h6p, Sp).cpu())
            if i == 0 and t >= 2:
                rc_ = []; DM.read_fixed(mem, hls[t - 1][ar, pos][:, None], Sp, rc_); rv_ = EV.read_vec(mem, h6p, Sp)
                chk = max(chk, float((rc_[0] - rv_[:, :DM.H_ADD * DM.DH]).abs().max() / rc_[0].abs().max().clamp_min(1e-30)))
        print("batch", i, "/", n, flush=True)
    us = sorted(set(sigs)); perm = np.random.default_rng(0).permutation(len(us)); rank = {us[j]: r for r, j in enumerate(perm)}; r = np.array([rank[s_] for s_ in sigs]) / len(us)
    tr, va, te = [np.where(c)[0] for c in (r < 0.6, (r >= 0.6) & (r < 0.8), r >= 0.8)]
    res = {"d": d, "n": n, "seed": SEED, "split": [len(tr), len(va), len(te)], "read_vec_check_rel": chk, "K": K}; pa = {}
    assert chk <= 1e-5, ("read_vec != read_fixed", chk)
    for t in range(1, K + 1):
        a = arg[:, t - 1]; vt = V[:, min(t, d)]; rec = {"eq_vt": float((a == vt).mean()), "majority_vt": float(np.bincount(vt, minlength=P).max() / n)}
        if t <= d:
            vm = V[:, t - 1]; vp = V[:, t + 1] if t < d else None
            rec.update({"eq_vtm1": float((a == vm).mean()), "eq_vtp1": float((a == vp).mean()) if vp is not None else None, "eq_vtm2": float((a == V[:, t - 2]).mean()) if t >= 2 else None,
                        "eq_v0": float((a == V[:, 0]).mean()), "eq_any_earlier_v": float(np.any(a[:, None] == V[:, :t], 1).mean()), "eq_any_later_v": float(np.any(a[:, None] == V[:, t + 1:], 1).mean()) if t < d else None})
            w = a != vt; rec["wrong_token_share"] = {str(k): float(((a == k) & w).mean()) for k in range(P) if ((a == k) & w).any()}
            rec["wrong_token_given_vt"] = {str(k): float(((a == k) & w).sum() / max(1, w.sum())) for k in range(P) if ((a == k) & w).any()}
            rec["wrong_minus_vt_mod11"] = {str(k): float((((a - vt) % P == k) & w).mean()) for k in range(1, P) if (((a - vt) % P == k) & w).any()}
        pa[str(t)] = rec
    res["argmax"] = pa; res["split_n"] = [len(tr), len(va), len(te)]; res["argmax_per_item"] = arg.tolist(); res["v_per_item"] = V.tolist(); res["test_idx"] = te.tolist(); pr = {}
    for t in range(1, d + 1):
        row = {"u|v_t": fit_probe(torch.cat(U[t]), V[:, t], tr, va, te), "S_t|v_t": fit_probe(torch.cat(SF[t]), V[:, t], tr, va, te)}
        if t >= 2:
            row["S_t-1|v_t-1"] = fit_probe(torch.cat(SF[t - 1]), V[:, t - 1], tr, va, te); row["read_at_colon|v_t-1"] = fit_probe(torch.cat(RD[t]), V[:, t - 1], tr, va, te)
        pr[str(t)] = row; print("t", t, {k: round(v["test_acc"], 3) for k, v in row.items()}, flush=True)
    opk = [str(it["ops"][4]) + "|src_j=" + str(it["src_j"][4]) for it in raw]; res["pass5_op_per_item"] = opk; by = {}
    a5 = arg[:, 4]; v5 = V[:, 5]; v4 = V[:, 4]; tei = np.array(te); pos_te = {j: r_ for r_, j in enumerate(te)}
    for k in sorted(set(opk)):
        idx = np.array([j for j in range(n) if opk[j] == k]); ti = np.array([j for j in te if opk[j] == k]); rec = {"n": int(len(idx)), "n_test": int(len(ti)), "argmax_eq_v5": float((a5[idx] == v5[idx]).mean()),
                                                                                  "argmax_eq_v4": float((a5[idx] == v4[idx]).mean()), "v5_eq_v4": float((v5[idx] == v4[idx]).mean())}
        if len(ti):
            for nm, tt, yy in (("u|v_5", "5", v5), ("S_5|v_5", "5", v5), ("S_4|v_4", "4", v4)):
                key = {"u|v_5": "u|v_t", "S_5|v_5": "S_t|v_t", "S_4|v_4": "S_t|v_t"}[nm]; pp = np.array(pr[tt][key]["test_pred"]); rec["probe_" + nm] = float((pp[[pos_te[j] for j in ti]] == yy[ti]).mean())
            pp = np.array(pr["5"]["read_at_colon|v_t-1"]["test_pred"]); rec["probe_read|v_4"] = float((pp[[pos_te[j] for j in ti]] == v4[ti]).mean())
        by[k] = rec
    res["pass5_by_op"] = by
    res["probes"] = pr; res["sec"] = round(time.time() - t0); return res


@torch.no_grad()
def main():
    t0 = time.time(); sets = IO.build_sets(2); out = {"device": dev, "dtype": "fp32", "d": D, "K": K, "seed": SEED, "law": "mul_s_p 2", "models": {}}
    for nm in RUNS:
        a, m, g = EV.load(os.path.join(ROOT, "boxes/runs_fsq", nm), dev); m.eval(); assert float(m.mem.gamma) == 1.0; assert law_ok(a)
        out["models"][nm] = run_d(D, m, sets); del m
    out["sec"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w", newline=chr(10)).write(json.dumps(out)); print("WROTE", OUT, out["sec"], "s")


main()
