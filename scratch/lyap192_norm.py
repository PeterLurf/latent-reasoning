"""note 192-2 POST-DATA DESCRIPTIVE addendum (Theory, after the result note; labelled so in the file): same items, draws, delta seeds and batching as
scratch/lyap192.py. Adds to <run>/eval_s192_lyapunov.json under "post_data_descriptive":
  g'(t) = (||S'_t - S_t|| / ||S_t||) / (||delta|| / ||S_t0||)  (the perturbation relative to the clean state's own norm), mean log g' by k and its LS slope
  per setting (A t0 2/3/4, T 8; B t0 = n + 1, T = n + 9), per-item slope mean / median / share > 0;
  the clean trajectory's log ||S_t|| LS slope over the same t range per setting, and over t = 1..16.
    python scratch/lyap192_norm.py <run dir> [device]"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch")); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.halt173 as H  # noqa: E402
from boxes.nested170 import test_set  # noqa: E402
from lyap192 import NDRAW, EPS, BS, slope  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def run_setting(m, items, dev, t0_of, T_of):
    gp = [dict() for _ in items]; ln = [dict() for _ in items]; groups = {}
    for i, it in enumerate(items):
        groups.setdefault((t0_of(it), T_of(it)), []).append(i)
    for (t0, T), ix in sorted(groups.items()):
        for b in range(0, len(ix), BS):
            bi = ix[b:b + BS]; ch = [items[i] for i in bi]; x, msk = m.think_inputs(ch, dev); S = m.mem.zero_state(len(ch), dev).float()
            for _ in range(t0):
                S = m.write(S, x, msk).float()
            nS = S.flatten(1).norm(dim=1); C = S.clone()
            for t in range(t0 + 1, T + 1):                                  # the clean log ||S_t||
                C = m.write(C, x, msk).float(); v = torch.log(C.flatten(1).norm(dim=1)).cpu().numpy()
                for r_, i in enumerate(bi):
                    ln[i][t - t0] = float(v[r_])
            for d in range(NDRAW):                                          # identical draws to lyap192.run_setting
                g_ = torch.Generator().manual_seed(19202 + d * 100003 + bi[0]); dl = torch.randn(S.shape, generator=g_).to(dev)
                dl = dl * (EPS * nS / dl.flatten(1).norm(dim=1)).view(-1, *[1] * (S.dim() - 1)); rel0 = dl.flatten(1).norm(dim=1) / nS
                A, B = S.clone(), S + dl; X2, M2 = torch.cat([x, x]), torch.cat([msk, msk])
                for t in range(t0 + 1, T + 1):
                    AB = m.write(torch.cat([A, B]), X2, M2).float(); A, B = AB[:len(ch)], AB[len(ch):]
                    lg = torch.log(((B - A).flatten(1).norm(dim=1) / A.flatten(1).norm(dim=1)) / rel0).cpu().numpy()
                    for r_, i in enumerate(bi):
                        gp[i].setdefault(t - t0, []).append(float(lg[r_]))
    return gp, ln


def summ(gp, ln, ks):
    mk = {k: float(np.mean([v for p in gp for v in p.get(k, [])])) for k in ks}; li = [slope(ks, [np.mean(p[k]) for k in ks]) for p in gp]
    lnk = {k: float(np.mean([p[k] for p in ln])) for k in ks}
    return {"mean_log_gprime_by_k": {str(k): v for k, v in mk.items()}, "lambda_hat_gprime": slope(ks, [mk[k] for k in ks]),
            "per_item_lambda_gprime": {"mean": float(np.mean(li)), "median": float(np.median(li)), "share_gt0": float(np.mean(np.array(li) > 0)), "n": len(li)},
            "clean_log_norm_by_k": {str(k): v for k, v in lnk.items()}, "clean_log_norm_slope": slope(ks, [lnk[k] for k in ks])}


@torch.no_grad()
def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"; kind, m, a = H.load_model(run, dev); m.float()
    TE = test_set(a.get("per_n", 1000)); pool = [it for it in TE if it["n"] in (6, 7, 8)]; its = [pool[i] for i in np.random.default_rng(19201).permutation(len(pool))[:256]]
    pdd = {"label": "POST-DATA DESCRIPTIVE (added after the 192-2 result note; Theory)"}
    for t0 in (2, 3, 4):
        gp, ln = run_setting(m, its, dev, lambda it, t0=t0: t0, lambda it: 8); pdd["A_t0_%d" % t0] = summ(gp, ln, list(range(1, 8 - t0 + 1)))
        print("A t0", t0, "lambda_gprime", round(pdd["A_t0_%d" % t0]["lambda_hat_gprime"], 4), "clean_log_norm_slope", round(pdd["A_t0_%d" % t0]["clean_log_norm_slope"], 4), flush=True)
    gp, ln = run_setting(m, its, dev, lambda it: it["n"] + 1, lambda it: it["n"] + 9); pdd["B_t0_n_plus_1"] = summ(gp, ln, list(range(1, 9)))
    print("B lambda_gprime", round(pdd["B_t0_n_plus_1"]["lambda_hat_gprime"], 4), "clean_log_norm_slope", round(pdd["B_t0_n_plus_1"]["clean_log_norm_slope"], 4), flush=True)
    nr = []
    for b in range(0, len(its), BS):                                          # the clean log ||S_t|| slope over t = 1..16
        ch = its[b:b + BS]; x, msk = m.think_inputs(ch, dev); S = m.mem.zero_state(len(ch), dev).float(); row = []
        for t in range(1, 17):
            S = m.write(S, x, msk).float(); row.append(torch.log(S.flatten(1).norm(dim=1)).cpu().numpy())
        nr.append(np.stack(row, 1))
    nr = np.concatenate(nr); mt = nr.mean(0); pdd["clean_log_norm_by_t_1_16"] = {str(t + 1): float(v) for t, v in enumerate(mt)}
    pdd["clean_log_norm_slope_t_1_16"] = slope(list(range(1, 17)), mt.tolist()); pdd["clean_log_norm_slope_t_8_16"] = slope(list(range(8, 17)), mt[7:].tolist())
    p_ = os.path.join(run, "eval_s192_lyapunov.json"); o = json.load(open(p_)); o["post_data_descriptive"] = pdd
    json.dump(o, open(p_, "w", newline="\n"), indent=1); print("DONE norm addendum", run, pdd["clean_log_norm_slope_t_1_16"], flush=True)


if __name__ == "__main__":
    main()
