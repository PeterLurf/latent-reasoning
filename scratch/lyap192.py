"""note 192-2 (Theory) finite-time Lyapunov read, eval-only, descriptive, fp32 (no autocast, TF32 off). The 256 test items of the plateau read (n 6-8, rng
19201). For each item and t0: S_t0 = passes 1..t0 from M_0 = 0; S' = S_t0 + delta, delta Gaussian scaled to ||delta|| = 1e-3 ||S_t0|| (per item; 8 seeded
draws, rng 19202 + draw); S and S' run on with the same input to T; g(t) = ||S'_t - S_t|| / ||delta|| (whole state: additive + delta heads).
  (A) t0 in {2, 3, 4}, T = 8;  (B) t0 = n + 1 (each item's own n), T = n + 9 (reported by k = t - t0 = 1..8).
Per setting: mean over items and draws of log g by t (by k for B); lambda_hat = least-squares slope of that mean vs t; per-item lambda_hat (slope of the
item's draw-mean log g) as mean / median / share > 0.
    python scratch/lyap192.py <run dir> [device]   -> <run>/eval_s192_lyapunov.json"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.halt173 as H  # noqa: E402
from boxes.nested170 import test_set  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
NDRAW, EPS, BS = 8, 1e-3, 32


def slope(ts, ys):
    ts = np.asarray(ts, float); ys = np.asarray(ys, float); tc = ts - ts.mean(); return float((tc * (ys - ys.mean())).sum() / (tc * tc).sum())


@torch.no_grad()
def run_setting(m, items, dev, t0_of, T_of):
    """-> per item: {k: [log g over draws]} with k = t - t0 (1..T - t0); items grouped so each batch shares t0 and T."""
    out = [dict() for _ in items]; groups = {}
    for i, it in enumerate(items):
        groups.setdefault((t0_of(it), T_of(it)), []).append(i)
    for (t0, T), ix in sorted(groups.items()):
        for b in range(0, len(ix), BS):
            bi = ix[b:b + BS]; ch = [items[i] for i in bi]; x, msk = m.think_inputs(ch, dev); S = m.mem.zero_state(len(ch), dev).float()
            for _ in range(t0):
                S = m.write(S, x, msk).float()
            nS = S.flatten(1).norm(dim=1)
            for d in range(NDRAW):
                g_ = torch.Generator().manual_seed(19202 + d * 100003 + bi[0]); dl = torch.randn(S.shape, generator=g_).to(dev)
                dl = dl * (EPS * nS / dl.flatten(1).norm(dim=1)).view(-1, *[1] * (S.dim() - 1)); nd = dl.flatten(1).norm(dim=1)
                A, B = S.clone(), S + dl; X2, M2 = torch.cat([x, x]), torch.cat([msk, msk])
                for t in range(t0 + 1, T + 1):
                    AB = m.write(torch.cat([A, B]), X2, M2).float(); A, B = AB[:len(ch)], AB[len(ch):]
                    lg = torch.log((B - A).flatten(1).norm(dim=1) / nd).cpu().numpy()
                    for r_, i in enumerate(bi):
                        out[i].setdefault(t - t0, []).append(float(lg[r_]))
    return out


def summarise(per_item, ks):
    mean_by_k = {k: float(np.mean([v for pi in per_item for v in pi.get(k, [])])) for k in ks}
    lam = slope(ks, [mean_by_k[k] for k in ks])
    li = [slope(ks, [np.mean(pi[k]) for k in ks]) for pi in per_item if all(k in pi for k in ks)]
    return {"mean_log_g_by_k": {str(k): v for k, v in mean_by_k.items()}, "lambda_hat": lam,
            "per_item_lambda": {"mean": float(np.mean(li)), "median": float(np.median(li)), "share_gt0": float(np.mean(np.array(li) > 0)), "n": len(li)}}


@torch.no_grad()
def step_ratio(m, items, dev, T=16):
    """the clean trajectory from M_0 = 0: s(t) = log(||S_{t+1} - S_t|| / ||S_t - S_{t-1}||), t = 2..T-1, mean over items (2607.09698's 'Lyapunov
    sensitivity' surrogate; descriptive, beside lambda_hat)."""
    acc = {t: [] for t in range(2, T)}
    for b in range(0, len(items), BS):
        ch = items[b:b + BS]; x, msk = m.think_inputs(ch, dev); S = [m.mem.zero_state(len(ch), dev).float()]
        for _ in range(T):
            S.append(m.write(S[-1], x, msk).float())
        d = [(S[t] - S[t - 1]).flatten(1).norm(dim=1) for t in range(1, T + 1)]   # d[t - 1] = ||S_t - S_{t-1}||
        for t in range(2, T):
            acc[t].extend(torch.log(d[t] / d[t - 1].clamp_min(1e-30)).cpu().numpy().tolist())
    return {str(t): float(np.mean(v)) for t, v in acc.items()}


def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"; kind, m, a = H.load_model(run, dev); m.float()
    TE = test_set(a.get("per_n", 1000)); pool = [it for it in TE if it["n"] in (6, 7, 8)]; its = [pool[i] for i in np.random.default_rng(19201).permutation(len(pool))[:256]]
    res = {}
    for t0 in (2, 3, 4):
        pi = run_setting(m, its, dev, lambda it, t0=t0: t0, lambda it: 8); r_ = summarise(pi, list(range(1, 8 - t0 + 1)))
        r_["mean_log_g_by_t"] = {str(t0 + int(k)): v for k, v in r_["mean_log_g_by_k"].items()}; res["A_t0_%d" % t0] = r_
        print("A t0", t0, "lambda_hat", round(r_["lambda_hat"], 4), r_["per_item_lambda"], flush=True)
    pi = run_setting(m, its, dev, lambda it: it["n"] + 1, lambda it: it["n"] + 9); res["B_t0_n_plus_1"] = summarise(pi, list(range(1, 9)))
    res["B_t0_n_plus_1"]["by_n"] = {str(n): summarise([p for p, it in zip(pi, its) if it["n"] == n], list(range(1, 9)))["lambda_hat"] for n in (6, 7, 8)}
    print("B lambda_hat", round(res["B_t0_n_plus_1"]["lambda_hat"], 4), res["B_t0_n_plus_1"]["per_item_lambda"], flush=True)
    sr = step_ratio(m, its, dev); print("step_ratio_by_t", {t: round(v, 3) for t, v in sr.items()}, flush=True)
    out = {"settings": res, "step_ratio_by_t": sr, "step_ratio_def": "clean trajectory from M_0 = 0 to T = 16: mean over items of log(||S_t+1 - S_t|| / ||S_t - S_t-1||), t = 2..15", "n_items": len(its), "n_draws": NDRAW, "eps_rel": EPS, "run": run, "kind": kind, "dtype": "float32 (no autocast, TF32 off)",
           "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "def": "items: test n in {6,7,8}, rng 19201 permutation[:256]; S_t0 from M_0 = 0; delta ~ N(0, I) scaled per item to 1e-3 ||S_t0|| (draw d seed 19202 + "
                  "100003 d + the batch's first item index); g(t) = ||S'_t - S_t|| / ||delta|| over the whole state; lambda_hat = LS slope of mean log g vs t "
                  "(per pass); per-item lambda from each item's draw-mean log g. A: t0 2/3/4, T 8; B: t0 = n + 1, T = n + 9, by k = t - t0"}
    json.dump(out, open(os.path.join(run, "eval_s192_lyapunov.json"), "w", newline="\n"), indent=1); print("DONE lyapunov", run, flush=True)


if __name__ == "__main__":
    main()
