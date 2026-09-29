"""PROTOCOL5 plan step 2 (exploratory): which measurable property of a trained model separates the models that
extrapolate (follow chains longer than any in training when given more latent steps) from those that do not?

    python scratch/extrap_predictors.py          -> results/extrap_predictors.json

Models: the 22 addendum-J-competent recurrent models of PROTOCOL4 addendum E (results/confirm_sweep.json).
Outcome: extrapolates = distance-7 delayed bal-acc >= 0.60 with bootstrap lower bound > 0.55 at K' = 12 or 16
(length_gen files; every model has at least one of the two rows).
Features come from data already on disk and are, except where marked "post-budget", measured inside the trained
regime (latent steps <= K, in-distribution graphs), so none restates the outcome:
  * confirm_sweep.json (confirm split, W = 2/4/8 layer-1 traces; delayed blocking at W = 4, steps 1..max(8, K))
  * competence / competence_matched files (accuracy at the training budget)
  * runs/<run>/log.jsonl (training dynamics)
"""
import glob
import json
import os

import numpy as np

CS = json.load(open("results/confirm_sweep.json"))
MODELS = list(CS)
DEPTHS = (1, 2, 3)


def find(prefix, run):
    """First results/<prefix>*.json containing the run key."""
    for f in sorted(glob.glob(f"results/{prefix}*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, dict) and run in d:
            return d[run]
    return None


def outcome(run):
    L = find("length_gen", run)
    best, ext, rows = None, False, {}
    for row in L["rows"]:
        if row["K'"] in (12, 16):
            a, ci = row["far"]["7"], row.get("far_ci", {}).get("7")
            rows[row["K'"]] = a
            best = a if best is None else max(best, a)
            ext |= bool(a >= 0.60 and ci is not None and ci[0] > 0.55)
    return {"extrapolates": ext, "d7_best": best, "d7": rows}


def training(run):
    f = f"{run}/log.jsonl"
    full, last = None, 0
    for line in open(f):
        x = json.loads(line)
        last = max(last, x.get("step", 0))
        if "curriculum_advance_to" in x and x["curriculum_advance_to"] is None:
            full = x.get("step")
    cfg = json.load(open(f"{run}/config.json"))
    return {"full_depth_step": full, "steps_at_full": None if full is None else last - full,
            "no_step_emb": bool(cfg.get("no_step_emb")), "K_train": cfg.get("K"),
            "K_random": bool(cfg.get("K_random")), "layers": cfg.get("layers", 2), "d": cfg.get("d", 256),
            "lr": cfg.get("lr")}


def competence(run):
    m = find("competence_matched", run)
    return {"matched5": m["5"]["bal_acc"] if m else None, "matched6": m["6"]["bal_acc"] if m else None}


def sweep(run):
    o = CS[run]
    K, Kp = o["K"], o["Kp"]
    T = o["trace"]["4"]
    R, D = np.array(T["reachable"]), np.array(T["decoy"])
    ex = R - D                                                # (step, depth) per-edge excess, W = 4
    ks = o["clock"]                                           # 1-based argmax step for depths 1..3
    Kin = min(K, Kp)
    peak = ex[:Kin, 1:].max()
    B = {int(k): v for k, v in o["block"]["downstream"].items()}
    off = [abs(B[k][x]) for x, kx in zip(DEPTHS, ks) for k in B if abs(k - kx) >= 2 and k <= K]
    diag = [B[kx][x] for x, kx in zip(DEPTHS, ks)]
    E = [int(np.argmax(R[k])) for k in range(min(6, Kp)) if R[k].max() >= 1.5 * D[k].mean()]
    f = {
        "E_confirm": max(E) if E else 0,
        "clock_rate": (ks[2] - ks[0]) / 2,                     # latent steps per depth
        "slack": K - ks[2],                                    # budget left after depth 3 is read
        "clock_same_W": all(o["clock_by_W"][w] == ks for w in ("2", "4", "8")),
        "offdiag_max": max(off) if off else 0.0,               # multi-step reading of a depth
        "diag_mean": float(np.mean(diag)),                     # causal strength of the sweep
        "amp_d1": float(ex[ks[0] - 1, 1]),                     # per-edge frontier attention, depth 1
        "end_activity": float(ex[Kin - 1, 1:].max() / peak) if peak > 0 else 0.0,  # sweep still active at step K
        "end_depth": int(np.argmax(ex[Kin - 1, 1:]) + 1),
        "block_last_step": float(-min(B[Kin])),                # blocking at step K still cuts deeper nodes
        "block_last2": float(-min(min(B[k]) for k in (Kin - 1, Kin))),
    }
    if Kp > K:                                                 # post-budget (steps K+1..Kp, in-distribution graphs)
        f["post_block"] = float(-min(min(B[k]) for k in range(K + 1, Kp + 1)))
        f["post_activity"] = float(ex[K:, 1:].max() / peak) if peak > 0 else 0.0
    return f


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))


def screen(rows, feat, rng):
    xs = [(r[feat], r["extrapolates"]) for r in rows if r.get(feat) is not None]
    x = np.array([float(a) for a, _ in xs]); y = np.array([b for _, b in xs])
    if y.sum() < 2 or (~y).sum() < 2:
        return None
    a = auc(x, y)
    sign = 1 if a >= 0.5 else -1
    a2 = max(a, 1 - a)
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(x), len(x))
        if y[i].sum() and (~y[i]).sum():
            b = auc(x[i], y[i]); boot.append(b if sign > 0 else 1 - b)
    perm = [max(auc(x, rng.permutation(y)), 1 - auc(x, rng.permutation(y))) for _ in range(0)]
    null = []
    for _ in range(5000):
        p = rng.permutation(y); b = auc(x, p); null.append(max(b, 1 - b))
    return {"n": int(len(x)), "n_pos": int(y.sum()), "auc": float(a2), "direction": "higher" if sign > 0 else "lower",
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p_perm": float((np.sum(np.array(null) >= a2) + 1) / (len(null) + 1)),
            "pos_values": sorted(float(v) for v in x[y]), "neg_values": sorted(float(v) for v in x[~y])}


def main():
    rng = np.random.default_rng(0)
    rows = []
    for run in MODELS:
        r = {"run": run, **outcome(run), **training(run), **competence(run), **sweep(run)}
        rows.append(r)
    feats = ["E_confirm", "clock_rate", "slack", "clock_same_W", "offdiag_max", "diag_mean", "amp_d1", "end_activity",
             "end_depth", "block_last_step", "block_last2", "matched5", "matched6", "full_depth_step",
             "steps_at_full", "K_train", "no_step_emb", "post_block", "post_activity"]
    S = {f: screen(rows, f, rng) for f in feats}
    in_budget = [f for f in feats if f not in ("post_block", "post_activity", "K_train", "no_step_emb") and S[f]]
    ps = sorted((S[f]["p_perm"], f) for f in in_budget)
    m = len(ps)
    holm, run_max = {}, 0.0
    for i, (p, f) in enumerate(ps):
        run_max = max(run_max, min(1.0, (m - i) * p)); holm[f] = run_max
    for f in in_budget:
        S[f]["p_holm"] = holm[f]
    out = {"models": rows, "screen": S, "in_budget_features": in_budget,
           "note": "Exploratory (plan step 2). AUC is direction-free (max(AUC, 1-AUC)); p_perm = label permutation, "
                   "p_holm over the in-budget features. post_* features use steps beyond K on in-distribution graphs "
                   "and exist only for models with K < 8."}
    json.dump(out, open("results/extrap_predictors.json", "w"), indent=1, default=float)
    print(f"{'feature':18s} {'n':>3s} {'pos':>3s} {'AUC':>5s}  dir     CI            p_perm  p_holm")
    for f in sorted([f for f in feats if S[f]], key=lambda f: -S[f]["auc"]):
        s = S[f]
        print(f"{f:18s} {s['n']:3d} {s['n_pos']:3d} {s['auc']:.2f}  {s['direction']:6s}  [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]"
              f"  {s['p_perm']:.4f}  {s.get('p_holm', float('nan')):.3f}")
    print()
    for r in sorted(rows, key=lambda r: -r["d7_best"]):
        print(f"{r['run'][5:]:20s} ext={int(r['extrapolates'])} d7={r['d7_best']:.3f} K={r['K_train']} clock={CS[r['run']]['clock']} "
              f"end_act={r['end_activity']:.2f} blockK={r['block_last_step']:.3f} off={r['offdiag_max']:.3f} "
              f"slack={r['slack']} m6={r['matched6']:.3f} full@{r['full_depth_step']}"
              + (f" post_block={r['post_block']:.3f}" if 'post_block' in r else ""))


if __name__ == "__main__":
    main()
