"""PROTOCOL7 T5: incremental validity of P7 over in-distribution covariates (retrospective, P30 + L competent runs).
Written 2026-09-28 ~23:59, before the analysis was run.
Covariates: delayed bal-acc, matched 5- and 6-hop bal-acc, full-depth arrival step, end activity (EA).
10-fold x 20-repeat cross-validated AUC of L2 logistic regression (standardised features, lambda = 1), covariates alone
vs covariates + P7. Supported if delta AUC >= 0.10 and covariate-only AUC <= 0.80; not supported if delta < 0.03.
    PYTHONPATH=. python scratch/p7s_t5.py
"""
import json

import numpy as np

from latent import protocol5_confirm as P5


def fit(X, y, lam=1.0, it=3000, lr=0.1):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + lam * w / len(y))
        b -= lr * (p - y).mean()
    return w, b


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def cv_auc(X, y, reps=20, k=10, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        idx = rng.permutation(len(y)); folds = np.array_split(idx, k); pred = np.zeros(len(y))
        for f in folds:
            tr = np.setdiff1d(idx, f)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            w, b = fit((X[tr] - mu) / sd, y[tr])
            pred[f] = (X[f] - mu) / sd @ w + b
        out.append(auc(pred, y))
    return float(np.mean(out)), float(np.std(out))


def main():
    EA = json.load(open("results/p6_endact.json"))
    rows = []
    for f in ("results/p30.json", "results/p_l.json"):
        d = json.load(open(f))
        for r in d["primary"]:
            p = P5.pilot_run(r)
            adv = [json.loads(l) for l in open(f"runs/{r}/log.jsonl") if "curriculum_advance_to" in l]
            full = next(x["step"] for x in adv if x["curriculum_advance_to"] is None)
            rows.append({"run": r, "extrap": d["rows"][r]["extrap"], "P7": d["rows"][r]["P7"], "delayed": p["delayed"],
                         "matched5": p["matched5"], "matched6": p["matched6"], "full_step": full,
                         "EA": EA[f"runs/{r}"]["end_activity"]})
    y = np.array([x["extrap"] for x in rows], float)
    cov = ["delayed", "matched5", "matched6", "full_step", "EA"]
    Xc = np.array([[x[c] for c in cov] for x in rows]); Xp = np.column_stack([Xc, [x["P7"] for x in rows]])
    a_c, s_c = cv_auc(Xc, y); a_p, s_p = cv_auc(Xp, y)
    single = {c: auc([x[c] for x in rows], y.astype(bool)) for c in cov + ["P7"]}
    d_ = a_p - a_c
    dec = "supported" if d_ >= 0.10 and a_c <= 0.80 else "not supported" if d_ < 0.03 else "inconclusive"
    out = {"n": len(rows), "n_extrap": int(y.sum()), "cv_auc_covariates": [a_c, s_c], "cv_auc_plus_P7": [a_p, s_p],
           "delta": d_, "single_auc": single, "T5": dec}
    json.dump({"rows": rows, "summary": out}, open("results/p7s_t5.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
