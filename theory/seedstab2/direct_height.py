"""SEEDSTAB2 check D (POST-HOC data): the one-shot accuracy ceiling by tree height. DIRECT (one trained GPT-2 pass on x, answer only, 48k steps)
acc_by_n (n = 1..8, its training range) = sum_h P(root height h | n) a_h; fit a_h (h = 1..8) by projected least squares on [0, 1] with a monotone
(non-increasing in h) constraint by isotonic projection. Also: the share of SVR targets (reduction t, its node height) with height >= 3 / >= 4 by t.
Reads boxes/runs_fsq/s170_direct_s{1,2}/eval.json (acc_by_n only). Output: direct_height.json beside this file."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heights as H                                                         # noqa: E402  (re-runs its small MC; fine)

rng = np.random.default_rng(170)
PH = np.zeros((8, 8)); share = {t: np.zeros(9) for t in range(1, 9)}
for n in range(1, 9):
    for _ in range(6000):
        r, _t = H.generate(rng, n); H.set_h(r); PH[n - 1, r.h - 1] += 1; hs = H.red_heights(r)
        for t, h in enumerate(hs, 1):
            share[t][h] += 1
PH /= PH.sum(1, keepdims=True)


def iso_dec(y):                                                             # pool-adjacent-violators for a non-increasing fit
    y = list(y); w = [1.0] * len(y); i = 0
    blocks = [[v, 1.0] for v in y]
    out = []
    for v in y:
        out.append([v, 1.0])
        while len(out) > 1 and out[-2][0] < out[-1][0]:
            a, b = out.pop(), out.pop(); out.append([(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1]])
    res = []
    for v, c in out:
        res += [v] * int(c)
    return np.array(res)


res = {"P_root_height_given_n": PH.round(4).tolist()}
for s in (1, 2):
    acc = json.load(open("D:/Latent/boxes/runs_fsq/s170_direct_s%d/eval.json" % s))["acc_by_n"]; y = np.array([acc[str(n)] for n in range(1, 9)])
    a = np.full(8, 0.5)
    for _ in range(20000):                                                  # projected gradient on 0.5 ||PH a - y||^2
        a = a - 0.5 * PH.T @ (PH @ a - y); a = np.clip(iso_dec(a), 0.0, 1.0)
    res["direct_s%d" % s] = {"a_h_fit": a.round(3).tolist(), "fit_resid": float(np.abs(PH @ a - y).max()), "acc_by_n": y.tolist()}
    print("DIRECT s%d one-shot acc by height 1..8:" % s, a.round(2), "max resid %.3f" % np.abs(PH @ a - y).max())
tab = {}
for t in range(1, 9):
    c = share[t]; tot = c.sum(); tab[str(t)] = {"h>=3": float(c[3:].sum() / tot), "h>=4": float(c[4:].sum() / tot)}
    print("t", t, "share h>=3 %.3f h>=4 %.3f" % (tab[str(t)]["h>=3"], tab[str(t)]["h>=4"]))
res["svr_target_height_share_by_t"] = tab
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "direct_height.json"), "w", newline="\n"), indent=1)
