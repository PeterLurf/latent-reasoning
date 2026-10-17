"""RT-4: adversarial fixtures through the draft's own reader (d4ec_copy.read = docs/.../d4_edge_collapse.py, copied unchanged).
FREEZE3   : one reduction per pass; at pass 9 an item with >= 3 operators alive freezes for good (STUCK); fewer -> finishes on schedule.
SHIFT12/13: the draft's DEADLINE fixture shifted by +3 / +4 (hazard 0.05/0.2/0.5/0.8 at t_d-4..t_d-1, collapse to 0 at t_d).
SIZEKEY   : N2-like hazard, logit = -10.67 + 0.454 t + 0.225 T (rt_hazard_keys N2 'fit t+T'), T from the replicated §170 test set.
TONLY     : hazard logit = -8.60 + 1.222 t (N1 'fit t').
Each: EC / DG by the draft rule; Audit (b)'s slope test (median t_fin on n 9-14 OLS slope >= 0.7, spread >= 3); my T-split (h_highT - h_lowT at fixed t 6-8, n 10-12)."""
import json, os, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import d4ec_copy as EC
from testset import test_set
TE = test_set(); Tn = {n: [it["T"] for it in TE if it["n"] == n] for n in range(1, 17)}
rng = np.random.default_rng(7)
def ols(xs, ys):
    xs = np.array(xs, float); ys = np.array(ys, float); return float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())
def traj_hazard(n, T, logit):
    al = [n]; a = n
    for t in range(1, 25):
        if a >= 3 and rng.random() < 1 / (1 + np.exp(-logit(t, T))):
            a = max(0, a - (2 + int(rng.integers(0, 2))))
        else:
            a = max(0, a - 1)
        al.append(a)
    return al
def make(kind):
    rows = []
    for n in range(9, 17):
        for k in range(1000):
            T = Tn[n][k]
            if kind == "FREEZE3":
                al = [max(0, n - t) for t in range(9)]; a = al[-1]
                al += [a if a >= 3 else max(0, a - (t - 8)) for t in range(9, 25)]
            elif kind.startswith("SHIFT"):
                td = int(kind[5:]); al = [n]; a = n
                for t in range(1, 25):
                    hz = {td - 4: 0.05, td - 3: 0.2, td - 2: 0.5, td - 1: 0.8}.get(t, 1.0 if t >= td else 0.0)
                    a = 0 if t >= td else (max(0, a - (2 + int(rng.integers(0, 2)))) if (rng.random() < hz and a >= 3) else max(0, a - 1)); al.append(a)
            elif kind == "SIZEKEY":
                al = traj_hazard(n, T, lambda t, T_: -10.67 + 0.454 * t + 0.225 * T_)
            elif kind == "TONLY":
                al = traj_hazard(n, T, lambda t, T_: -8.60 + 1.222 * t)
            rows.append({"n": n, "item": 1000 * (n - 1) + k, "alive": al})
    return rows
def tsplit(rows):
    out = []
    for t in (6, 7, 8):
        for n in (10, 11, 12):
            ev = [(TE[r["item"]]["T"], int(r["alive"][t - 1] - r["alive"][t] >= 2)) for r in rows if r["n"] == n and r["alive"][t - 1] == n - t + 1 and r["alive"][t - 1] >= 3]
            if len(ev) < 90:
                continue
            Ts = np.array([e[0] for e in ev]); c = np.array([e[1] for e in ev]); q1, q2 = np.quantile(Ts, [1 / 3, 2 / 3])
            if (Ts <= q1).sum() >= 20 and (Ts > q2).sum() >= 20:
                out.append(float(c[Ts > q2].mean() - c[Ts <= q1].mean()))
    return round(float(np.mean(out)), 3) if out else None
res = {}
for kind in ("FREEZE3", "SHIFT12", "SHIFT13", "SIZEKEY", "TONLY"):
    rows = make(kind); d = tempfile.mkdtemp(dir=HERE); json.dump({"rows": rows}, open(os.path.join(d, "eval_vbhalt.json"), "w"))
    r = EC.read(d); os.remove(os.path.join(d, "eval_vbhalt.json")); os.rmdir(d)
    med = {int(k): v for k, v in r["median_t_fin"].items()}; ns = list(range(9, 15))
    res[kind] = {"EC": r["EDGE_COLLAPSE"], "DG": r["DEADLINE_GONE"], "h_7_9": round(r["h_7_9"], 3) if r["h_7_9"] is not None else None, "events_7_9": r["events_7_9"],
                 "median_t_fin": med, "audit_slope_9_14": round(ols(ns, [med[n] for n in ns]), 3), "audit_spread": med[14] - med[9],
                 "audit_pass": bool(ols(ns, [med[n] for n in ns]) >= 0.7 and med[14] - med[9] >= 3), "Tsplit_mean_diff": tsplit(rows)}
json.dump(res, open(os.path.join(HERE, "rt_fixtures.json"), "w", newline="\n"), indent=1)
for k, v in res.items():
    print(k, v)
