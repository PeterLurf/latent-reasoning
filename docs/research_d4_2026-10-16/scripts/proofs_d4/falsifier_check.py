"""PROOFS_D4 check 6: Audit's proposed shifted-deadline falsifier.
F_AUDIT: OLS slope of median t_fin on n over n 9..14 >= 0.7 AND spread (max - min of the medians over n 9..14) >= 3.
t_fin = first pass whose decoded output has 0 operators alive (d4_edge_collapse.py's definition; 99 if never within K' 24).
Models of plain-pass test behaviour (no markers at test):
  CONTENT(q): one reduction per pass, plus a state-independent over-reduction (drop 2) with prob q per pass when >= 3 alive.
  HARD(t_d): constant output deadline: t_fin = min(n, t_d) (t_d half-integer = 50/50 mix of floor / ceil).
  SOFT(Delta): N1's measured on-schedule hazard h(t) (refs: t 2..9, n 10..14 pooled), shifted to h(t - Delta), with N1's empirical drop-size law
               for collapse events (read from N1's eval_vbhalt rows); validated at Delta = 0 against N1's measured medians.
  GATED-9: history- or marker-gated deadline (D0b): at test the history is pause-free -> behaves like SOFT(0) (N1-like).
Also evaluates (i) N1 / N2 measured medians, (ii) alternative ranges n 12..16, (iii) a START-D variant: test-time start pauses D, an a-keyed
constant deadline gives t_fin = min(n + D, t_d) in bank index, content gives n + D."""
import json, sys
import numpy as np
rng = np.random.default_rng(0)
REF = json.load(open("D:/Latent/docs/research_d4_2026-10-16/scripts/d4_edge_collapse_refs.json"))
N1 = REF["D:/Latent/boxes/runs_fsq/s211_vb_carry_s1"]; N2 = REF["D:/Latent/boxes/runs_fsq/s211_vb_pa_s1"]
h_on = {int(t): v for t, v in N1["h_by_t"].items()}
def h_at(t):
    if t < 2:
        return 0.0
    if t > 9:
        return 0.995
    lo = int(np.floor(t)); hi = lo + 1; f = t - lo
    return (1 - f) * h_on.get(lo, 0.0) + f * h_on.get(hi, 0.995) if hi <= 9 else (1 - f) * h_on[lo] + f * 0.995
# N1 empirical drop sizes (on-schedule collapse events, n 10..14, >= 3 alive), from the raw rows (explicit file, read-only)
rows = json.load(open("D:/Latent/boxes/runs_fsq/s211_vb_carry_s1/eval_vbhalt.json"))["rows"]
drops = []; frac_to_zero = []
for r in rows:
    n, al = r["n"], r["alive"]
    if 10 <= n <= 14:
        for t in range(2, len(al)):
            if al[t - 1] == n - t + 1 and al[t - 1] >= 3 and al[t - 1] - al[t] >= 2:
                drops.append(al[t - 1] - al[t]); frac_to_zero.append(al[t] == 0)
drops = np.array(drops); dvals, dcnt = np.unique(drops, return_counts=True); dlaw = dcnt / dcnt.sum()
# N1 measured medians per n (raw rows)
def med_tfin(rows_, n):
    v = []
    for r in rows_:
        if r["n"] == n:
            al = r["alive"]; t_ = next((t for t in range(len(al)) if al[t] == 0), None); v.append(99 if t_ is None else t_)
    return float(np.median(v))
def ols(ns, ys):
    x = np.array(ns, float); y = np.array(ys, float); return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
def verdict(med, lo=9, hi=14):
    ns = list(range(lo, hi + 1)); ys = [med[n] for n in ns]; s = ols(ns, ys); sp = max(ys) - min(ys)
    return {"slope": round(s, 3), "spread": sp, "PASS(content-tracking)": bool(s >= 0.7 and sp >= 3)}
def sim(model, n, M=4000, K=24, D=0):
    tf = []
    for _ in range(M):
        a = n; t_ = None
        for t in range(1, K + 1):
            if t <= D:                                     # START-D pause passes (a-keyed age advances; no reduction)
                continue
            if model[0] == "CONTENT":
                d = 2 if (a >= 3 and rng.random() < model[1]) else 1
            elif model[0] == "SOFT":
                if a >= 3 and rng.random() < h_at(t - model[1]):
                    d = int(rng.choice(dvals, p=dlaw))
                else:
                    d = 1
            a = max(0, a - d)
            if a == 0:
                t_ = t; break
        tf.append(99 if t_ is None else t_)
    return float(np.median(tf))
def hard(n, td, D=0):
    lo, hi = int(np.floor(td)), int(np.ceil(td))
    return 0.5 * (min(n + D, lo) + min(n + D, hi))       # median of a 50/50 mix is ambiguous; use the mean of the two medians
out = {"N1_drop_law": {int(k): float(p) for k, p in zip(dvals, dlaw)}, "N1_collapse_to_zero_frac": float(np.mean(frac_to_zero))}
meas = {"N1_measured": {n: med_tfin(rows, n) for n in range(9, 17)},
        "N2_measured": {int(k): v for k, v in N2["median_t_fin"].items()}}
models = {}
for q in (0.0, 0.05, 0.10):
    models["CONTENT(q=%.2f)" % q] = {n: sim(("CONTENT", q), n) for n in range(9, 17)}
for dl in (0.0, 2.0, 3.0, 4.5, 6.5, 8.0):
    models["SOFT(Delta=%.1f)" % dl] = {n: sim(("SOFT", dl), n) for n in range(9, 17)}
for td in (9, 11, 12, 12.5, 13, 13.5, 14, 15.5, 16, 17, 20):
    models["HARD(t_d=%s)" % td] = {n: hard(n, td) for n in range(9, 17)}
res = {}
for name, med in {**meas, **models}.items():
    res[name] = {"medians": med, "F_AUDIT n9-14": verdict(med), "alt n12-16": verdict(med, 12, 16), "alt n9-16": verdict(med, 9, 16)}
out["models"] = res
# smallest constant deadline that passes F_AUDIT
grid = np.arange(9, 17.01, 0.25); first = None
for td in grid:
    if verdict({n: hard(n, td) for n in range(9, 15)})["PASS(content-tracking)"]:
        first = float(td); break
out["HARD_min_t_d_passing_F_AUDIT"] = first
# START-D variant: n 9..12, D in {0, 2, 4, 6, 8}; regress median (t_fin) on (n + D) over cells with n + D in [12, 20]
sd = {}
for name, f in {"CONTENT": lambda n, D: n + D, "HARD(13.5)": lambda n, D: hard(n, 13.5, D), "HARD(16)": lambda n, D: hard(n, 16, D),
                "HARD(20)": lambda n, D: hard(n, 20, D), "GATED-9 (unlocked by any pause)": lambda n, D: (min(n, 9) if D == 0 else n + D)}.items():
    xs = []; ys = []
    for n in range(9, 13):
        for D in (0, 2, 4, 6, 8, 12):
            if 12 <= n + D <= 20:
                xs.append(n + D); ys.append(f(n, D))
    sd[name] = {"slope_on_n_plus_D (cells n+D in 12..20)": round(ols(xs, ys), 3), "max_minus_min": max(ys) - min(ys)}
out["START_D_variant"] = sd
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1, default=float)
print("N1 drop law", out["N1_drop_law"], "to-zero frac %.3f" % out["N1_collapse_to_zero_frac"])
for name, v in res.items():
    print("%-22s med %s | F_AUDIT %s | n12-16 %s" % (name, [v["medians"][n] for n in range(9, 17)], v["F_AUDIT n9-14"], v["alt n12-16"]))
print("min constant deadline passing F_AUDIT:", first)
for k, v in sd.items():
    print("START-D", k, v)
