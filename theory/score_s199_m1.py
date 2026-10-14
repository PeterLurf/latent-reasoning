"""THEORY §199 M1 scoring (rules fixed in §199 M1, 2026-10-13 21:23 EDT, before any alpha value was computed; Theory is the only scorer).
x = alpha_pass(model, n) (per-pass geometric-mean delta decay over passes 1..n; CPU subagent, input file theory/data/s199_m1_alpha.json:
    {"<run>": {"9": {"alpha_pass": float, "p_mean_logsv": float | null}, ...}, ...}); y = onset(model, n) on the 187-3 rows (nesting <= 2,
first t with per-n mean handle < 0.80, else T + 1; score_s194.rows_table); y' = mean handle over t 6..min(9, n) (continuous alternative, Audit).
rho_bar = mean over n in 9..12 of Spearman rho_n across models (average ranks). PREDICTED rho_bar > 0; FALSIFIED if rho_bar <= 0 or rho_bar < 0.3.
Reported beside it: per-n rho_n, the y' version, the P-spectrum x version, and the converged-only (9-model) version. Descriptive.
    python theory/score_s199_m1.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s194 as S194  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
CONV = ["s182_window_s1", "s182_window_s2", "s184_reweight_s2", "s185_rms_s1", "s185_rms_s2", "s187_nopass_s1", "s187_nopass_s2", "s190_rmsnopass_s1", "s190_rmsnopass_s2"]
FAIL = ["s185_den_s1", "s185_rms_s3", "s188_rwwin_s1"]
NS, EPS = (9, 10, 11, 12), 1e-9
XFILE = os.path.join(ROOT, "theory", "data", "s199_m1_alpha.json")


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y); n = len(x); mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); sx = sum((a - mx) ** 2 for a in rx) ** 0.5; sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if sx < EPS or sy < EPS else sxy / (sx * sy)


def handle_curve(d, n):
    rows = json.load(open(os.path.join(R, d, "eval_s187_deepaux_items_n9_12.json")))["items"]
    it = [r for r in rows if r["n"] == n and r["nest_depth"] <= 2]; T = max(int(t) for r in it for t in r["handle_by_t"])
    return [sum(r["handle_by_t"][str(t)] for r in it if str(t) in r["handle_by_t"]) / max(1, sum(str(t) in r["handle_by_t"] for r in it)) for t in range(1, T + 1)]


def rho_bar(models, X, xkey, ykind):
    per = {}
    for n in NS:
        xs, ys = [], []
        for d in models:
            xv = X[d][str(n)].get(xkey)
            if xv is None:
                continue
            if ykind == "onset":
                yv = S194.rows_table(d)[n]["onset"]
            else:
                h = handle_curve(d, n); yv = sum(h[5:min(9, n)]) / len(h[5:min(9, n)])
            xs.append(xv); ys.append(yv)
        per[n] = spearman(xs, ys) if len(xs) >= 3 else None
    vals = [v for v in per.values() if v is not None]                     # M1a (Audit): a constant side (all ties) drops that n; reported, never silent
    return (sum(vals) / len(vals) if vals else None), per


def main():
    if not os.path.exists(XFILE):
        print("READING: pending (no %s)" % os.path.relpath(XFILE, ROOT)); return
    X = json.load(open(XFILE)); allm = CONV + FAIL; miss = [d for d in allm if d not in X or any(str(n) not in X[d] for n in NS)]
    assert not miss, ("M1: models / n missing from the alpha file", miss)
    res = {}
    for tag, models in (("all12", allm), ("converged9", CONV)):
        for xkey in ("alpha_pass", "p_mean_logsv"):
            for yk in ("onset", "handle_t6_9"):
                rb, per = rho_bar(models, X, xkey, yk); res["%s/%s/%s" % (tag, xkey, yk)] = {"rho_bar": rb, "per_n": per}
    m = res["all12/alpha_pass/onset"]["rho_bar"]; used = [n for n, v in res["all12/alpha_pass/onset"]["per_n"].items() if v is not None]
    dropped = [n for n in NS if n not in used]
    lab = "INSUFFICIENT (%d of 4 n usable; M1a needs >= 3)" % len(used) if len(used) < 3 else ("PREDICTED (rho_bar > 0, >= 0.3)" if m >= 0.3 - EPS else "FALSIFIED")
    res["n_used"], res["n_dropped"] = used, dropped
    rd = "M1 %s: rho_bar %s over n %s (dropped, a constant side: %s) (per n %s) | continuous y' %s | P-spectrum x %s | converged-only %s" % (
        lab, None if m is None else round(m, 3), used, dropped or "none", {n: None if v is None else round(v, 3) for n, v in res["all12/alpha_pass/onset"]["per_n"].items()},
        res["all12/alpha_pass/handle_t6_9"]["rho_bar"], res["all12/p_mean_logsv/onset"]["rho_bar"], res["converged9/alpha_pass/onset"]["rho_bar"])
    res["reading"] = rd; print("READING:", rd)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s199_m1.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
