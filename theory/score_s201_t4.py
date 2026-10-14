"""THEORY §201 note 201-7 scoring (T4 SNAP-CHAIN; rules fixed before any output; Theory is the only scorer).
<run>/eval_s201_snap.json = {"policies": {"LATENT_BEST"|"LATENT_HALT"|"TIMEOUT7"|"FIX2"|"FIX4"|"FIX7"|"LEN_IN_RANGE": {"acc_by_n": {n: acc},
  "mean_passes_by_n": {n: p}, "mean_tokens_by_n": {n: t}}}, "qW_kw7_by_n": {n: q}, "parse_valid_by_n", "segment_k0_by_n", "best_fixed_k_train": K}.
Per model, n 11-12 pooled = mean(acc n11, acc n12). LAT = max(LATENT_BEST, LATENT_HALT). PRIMARY TIMEOUT7: SNAP-HELPS (>= LAT + 0.15 and mean
total passes (n 11-12 mean) <= best_fixed_k_train + 4) / SNAP-NEUTRAL (within +-0.05) / SNAP-HURTS (<= LAT - 0.05) / mixed.
T4-a (§190 s1, if qW at 7 (n 11-12 mean) >= 0.6): SNAP-HELPS. T4-b: FIX7 >= FIX4 >= FIX2 at n 11-12; falsified if FIX4 > FIX7 + 0.05.
T4-c: n 14 helped (TIMEOUT7 >= LAT_14 + 0.15) and n 16 not; falsified if n 16 helped by >= 0.15.
    python theory/score_s201_t4.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
MODELS = ["s190_rmsnopass_s1", "s185_rms_s1", "s185_rms_s2", "s184_reweight_s2"]
EPS = 1e-9


def read(d):
    p = os.path.join(R, d, "eval_s201_snap.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p)); P = j["policies"]
    acc = lambda pol, ns: sum(P[pol]["acc_by_n"][str(n)] for n in ns) / len(ns)   # noqa: E731
    lat = lambda ns: max(acc("LATENT_BEST", ns), acc("LATENT_HALT", ns))   # noqa: E731
    t7, L = acc("TIMEOUT7", (11, 12)), lat((11, 12))
    passes = sum(P["TIMEOUT7"]["mean_passes_by_n"][str(n)] for n in (11, 12)) / 2
    if t7 >= L + 0.15 - EPS and passes <= j["best_fixed_k_train"] + 4 + EPS:
        lab = "SNAP-HELPS"
    elif abs(t7 - L) <= 0.05 + EPS:
        lab = "SNAP-NEUTRAL"
    elif t7 <= L - 0.05 + EPS:
        lab = "SNAP-HURTS"
    else:
        lab = "mixed"
    qw = sum(j["qW_kw7_by_n"][str(n)] for n in (11, 12)) / 2
    f2, f4, f7 = (acc(p_, (11, 12)) for p_ in ("FIX2", "FIX4", "FIX7"))
    h14 = acc("TIMEOUT7", (14,)) >= lat((14,)) + 0.15 - EPS
    h16 = acc("TIMEOUT7", (16,)) >= lat((16,)) + 0.15 - EPS
    return {"label": lab, "timeout7_11_12": t7, "LAT_11_12": L, "passes_11_12": passes, "qW7_11_12": qw, "fix": (f2, f4, f7),
            "T4b_order_holds": f7 >= f4 - EPS and f4 >= f2 - EPS, "T4b_falsified": f4 > f7 + 0.05 + EPS, "n14_helped": h14, "n16_helped": h16}


def main():
    res = {d: read(d) for d in MODELS}; lines = []
    for d, r in res.items():
        if r is None:
            lines.append("%s pending" % d); continue
        lines.append("%s %s (TIMEOUT7 %.3f vs LAT %.3f, passes %.1f, qW7 %.2f; FIX2/4/7 %.3f/%.3f/%.3f; n14 %s n16 %s)" % (
            d, r["label"], r["timeout7_11_12"], r["LAT_11_12"], r["passes_11_12"], r["qW7_11_12"], r["fix"][0], r["fix"][1], r["fix"][2], r["n14_helped"], r["n16_helped"]))
    pr = {}
    a = res["s190_rmsnopass_s1"]
    if a:
        pr["T4-a"] = "not applicable (qW7 < 0.6)" if a["qW7_11_12"] < 0.6 - EPS else ("holds" if a["label"] == "SNAP-HELPS" else "FALSIFIED")
    done = [r for r in res.values() if r]
    if len(done) == len(MODELS):
        pr["T4-b"] = "FALSIFIED" if any(r["T4b_falsified"] for r in done) else ("holds" if all(r["T4b_order_holds"] for r in done) else "order not monotone (not falsified)")
        pr["T4-c"] = "FALSIFIED (n16 helped)" if any(r["n16_helped"] for r in done) else ("holds" if all(r["n14_helped"] for r in done) else "n14 not helped everywhere (n16 boundary holds)")
    out = {"models": res, "predictions": pr, "reading": " | ".join(lines)}
    print("READING:", out["reading"]); print("PREDICTIONS:", pr)
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s201_t4.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
