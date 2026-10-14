"""THEORY §199 note 199-M2 scoring (same-init linear mode connectivity; rules fixed before any interpolation was computed; Theory is the only scorer).
Inputs boxes/runs_fsq/s199_lmc/lmc_<A>_<B>.json = {lam: [...], loss_train: [...], loss_test: [...], path_indep: [{delta_cos_final, answer_agree, ...}], gpu, dtype}.
B = max over lam of [L(lam) - ((1 - lam) L(0) + lam L(1))] on TRAIN-DIST (TEST beside).
SEPARATE-BASIN: B(s1-s3) > 3 B(s1-s2) and > 0.5, and the same for s2-s3. CONNECTED: for both s1-s3 and s2-s3, B <= 0.1 or <= 1.5 B(s1-s2). Else UNCLEAR.
Descriptive: the lam where delta_cos_final crosses 0.65 on each path (linear interpolation between grid points), and the barrier argmax.
    python theory/score_s199_m2.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "boxes", "runs_fsq", "s199_lmc")
PAIRS = (("s185_rms_s1", "s185_rms_s2"), ("s185_rms_s1", "s185_rms_s3"), ("s185_rms_s2", "s185_rms_s3"))
EPS = 1e-9


def barrier(lam, L):
    assert abs(lam[0]) < EPS and abs(lam[-1] - 1) < EPS and len(lam) == len(L)
    ex = [l_ - ((1 - x) * L[0] + x * L[-1]) for x, l_ in zip(lam, L)]; k = max(range(len(ex)), key=lambda i: ex[i])
    return ex[k], lam[k]


def crossings(lam, pi, thr=0.65):
    v = [p["delta_cos_final"] for p in pi]; out = []
    for i in range(len(v) - 1):
        if (v[i] - thr) * (v[i + 1] - thr) < 0:
            out.append(lam[i] + (thr - v[i]) / (v[i + 1] - v[i]) * (lam[i + 1] - lam[i]))
    return out


def main():
    res = {}
    for a, b in PAIRS:
        p = os.path.join(D, "lmc_%s_%s.json" % (a, b))
        if not os.path.exists(p):
            res["%s-%s" % (a[-2:], b[-2:])] = None; continue
        j = json.load(open(p)); bt, at = barrier(j["lam"], j["loss_train"]); bs, _ = barrier(j["lam"], j["loss_test"])
        res["%s-%s" % (a[-2:], b[-2:])] = {"B_train": bt, "argmax_lam": at, "B_test": bs, "L_end_train": (j["loss_train"][0], j["loss_train"][-1]),
                                           "dcos_cross_0.65": crossings(j["lam"], j["path_indep"]) if j.get("path_indep") else None, "gpu": j.get("gpu"), "dtype": j.get("dtype")}
    r12, r13, r23 = res.get("s1-s2"), res.get("s1-s3"), res.get("s2-s3")
    if not (r12 and r13 and r23):
        lab = "pending (%s)" % [k for k, v in res.items() if v is None]
    else:
        sep = lambda r: r["B_train"] > 3 * r12["B_train"] + EPS and r["B_train"] > 0.5 + EPS   # noqa: E731
        cn = lambda r: r["B_train"] <= 0.1 + EPS or r["B_train"] <= 1.5 * r12["B_train"] + EPS   # noqa: E731
        con = cn(r13) and cn(r23)
        lab = "SEPARATE-BASIN" if (sep(r13) and sep(r23)) else ("CONNECTED" if con else "UNCLEAR")
    rd = "M2 %s :: %s" % (lab, " | ".join("%s B_train %s (argmax lam %s), B_test %s, dcos crosses 0.65 at %s" % (
        k, None if v is None else round(v["B_train"], 4), None if v is None else v["argmax_lam"], None if v is None else round(v["B_test"], 4),
        None if v is None else v["dcos_cross_0.65"]) for k, v in res.items()))
    print("READING:", rd)
    json.dump({"reading": rd, "pairs": res}, open(os.path.join(ROOT, "theory", "results", "score_s199_m2.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
