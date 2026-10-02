"""THEORY §70 scoring (rules fixed in §70): A candidate chosen on development seeds 1300-1302 from {mul, opb12} by
certified count, then mean accuracy at N 128, then opb; A passes if the candidate certifies >= 9/10 on 1400-1409 and >= 3
more than add (same seeds). B passes if temper certifies >= 9/10 and >= 3 more than mml2. Wilson 95 % intervals; paired
outcomes; entropy-collapse step (first logged update with mean transition-row entropy < 0.1 nats) descriptive.
    python theory/c1_acq_score.py [--dir theory/results/c1_acq] [--out theory/results/score_s70.json]
"""
import argparse
import json
import math
import os

import numpy as np

DEV, CONF = list(range(1300, 1303)), list(range(1400, 1410))
ARMS = ["A_add", "A_mul", "A_opb12", "A_opb24", "B_temper", "B_eta1", "B_mml2", "B_mml", "B_exr"]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="theory/results/c1_acq"); ap.add_argument("--out", default="theory/results/score_s70.json")
    a = ap.parse_args()
    R = {}
    for arm in ARMS:
        for s in DEV + CONF:
            f = os.path.join(a.dir, "%s_s%d.json" % (arm, s))
            R[arm, s] = next(iter(json.load(open(f)).values())) if os.path.exists(f) else None
    cert = lambda arm, seeds: [bool(R[arm, s] and R[arm, s]["certified"]) for s in seeds]          # noqa: E731
    acc = lambda arm, seeds: float(np.mean([R[arm, s]["acc"]["128"] for s in seeds if R[arm, s]]))  # noqa: E731
    out = {"arms": {}}
    for arm in ARMS:
        cd, cc = cert(arm, DEV), cert(arm, CONF)
        col = []
        for s in CONF:
            tr = (R[arm, s] or {}).get("ent_trace", {})
            col.append(next((int(u) for u, v in sorted(tr.items(), key=lambda x: int(x[0])) if v < 0.1), None))
        out["arms"][arm] = {"dev": sum(cd), "conf": sum(cc), "conf_wilson": wilson(sum(cc), 10), "conf_per_seed": cc,
                            "acc128_conf": round(acc(arm, CONF), 3), "missing": sum(R[arm, s] is None for s in DEV + CONF),
                            "params": (R[arm, CONF[0]] or {}).get("params"), "entropy_collapse_step": col}
        print("%-9s dev %d/3  conf %2d/10 %s  acc128 %.3f  params %s  collapse %s" % (
            arm, sum(cd), sum(cc), wilson(sum(cc), 10), acc(arm, CONF), out["arms"][arm]["params"], col))
    key = lambda arm: (sum(cert(arm, DEV)), acc(arm, DEV), arm == "A_opb12")                       # noqa: E731
    cand = max(["A_mul", "A_opb12"], key=key)
    kA, kadd = sum(cert(cand, CONF)), sum(cert("A_add", CONF))
    kB, kc = sum(cert("B_temper", CONF)), sum(cert("B_mml2", CONF))
    pair = lambda x, y: [(int(i), int(j)) for i, j in zip(cert(x, CONF), cert(y, CONF))]           # noqa: E731
    out.update(A_candidate=cand, A_pass=kA >= 9 and kA - kadd >= 3, A_paired_vs_add=pair(cand, "A_add"),
               B_pass=kB >= 9 and kB - kc >= 3, B_paired_vs_mml2=pair("B_temper", "B_mml2"))
    out["close_line"] = not (out["A_pass"] or out["B_pass"])
    print("A candidate", cand, "%d/10 vs add %d/10 -> pass %s" % (kA, kadd, out["A_pass"]))
    print("B temper %d/10 vs mml2 %d/10 -> pass %s" % (kB, kc, out["B_pass"]), "| close line:", out["close_line"])
    json.dump(out, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
