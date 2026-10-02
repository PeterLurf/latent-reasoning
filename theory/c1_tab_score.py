"""THEORY §76(A) scoring (rules fixed in §76 and amendment 76-5): per (group, curriculum), certified seeds out of 10
(2051-2060) with accuracy >= 0.99 at N 8-128; a cell succeeds at >= 9/10. Descriptive: problems-to-certificate (first
logged update with a certificate x 256 problems), the A5 ge3 M sweep and tau0 = 1 arms, and the accuracy plateau of failed
seeds (nearest 1/k). One completed run per (group, curriculum, seed) in theory/results/c1_tab_local (76-5).
    python theory/c1_tab_score.py [--dir theory/results/c1_tab_local] [--out theory/results/score_s76a.json]
"""
import argparse
import json
import os

import numpy as np

SEEDS = list(range(2051, 2061))


def load(d, name):
    f = os.path.join(d, name + ".json")
    return next(iter(json.load(open(f)).values())) if os.path.exists(f) else None


def summarise(d, g, cur, tag="", seeds=SEEDS):
    rs = {s: load(d, "%s_%s%s_s%d" % (g, cur, tag, s)) for s in seeds}
    ok = {s: bool(r and r["certified"] and min(r["acc"][k] for k in ("8", "32", "128")) >= 0.99) for s, r in rs.items()}
    first = [next((int(u) for u, v in sorted(r.get("cert_trace", {}).items(), key=lambda x: int(x[0])) if v), None)
             for r in rs.values() if r and r["certified"]]
    plateaus = [round(1 / r["acc"]["128"]) if r["acc"]["128"] > 0 else None for r in rs.values() if r and not r["certified"]]
    return {"certified": sum(ok.values()), "n": sum(r is not None for r in rs.values()), "per_seed": {str(s): ok[s] for s in seeds},
            "first_cert_update_median": float(np.median([f for f in first if f])) if any(first) else None,
            "problems_to_cert_median": float(np.median([f for f in first if f]) * 256) if any(first) else None,
            "failed_plateau_1_over": plateaus}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="theory/results/c1_tab_local"); ap.add_argument("--out", default="theory/results/score_s76a.json")
    a = ap.parse_args(); res = {}
    for g in ("S4", "A5", "S5"):
        for cur in ("ge3", "std"):
            r = summarise(a.dir, g, cur); r["succeeds"] = r["certified"] >= 9 and r["n"] == 10; res["%s_%s" % (g, cur)] = r
            print("%s %-4s certified %d/%d %s  problems-to-cert median %s  failed plateaus 1/%s" % (
                g, cur, r["certified"], r["n"], "SUCCEEDS" if r["succeeds"] else "", r["problems_to_cert_median"], r["failed_plateau_1_over"]))
    for tag in ("_M60", "_M90", "_M240", "_tau1"):
        r = summarise(a.dir, "A5", "ge3", tag, SEEDS[:5]); res["A5_ge3" + tag] = r
        print("A5 ge3%-6s certified %d/%d (descriptive)" % (tag, r["certified"], r["n"]))
    json.dump(res, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
