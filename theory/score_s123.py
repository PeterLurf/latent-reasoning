"""THEORY §123 scoring (rules fixed in §123; written before any §123 run; Theory is the only scorer).
Runs: boxes/runs_fsq/s123_s5_dither{000,010,025,050}_s{1,2}/ and s123_ff_dither{000,<sigma*>}_s1/ (continuous, no
rounding; flip-flop's sigma* = the S5 sigma > 0 with the lowest mean rho) and
s123_{s5,ff}_fsqdither_s{1,2}/ (FSQ + pre-rounding dither). Keys: eval.json long_acc, rho (geometric mean over 8 hand-offs
of the median normalised distance ratio at eps 0.1), basin, grid_share (implicit quantisation); FSQ+DITHER also
eval_growth.json (codes per state by position, '2048') and eval_cert.json (cert {status, holds}).
CAUSAL: (i) S5 mean rho falls from sigma 0 to 0.5, and flip-flop rho at sigma* < its sigma-0 control; (ii) every continuous cell with rho < 0.3 has long_acc >= 0.99;
(iii) every continuous cell with rho > 0.6 has long_acc < 0.99. Continuous-stabiliser reading needs passing cells'
grid_share < 0.5. CANONICAL: codes per state at 2048 below §120 FINAL's per family (S5 34.2; FF 14.0, note 123-2) in both
seeds AND CERT done & holds in >= 1 seed (FF: the task-language CERT, 120-8; the registered CERT printed beside).
    python theory/score_s123.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
SIG = {"000": 0.0, "010": 0.1, "025": 0.25, "050": 0.5}


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    res = {"continuous": {}, "fsqdither": {}}
    cells = []
    for fam, seeds in (("s5", (1, 2)), ("ff", (1,))):
        for tag, sig in SIG.items():
            for sd in seeds:
                r = load("s123_%s_dither%s_s%d" % (fam, tag, sd))
                if r is None:
                    continue
                c = {"fam": fam, "sigma": sig, "seed": sd, "long_acc": r.get("long_acc"), "rho": r.get("rho"), "basin": r.get("basin"), "grid_share": r.get("grid_share")}
                res["continuous"]["%s_%s_s%d" % (fam, tag, sd)] = c; cells.append(c)
    mrho = lambda fam, sig: (lambda v: sum(v) / len(v) if v else None)([c["rho"] for c in cells if c["fam"] == fam and c["sigma"] == sig and c["rho"] is not None])  # noqa: E731
    s5_ok = mrho("s5", 0.5) is not None and mrho("s5", 0.0) is not None and mrho("s5", 0.5) < mrho("s5", 0.0)
    pos = [sg for sg in (0.1, 0.25, 0.5) if mrho("s5", sg) is not None]
    sigma_star = min(pos, key=lambda sg: mrho("s5", sg)) if pos else None; res["sigma_star"] = sigma_star
    ff_ok = sigma_star is not None and mrho("ff", sigma_star) is not None and mrho("ff", 0.0) is not None and mrho("ff", sigma_star) < mrho("ff", 0.0)
    i_ok = s5_ok and ff_ok
    ii = [c for c in cells if c["rho"] is not None and c["rho"] < 0.3]; iii = [c for c in cells if c["rho"] is not None and c["rho"] > 0.6]
    ii_ok = all((c["long_acc"] or 0) >= 0.99 for c in ii); iii_ok = all((c["long_acc"] or 0) < 0.99 for c in iii)
    res["CAUSAL"] = (len(cells) == 10) and i_ok and ii_ok and iii_ok
    res["CAUSAL_parts"] = {"i": i_ok, "ii": ii_ok, "iii": iii_ok, "n_cells": len(cells)}
    passing = [c for c in cells if (c["long_acc"] or 0) >= 0.99]
    res["continuous_stabiliser_reading"] = bool(passing) and all((c["grid_share"] or 1) < 0.5 for c in passing)
    canon = {}
    for fam in ("s5", "ff"):
        seeds = {}
        for s in (1, 2):
            d = "s123_%s_fsqdither_s%d" % (fam, s); r = load(d)
            if r is None:
                continue
            g = load(d, "eval_growth.json") or {}; c = load(d, "eval_cert.json") or {}
            reg = c.get("cert", c); task = c.get("cert_task")
            cert = task if (fam == "ff" and task is not None) else reg                 # note 120-8 / 123-2: FF uses the task-language CERT
            cps = ((g.get("codes_per_state") or {}).get("2048")) if isinstance(g.get("codes_per_state"), dict) else g.get("codes_per_state_2048")
            if cps is None:
                cps = g.get("codes_per_state_2048")
            seeds[s] = {"long_acc": r.get("long_acc"), "codes_per_state_2048": cps,
                        "cert_holds": cert.get("status") == "done" and bool(cert.get("holds")),
                        "cert_registered_holds": reg.get("status") == "done" and bool(reg.get("holds"))}
        base = {"s5": 34.2, "ff": 14.0}[fam]                                           # note 123-2: §120 FINAL's codes per state, per family
        canon[fam] = {"seeds": seeds, "codes_per_state_baseline": base,
                      "CANONICAL": len(seeds) == 2 and all((v["codes_per_state_2048"] or 1e9) < base for v in seeds.values()) and any(v["cert_holds"] for v in seeds.values())}
    res["fsqdither"] = canon
    print("continuous cells:", {k: (v["rho"], v["long_acc"], v["grid_share"]) for k, v in res["continuous"].items()})
    print("CAUSAL", res["CAUSAL"], res["CAUSAL_parts"], "| continuous-stabiliser reading", res["continuous_stabiliser_reading"])
    print("FSQ+DITHER", {f: (v["CANONICAL"], v["seeds"]) for f, v in canon.items()})
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s123.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
