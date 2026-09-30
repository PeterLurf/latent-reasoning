"""Score PROTOCOL9 (anytime supervision across architectures, paired with PROTOCOL7 T9/T10 controls). Written
2026-09-29 before any PROTOCOL9 run started and before any of the paired T9/T10 controls was scored.

Pairs: p9at_l3_s<s> vs t9l3_s<s> (s 500-507), p9at_l4 vs t9l4 (520-527), p9at_w512 vs t9w512 (540-547),
p9at_fork vs t10fork (560-567). Competence and extrapolation: PROTOCOL5 rules (latent.protocol5_confirm.pilot_run);
a run without a model (stage-1 failure after its relaunch budget) is not competent. d8 = length_gen accuracy at
distance 8, K' = 12 (trained distance + 2).
AP1 (primary, pooled): evaluable = both competent and d8 not exactly tied; needs >= 10. Supported if anytime > control
  in >= 70 % of evaluable pairs and one-sided sign p < 0.05; not supported if <= 50 %; else inconclusive.
AP2: extrapolation success (competent and extrap_K12): l = anytime-only, b = control-only successes; supported if
  l - b >= 4 and one-sided sign p <= 0.05; not supported if l <= b; else inconclusive.
AP3 (learnability cost): pooled competent anytime vs control; "cost" if anytime has >= 6 fewer of 32; per family
  descriptive.
Amendment 1 (ceiling rule, before any run): per family, if the competent controls' median d8 is >= 0.85, that family's
beyond-depth measure is the mean of far 10-12 at K' = 16 instead of d8; if that median is also >= 0.85, the family
leaves AP1 (reported as "controls never fail beyond depth"). Decided from the controls alone.
Descriptive: AP1 share per family; d7/d8/d9 means by arm; P7 by arm is added by a separate frozen far_sweep pass.
    PYTHONPATH=. python scratch/score_p9.py [--final]
"""
import argparse
import json
import os
from math import comb

import numpy as np

from latent import protocol5_confirm as P5

FAM = {"l3": ("t9l3", 500), "l4": ("t9l4", 520), "w512": ("t9w512", 540), "fork": ("t10fork", 560)}


def far(r, d, Kp=12):
    f = f"results/length_gen_{r}.json"
    if not os.path.exists(f):
        return None
    rows = json.load(open(f))[f"runs/{r}"]["rows"]
    row = next((x for x in rows if x["K'"] == Kp), None)
    return None if row is None else row["far"].get(str(d))


def status(r):
    done = os.path.exists(f"results/competence_matched_{r}.json") and os.path.exists(f"results/length_gen_{r}.json")
    if done:
        p = P5.pilot_run(r)
        f16 = [far(r, d, 16) for d in (10, 11, 12)]
        return {"done": True, "competent": bool(p.get("gate")), "extrap": bool(p.get("gate") and p.get("extrap_K12")),
                "d7": far(r, 7), "d8": far(r, 8), "d9": far(r, 9),
                "far10_12_K16": None if None in f16 else float(np.mean(f16))}
    no_model = os.path.isdir(f"runs/{r}") and not os.path.exists(f"runs/{r}/model.pt") and \
        any(f"evaluated {r}" in open(q, errors="ignore").read() for q in
            [os.path.join(dp, fn) for dp, _, fs in os.walk("logs") for fn in fs if fn.startswith("queue_")])
    return {"done": no_model, "competent": False, "extrap": False, "d7": None, "d8": None, "d9": None, "far10_12_K16": None}


def sign_p(k, n):
    return None if n == 0 else sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--ext", action="store_true", help="amendment 6: pool the extension seeds (8-15 per family) with the primary 32")
    a = ap.parse_args()
    rows, pending = [], []
    for fam, (ctl, base) in FAM.items():
        for s in range(base, base + (16 if a.ext else 8)):
            at, co = f"p9at_{fam}_s{s}", f"{ctl}_s{s}"
            sa, sc = status(at), status(co)
            if not (sa["done"] and sc["done"]):
                pending.append((at, co)); continue
            rows.append({"family": fam, "seed": s, "anytime": at, "control": co, "a": sa, "c": sc})
    measure = {}
    for fam in FAM:
        cd8 = [r["c"]["d8"] for r in rows if r["family"] == fam and r["c"]["competent"] and r["c"]["d8"] is not None]
        cfar = [r["c"]["far10_12_K16"] for r in rows if r["family"] == fam and r["c"]["competent"]
                and r["c"]["far10_12_K16"] is not None]
        measure[fam] = ("d8" if not cd8 or np.median(cd8) < 0.85 else
                        "far10_12_K16" if cfar and np.median(cfar) < 0.85 else None)
    for r in rows:
        m = measure[r["family"]]
        r["measure"] = m
        r["a"]["m"] = r["a"][m] if m else None
        r["c"]["m"] = r["c"][m] if m else None
    ev = [r for r in rows if r["measure"] and r["a"]["competent"] and r["c"]["competent"]
          and r["a"]["m"] is not None and r["c"]["m"] is not None and r["a"]["m"] != r["c"]["m"]]
    k = sum(r["a"]["m"] > r["c"]["m"] for r in ev)
    share = k / len(ev) if ev else None
    p = sign_p(k, len(ev))
    ap1 = ("unresolved" if len(ev) < 10 else "supported" if share >= 0.70 and p < 0.05 else
           "not supported" if share <= 0.50 else "inconclusive")
    l = sum(r["a"]["extrap"] and not r["c"]["extrap"] for r in rows)
    b = sum(r["c"]["extrap"] and not r["a"]["extrap"] for r in rows)
    p2 = sign_p(l, l + b)
    ap2 = "supported" if l - b >= 4 and p2 is not None and p2 <= 0.05 else "not supported" if l <= b else "inconclusive"
    ca, cc = sum(r["a"]["competent"] for r in rows), sum(r["c"]["competent"] for r in rows)
    fam_out = {}
    for fam in FAM:
        fr = [r for r in rows if r["family"] == fam]
        fe = [r for r in fr if r in ev]
        fam_out[fam] = {"pairs": len(fr), "evaluable": len(fe),
                        "measure": measure[fam], "anytime_better": sum(r["a"]["m"] > r["c"]["m"] for r in fe),
                        "competent_anytime": sum(r["a"]["competent"] for r in fr),
                        "competent_control": sum(r["c"]["competent"] for r in fr),
                        "extrap_anytime": sum(r["a"]["extrap"] for r in fr), "extrap_control": sum(r["c"]["extrap"] for r in fr)}
    mean = lambda arm, key: float(np.mean([r[arm][key] for r in rows if r[arm]["competent"] and r[arm][key] is not None]
                                          or [np.nan]))
    out = {"final": a.final, "pairs_scored": len(rows), "pending": pending,
           "AP1": {"evaluable": len(ev), "anytime_better": k, "share": share, "sign_p": p,
                   "decision": ap1 if (a.final or not pending) else f"incomplete ({len(pending)} pairs pending)"},
           "AP2": {"l": l, "b": b, "sign_p": p2, "decision": ap2 if (a.final or not pending) else "incomplete"},
           "AP3": {"competent_anytime": ca, "competent_control": cc, "cost": (cc - ca) >= 6},
           "families": fam_out,
           "means_competent": {arm: {d: mean(arm, d) for d in ("d7", "d8", "d9")} for arm in ("a", "c")},
           "rows": rows}
    json.dump(out, open("results/p9_pooled64.json" if a.ext else "results/p9.json", "w"), indent=1)
    print({k2: v for k2, v in out.items() if k2 != "rows"})


if __name__ == "__main__":
    main()
