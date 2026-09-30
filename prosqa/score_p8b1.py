"""Score PROTOCOL8b step 1 (docs/PROTOCOL8B_STEP1.md). Written before any step-1 run started.
Inputs: prosqa/res_p8b1/p8b1_<arm>_s<seed>.json (reanchor_p8.py --random_pos output), arms ckpt / ctl, seeds 1-3.
Per seed, over held-out test problems with answer depth 7..10 at K' = 10:
  OG   = acc(oracle_rp) - acc(base)      oracle checkpoint at a random statement position (primary; training distribution)
  OGe  = acc(oracle)    - acc(base)      end-appended (as in the pilot; secondary)
  RG   = acc(random_rp) - acc(base)      random-concept checkpoints (false statements), random position
  REG  = acc(reanchor_rp) - acc(base)    the model's own decoded anchors
  BASE = acc(base)                       plain deep accuracy
P8b1-1 (primary): supported if mean OG(ckpt) >= 0.10, mean OG(ckpt) - mean OG(ctl) >= 0.05, and every ckpt seed's OG
  exceeds every ctl seed's OG (complete separation: exact one-sided MW p = 1/20 = 0.05); not supported if
  mean OG(ckpt) - mean OG(ctl) <= 0; otherwise inconclusive.
P8b1-2 (control, two-sided): per arm, mean RG in [-0.02, 0.02] = no gain from false checkpoints; RG > 0.02 = spurious
  gain; RG < -0.02 = harm (blind trust in written statements, motivating step 2's verification).
Secondary: OGe, REG and BASE by arm; the same by answer depth.
    python prosqa/score_p8b1.py
"""
import json
import os

import numpy as np

R = "prosqa/res_p8b1"


def acc(rows, key, lo=7, hi=10):
    sel = [r for r in rows if lo <= r["depth"] <= hi and key in r]
    return float(np.mean([r[key] > 0 for r in sel])) if sel else None


def main():
    per = {}
    for arm in ("ckpt", "ctl"):
        for s in (1, 2, 3):
            f = f"{R}/p8b1_{arm}_s{s}.json"
            if not os.path.exists(f):
                continue
            rows = json.load(open(f))["rows"]
            b = acc(rows, "base")
            per[f"{arm}_s{s}"] = {"BASE": b, "OG": acc(rows, "oracle_rp") - b, "OGe": acc(rows, "oracle") - b,
                                  "RG": acc(rows, "random_rp") - b, "REG": acc(rows, "reanchor_rp") - b,
                                  "by_depth": {d: {k: acc(rows, k, d, d) for k in ("base", "oracle_rp", "random_rp", "reanchor_rp")}
                                               for d in range(3, 15)}}
    get = lambda arm, k: [v[k] for n, v in per.items() if n.startswith(arm + "_")]
    out = {"per_seed": per}
    og_c, og_t = get("ckpt", "OG"), get("ctl", "OG")
    if len(og_c) >= 2 and len(og_t) >= 2:
        d = float(np.mean(og_c) - np.mean(og_t))
        sep = min(og_c) > max(og_t)
        dec = ("supported" if (np.mean(og_c) >= 0.10 and d >= 0.05 and sep and len(og_c) == 3 and len(og_t) == 3)
               else "not supported" if d <= 0 else "inconclusive")
        out["P8b1-1"] = {"OG_ckpt": og_c, "OG_ctl": og_t, "mean_diff": d, "complete_separation": sep, "decision": dec}
    for arm in ("ckpt", "ctl"):
        rg = get(arm, "RG")
        if rg:
            m = float(np.mean(rg))
            out[f"P8b1-2_{arm}"] = {"RG": rg, "mean": m,
                                    "reading": "no gain" if -0.02 <= m <= 0.02 else "spurious gain" if m > 0.02 else "harm (blind trust)"}
        out[f"secondary_{arm}"] = {k: get(arm, k) for k in ("BASE", "OGe", "REG")}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/p8b1.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=1))


if __name__ == "__main__":
    main()
