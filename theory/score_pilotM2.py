"""Score THEORY §38 amendment M-2 (M2a-M2d): M-T8F on all 20 models vs M-T8, as registered.
Deep measure as §38 (theory/results/exp18_pilotM.json, single24, mean of d13-18); competence = eval_arch gate.
    python theory/score_pilotM2.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pilotFG import ev, pair_rule  # noqa: E402

H = [102, 109, 111, 112, 113, 115, 118, 119, 125, 126, 130, 131, 132, 135, 140, 145, 146, 147, 157, 158]


def main():
    X = json.load(open("theory/results/exp18_pilotM.json"))
    deep = lambda r: float(np.mean(X[r]["single24"][12:18]))
    comp = lambda r: ev(r) is not None and ev(r)["gate"]
    have = [s for s in H if f"pm_t8f_s{s}" in X and ev(f"pm_t8f_s{s}")]
    pairs = [(deep(f"pm_t8f_s{s}"), deep(f"pm_t8_s{s}")) for s in have if comp(f"pm_t8f_s{s}") and comp(f"pm_t8_s{s}")]
    d8 = [ev(f"pm_t8f_s{s}")["d8"] for s in have]
    out = {"n_t8f": len(have), "missing": [s for s in H if s not in have],
           "M2a": pair_rule(pairs, lambda x, y: x > y),
           "M2b": {"median_d8_t8f": float(np.median(d8)) if d8 else None, "met": bool(d8) and float(np.median(d8)) >= 0.608},
           "M2c": {"t8f_deep_ge_0.60": sum(deep(f"pm_t8f_s{s}") >= 0.60 for s in have), "of": len(have)},
           "M2d": {"competent_t8f": sum(comp(f"pm_t8f_s{s}") for s in have), "competent_t8": sum(comp(f"pm_t8_s{s}") for s in H)}}
    out["M2c"]["met"] = out["M2c"]["t8f_deep_ge_0.60"] >= 10
    out["M2d"]["met"] = out["M2d"]["competent_t8f"] >= out["M2d"]["competent_t8"]
    out["hardware_note"] = "odd-indexed T8F on local RTX 2070 fp32; even-indexed on Vast bf16 (Pilot M)"
    json.dump(out, open("theory/results/score_pilotM2.json", "w", newline="\n"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
