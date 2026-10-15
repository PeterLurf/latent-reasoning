"""QUESTION-ONCE draft: kept fractions of the PROGRESS-TRANSPLANT hybrids on TRAIN-REDRAW template pairs (pairs_calc's pool and rules).
Transplant after pass t (t = 2, 3): the recipient's pass-1 content plus the donor's passes 2..t. Two schedule conventions:
  C (pass 1 copies only; pass s >= 2 computes step s-1): donor steps {1..t-1} fixed;
  S (pass 1 also computes step 1; pass s computes step s): donor steps {2..t} fixed, the recipient's v_1 kept.
Kept = y_hyb not in {y_own, y_don} (y_own != y_don); strict = kept and renderable (>= 0, <= 2 decimals). Writes theory/ansbottle/qonce_pairs.json (LF)."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairs_calc as P  # noqa: E402  (sets Idle priority and one thread on import)

import numpy as np  # noqa: E402

G = P.G
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qonce_pairs.json")
SETS = {"C_t2": [0], "C_t3": [0, 1], "S_t2": [1], "S_t3": [1, 2]}         # 0-indexed donor steps fixed in the hybrid


def main():
    tr, _ = G.load_aug("train"); rng = np.random.default_rng(20711); idx = rng.permutation(len(tr))[:3000]
    res = {k: {"n": 0, "kept": 0, "strict": 0, "eq_own": 0, "eq_don": 0, "by_m": {}} for k in SETS}; n_pairs = 0
    for i in idx:
        r = tr[i]; t, _ = G.parse_recomputable(r)
        if t is None:
            continue
        base = P.redraw(r, t, rng)
        if base is None:
            continue
        qv, vr = base
        for _ in range(P.D_DRAWS - 1):
            d = P.redraw(r, t, rng)
            if d is None or P.tol_eq(d[1][-1], vr[-1]):
                continue
            n_pairs += 1; vd = d[1]; m = len(t["eqt"])
            for k, steps in SETS.items():
                fixed = {s: vd[s] for s in steps if s < m}
                ch = P.chain(t, qv, fixed); y = ch[-1] if ch else None
                kept = y is not None and math.isfinite(y) and not P.tol_eq(y, vr[-1]) and not P.tol_eq(y, vd[-1])
                st = kept and P.render_ans(y) is not None; rr = res[k]; rr["n"] += 1; rr["kept"] += kept; rr["strict"] += st
                rr["eq_own"] += P.tol_eq(y, vr[-1]); rr["eq_don"] += P.tol_eq(y, vd[-1])
                b = rr["by_m"].setdefault("m5+" if m >= 5 else "m%d" % m, [0, 0]); b[0] += 1; b[1] += st
    out = {"n_pairs": n_pairs}
    for k, rr in res.items():
        out[k] = {"kept": rr["kept"] / rr["n"], "strict": rr["strict"] / rr["n"], "eq_own": rr["eq_own"] / rr["n"], "eq_don": rr["eq_don"] / rr["n"],
                  "strict_by_m": {mk: v[1] / v[0] for mk, v in sorted(rr["by_m"].items())}}
    with open(OUT, "w", newline="\n") as f_:
        json.dump(out, f_, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
