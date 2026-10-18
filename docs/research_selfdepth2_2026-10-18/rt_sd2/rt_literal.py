"""RT-SD2 check 3 ($0, arithmetic): a literal C2 row for round 2 and an H_prop-vs-H_add discriminator, on the draft's own panel construction
(sd2_numbers.py's K1x / K1xf / panel replicated here, NOT imported: importing it would rewrite the draft's sd2_numbers.json).
  LIT-C2 = mean_{n 22..24}[a(n) - a_20(n)]: accuracy gained by passes 21..n+2 > trained K 20 on items that need > K + 1 reductions (the coda finishes
  at most one reduction: FT-SELF a_{n-1}(n) ~ a(n) at n 15-16, rt_curves). K'-BEYOND-2's decomposition: gain at passes <= 20 vs > 20 (n 19, 20).
  WIDTH = n50 - n90 (H_prop stretches it by 16/12; H_shift / H_add keep it)."""
import json
import math
import os

R = "D:/Latent/boxes/runs_fsq/"; HERE = os.path.dirname(os.path.abspath(__file__)); FLOOR = 0.11
e1 = json.load(open(R + "s215_ftself_s1/eval_vb.json"))
a1 = {int(n): v for n, v in e1["acc_n2"].items()}; K1 = {int(k): {int(n): v for n, v in d.items()} for k, d in e1["acc_k"].items()}
ex = {n: a1[n] - FLOOR for n in (14, 15, 16)}; r1, r2 = ex[15] / ex[14], ex[16] / ex[15]; r3 = r2 * (r2 / r1)
A1x = dict(a1); A1x[17] = FLOOR + ex[16] * r3; A1x[18] = FLOOR + (A1x[17] - FLOOR) * r3 * (r2 / r1)
for n in range(19, 40):
    A1x[n] = FLOOR


def K1x(k, n):
    if n > 16:
        return A1x[n] if k >= n - 1 else FLOOR
    if k <= 16:
        return K1[k][n]
    return K1[16][n] + (K1[24][n] - K1[16][n]) * min(1.0, (k - 16) / 8)


def K1xf(k, n):
    if k < 0:
        return 0.0
    k0, n0 = int(math.floor(k)), int(math.floor(n)); fk, fn = k - k0, n - n0
    g = lambda kk, nn: K1x(kk, max(1, nn))  # noqa: E731
    return (1 - fk) * (1 - fn) * g(k0, n0) + fk * (1 - fn) * g(k0 + 1, n0) + (1 - fk) * fn * g(k0, n0 + 1) + fk * fn * g(k0 + 1, n0 + 1)


SH = 4 + 1.1532                                                           # H_add: translate round 1 by 5.15 in n and K' (same n50 as H_prop)


def A2(kind, k, n):
    if kind == "H_shift":
        return K1x(k - 4, n - 4) if k >= 4 else FLOOR
    if kind == "H_prop":
        return K1xf(12 + 0.75 * (k - 16), 12 + 0.75 * (n - 16))
    if kind == "H_add":
        return K1xf(k - SH, n - SH) if k >= SH else FLOOR


def crossing(a, level, lo, hi):
    for n in range(lo, hi):
        if a[n] >= level > a[n + 1]:
            return n + (a[n] - level) / (a[n] - a[n + 1])
    return None


out = {}
for kind in ("H_shift", "H_prop", "H_add"):
    a = {n: A2(kind, n + 2, n) for n in range(13, 25)}
    lit = {n: a[n] - A2(kind, 20, n) for n in (21, 22, 23, 24)}
    kb_le = sum(A2(kind, 20, n) - A2(kind, 18, n) for n in (19, 20)) / 2; kb_gt = sum(a[n] - A2(kind, 20, n) for n in (19, 20)) / 2
    full = {**{n: 1.0 for n in range(1, 13)}, **a}; c50, c90 = crossing(full, 0.5, 13, 24), crossing(full, 0.9, 13, 24)
    out[kind] = {"a": {n: round(v, 4) for n, v in a.items()}, "M2": sum(a[n] for n in range(17, 21)) / 4, "LIT_C2_by_n": lit, "LIT_C2_n22_24": sum(lit[n] for n in (22, 23, 24)) / 3,
                 "KB2": kb_le + kb_gt, "KB2_gain_passes_le_20": kb_le, "KB2_gain_passes_gt_20": kb_gt, "n50": c50, "n90": c90, "WIDTH_n50_minus_n90": None if (c50 is None or c90 is None) else c50 - c90}
out["H_full_floor_LIT_C2_n22_24_approx"] = sum(v - FLOOR for v in (0.85, 0.82, 0.80)) / 3
out["round1_WIDTH"] = {"base": 10.3056 - 9.0719, "FT-SELF": 15.4602 - 13.9340}
json.dump(out, open(os.path.join(HERE, "rt_literal.json"), "w", newline="\n"), indent=1, default=float)
for k, v in out.items():
    print(k, v if not isinstance(v, dict) else {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "a"})
