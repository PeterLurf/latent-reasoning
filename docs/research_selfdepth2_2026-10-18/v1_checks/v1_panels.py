"""SD2_v1 check 2 ($0, CPU, stdlib only; read-only on D:/Latent). Inputs: boxes/runs_fsq/{s213_d3t_s1, s215_ftself_s1}/eval_vb.json (acc_n2, acc_k).
(1) The per-lineage bar formulas (D1 / RT B1), applied to s1: r1_bars(base) -> round-1 bars (must give v1's 0.63 / 0.40), r2_bars(base, R1) -> round-2
    bars (must give 0.785 / 0.60 / FULL 0.80 / M_far 0.60). Only measured n <= 16 values enter a bar. These functions are the reference implementation
    score_s216 must reproduce (G-S12 fixture).
(2) Round-2 panels per n (n 13..24): H_shift, H_prop, constant growth (H_add, round 1 translated by 4 + delta, delta = m50_1 - m50_0), SHRINK, H_full floor;
    readings M2, M_far, n50, n90, WIDTH-2 = n50 - n90 (D9), on per-n tables (the definitions' own form; PROOFS: 'the per-n tables are the right ones').
(3) K'-dependent readings on 2-D (K', n) tables (PROOFS p1_panels' interpolation through K' 16, K' n + 2, K' 24, replicated here, not imported: importing
    p1_panels.py rewrites its JSON): STOP-LOC-2 G(k), KB2 (+ decomposition into passes <= 20 / > 20), RISE-PAST-K-2, LIT-C2-2 = mean_{n 22..24}
    [a(n) - a_20(n)], on both interpolation axes ((K', n) and (lag, n), PROOFS p1b) for the fractional maps.
(4) Lineage-noise OC (RT M3): Gaussian sigma on M2 -> P(M2 >= 0.80), P(M2 >= 0.785), P(M2 < 0.60) per hypothesis; mislabel rates.
Round-1 values beyond n 16 are unread: the draft's assumed extension (FLOOR 0.11, excess decaying at the last ratios' trend) is reproduced and flagged."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__)); R = "D:/Latent/boxes/runs_fsq/"; FLOOR = 0.11; EPS = 1e-9


def load(c):
    e = json.load(open(R + c + "/eval_vb.json"))
    return {int(n): v for n, v in e["acc_n2"].items()}, {int(k): {int(n): v for n, v in d.items()} for k, d in e["acc_k"].items()}


def ceil005(x):
    return round(math.ceil(x / 0.005 - 1e-6) * 0.005, 3)


def ip(a, x):
    lo = math.floor(x); f = x - lo
    return a[lo] if f < 1e-12 else (1 - f) * a[lo] + f * a[lo + 1]


def mean(v):
    v = list(v); return sum(v) / len(v)


# ---- (1) the registered per-lineage formulas (measured n <= 16 only) ----
def r1_bars(a0):
    """Round-1 (B 8 -> 12) bars from the lineage's own base eval_vb: H_shift M = mean a0(9..12) (exact translation by +4)."""
    M0 = mean(a0[n] for n in range(9, 13)); MG = ceil005(M0 + 0.15)
    return {"M_shift": M0, "MARGIN_GROWS": MG, "SHRINK_below": ceil005(M0 - 0.08), "DBS": 0.80, "FULL_M": max(0.80, MG), "K_BEYOND": 0.25}


def r2_bars(a0, a1):
    """Round-2 (B 12 -> 16) bars from the lineage's own base and round-1 eval_vb (acc_n2):
    M2_shift = mean a1(13..16); M2_prop = mean_{d 1..4} a1(12 + 0.75 d) (linear in n); M2_shrink = mean a0(9..12);
    MG2 = ceil005(midpoint shift / prop); SH2 = ceil005(midpoint shrink / shift); FULL needs M2 >= max(0.80, MG2) and M_far >= 0.60 (absolute);
    separability flag: H_prop's M_far upper bound (a1(x > 16) := a1(16), monotone [ASSUMED]) >= 0.60 -> FULL 'not separable from H_prop'."""
    Ms = mean(a1[n] for n in range(13, 17)); Mp = mean(ip(a1, 12 + 0.75 * d) for d in range(1, 5)); Mk = mean(a0[n] for n in range(9, 13))
    MG = ceil005((Ms + Mp) / 2); SH = ceil005((Ms + Mk) / 2)
    mono = lambda x: a1[16] if x >= 16 else ip(a1, x)  # noqa: E731
    up = mean(mono(12 + 0.75 * d) for d in range(5, 9))
    return {"M2_shift": Ms, "M2_prop": Mp, "M2_shrink": Mk, "MARGIN_GROWS_2": MG, "SHRINK_below": SH, "FULL_M2": max(0.80, MG), "FULL_M_far": 0.60,
            "DBS_2": 0.80, "LIT_C2_2": 0.10, "Hprop_M_far_upper": up, "FULL_not_separable_flag": up >= 0.60 - EPS, "degenerate_SH_ge_MG": SH >= MG}


a0, K0 = load("s213_d3t_s1"); a1, K1 = load("s215_ftself_s1")
assert all(abs(a1[n] - K1[n + 2][n]) < 1e-12 for n in range(1, 15)) and all(abs(a0[n] - K0[n + 2][n]) < 1e-12 for n in range(1, 15))
out = {"r1_bars_s1_instance": r1_bars(a0), "r2_bars_s1_instance": r2_bars(a0, a1)}
assert out["r1_bars_s1_instance"]["MARGIN_GROWS"] == 0.63 and out["r1_bars_s1_instance"]["SHRINK_below"] == 0.40
assert out["r2_bars_s1_instance"]["MARGIN_GROWS_2"] == 0.785 and out["r2_bars_s1_instance"]["SHRINK_below"] == 0.60


def cross(a, L, lo, hi):
    for n in range(lo, hi):
        if n + 1 in a and a[n] >= L - EPS and L > a[n + 1]:
            return n + (a[n] - L) / (a[n] - a[n + 1])
    return None


# ---- the draft's assumed extension of a1 beyond 16 (flagged) ----
ex = {n: a1[n] - FLOOR for n in (14, 15, 16)}; q1, q2 = ex[15] / ex[14], ex[16] / ex[15]; q3 = q2 * q2 / q1
A1 = dict(a1); A1[17] = FLOOR + ex[16] * q3; A1[18] = FLOOR + (A1[17] - FLOOR) * q3 * (q2 / q1)
for n in range(19, 41):
    A1[n] = FLOOR
A0 = dict(a0)
for n in range(17, 41):
    A0[n] = FLOOR
for n in range(-12, 1):
    A0[n] = 1.0; A1[n] = 1.0
delta = cross(a1, 0.5, 12, 16) - 12 - (cross(a0, 0.5, 8, 16) - 8)


def tab(a, K, nmax_read):                                                   # PROOFS p1_panels' 2-D table (replicated)
    def f(k, n):
        if n <= 0:
            return 1.0
        if k < 0:
            return FLOOR
        if n > nmax_read:
            return a[n] if k >= n - 1 else FLOOR
        if k in K:
            return K[k][n]
        pts = sorted({p[0]: p for p in sorted([(kk, K[kk][n]) for kk in K] + [(n + 2, a[n])])}.values())
        if k > pts[-1][0]:
            return pts[-1][1]
        for (ka, va), (kb, vb) in zip(pts, pts[1:]):
            if ka <= k <= kb:
                return va + (vb - va) * (k - ka) / (kb - ka)
    return f


T1, T0 = tab(A1, K1, 16), tab(A0, K0, 16)


def bil(f, k, n):
    k0, n0 = math.floor(k), math.floor(n); fk, fn = k - k0, n - n0
    return (1 - fk) * (1 - fn) * f(k0, n0) + fk * (1 - fn) * f(k0 + 1, n0) + (1 - fk) * fn * f(k0, n0 + 1) + fk * fn * f(k0 + 1, n0 + 1)


def lagf(j, n):                                                             # bilinear in (lag j = K' - n, n) on round 1's table (PROOFS p1b)
    n0 = math.floor(n); j0 = math.floor(j); fn, fj = n - n0, j - j0
    g = lambda jj, nn: T1(nn + jj, nn)  # noqa: E731
    return (1 - fn) * ((1 - fj) * g(j0, n0) + fj * g(j0 + 1, n0)) + fn * ((1 - fj) * g(j0, n0 + 1) + fj * g(j0 + 1, n0 + 1))


NS = range(13, 25)
PAN = {"H_shift": (lambda n: A1[n - 4], {"kn": lambda k, n: T1(k - 4, n - 4)}),
       "H_prop": (lambda n: ip(A1, 12 + 0.75 * (n - 16)), {"kn": lambda k, n: bil(T1, 12 + 0.75 * (k - 16), 12 + 0.75 * (n - 16)),
                                                             "lag": lambda k, n: lagf(0.75 * (k - n), 12 + 0.75 * (n - 16))}),
       "H_add_constant_growth": (lambda n: ip(A1, n - 4 - delta), {"kn": lambda k, n: bil(T1, k - 4 - delta, n - 4 - delta), "lag": lambda k, n: lagf(k - n, n - 4 - delta)}),
       "SHRINK": (lambda n: A0[n - 8], {"kn": lambda k, n: T0(k - 8, n - 8)})}


def readings(a2, A2s):
    full = {**{n: 1.0 for n in range(1, 13)}, **a2}; c50, c90 = cross(full, 0.5, 12, 24), cross(full, 0.9, 12, 24)
    r = {"a": {n: round(a2[n], 4) for n in NS}, "M2": mean(a2[n] for n in range(17, 21)), "M_far": mean(a2[n] for n in range(21, 25)),
         "n50": c50, "n90": c90, "WIDTH_2": None if (c50 is None or c90 is None) else c50 - c90}
    se = {n: math.sqrt(max(a2[n], 0) * (1 - a2[n]) / 1000) for n in NS}; r["se_M2"] = math.sqrt(sum(se[n] ** 2 for n in range(17, 21))) / 4
    for ax, A2 in A2s.items():
        G = {k: mean(a2[n] - A2(k, n) for n in range(k + 1, min(k + 4, 24) + 1)) for k in range(13, 24)}
        stop = next((k for k in range(13, 24) if G[k] <= 0.02 + EPS), "24+")
        kb_le = mean(A2(20, n) - A2(18, n) for n in (19, 20)); kb_gt = mean(a2[n] - A2(20, n) for n in (19, 20))
        lit = {n: a2[n] - A2(20, n) for n in (21, 22, 23, 24)}
        r[ax] = {"G": {k: round(v, 4) for k, v in G.items()}, "STOP": stop, "KB2": mean(a2[n] - A2(18, n) for n in (19, 20)), "KB2_gain_passes_le_20": kb_le,
                 "KB2_gain_passes_gt_20": kb_gt, "RISE_PAST_K_2": kb_gt, "LIT_C2_2": mean(lit[n] for n in (22, 23, 24)), "LIT_by_n": {n: round(v, 4) for n, v in lit.items()}}
    return r


pan = {h: readings({n: f1(n) for n in NS}, f2) for h, (f1, f2) in PAN.items()}
floor = {17: 0.97, 18: 0.95, 19: 0.93, 20: 0.90, 21: 0.88, 22: 0.85, 23: 0.82, 24: 0.80}
pan["H_full_floor"] = {"a": floor, "M2": mean(floor[n] for n in range(17, 21)), "M_far": mean(floor[n] for n in range(21, 25)), "n50": "> 24", "STOP": "24+",
                       "LIT_C2_2": mean(floor[n] for n in (22, 23, 24)) - FLOOR, "note": "a_20(n) at n >= 22 ~ chance (the coda finishes at most one reduction) [ASSUMED]"}
out["delta_m50"] = delta; out["a1_extension_assumed"] = {n: round(A1[n], 4) for n in range(16, 21)}; out["panels"] = pan
out["round1_width"] = {"base": cross(a0, 0.5, 8, 16) - cross(a0, 0.9, 8, 16), "FT_SELF": cross(a1, 0.5, 12, 16) - cross(a1, 0.9, 12, 16)}
out["Hshift_G19_measured_a16"] = pan["H_shift"]["kn"]["G"][19]


# ---- (4) lineage noise on M2 ----
def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


Ms = {h: pan[h]["M2"] for h in ("H_shift", "H_prop", "H_add_constant_growth", "SHRINK")}
noc = {}
for s in (0.03, 0.06, 0.09, 0.12):
    d = {h: {"P_ge_0.80": 1 - Phi((0.80 - m) / s), "P_ge_0.785": 1 - Phi((0.785 - m) / s), "P_lt_0.60": Phi((0.60 - m) / s)} for h, m in Ms.items()}
    d["LR_DBS_Hprop_vs_Hshift"] = d["H_prop"]["P_ge_0.80"] / d["H_shift"]["P_ge_0.80"]
    d["MG_mislabel_Hshift"] = d["H_shift"]["P_ge_0.785"]; d["MG_mislabel_Hprop"] = 1 - d["H_prop"]["P_ge_0.785"]
    noc[str(s)] = d
out["lineage_noise_M2"] = noc
json.dump(out, open(os.path.join(HERE, "v1_panels.json"), "w", newline="\n"), indent=1, default=float)
print("r1 bars", out["r1_bars_s1_instance"]); print("r2 bars", out["r2_bars_s1_instance"]); print("delta", delta, "ext", out["a1_extension_assumed"])
for h, v in pan.items():
    if h == "H_full_floor":
        print(h, {k: v[k] for k in ("M2", "M_far", "LIT_C2_2")}); continue
    print(h, "M2 %.4f Mfar %.4f n50 %s n90 %s W %s seM2 %.4f" % (v["M2"], v["M_far"], v["n50"], v["n90"], v["WIDTH_2"], v["se_M2"]))
    for ax in ("kn", "lag"):
        if ax in v:
            print("   ", ax, {k: v[ax][k] for k in ("STOP", "KB2", "KB2_gain_passes_le_20", "KB2_gain_passes_gt_20", "LIT_C2_2")}, "G19", v[ax]["G"].get(19))
print("width r1", out["round1_width"])
for s, d in noc.items():
    print("sigma", s, {h: {k: round(x, 3) for k, x in v.items()} if isinstance(v, dict) else round(v, 3) for h, v in d.items()})
