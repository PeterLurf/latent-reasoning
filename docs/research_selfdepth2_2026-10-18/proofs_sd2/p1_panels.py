"""PROOFS_SD2 check 1 ($0, CPU; independent rewrite). Inputs: boxes/runs_fsq/{s213_d3t_s1, s215_ftself_s1}/eval_vb.json (acc_n2, acc_k).
(1) crossings at levels 0.3 / 0.5 / 0.7 / 0.9 for D3-T (B 8) and FT-SELF (B 12): n_L, margin m_L = n_L - B, m_L / B; transition widths.
(2) round-2 panels H_shift / H_prop / SHRINK and a rigid-translation 'H_add' (round 1 translated by 4 + delta, delta = m50_1 - m50_0 = the
    'constant +1.15 growth'), each (a) per-n from acc_n2 directly and (b) on a 2-D (K', n) table; readings M2, M_far, n50, STOP-LOC-2, KB2; SEs.
(3) retro-test: each hypothesis applied to D3-T -> FT-SELF at n 13-16 (H_prop-retro: a1(12+d) = a0(8 + 2d/3); H_add-retro: a1(n) = a0(n - 4 - delta)).
(4) the H_prop fit at other crossing levels (does margin/B constancy hold away from 50 %?).
Round-1 values beyond n 16 are unread: the draft's assumed extension (A1x) is reproduced and flagged; FLOOR 0.11 [ASSUMED, as the draft]."""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = "D:/Latent/boxes/runs_fsq/"
FLOOR = 0.11


def load(c):
    e = json.load(open(R + c + "/eval_vb.json"))
    return {int(n): v for n, v in e["acc_n2"].items()}, {int(k): {int(n): v for n, v in d.items()} for k, d in e["acc_k"].items()}


a0, K0 = load("s213_d3t_s1"); a1, K1 = load("s215_ftself_s1")
assert all(abs(a1[n] - K1[n + 2][n]) < 1e-12 for n in range(1, 15)) and all(abs(a0[n] - K0[n + 2][n]) < 1e-12 for n in range(1, 15))


def cross(a, L, lo, hi):
    for n in range(lo, hi):
        if n + 1 in a and a[n] >= L > a[n + 1]:
            return n + (a[n] - L) / (a[n] - a[n + 1])
    return None


# draft's assumed extension of a1 beyond 16 (excess over FLOOR decays with the trend of the last ratios)
ex = {n: a1[n] - FLOOR for n in (14, 15, 16)}; r1, r2 = ex[15] / ex[14], ex[16] / ex[15]; r3 = r2 * r2 / r1
A1 = dict(a1); A1[17] = FLOOR + ex[16] * r3; A1[18] = FLOOR + (A1[17] - FLOOR) * r3 * (r2 / r1)
for n in range(19, 40):
    A1[n] = FLOOR
A0 = dict(a0)
for n in range(17, 40):
    A0[n] = FLOOR
for n in range(-10, 1):
    A0[n] = 1.0; A1[n] = 1.0

# (1) crossings
LV = (0.3, 0.5, 0.7, 0.9); cr = {}
for tag, a, B in (("D3T", A0, 8), ("FTSELF_measured", a1, 12), ("FTSELF_with_extension", A1, 12)):
    cr[tag] = {}
    for L in LV:
        c = cross(a, L, B, 30)
        cr[tag][str(L)] = None if c is None else {"n": c, "m": c - B, "m_over_B": (c - B) / B}
ratio = {str(L): (cr["FTSELF_with_extension"][str(L)]["m"] / cr["D3T"][str(L)]["m"]) for L in LV}
width = {tag: cr[tag]["0.9"]["n"] - cr[tag]["0.5"]["n"] for tag in ("D3T", "FTSELF_with_extension")}
delta = cr["FTSELF_measured"]["0.5"]["m"] - cr["D3T"]["0.5"]["m"]


def ip1(a, x):
    lo = math.floor(x); f = x - lo
    return a[lo] if f == 0 else (1 - f) * a[lo] + f * a[lo + 1]


# 2-D round-1 / base tables a_k(n) for integer k, n (k 0..40)
def tab(a, K, nmax_read):
    def f(k, n):
        if n <= 0:
            return 1.0
        if k < 0:
            return FLOOR
        if n > nmax_read:                                                   # unread: draft's rule (flat once k >= n - 1, else FLOOR)
            return a[n] if k >= n - 1 else FLOOR
        if k in K:
            return K[k][n]
        pts = sorted([(kk, K[kk][n]) for kk in K] + [(n + 2, a[n])])     # interpolate K' 17..23 through K'16, K' n+2 and K'24
        pts = sorted({p[0]: p for p in pts}.values())
        if k > pts[-1][0]:
            return pts[-1][1]
        for (ka, va), (kb, vb) in zip(pts, pts[1:]):
            if ka <= k <= kb:
                return va + (vb - va) * (k - ka) / (kb - ka)
    return f


T1, T0 = tab(A1, K1, 16), tab(A0, K0, 16)


def bil(f, k, n):
    k0, n0 = math.floor(k), math.floor(n); fk, fn = k - k0, n - n0
    return ((1 - fk) * (1 - fn) * f(k0, n0) + fk * (1 - fn) * f(k0 + 1, n0) + (1 - fk) * fn * f(k0, n0 + 1) + fk * fn * f(k0 + 1, n0 + 1))


NS = range(13, 25); KS = range(0, 29)
H = {"H_shift": (lambda k, n: T1(k - 4, n - 4), lambda n: A1[n - 4]),
     "H_prop": (lambda k, n: bil(T1, 12 + 0.75 * (k - 16), 12 + 0.75 * (n - 16)), lambda n: ip1(A1, 12 + 0.75 * (n - 16))),
     "H_add_rigid_delta": (lambda k, n: bil(T1, k - 4 - delta, n - 4 - delta), lambda n: ip1(A1, n - 4 - delta)),
     "SHRINK": (lambda k, n: T0(k - 8, n - 8), lambda n: A0[n - 8])}


def readings(a2, A2):
    M2 = sum(a2[n] for n in range(17, 21)) / 4; Mf = sum(a2[n] for n in range(21, 25)) / 4
    aa = {**{n: 1.0 for n in range(1, 13)}, **a2}; c50 = cross(aa, 0.5, 13, 24)
    G = {k: sum(a2[n] - A2(k, n) for n in range(k + 1, min(k + 4, 24) + 1)) / len(range(k + 1, min(k + 4, 24) + 1)) for k in range(13, 24)}
    stop = next((k for k in range(13, 24) if G[k] <= 0.02), "24+")
    KB = sum(a2[n] - A2(18, n) for n in (19, 20)) / 2; RISE = sum(a2[n] - A2(20, n) for n in (19, 20)) / 2
    se = {n: math.sqrt(max(a2[n], 0) * (1 - a2[n]) / 1000) for n in NS}
    seM = math.sqrt(sum(se[n] ** 2 for n in range(17, 21))) / 4
    g = {n: a2[n] - A2(18, n) for n in (19, 20)}; seKB = math.sqrt(sum(abs(g[n]) * (1 - abs(g[n])) / 1000 for n in (19, 20))) / 2   # one-way discordance (minimum)
    return {"a2": {n: round(a2[n], 4) for n in NS}, "M2": M2, "M_far": Mf, "n50": c50, "G": {k: round(v, 4) for k, v in G.items()}, "STOP": stop,
            "KB2": KB, "RISE": RISE, "se_n": {n: round(v, 4) for n, v in se.items()}, "se_M2": seM, "se_KB2_oneway": seKB,
            "gap_SE_to_0.80": (0.80 - M2) / seM, "gap_SE_to_0.785": (0.785 - M2) / seM, "gap_SE_KB_to_0.25": (0.25 - KB) / seKB}


pan = {}
for h, (f2, f1) in H.items():
    a2_2d = {n: f2(n + 2, n) for n in NS}
    a2_pn = {n: f1(n) for n in NS}
    pan[h] = {"per_n": readings(a2_pn, f2), "two_d": readings(a2_2d, f2)}

# (3) retro-test on round 1 (D3-T -> FT-SELF, n 13..16)
retro = {"measured": {n: a1[n] for n in range(13, 17)},
         "H_shift": {n: A0[n - 4] for n in range(13, 17)},
         "H_prop": {n: ip1(A0, 8 + (8 / 12) * (n - 12)) for n in range(13, 17)},
         "H_add_rigid_delta": {n: ip1(A0, n - 4 - delta) for n in range(13, 17)}}
rt = {}
for h in ("H_shift", "H_prop", "H_add_rigid_delta"):
    d = [retro[h][n] - a1[n] for n in range(13, 17)]
    rt[h] = {"pred": {n: round(v, 4) for n, v in retro[h].items()}, "M": sum(retro[h].values()) / 4, "err": [round(x, 4) for x in d], "rmse": math.sqrt(sum(x * x for x in d) / 4)}
rt["measured_M"] = sum(a1[n] for n in range(13, 17)) / 4
# (4) H_prop at other crossing levels: prediction for round-1 n_L from D3-T under H_prop (m_L,1 = 1.5 m_L,0) and under H_add (m_L,1 = m_L,0 + delta)
lev = {}
for L in LV:
    m0 = cr["D3T"][str(L)]["m"]; m1 = cr["FTSELF_with_extension"][str(L)]["m"]
    lev[str(L)] = {"m0": m0, "m1": m1, "m0_over_B": m0 / 8, "m1_over_B": m1 / 12, "H_prop_pred_m1": 1.5 * m0, "H_add_pred_m1": m0 + delta,
                   "m1_needs_extension": cr["FTSELF_measured"][str(L)] is None}
out = {"crossings": cr, "margin_ratio_r1_over_r0": ratio, "width_n90_to_n50": width, "delta_m50": delta, "a1_extension": {n: round(A1[n], 4) for n in range(16, 21)},
       "panels": pan, "retro_round1": rt, "levels": lev}
json.dump(out, open(os.path.join(HERE, "p1_panels.json"), "w", newline="\n"), indent=1, default=float)
print("crossings", json.dumps(cr, default=float)); print("ratio", ratio, "width", width, "delta", delta)
for h, v in pan.items():
    for kind in ("per_n", "two_d"):
        r = v[kind]; print(h, kind, "a2", r["a2"], "M2 %.4f Mfar %.4f n50 %s STOP %s KB2 %.4f RISE %.4f seM %.4f seKB %.4f gap80 %.2f gap785 %.2f gapKB %.2f" % (
            r["M2"], r["M_far"], r["n50"], r["STOP"], r["KB2"], r["RISE"], r["se_M2"], r["se_KB2_oneway"], r["gap_SE_to_0.80"], r["gap_SE_to_0.785"], r["gap_SE_KB_to_0.25"]))
        print("    G", r["G"])
print("retro", json.dumps(rt, default=float)); print("levels", json.dumps(lev, default=float))
