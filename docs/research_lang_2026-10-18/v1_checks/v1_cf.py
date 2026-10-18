"""LANG v1 $0: CONSUMPTION-FRONT (Audit's research-pass addition, relayed by Theory 10-18; an extension of Liang & Pan 2602.00449).
(1) EXPOSURE on SYM v1 (leading-space rendering): for each test-law item and hop j, the donor x' rewires ONE fact, R's hop-j statement
    "Every a_{j-1} is a a_j^R" (j = 1: "<R> is a a_1^R") -> object a_j^P (a same-length 2-token name). Exposed iff (a) in x' R reaches o_P at
    distance D and does not reach o_R (y' = o_P != y), (b) x and x' have equal token length and every concept's first-mention slot token at the
    same index. Also the descriptive double-swap variant (R's and P's hop-j objects exchanged).
(2) CALIBRATION of the pass rule by Monte Carlo over model classes (ideal crossover t* per pair + per-t readout flips eps), with the crossover
    estimated as a least-violations change point on t in {0..K'}, K' = D.
    python -B v1_cf.py  -> v1_cf.json
"""
import collections
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]]
import v1_sym as V  # noqa: E402

V.LEAD = " "
P = {"dlo": 0, "dhi": 0, "klo": "half", "rmax": 4, "cap": (lambda D: None if D < 9 else 2 * D + 6), "jmax": 2, "leaves": "last",
     "pextra": 0.1, "toplev": 0, "mlo": 1, "mhi": 1, "rr": 4, "rr_small": 5}


def render(x, E):
    sym = x["idx_to_symbol"]
    stm = [("%s is a %s." if sym[E[j][0]][:1].isupper() else "Every %s is a %s.") % (sym[E[j][0]], sym[E[j][1]]) for j in x["order"]]
    return V.LEAD + " ".join(stm) + " Is %s a %s or %s?" % (sym[0], sym[x["o1"]], sym[x["o2"]])


def reach(E, src):
    adj = collections.defaultdict(list)
    for a, b in E:
        adj[a].append(b)
    return V.bfs(adj, [src])


def exposure(n=1000, Ds=(4, 5, 6, 7, 8), seed=24300):
    rng = random.Random(seed); out = {}
    for D in Ds:
        cnt = collections.Counter(); cnt2 = collections.Counter()
        for _ in range(n):
            x = V.make(rng, D, P); E = [list(e) for e in x["edges"]]; lv = x["levels"]; sym = x["idx_to_symbol"]
            conc = [c for c in range(len(sym)) if not sym[c][:1].isupper()]
            base_pos = V.first_mention_last_tok(x["question"], sym, conc); base_len = len(V.ENC.encode(x["question"]))
            oR, oP = x["target"], x["neg_target"]
            for j in range(1, D + 1):
                src = 0 if j == 1 else lv[j - 2][0]
                k = [i for i, e in enumerate(E) if e == [src, lv[j - 1][0]]]
                if len(k) != 1:
                    continue
                k = k[0]
                for variant in ("single", "double"):
                    E2 = [list(e) for e in E]; E2[k] = [src, lv[j - 1][1]]
                    if variant == "double":
                        srcP = 1 if j == 1 else lv[j - 2][1]
                        kp = [i for i, e in enumerate(E) if e == [srcP, lv[j - 1][1]]]
                        if len(kp) != 1:
                            continue
                        E2[kp[0]] = [srcP, lv[j - 1][0]]
                    dR = reach(E2, 0)
                    flip = dR.get(oP) == D and oR not in dR
                    q2 = render(x, E2)
                    same = len(V.ENC.encode(q2)) == base_len and V.first_mention_last_tok(q2, sym, conc) == base_pos
                    c_ = cnt if variant == "single" else cnt2
                    c_["n_%d" % j] += 1; c_["flip_%d" % j] += flip; c_["exposed_%d" % j] += flip and same
        out[D] = {"single": {j: {"n": cnt["n_%d" % j], "flip": cnt["flip_%d" % j], "exposed": cnt["exposed_%d" % j]} for j in range(1, D + 1)},
                  "double": {j: {"n": cnt2["n_%d" % j], "flip": cnt2["flip_%d" % j], "exposed": cnt2["exposed_%d" % j]} for j in range(1, D + 1)}}
        print("D", D, "single exposed by j", [cnt["exposed_%d" % j] for j in range(1, D + 1)], "double", [cnt2["exposed_%d" % j] for j in range(1, D + 1)], flush=True)
    return out


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        k = i
        while k + 1 < len(v) and v[o[k + 1]] == v[o[i]]:
            k += 1
        for m in range(i, k + 1):
            r[o[m]] = (i + k) / 2.0
        i = k + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    va = sum((x - ma) ** 2 for x in ra); vb = sum((x - mb) ** 2 for x in rb)
    if va == 0 or vb == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(va * vb)


def crossover(ans, K):
    """least-violations change point: t* minimising #{t < t*: ans = y} + #{t >= t*: ans = y'}; ties -> smallest t*"""
    ny_before = 0; nyp_after = sum(1 for a_ in ans if a_ == 0); best = (nyp_after, 0)
    for c in range(1, K + 2):
        if ans[c - 1] == 1:
            ny_before += 1
        else:
            nyp_after -= 1
        if ny_before + nyp_after < best[0]:
            best = (ny_before + nyp_after, c)
    return best[1]


def ideal(cls, D, j, rng, q_la):
    return {"serial": j - (1 if rng.random() < q_la else 0), "parc": 1, "recompute": D, "clock_D-2": D - 2, "clock_4": 4,
            "doubling": math.ceil(j / 2), "partial_c0_3": max(1, j - 2)}[cls]


RULES = {"asym {j-1, j}, j in 3..D-1; share>=0.80, rho>=0.80": (lambda ts, j: j - 1 <= ts <= j, lambda D: range(3, D), 0.80, 0.80),
         "+-1, j in 3..D-2 (Audit); share>=0.80, rho>=0.80": (lambda ts, j: abs(ts - j) <= 1, lambda D: range(3, D - 1), 0.80, 0.80)}
EPS = (0.0, 0.05, 0.10)


def calibrate(reps=150, per_cell=100, seed=7, rules=None):
    rng = random.Random(seed); res = {}
    rules = rules or RULES
    for rname, (win, jr, b_share, b_rho) in rules.items():
        res[rname] = {}
        for cls in ("serial", "parc", "recompute", "clock_D-2", "clock_4", "doubling", "partial_c0_3"):
            for q_la in ((0.0, 0.5) if cls == "serial" else (0.0,)):
                for eps in EPS:
                    passes = 0; fr_m = 0.0; sp_m = 0.0
                    for _ in range(reps):
                        ok = 0; tot = 0; byD = collections.defaultdict(lambda: ([], []))
                        for D in (5, 6, 7, 8):
                            for j in jr(D):
                                for _ in range(per_cell):
                                    t0 = ideal(cls, D, j, rng, q_la)
                                    ans = [(1 if t >= t0 else 0) for t in range(D + 1)]
                                    ans = [a_ if rng.random() >= eps else 1 - a_ for a_ in ans]
                                    ts = crossover(ans, D); ok += win(ts, j); tot += 1
                                    byD[D][0].append(ts); byD[D][1].append(j)
                        frac = ok / tot
                        w = [(len(byD[D][0]), spearman(byD[D][0], byD[D][1])) for D in byD if len(set(byD[D][1])) >= 2]
                        sp = sum(n_ * s_ for n_, s_ in w) / sum(n_ for n_, _ in w)
                        passes += (frac >= b_share and sp >= b_rho); fr_m += frac; sp_m += sp
                    key = "%s q_la %.1f eps %.2f" % (cls, q_la, eps)
                    res[rname][key] = {"P(pass)": round(passes / reps, 3), "mean frac in window": round(fr_m / reps, 3), "mean within-D Spearman": round(sp_m / reps, 3)}
                    print(rname, key, res[rname][key], flush=True)
    return res


if __name__ == "__main__" and len(sys.argv) == 1 and os.environ.get("CF_REG") == "1":
    EPS = (0.0, 0.05, 0.10, 0.15)
    reg = {"REGISTERED asym {j-1, j}, j in 3..D-1; share>=0.75, rho>=0.60": (lambda ts, j: j - 1 <= ts <= j, lambda D: range(3, D), 0.75, 0.60)}
    out = json.load(open(os.path.join(HERE, "v1_cf.json")))
    out["calibration_registered"] = calibrate(rules=reg)
    json.dump(out, open(os.path.join(HERE, "v1_cf.json"), "w", newline="\n"), indent=1)
elif __name__ == "__main__":
    out = {"exposure": exposure()}
    json.dump(out, open(os.path.join(HERE, "v1_cf.json"), "w", newline="\n"), indent=1)
    out["calibration"] = calibrate()
    json.dump(out, open(os.path.join(HERE, "v1_cf.json"), "w", newline="\n"), indent=1)
