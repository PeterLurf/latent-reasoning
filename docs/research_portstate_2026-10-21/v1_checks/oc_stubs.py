"""PORT-STATE v1: OC per co-primary + known-answer stubs of the reading LOGIC (CPU, numpy; no model). rng 22801."""
import numpy as np, json
rng = np.random.default_rng(22801)
BAND = [4, 5, 6, 7]                 # assumed C1 band for the simulation (>= 3 depths)
N_PAIRS = 1000
def avgrank(a):
    a = np.asarray(a, dtype=float); o = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); r[o] = np.arange(len(a), dtype=float)
    for v in np.unique(a):
        m = a == v; r[m] = r[m].mean()
    return r
def spearman(x, y):
    rx, ry = avgrank(x), avgrank(y); return float(np.corrcoef(rx, ry)[0, 1])
def front_reading(kind, noise_p=0.3, max_b=12):
    """v1.1 rule: pooled over the band with depth fixed effects -- b* residualised on d (within-d centring), t centred within d;
    pass iff tie-aware Spearman(t_c, b_c) <= -0.6 AND the within-d OLS slope <= -0.4 (n >= 1,000 pairs)."""
    T, B = [], []
    for d in BAND:
        n = N_PAIRS // len(BAND); ts = rng.integers(2, d, n)
        extra = rng.geometric(1 - noise_p, n) - 1
        if kind == "serial": b = d - ts + extra
        elif kind == "relay": b = d + extra
        elif kind == "precompute": b = 0 + extra
        elif kind == "partial": b = np.where(rng.random(n) < 0.5, d - ts, d) + extra
        b = np.minimum(b, max_b + 1).astype(float)
        T.append(ts - ts.mean()); B.append(b - b.mean())
    T, B = np.concatenate(T), np.concatenate(B)
    rho = spearman(T, B); slope = float(np.polyfit(T, B, 1)[0])
    return rho <= -0.6 and slope <= -0.4
def onset_reading(kind, floor=0.09, n_items=1000):
    """answer decodable from z_t (t <= d-2) at <= floor + 0.05, per t pooled over the band."""
    rates = []
    for d in BAND:
        for t in range(1, d - 1):
            p = floor if kind != "precompute" else 0.8
            rates.append(rng.binomial(n_items // len(BAND), p) / (n_items // len(BAND)))
    return all(r <= floor + 0.05 for r in rates)
def follow_reading(true_follow, floor=0.09, n=N_PAIRS, deff=1.3):
    se = np.sqrt(true_follow * (1 - true_follow) / (n / deff)); obs = rng.normal(true_follow, se)
    return obs >= floor + 0.30
R = 2000
out = {}
for kind in ("serial", "relay", "precompute", "partial"):
    out["BUDGET-FRONT | " + kind] = float(np.mean([front_reading(kind) for _ in range(R)]))
for kind in ("serial", "precompute"):
    out["ANSWER-ONSET | " + kind] = float(np.mean([onset_reading(kind) for _ in range(R)]))
for tf in (0.09, 0.30, 0.39, 0.42, 0.45, 0.60):
    out["PORT-FOLLOW | true %.2f" % tf] = float(np.mean([follow_reading(tf) for _ in range(R)]))
# joint: PORT-STATE needs follow & front & onset (DECODE-VAL / COPY-CONTROL gate separately)
out["JOINT | portable serial (follow 0.45)"] = out["PORT-FOLLOW | true 0.45"] * out["BUDGET-FRONT | serial"] * out["ANSWER-ONSET | serial"]
out["JOINT | relay with high follow (0.60)"] = out["PORT-FOLLOW | true 0.60"] * out["BUDGET-FRONT | relay"] * out["ANSWER-ONSET | serial"]
out["JOINT | precompute with high follow (0.60)"] = out["PORT-FOLLOW | true 0.60"] * out["BUDGET-FRONT | precompute"] * out["ANSWER-ONSET | precompute"]
out["JOINT | no portability (follow = floor)"] = out["PORT-FOLLOW | true 0.09"]
print(json.dumps(out, indent=1))
json.dump(out, open(__file__.replace(".py", ".json"), "w", newline=chr(10)), indent=1)
