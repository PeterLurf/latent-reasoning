"""OC for GSM-CACHE-SWAP stage 0: pass iff follow_obs >= floor_obs + 0.30 at n pairs; carrier leak control: competing <= 0.25.
floor_obs modelled conservatively as an independent binomial mean over n (the permutation floor's real SE is smaller)."""
import math, sys, re, collections
import numpy as np
def pmf_vec(n, p):
    x = np.arange(n + 1)
    if p <= 0: v = np.zeros(n + 1); v[0] = 1; return v
    if p >= 1: v = np.zeros(n + 1); v[-1] = 1; return v
    lg = np.array([math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in x])
    return np.exp(lg + x * math.log(p) + (n - x) * math.log1p(-p))
def p_pass(n, f, phi, margin=0.30):
    pf, pl = pmf_vec(n, f), pmf_vec(n, phi)          # follow count X, floor count Y
    cf = np.cumsum(pf[::-1])[::-1]                    # P(X >= k)
    tot = 0.0
    for y in range(n + 1):
        k = math.ceil(y + margin * n - 1e-9)
        tot += pl[y] * (cf[k] if k <= n else 0.0)
    return tot
for n in (500, 300, 150):
    print("n pairs", n)
    for phi in (0.0, 0.01, 0.03, 0.05):
        row = {d: round(p_pass(n, min(1, phi + d), phi), 4) for d in (0.0, 0.10, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.60)}
        print("  floor %.2f  P(pass | follow = floor + d):" % phi, row)
n = 500
for c in (0.15, 0.20, 0.25, 0.30):
    k = math.floor(0.25 * n + 1e-9); print("competing true %.2f: P(obs <= 0.25) = %.4f" % (c, pmf_vec(n, c)[:k + 1].sum()))
for o in (0.75, 0.80, 0.85, 0.90):
    k = math.ceil(0.80 * n - 1e-9); print("oracle true %.2f: P(obs >= 0.80) = %.4f" % (o, pmf_vec(n, o)[k:].sum()))
# GSM8K natural pairs: solution-graph signatures from <<expr=c>> annotations
import pandas as pd
df = pd.read_parquet(sys.argv[1])
def sig(ans):
    steps = re.findall(r'<<([^=<>]+)=([^<>]+)>>', ans.replace(',', ''))
    res, out = [], []
    for expr, val in steps:
        toks = re.findall(r'\d+\.?\d*|[-+*/()]', expr); s = []
        for t in toks:
            if re.fullmatch(r'\d+\.?\d*', t):
                v = float(t); s.append('r%d' % res.index(v) if v in res else 'q')
            else: s.append(t)
        out.append(''.join(s))
        try: res.append(float(val))
        except ValueError: res.append(None)
    return tuple(out)
S_ = [sig(a) for a in df.answer]; G = [float(a.split('####')[-1].replace(',', '')) for a in df.answer]
c = collections.Counter(S_); n = len(S_)
print("\ndistinct graph signatures", len(c), "of", n, "; top", c.most_common(3))
pmatch = sum(v * (v - 1) for v in c.values()) / (n * (n - 1)); print("P(random pair has identical signature) = %.4f -> expected matched among 500 random pairs: %.1f" % (pmatch, 500 * pmatch))
have = sum(1 for i in range(n) if any(S_[j] == S_[i] and G[j] != G[i] for j in range(n) if j != i))
print("items with >= 1 same-signature partner with a different gold: %d (%.2f); x0.556 correct-pool approx %d recipients (if both correct ~ x0.556 again for partners)" % (have, have / n, round(have * 0.556)))
multi = [i for i in range(n) if len(S_[i]) >= 2]; print("items with >= 2 steps: %d (%.2f)" % (len(multi), len(multi) / n))
