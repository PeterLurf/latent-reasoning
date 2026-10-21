import sys, json, re, collections, math, hashlib
import numpy as np, pandas as pd

class _B:
    @staticmethod
    def pmf(x, n, p):
        if p <= 0: return 1.0 if x == 0 else 0.0
        return math.exp(math.lgamma(n+1)-math.lgamma(x+1)-math.lgamma(n-x+1)+x*math.log(p)+(n-x)*math.log1p(-p))
    @staticmethod
    def cdf(k, n, p):
        return sum(_B.pmf(x, n, p) for x in range(0, int(k)+1)) if k >= 0 else 0.0
    @staticmethod
    def sf(k, n, p):
        return sum(_B.pmf(x, n, p) for x in range(int(k)+1, n+1))
binom = _B
pq, tokp, local = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_parquet(pq)
print("HF test n", len(df), "sha256", hashlib.sha256(open(pq,'rb').read()).hexdigest()[:16])
def gold(a):
    a = a.split('####')[-1].replace(',', '')
    try: return float(a)
    except ValueError: return float('inf')
G = np.array([gold(a) for a in df.answer]); Q = [q.strip().replace('  ', ' ') for q in df.question]
# local json used by our GPT-2 harness: same items/order?
d = json.load(open(local, encoding='utf-8'))
lq = [q.strip().replace('  ', ' ') for q in d['question']]; lg = [float(str(a).replace(',', '').split('####')[-1].strip()) for a in d['answer']]
print("local json n", len(lq), "same questions in order:", lq == Q, "same golds:", np.allclose(lg, G))
c = collections.Counter(G.tolist()); n = len(G)
p = np.array(list(c.values())) / n
print("distinct golds", len(c), "top10", c.most_common(10))
coll = float((p**2).sum()); print("collision prob sum p^2 = %.4f" % coll)
# P(random other item's gold == g | gold != recipient's)  averaged over recipients
cond = np.mean([(c[g2] if False else 0) for g2 in []] or [0])
vals = np.array([c[g] for g in G.tolist()])
# for recipient r (gold gr) and donor drawn uniformly among items with gold != gr: coincidence of an output o with donor gold
# null A: output distributed as the gold marginal restricted to != gr  -> P(o == gd) = sum_{a != gr} p_a^2 / (1-p_gr)^2
nullA = np.mean([(coll - (c[g]/n)**2) / (1 - c[g]/n)**2 for g in G.tolist()])
print("null A (garbage output ~ gold marginal, != own gold): %.4f" % nullA)
# null B: output = a number copied from the recipient's question; P(donor gold appears among recipient question numbers)
nums = [set(float(x) for x in re.findall(r'-?\d+\.?\d*', q.replace(',', ''))) for q in Q]
rng = np.random.default_rng(0); hits = []
for r in range(n):
    pool = np.where(G != G[r])[0]
    for _ in range(20):
        dd = rng.choice(pool); hits.append(G[dd] in nums[r])
print("null B (donor gold among recipient's question numbers): %.4f" % np.mean(hits))
print("P(gold in own question numbers): %.4f" % np.mean([G[i] in nums[i] for i in range(n)]))
small = np.mean((G == np.round(G)) & (np.abs(G) <= 20)); print("share of golds integer <= 20: %.3f" % small)
# question length in Llama tokens (BOS + text) and spread of donor-recipient length gaps
from tokenizers import Tokenizer
tk = Tokenizer.from_file(tokp); L = np.array([len(tk.encode(q, add_special_tokens=True).ids) for q in Q])
print("Llama tokens (incl BOS): max %d p99 %d mean %.1f median %d" % (L.max(), np.percentile(L, 99), L.mean(), np.median(L)))
gap = np.abs(L[:, None] - L[None, :]); print("random pair |len gap|: median %d, P(gap<=2)=%.3f, P(gap==0)=%.3f" % (np.median(gap), (gap <= 2).mean(), (gap == 0).mean()))
# OC
print("\nOC (bar = 0.30, floor ~0):")
for acc in (0.53, 0.556, 0.58):
    N = int(round(acc * n)); k = math.ceil(0.30 * N - 1e-9)
    row = {f: float(binom.sf(k - 1, N, f)) for f in (0.02, 0.05, 0.10, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40)}
    print(" acc %.3f pool %d need >= %d follows; P(pass|follow):" % (acc, N, k), {f: round(v, 4) for f, v in row.items()})
    print("   SE at 0.02 %.4f, 0.32 %.4f, 0.40 %.4f" % tuple(math.sqrt(f*(1-f)/N) for f in (0.02, 0.32, 0.40)))
    kc = math.ceil(0.10 * N - 1e-9)  # context-bound: follow < 0.10 -> X <= kc-1
    print("   P(follow_obs < 0.10 | true):", {f: round(float(binom.cdf(kc - 1, N, f)), 4) for f in (0.04, 0.06, 0.08, 0.10, 0.12)})
N = 300; k = 90; print(" min pool 300: P(pass|0.32)=%.3f P(pass|0.40)=%.4f" % (binom.sf(k-1, N, .32), binom.sf(k-1, N, .40)))
# intact gate
for acc in (0.556,):
    se = math.sqrt(acc*(1-acc)/n); print("\nintact SE at n 1319 = %.4f; 0.025 band = %.2f SE; P(|acc_obs-0.556|>0.025 | true 0.556) = %.4f" % (se, 0.025/se, math.erfc((0.025/se)/math.sqrt(2))))
    print(" P(pass gate | true acc) :", {t: round(float(binom.cdf(math.floor((0.556+0.025)*n), n, t) - binom.cdf(math.ceil((0.556-0.025)*n)-1, n, t)), 3) for t in (0.50, 0.52, 0.531, 0.54, 0.556, 0.57, 0.58, 0.60)})
