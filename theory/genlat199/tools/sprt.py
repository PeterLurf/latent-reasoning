# Exact operating characteristics of Wald's Bernoulli SPRT (H0 p = p0 vs H1 p = p1), fully sequential or in waves,
# truncated at n_max (no crossing by n_max => NO-CLAIM, conservative). Dynamic programming over (successes, failures).
import math, sys
sys.stdout.reconfigure(encoding="utf-8")
def sprt_oc(p0, p1, alpha, beta, p, wave=1, n_max=30):
    A, B = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    ws, wf = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
    claim = 0.0; en = 0.0; dist = {(0, 0): 1.0}; n = 0
    while dist and n < n_max:
        k = min(wave, n_max - n); new = {}
        for (s, f), pr in dist.items():
            for j in range(k + 1):       # j successes among the k seeds of this wave
                q = pr * math.comb(k, j) * p ** j * (1 - p) ** (k - j)
                s2, f2 = s + j, f + k - j; lam = s2 * ws + f2 * wf
                if lam >= A: claim += q; en += q * (n + k)
                elif lam <= B: en += q * (n + k)
                else: new[(s2, f2)] = new.get((s2, f2), 0.0) + q
        n += k; dist = new
    en += sum(dist.values()) * n
    return claim, en
p0, p1, a, b = 0.6, 0.9, 0.05, 0.2
A, B = math.log((1 - b) / a), math.log(b / (1 - a)); ws, wf = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
print("SPRT p0 %.2f p1 %.2f alpha %.2f beta %.2f: A = ln((1-b)/a) = %.4f, B = ln(b/(1-a)) = %.4f; +%.4f per entry, %.4f per failure" % (p0, p1, a, b, A, B, ws, wf))
print("  claim after s entries and f failures iff s >= (A - f*wf)/ws:", [math.ceil((A - f * wf) / ws - 1e-12) for f in range(4)])
print("  stop for futility iff s <= (B - f*wf)/ws:", [math.floor((B - f * wf) / ws + 1e-12) for f in range(1, 5)])
for wave in (1, 3):
    for p in (0.6, 2 / 3, 0.9, 0.95):
        c, en = sprt_oc(p0, p1, a, b, p, wave=wave, n_max=30)
        print("  wave %d, p %.3f: P(claim) %.4f, E[N] %.2f" % (wave, p, c, en))
# Wald approximations (no overshoot)
for p, L in ((0.6, 1 - a), (0.9, b)):
    ez = p * ws + (1 - p) * wf
    print("  Wald ASN approx at p %.1f: E[z] %.4f, E[N] ~ %.2f" % (p, ez, (L * B + (1 - L) * A) / ez))
