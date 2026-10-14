"""seedstats: exact statistics for binary per-seed outcomes (e.g. basin entry), pure Python, no scipy.
Proposal for Theory (G3_methodology.md); not in the repo. Self-test: python -I seedstats.py

  exact_ci(s, n)                 Clopper-Pearson two-sided interval (and one-sided lower bound via conf).
  post_tail(s, n, x, a=1, b=1)   P(p >= x | s of n) under a Beta(a, b) prior (integer a, b: exact binomial identity).
  simon(p0, p1, alpha, beta)     Simon (1989) optimal and minimax two-stage designs (futility stop after stage 1).
  read_sequential(outcomes, d)   CLAIM / NO-CLAIM / CONTINUE for a curtailed design d = (n1, r1, n, r);
                                 outcomes are 1 (entered), 0 (failed), or 'F' (flagged by a frozen detector:
                                 scored as a failure, the conservative-censoring rule).
"""
from math import comb


def pmf(k, n, p):
    return comb(n, k) * p ** k * (1 - p) ** (n - k)


def tail_ge(k, n, p):
    return sum(pmf(j, n, p) for j in range(max(k, 0), n + 1))


def cdf_le(k, n, p):
    return sum(pmf(j, n, p) for j in range(0, min(k, n) + 1)) if k >= 0 else 0.0


def _bisect(f, lo=0.0, hi=1.0, it=200):
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def exact_ci(s, n, conf=0.95, one_sided=False):
    a = 1 - conf if one_sided else (1 - conf) / 2
    lo = 0.0 if s == 0 else _bisect(lambda p: tail_ge(s, n, p) >= a)
    hi = 1.0 if s == n else _bisect(lambda p: cdf_le(s, n, p) <= a)
    return (lo, 1.0) if one_sided else (lo, hi)


def post_tail(s, n, x, a=1, b=1):
    """P(p >= x) for the Beta(s + a, n - s + b) posterior; integer a, b only.
    Identity: for Beta(A, B) with integer A, B, P(p <= x) = P(Bin(A + B - 1, x) >= A)."""
    A, B = s + a, n - s + b
    return cdf_le(A - 1, A + B - 1, x)


def design_oc(d, p):
    """(P(claim), P(stop at stage 1), E[N]) for d = (n1, r1, n, r) without curtailment."""
    n1, r1, n, r = d
    rej = 0.0
    for x1 in range(r1 + 1, n1 + 1):
        need = r + 1 - x1
        rej += pmf(x1, n1, p) * (tail_ge(need, n - n1, p) if need > 0 else 1.0)
    pet = cdf_le(r1, n1, p)
    return rej, pet, n1 + (1 - pet) * (n - n1)


def simon(p0, p1, alpha=0.05, beta=0.2, nmax=30):
    opt = mm = None
    for n in range(2, nmax + 1):
        for n1 in range(1, n):
            for r1 in range(-1, n1):
                for r in range(max(r1, 0), n):
                    a0, pet0, en0 = design_oc((n1, r1, n, r), p0)
                    if a0 > alpha:
                        continue
                    pw = design_oc((n1, r1, n, r), p1)[0]
                    if pw >= 1 - beta:
                        rec = dict(n1=n1, r1=r1, n=n, r=r, size=a0, power=pw, pet0=pet0, en0=en0)
                        if opt is None or en0 < opt["en0"] - 1e-12:
                            opt = rec
                        if mm is None or (n, en0) < (mm["n"], mm["en0"] - 1e-12):
                            mm = rec
                    break
    return opt, mm


def read_sequential(outcomes, d):
    """Curtailed reading of a Simon-type design: claim needs > r1 successes in the first n1 and > r in n."""
    n1, r1, n, r = d
    xs = [1 if o == 1 else 0 for o in outcomes]
    f1_allowed, f_allowed = n1 - (r1 + 1), n - (r + 1)
    fails1 = sum(1 - x for x in xs[:n1]); fails = sum(1 - x for x in xs)
    if fails1 > f1_allowed or fails > f_allowed:
        return "NO-CLAIM"
    if len(xs) >= n:
        return "CLAIM"
    return "CONTINUE"


def _selftest():
    # Simon (1989) classic table entry: p0 0.10, p1 0.30, alpha 0.05, beta 0.20 -> optimal 1/10, 5/29 (E[N|p0] 15.0); minimax 1/15, 5/25.
    o, m = simon(0.10, 0.30, 0.05, 0.20, nmax=30)
    assert (o["r1"], o["n1"], o["r"], o["n"]) == (1, 10, 5, 29), o
    assert abs(o["en0"] - 15.0) < 0.05, o
    assert (m["r1"], m["n1"], m["r"], m["n"]) == (1, 15, 5, 25), m
    # Clopper-Pearson: 6/6 one-sided 95% lower bound = 0.05 ** (1/6)
    assert abs(exact_ci(6, 6, one_sided=True)[0] - 0.05 ** (1 / 6)) < 1e-9
    lo, hi = exact_ci(2, 3)
    assert abs(lo - 0.0943) < 1e-3 and abs(hi - 0.9916) < 1e-3, (lo, hi)
    # Beta posterior: 6/6 under a uniform prior, P(p >= 0.9) = 1 - 0.9 ** 7
    assert abs(post_tail(6, 6, 0.9) - (1 - 0.9 ** 7)) < 1e-12
    # Sequential reading of the recommended design (6, 4, 13, 10) with flags scored as failures
    D = (6, 4, 13, 10)
    assert read_sequential([1, 1, 1], D) == "CONTINUE"
    assert read_sequential([1, "F", 1, 0], D) == "NO-CLAIM"
    assert read_sequential([1] * 11 + [0, 0], D) == "CLAIM"
    assert read_sequential([1] * 10 + [0, 0, 0], D) == "NO-CLAIM"
    print("seedstats self-test OK")
    for p0 in (0.6, 2 / 3):
        o, m = simon(p0, 0.9, 0.05, 0.2, nmax=26)
        print("p0 %.3f p1 0.90 a 0.05 b 0.20 optimal %s" % (p0, {k: round(v, 3) for k, v in o.items()}))
    print("RMS base 2/3 entered: exact 95%% CI %.3f-%.3f" % exact_ci(2, 3))
    print("pooled 10/15 entered by 48k: exact 95%% CI %.3f-%.3f; 9/15 by 30k: %.3f-%.3f" % (exact_ci(10, 15) + exact_ci(9, 15)))


if __name__ == "__main__":
    _selftest()
