# DEPTH vs basin entry (THEORY §187 s2 note table; entry from that table) -- descriptive variance decomposition.
import statistics as st, math, sys
sys.stdout.reconfigure(encoding="utf-8")
rows = [("WINDOW s2",15.8,.659),("RMS s2",17.3,.662),("RMS s1",17.9,.587),("187 s1",18.0,.623),("184 s2",18.7,.665),
        ("WINDOW s1",23.2,.465),("187 s2",26.2,.493),("184 s1",41.1,.486)]
d = [r[2] for r in rows]; e = [r[1] for r in rows]
print("raw SD across 8 runs: %.3f" % st.stdev(d))
early = [r[2] for r in rows if r[1] <= 20]; late = [r[2] for r in rows if r[1] > 20]
pooled = math.sqrt(((len(early)-1)*st.variance(early) + (len(late)-1)*st.variance(late)) / (len(d)-2))
print("early (<=20k) n=%d mean %.3f SD %.3f; late n=%d mean %.3f SD %.3f; pooled within-stratum SD %.3f" % (len(early), st.mean(early), st.stdev(early), len(late), st.mean(late), st.stdev(late), pooled))
# linear regression on log entry
x = [math.log(v) for v in e]; mx, my = st.mean(x), st.mean(d)
b = sum((xi-mx)*(yi-my) for xi, yi in zip(x, d)) / sum((xi-mx)**2 for xi in x)
res = [yi - (my + b*(xi-mx)) for xi, yi in zip(x, d)]
print("regression on log(entry): slope %.3f per log-unit, residual SD %.3f (df 6)" % (b, math.sqrt(sum(r*r for r in res)/6)))
print("variance ratio raw/within-stratum: %.1f" % (st.variance(d)/pooled**2))
# single-seed vs single-seed: P(|diff| >= 0.10) under no effect, normal approx
from math import erf
Phi = lambda z: 0.5*(1+erf(z/math.sqrt(2)))
for sd, lab in ((st.stdev(d), "raw"), (pooled, "entry-stratified")):
    s = math.sqrt(2)*sd
    print("%s: one-sided P(diff >= +0.10 | no effect) = %.3f ; P(diff >= +0.15) = %.3f" % (lab, 1-Phi(0.10/s), 1-Phi(0.15/s)))
# eval noise
for n_items in (1000, 4000):
    print("eval binomial SE at p=0.6, %d items: %.4f" % (n_items, math.sqrt(.24/n_items)))
