# J-3: LR-exposure matching of SO steps in the joint run vs LOOP-SI (lr_at, boxes/fsq120.py:740-743, warm 100, cosine over --steps)
import math
def lr(s, T): return min(1, s/100) * 0.5 * (1 + math.cos(math.pi * s / T))
def S(T, stop): return sum(lr(s, T) for s in range(1, stop+1))
p = 0.5
for (Tj, stopj, Ts, stops, tag) in [(16000, 8000, 8000, 4000, "pilot: joint 16k@8k vs LOOP-SI 8k@4k"), (16000, 16000, 8000, 8000, "full: joint 16k vs LOOP-SI 8k")]:
    ej = p * S(Tj, stopj); es = S(Ts, stops)
    print(tag, "E[sum lr over SO steps] joint %.2f, single %.2f, ratio %.4f" % (ej, es, ej/es))
    # per-SO-step mean LR
    print("   mean lr per SO step: joint %.4f single %.4f" % (S(Tj, stopj)/stopj, es/stops))
import numpy as np
for N in (8000, 16000):
    print("N", N, "SO steps mean", N*p, "sd %.1f" % (math.sqrt(N*p*(1-p))), "P(share in [0.45,0.55] per 1k bin) %.4f" % (sum(math.comb(1000,k) for k in range(450,551))/2**1000))
