import numpy as np
from math import sqrt

rng=np.random.default_rng(22100)
def p_pass(f, floor, N, deff=1.3, bar=0.30, nsim=200000):
    # binomial with design effect via beta-binomial approx: effective N
    Ne=int(N/deff)
    k=rng.binomial(Ne, f, nsim)/Ne
    # floor estimated by 1000 permutations -> negligible SE; treat as known
    return np.mean(k>=floor+bar)
for N in (300,500,1000):
    row=[]
    for d in (0.0,0.10,0.20,0.25,0.28,0.30,0.32,0.35,0.40,0.50):
        row.append((d, round(p_pass(0.09+d,0.09,N),4)))
    print(N,row)
# mixture prior: P(pass) under shaded priors
# H0 none/carriers (follow=floor), H_partial (floor+0.15), H_port (floor+U[0.3,0.8])
for N in (500,1000):
    ph=[0.80,0.08,0.12]
    pp0=p_pass(0.09,0.09,N); pp1=p_pass(0.24,0.09,N)
    fs=rng.uniform(0.30,0.80,2000); pp2=np.mean([p_pass(0.09+x,0.09,N,nsim=4000) for x in fs])
    print(N,'P(pass|none)',pp0,'P(pass|partial .15)',pp1,'P(pass|port U.3-.8)',round(pp2,3),'total',round(ph[0]*pp0+ph[1]*pp1+ph[2]*pp2,3))
