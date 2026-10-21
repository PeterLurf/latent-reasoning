import numpy as np
rng=np.random.default_rng(22101)
P=11
def gen(n, p_bin=0.5, p_const_ref=0.0, mode="uniform"):
    # q0 = constant; step i (1..n) defines q_i from 1 or 2 earlier quantities (uniform over q_0..q_{i-1}) or prev (chain)
    par=[[]]; 
    for i in range(1,n+1):
        if mode=="chain": ps=[i-1] + ([int(rng.integers(0,i))] if rng.random()<p_bin else [])
        else:
            k=2 if (rng.random()<p_bin and i>=2) else 1
            ps=list(rng.choice(i, size=k, replace=False))
        par.append(ps)
    depth=[0]*(n+1)
    for i in range(1,n+1): depth[i]=1+max(depth[p] for p in par[i])
    # ancestors of q_n
    anc=set(); st=[n]
    while st:
        x=st.pop()
        if x in anc: continue
        anc.add(x); st+=par[x]
    return depth[n], len([a for a in anc if a>0])
for mode in ("uniform","chain"):
    for n in (4,6,8):
        D=[gen(n,mode=mode) for _ in range(20000)]
        d=np.array([x[0] for x in D]); r=np.array([x[1] for x in D])
        print(mode,n,"depth mean %.2f"%d.mean(),"P(depth=n) %.3f"%np.mean(d==n),"P(depth>=n-1) %.3f"%np.mean(d>=n-1),"relevant steps mean %.2f"%r.mean())
# outcome distinctness for random graph-matched pairs (chain mode, ops affine +c, -c, *c, + other var)
def run(ops, consts, par, x0):
    q=[x0]
    for i,(o,c) in enumerate(zip(ops,consts)):
        ps=par[i+1]; a=q[ps[0]]; b=q[ps[1]] if len(ps)>1 else c
        q.append((a+b)%P if o=='+' else (a-b)%P if o=='-' else (a*b)%P)
    return q
cnt=0; dist=0; dc=0
for _ in range(20000):
    n=int(rng.integers(4,9)); t=(n+1)//2
    par=[[]]+[[i-1]+([int(rng.integers(0,i))] if rng.random()<0.5 else []) for i in range(1,n+1)]
    def rnd(): return [str(rng.choice(list('+-*'))) for _ in range(n)], [int(rng.integers(1,P)) for _ in range(n)], int(rng.integers(0,P))
    oR,cR,xR=rnd(); oD,cD,xD=rnd()
    qR=run(oR,cR,par,xR); qD=run(oD,cD,par,xD)
    # cf: donor's computed values q_0..q_t, recipient's ops/consts for steps t+1..n
    qc=qD[:t+1]
    for i in range(t,n):
        o,c=oR[i],cR[i]; ps=par[i+1]; a=qc[ps[0]]; b=qc[ps[1]] if len(ps)>1 else c
        qc.append((a+b)%P if o=='+' else (a-b)%P if o=='-' else (a*b)%P)
    yo,yd,yc=qR[n],qD[n],qc[n]
    cnt+=1; dist+= (len({yo,yd,yc})==3); dc+= (yd==yc)
print("random-program pairs: P(all three distinct) %.3f  P(donor==cf) %.3f"%(dist/cnt, dc/cnt))
