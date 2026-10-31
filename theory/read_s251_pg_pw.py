import sys,re,collections
sys.path.insert(0,'D:/Latent')
import boxes.nested_words as NW
ops6=('add_c','sub_c','mul_c','add_v','sub_v','mul_v')
def toks(t):
    t=re.sub(r'\{[A-Za-z]\}','',t).replace("'s"," ").lower()
    return set(re.findall(r"[a-z]+",t))
def bigr(t):
    t=re.sub(r'\{[A-Za-z]\}','X',t).replace("'s","").lower()
    w=re.findall(r"[a-zX]+",t); return set(zip(w,w[1:]))
keys=['by','times','more','fewer','less','plus','as','many','together','beyond','exceeds','minus','multiplied','multiply','product','sum','difference','combined','than','extra','short','take','add','of','those','for','each','with','number','increased','decreased','scale','subtract','excess','jointly','pool','put']
for name,frs in [('0-5',range(6)),('0-5+8-11',list(range(6))+list(range(8,12)))]:
    # weight ops by training frequency: OPS_REG uniform, add_s->add_v templates, mul_s forced ~1 per item
    w={'add_c':1,'sub_c':1,'mul_c':1,'add_v':2,'sub_v':1,'mul_v':1}  # add_s aliases add_v; mul_s (forced) ignored here
    for wname,ww in [('uniform',{o:1 for o in ops6}),('freq',w)]:
        cnt=collections.defaultdict(lambda:collections.Counter()); cntb=collections.defaultdict(lambda:collections.Counter())
        for op in ops6:
            for f in frs:
                for k in toks(NW.FRAMES[op][f]): cnt[k][op]+=ww[op]
                for k in toks(NW.FRAMES[op][f]): cntb[k][NW.BASE[op]]+=ww[op]
        print('==',name,wname)
        for k in keys:
            if k in cntb:
                c=cntb[k]; n=sum(c.values()); print(f"  {k:12s} n={n:3d} base:"+" ".join(f"{o}={c[o]/n:.2f}" for o in ('add','sub','mul')), "| op max %.2f"%(max(cnt[k].values())/n))
        # fraction of content words with max P(base|word)>=0.9
        allw=[k for k in cntb if sum(cntb[k].values())>=2]
        hi=[k for k in allw if max(cntb[k].values())/sum(cntb[k].values())>=0.99]
        print('  words n>=2:',len(allw),' base-pure:',len(hi), sorted(hi))
    # bigram purity
    cb=collections.defaultdict(collections.Counter)
    for op in ops6:
        for f in frs:
            for b in bigr(NW.FRAMES[op][f]): cb[b][NW.BASE[op]]+=1
    for b in [('as','many'),('times','as'),('X','by'),('by','X'),('fewer','u'),('more','u'),('more','than'),('fewer','than')]:
        if b in cb: print('  bigram',b,dict(cb[b]))
