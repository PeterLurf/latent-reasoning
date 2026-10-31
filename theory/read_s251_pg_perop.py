import os; os.environ["OMP_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import sys, json, collections
sys.path.insert(0, "D:/Latent")
import boxes.ow_words as OW, boxes.inject_ow as IO
M=11
FR=list(range(6,16))
sets={}
for f in FR:
    sf=OW.Stream(IO.SEED_FRAMESET+f, frames=(f,), mul_s_p=2, **IO.OWKW); sets[f]={d:[sf.draw(d) for _ in range(IO.TEST_N)] for d in (1,2,3,4)}
cand={"add":lambda a,o:(a+o)%M,"sub":lambda a,o:(a-o)%M,"rsub":lambda a,o:(o-a)%M,"mul":lambda a,o:(a*o)%M}
res={}
for run in ["lh_fh_s1","lh_fh_s2","fd_ctrl_s1","fd_ctrl_s2","fd_div_s1","fd_div_s2"]:
    e=json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_frames.json"%run,encoding="utf-8"))
    for f in FR:
        if str(f) not in e: continue
        for d in (2,3,4):
            it=e[str(f)][str(d)]["items"]; regen=sets[f][d]
            if [x["sig"] for x in regen]!=it["sig"]: print("SIG MISMATCH",run,f,d); break
            w1=it["ws_pred"]["1"]
            for i,x in enumerate(regen):
                op=x["ops"][0]; a,o=x["values"][0],x["operands"][0]
                k=(run[:-3],f,op); r=res.setdefault(k,collections.Counter()); r["n"]+=1; r["ok"]+=(w1[i]==x["values"][1])
                for nm,fn in cand.items(): r[nm]+=(w1[i]==fn(a,o))
ops=['add_c','sub_c','mul_c','add_v','sub_v','mul_v','add_s']
for f in FR:
    print("frame",f)
    for op in ops:
        row=[]
        for g in ["lh_fh","fd_ctrl","fd_div"]:
            r=res.get((g,f,op))
            if not r: row.append("   -   "); continue
            comp=max(cand,key=lambda nm:r[nm]); row.append("%.2f(%s %.2f)"%(r["ok"]/r["n"],comp,r[comp]/r["n"]))
        print("  %-6s"%op,"  ".join(row), "n=%d"%(res.get(("fd_div",f,op),{"n":0})["n"]))
