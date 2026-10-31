import os; os.environ["OMP_NUM_THREADS"]="2"
import sys, json, collections
sys.path.insert(0, "D:/Latent")
import boxes.ow_words as OW, boxes.inject_ow as IO
M=11
cand={"add":lambda a,o:(a+o)%M,"sub":lambda a,o:(a-o)%M,"rsub":lambda a,o:(o-a)%M,"mul":lambda a,o:(a*o)%M}
for f,opsel in [(7,('add_c','sub_c')),(12,('mul_v','sub_v','add_c')),(14,('mul_v',)),(6,('sub_v','mul_v'))]:
    sf=OW.Stream(IO.SEED_FRAMESET+f, frames=(f,), mul_s_p=2, **IO.OWKW); S={d:[sf.draw(d) for _ in range(IO.TEST_N)] for d in (1,2,3,4)}
    for run in ["fd_ctrl_s1","fd_ctrl_s2","fd_div_s1","fd_div_s2"]:
        e=json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_frames.json"%run,encoding="utf-8"))
        r=collections.defaultdict(collections.Counter)
        for d in (2,3,4):
            it=e[str(f)][str(d)]["items"]; w1=it["ws_pred"]["1"]
            assert [x["sig"] for x in S[d]]==it["sig"]
            for i,x in enumerate(S[d]):
                op=x["ops"][0]
                if op not in opsel: continue
                a,o=x["values"][0],x["operands"][0]; r[op]["n"]+=1
                for nm,fn in cand.items(): r[op][nm]+=(w1[i]==fn(a,o))
        print(f,run,{op:{k:round(v/c["n"],2) for k,v in c.items() if k!="n"} for op,c in r.items()})
