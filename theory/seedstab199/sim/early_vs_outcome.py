import json, os, sys
import numpy as np
R = "D:/Latent/boxes/runs_fsq"
runs = ["s182_window_s1","s182_window_s2","s182_hold_s1","s182_nofade_s1","s184_reweight_s1","s184_reweight_s2",
        "s185_den_s1","s185_rms_s1","s185_rms_s2","s185_rms_s3","s187_nopass_s1","s187_nopass_s2","s188_rwwin_s1",
        "s190_rmsnopass_s1","s190_rmsnopass_s2"]
marks = [1000,2000,4000,6000,8000,10000,12000,16000,20000]
rows = []
for r in runs:
    p = os.path.join(R, r, "log.jsonl"); a = json.load(open(os.path.join(R, r, "args.json")))
    L = [json.loads(l) for l in open(p) if l.strip().startswith("{")]
    L = [l for l in L if "step" in l and "loss" in l]
    st = np.array([l["step"] for l in L]); lo = np.array([l["loss"] for l in L])
    an = np.array([l.get("answer_ce", np.nan) for l in L]); ax = np.array([l.get("aux_ce", np.nan) for l in L])
    def at(arr, s):
        i = np.where((st > s - 500) & (st <= s + 500))[0]
        return float(np.nanmean(arr[i])) if len(i) else float("nan")
    ent = st[np.where(lo < 0.5)[0][0]] if (lo < 0.5).any() else -1
    fin = float(np.mean(lo[st > st.max() - 1000]))
    rows.append((r, a.get("train_gpu","?")[-4:], a.get("add_norm"), a.get("aux_reweight"), a.get("k_window"), a.get("aux_nopass"), a.get("aux_hold"), a.get("aux_fade"),
                 ent, fin, [round(at(an, m),3) for m in marks], [round(at(ax, m),3) for m in marks], int(st.max())))
for x in rows:
    print(x[0], x[1], "norm", x[2], "rw", x[3], "win", x[4], "nop", x[5], "hold", x[6], "fade", x[7], "| entry", x[8], "final %.3f" % x[9], "maxstep", x[12])
    print("   answer_ce @", marks, x[10]); print("   aux_ce    @", marks, x[11])
