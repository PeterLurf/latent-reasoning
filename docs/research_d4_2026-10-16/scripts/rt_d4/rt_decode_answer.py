"""RT-1: is EDGE-COLLAPSE (decoded alive drop >= 2) a real over-reduction or an R-decode artefact?
Per item: collapse = any alive drop >= 2 (collapse_t non-empty). If the TRUE state over-reduced, the coda answer is wrong (~chance 1/11 + luck)
from t_fin on and stays frozen; if only the decode mis-reads, later coda answers recover / keep moving (the core keeps computing).
Reads eval_vbhalt rows of N1 (carry) / N2 (pa). CPU, read-only."""
import json, sys, collections
import numpy as np
sys.path.insert(0, sys.argv[2])
from testset import test_set
OUT = sys.argv[1]
TE = test_set()
res = {}
for name, run in (("N1_carry", "D:/Latent/boxes/runs_fsq/s211_vb_carry_s1"), ("N2_pa", "D:/Latent/boxes/runs_fsq/s211_vb_pa_s1")):
    rows = json.load(open(run + "/eval_vbhalt.json"))["rows"]
    assert all(TE[r["item"]]["t"] == r["y"] and TE[r["item"]]["n"] == r["n"] for r in rows), "test-set replica mismatch"
    R = {}
    for n in range(6, 17):
        rs = [r for r in rows if r["n"] == n]
        grp = {"collapse": [], "clean": []}
        for r in rs:
            grp["collapse" if r["collapse_t"] else "clean"].append(r)
        o = {}
        for g, L in grp.items():
            if not L:
                continue
            al = lambda r: r["alive"]
            tf = [next((t for t in range(25) if r["alive"][t] == 0), None) for r in L]
            ok24 = np.mean([r["coda"][24] == r["y"] for r in L]); okn2 = np.mean([r["coda"][n + 2] == r["y"] for r in L])
            fin = [(r, t) for r, t in zip(L, tf) if t is not None]
            ok_tf = np.mean([r["coda"][t] == r["y"] for r, t in fin]) if fin else None
            frozen_after = np.mean([all(r["coda"][u] == r["coda"][t] for u in range(t, 25)) for r, t in fin]) if fin else None
            # answer changes after t_fin and becomes correct later (recovery)
            recover = np.mean([(r["coda"][t] != r["y"]) and (r["coda"][24] == r["y"]) for r, t in fin]) if fin else None
            o[g] = {"items": len(L), "acc_n2": round(float(okn2), 3), "acc_24": round(float(ok24), 3), "acc_at_tfin": None if ok_tf is None else round(float(ok_tf), 3),
                    "answer_frozen_tfin_to_24": None if frozen_after is None else round(float(frozen_after), 3), "wrong_at_tfin_right_at_24": None if recover is None else round(float(recover), 3)}
        R[str(n)] = o
    res[name] = R
json.dump(res, open(OUT, "w", newline="\n"), indent=1)
for name, R in res.items():
    print("==", name)
    for n, o in R.items():
        print(" n", n, " | ".join("%s: %s" % (g, " ".join("%s=%s" % (k, v) for k, v in x.items())) for g, x in o.items()))
