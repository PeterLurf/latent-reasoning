"""$0 descriptive: the pass at which the decoded state first has 0 operators alive (t_fin), vs n, from eval_vbhalt rows (N1 / N2);
and coda-answer correctness at t_fin and at 24; collapses (alive drop >= 2) by pass. A deadline clock predicts t_fin clustering at a fixed pass."""
import json, sys, collections
import numpy as np
OUT = sys.argv[1]; res = {}
for name, run in (("N1_carry", "D:/Latent/boxes/runs_fsq/s211_vb_carry_s1"), ("N2_pa", "D:/Latent/boxes/runs_fsq/s211_vb_pa_s1")):
    d = json.load(open(run + "/eval_vbhalt.json")); rows = d["rows"]; R = {}
    for n in range(6, 17):
        rs = [r for r in rows if r["n"] == n]
        tf = []; ok_fin = []; ncol = []; colt = collections.Counter()
        for r in rs:
            al = r["alive"]; t_ = next((t for t in range(len(al)) if al[t] == 0), None); tf.append(t_)
            ok_fin.append(None if t_ is None else int(r["coda"][t_] == r["y"]))
            ncol.append(len(r["collapse_t"]))
            for c in r["collapse_t"]:
                colt[c] += 1
        c_ = collections.Counter(["never" if t_ is None else (t_ - n) for t_ in tf])
        acc_by_off = {}
        for off in sorted(set(k for k in c_ if k != "never")):
            v = [o for t_, o in zip(tf, ok_fin) if t_ is not None and t_ - n == off]
            acc_by_off[str(off)] = {"items": len(v), "coda_acc_at_tfin": float(np.mean(v))}
        R[str(n)] = {"tfin_minus_n": {str(k): v for k, v in sorted(c_.items(), key=lambda kv: (kv[0] == "never", kv[0] if kv[0] != "never" else 0))},
                     "acc_by_offset": acc_by_off, "mean_collapses": float(np.mean(ncol)), "collapse_pass_hist": {str(k): v for k, v in sorted(colt.items())}}
    res[name] = R
json.dump(res, open(OUT, "w", newline="\n"), indent=1)
for name, R in res.items():
    print("==", name)
    for n, v in R.items():
        print(" n", n, "t_fin-n:", v["tfin_minus_n"], "| acc@t_fin by offset:", {k: "%d:%.2f" % (x["items"], x["coda_acc_at_tfin"]) for k, x in v["acc_by_offset"].items()}, "| mean collapses %.2f" % v["mean_collapses"])
        print("     collapse pass hist:", v["collapse_pass_hist"])
