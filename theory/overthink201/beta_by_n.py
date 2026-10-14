# read-only: per-n per-pass beta mean / state retention / additive cos_prev & rel_inc from emp_<run>.json (POST-HOC tabulation)
import json, sys, os
D = "D:/Latent/theory/seedstab199/emp"
for r in ["s185_rms_s1", "s185_rms_s2", "s190_rmsnopass_s1", "s184_reweight_s2", "s185_rms_s3"]:
    d = json.load(open(os.path.join(D, "emp_%s.json" % r)))
    P = d["passes"]; ks = sorted(P, key=int)
    k0 = list(P["2"]["by_n"]["4"].keys())
    if r == "s185_rms_s1": print("by_n keys:", k0)
    print("==", r)
    for n in ["4", "5", "6", "7", "8"]:
        for f in ("beta_mean", "state_retention", "rel_inc_add", "cos_prev_add", "rel_inc_delta"):
            row = []
            for t in ks:
                bn = P[t]["by_n"].get(n, {}); v = bn.get(f)
                if isinstance(v, dict): v = v.get("mean", v.get("median"))
                row.append("%.3f" % v if isinstance(v, (int, float)) else "  -  ")
            print("  n%s %-16s %s" % (n, f, " ".join(row)))
