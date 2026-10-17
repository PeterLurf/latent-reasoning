"""RT-1b: decode faithfulness of collapse events, per n (N1/N2 eval_vbhalt rows).
For items with a collapse (alive drop >= 2) and t_fin < n + 2:
  right_tfin   = coda[t_fin] == y            (the compressed state already answers right: a correct multi-reduction or a decode lag)
  recov_n2     = coda[t_fin] != y and coda[n+2] == y   (the decode said 'done' but computation continued and fixed the answer: decode artefact)
  wrong_both   = wrong at t_fin and at n + 2
  alive_rises  = alive goes back above 0 after t_fin (decode non-monotone)
Also the first collapse event alone: P(coda[n+2] == y | first collapse at t, on-schedule before it)."""
import json, sys
import numpy as np
OUT = sys.argv[1]; res = {}
for name, run in (("N1_carry", "D:/Latent/boxes/runs_fsq/s211_vb_carry_s1"), ("N2_pa", "D:/Latent/boxes/runs_fsq/s211_vb_pa_s1")):
    rows = json.load(open(run + "/eval_vbhalt.json"))["rows"]; R = {}
    for n in range(9, 17):
        c = {"items": 0, "right_tfin": 0, "recov_n2": 0, "wrong_both": 0, "alive_rises": 0, "any_alive_increase": 0}
        allrows = [r for r in rows if r["n"] == n]
        for r in allrows:
            al = r["alive"]; c["any_alive_increase"] += any(al[t] > al[t - 1] for t in range(1, 25))
            if not r["collapse_t"]:
                continue
            tf = next((t for t in range(25) if al[t] == 0), None)
            if tf is None or tf >= n + 2:
                continue
            c["items"] += 1; y = r["y"]; a_t = r["coda"][tf] == y; a_n2 = r["coda"][n + 2] == y
            c["right_tfin"] += a_t; c["recov_n2"] += (not a_t) and a_n2; c["wrong_both"] += (not a_t) and (not a_n2)
            c["alive_rises"] += any(al[u] > 0 for u in range(tf + 1, n + 3))
        k = c["items"]
        R[str(n)] = {"collapse_items_tfin_lt_n2": k, **{x: round(c[x] / k, 3) for x in ("right_tfin", "recov_n2", "wrong_both", "alive_rises")},
                     "any_alive_increase_all_items": round(c["any_alive_increase"] / len(allrows), 3), "acc_n2_all": round(float(np.mean([r["coda"][n + 2] == r["y"] for r in allrows])), 3)}
    res[name] = R
json.dump(res, open(OUT, "w", newline="\n"), indent=1)
for name, R in res.items():
    print("==", name)
    for n, v in R.items():
        print(" n", n, v)
