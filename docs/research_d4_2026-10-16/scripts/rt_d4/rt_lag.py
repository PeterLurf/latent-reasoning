"""RT-7: is the CLOCK-KEY 'behind => collapse more' (g_dev +0.53) a decode-lag artefact?
LAG-CATCH pattern at t: on schedule at t-2 (alive = n-t+2), drop 0 at t-1 (decoded 'behind'), drop >= 2 at t landing exactly on schedule (alive[t] = n - t).
If the decode merely lagged, the true state was on schedule throughout and these items answer like clean on-schedule items; a genuine stall +
double reduction (catch-up) is riskier. Compare acc_n2 at n 9 / 10 (answers informative) for items whose FIRST off-schedule event is a lag-catch
vs items never off schedule through t = n - 2 vs items whose first off event is an over-reduction from on-schedule. Also: share of behind->collapse events
(dev +1 at t-1, drop >= 2 at t) that land exactly on schedule."""
import json, sys
import numpy as np
OUT = sys.argv[1]; res = {}
for name, run in (("N1_carry", "D:/Latent/boxes/runs_fsq/s211_vb_carry_s1"), ("N2_pa", "D:/Latent/boxes/runs_fsq/s211_vb_pa_s1")):
    rows = json.load(open(run + "/eval_vbhalt.json"))["rows"]; R = {}
    land = [0, 0]
    for r in rows:
        n, al = r["n"], r["alive"]
        if not 10 <= n <= 16:
            continue
        for t in range(2, 13):
            if al[t - 1] >= 3 and al[t - 1] - (n - t + 1) == 1 and al[t - 1] - al[t] >= 2:
                land[0] += 1; land[1] += al[t] == n - t
    for n in (9, 10, 11):
        grp = {"lagcatch_first": [], "overreduce_first": [], "clean_to_n_minus_2": [], "other_first": []}
        for r in [r for r in rows if r["n"] == n]:
            al = r["alive"]; ok = r["coda"][n + 2] == r["y"]; kind = "clean_to_n_minus_2"
            for t in range(1, n - 1):
                if al[t] != n - t:                                   # first off-schedule pass t
                    if al[t] == n - t + 1 and t + 1 <= 24 and al[t + 1] == n - t - 1:
                        kind = "lagcatch_first"
                    elif al[t] < n - t:
                        kind = "overreduce_first"
                    else:
                        kind = "other_first"
                    break
            grp[kind].append(ok)
        R[str(n)] = {k: {"items": len(v), "acc_n2": round(float(np.mean(v)), 3) if v else None} for k, v in grp.items()}
    res[name] = {"behind_collapse_events": land[0], "share_landing_exactly_on_schedule": round(land[1] / max(1, land[0]), 3), "by_n": R}
json.dump(res, open(OUT, "w", newline="\n"), indent=1)
for k, v in res.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != "by_n"})
    for n, x in v["by_n"].items():
        print("   n", n, x)
