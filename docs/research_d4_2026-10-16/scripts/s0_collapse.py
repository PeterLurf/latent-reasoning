"""$0 descriptive: decoded alive-operator trajectories (eval_vbhalt rows) on N1 / N2: where do beyond-range trajectories leave schedule?
ahead = alive[t] < n - t (over-reduction / collapse), behind = alive[t] > n - t. First collapse pass histogram; alive at pass 8/9/10."""
import json, sys, collections
import numpy as np
OUT = sys.argv[1]
res = {}
for name, run in (("N1_carry", "D:/Latent/boxes/runs_fsq/s211_vb_carry_s1"), ("N2_pa", "D:/Latent/boxes/runs_fsq/s211_vb_pa_s1")):
    d = json.load(open(run + "/eval_vbhalt.json")); rows = d["rows"]; R = {}
    for n in range(7, 17):
        rs = [r for r in rows if r["n"] == n]
        if not rs:
            continue
        A = np.array([r["alive"][:25] for r in rs])  # (items, 25)
        sched = np.array([max(0, n - t) for t in range(25)])
        dev_ = A - sched[None, :]
        fc = collections.Counter([r["collapse_t"][0] if r["collapse_t"] else 0 for r in rs])
        first_off = []
        for row in dev_:
            nz = np.nonzero(row[1:] != 0)[0]
            first_off.append(int(nz[0]) + 1 if len(nz) else 0)
        fo = collections.Counter(first_off)
        # direction at first off-schedule pass
        dirs = collections.Counter()
        for k, row in enumerate(dev_):
            t0 = first_off[k]
            if t0:
                dirs["ahead" if row[t0] < 0 else "behind"] += 1
        # drop at each pass: alive[t-1]-alive[t] distribution, pooled over items still on schedule at t-1
        drop_by_t = {}
        for t in range(1, min(n + 3, 25)):
            on = dev_[:, t - 1] == 0
            if on.sum() < 30:
                continue
            dr = A[on, t - 1] - A[on, t]
            drop_by_t[t] = {"on_sched_items": int(on.sum()), "p_drop0": float((dr == 0).mean()), "p_drop1": float((dr == 1).mean()), "p_drop2plus": float((dr >= 2).mean()), "p_neg": float((dr < 0).mean())}
        R[str(n)] = {"items": len(rs), "first_collapse_t": {str(k): v for k, v in sorted(fc.items())}, "first_offsched_t": {str(k): v for k, v in sorted(fo.items())},
                     "first_off_dir": dict(dirs), "mean_alive_by_t": [round(float(x), 2) for x in A.mean(0)[:min(n + 4, 25)]],
                     "drop_given_on_schedule": drop_by_t, "acc_n2": d["by_n"][str(n)]["acc_n2"]}
    res[name] = R
json.dump(res, open(OUT, "w", newline="\n"), indent=1)
for name, R in res.items():
    print("==", name)
    for n, v in R.items():
        print(" n", n, "acc_n2 %.3f" % v["acc_n2"], "first_off_dir", v["first_off_dir"], "first_off_t", v["first_offsched_t"])
        print("    drop|on-sched:", " ".join("t%s:d0 %.2f d1 %.2f d2+ %.2f (%d)" % (t, x["p_drop0"], x["p_drop1"], x["p_drop2plus"], x["on_sched_items"]) for t, x in v["drop_given_on_schedule"].items()))
