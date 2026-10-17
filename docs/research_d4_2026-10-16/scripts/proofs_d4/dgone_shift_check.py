"""PROOFS_D4 check 8: does D4's DEADLINE-GONE reading (d4_edge_collapse.py, used verbatim via import) fire on a SHIFTED deadline?
Synthetic alive trajectories (n 9..16, 1000 items per n, K' 24) under SOFT(Delta) (N1 hazard shifted by Delta, N1 drop law) and HARD(t_d)
(one reduction per pass, then collapse to 0 at pass t_d), plus CONTENT. The reader runs on a temp dir inside this scratch folder."""
import json, os, sys, tempfile, importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location("ec", "D:/Latent/docs/research_d4_2026-10-16/scripts/d4_edge_collapse.py"); EC = importlib.util.module_from_spec(spec)
src = open("D:/Latent/docs/research_d4_2026-10-16/scripts/d4_edge_collapse.py").read().split('if __name__ == "__main__":')[0]
exec(compile(src, "d4_edge_collapse.py", "exec"), EC.__dict__)
REF = json.load(open("D:/Latent/docs/research_d4_2026-10-16/scripts/d4_edge_collapse_refs.json"))
h_on = {int(t): v for t, v in REF["D:/Latent/boxes/runs_fsq/s211_vb_carry_s1"]["h_by_t"].items()}
fc = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "falsifier_check.json")))
dl = {int(k): v for k, v in fc["N1_drop_law"].items()}; dvals = np.array(sorted(dl)); dlaw = np.array([dl[k] for k in dvals]); dlaw = dlaw / dlaw.sum()
rng = np.random.default_rng(1)
def h_at(t):
    if t < 2:
        return 0.0
    if t >= 9:
        return h_on[9] if t < 10 else 0.995
    lo = int(np.floor(t)); f = t - lo; return (1 - f) * h_on.get(lo, 0.0) + f * h_on.get(lo + 1, 0.995)
def traj(model, n, K=24):
    al = [n]; a = n
    for t in range(1, K + 1):
        if model[0] == "CONTENT":
            d = 1
        elif model[0] == "SOFT":
            d = int(rng.choice(dvals, p=dlaw)) if (a >= 3 and rng.random() < h_at(t - model[1])) else 1
        else:
            td = model[1]; tdi = int(np.floor(td)) + (1 if (td % 1 and rng.random() < 0.5) else 0)
            d = a if t >= tdi else 1
        a = max(0, a - d); al.append(a)
    return al
res = {}
models = [("CONTENT", 0)] + [("SOFT", d) for d in (0.0, 2.0, 3.0, 4.0, 4.5, 6.5, 8.0)] + [("HARD", t) for t in (9, 11, 12, 12.5, 13.5, 15.5, 16, 20)]
for m in models:
    rows = [{"n": n, "alive": traj(m, n)} for n in range(9, 17) for _ in range(1000)]
    d = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))
    json.dump({"rows": rows}, open(os.path.join(d, "eval_vbhalt.json"), "w")); r = EC.read(d); os.remove(os.path.join(d, "eval_vbhalt.json")); os.rmdir(d)
    res["%s(%s)" % m] = {"EDGE_COLLAPSE": r["EDGE_COLLAPSE"], "DEADLINE_GONE": r["DEADLINE_GONE"], "h_7_9": round(r["h_7_9"], 4) if r["h_7_9"] is not None else None,
                          "events_7_9": r["events_7_9"], "median_t_fin": r["median_t_fin"]}
json.dump(res, open(sys.argv[1], "w", newline="\n"), indent=1)
for k, v in res.items():
    print("%-16s EC %-5s DG %-5s h79 %s ev %d t_fin %s" % (k, v["EDGE_COLLAPSE"], v["DEADLINE_GONE"], v["h_7_9"], v["events_7_9"], [v["median_t_fin"][str(n)] for n in range(9, 17)]))
