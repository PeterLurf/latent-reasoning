"""Note 259-9 (post-hoc, pre-stated f7296836): (a) r = 2 miss classes at d 5-7; (b) r = 3 halting via score_c5c's rule, with passes_run - (d - 1)."""
import json, os, sys
sys.path.insert(0, "D:/Latent/theory")
import score_c5c as C, score_hx as HX
R = "D:/Latent/boxes/runs_fsq"
out = {}
for run in C.RUNS:
    h = json.load(open(os.path.join(R, run, "eval_halthead2.json")))
    miss = {}
    for d in (5, 6, 7):
        it = h[str(d)]["items"]; cls = {"PREMATURE": 0, "WRONG-ANYWAY": 0, "LATE-DRIFT": 0, "other": 0}; n = 0
        for p, y in zip(it["pred"], it["ans"]):
            k, _ = C.halt_items(p, 2)
            if p[k] == y:
                continue
            n += 1
            c = "PREMATURE" if p[14] == y else "WRONG-ANYWAY" if p[8] != y else "LATE-DRIFT" if p[14] != y else "other"
            cls[c] += 1
        miss[d] = {"n_miss": n, **{k: (v / n if n else 0.0) for k, v in cls.items()}}
    e = {"head": {"val_acc": 1.0}}
    for d in range(1, 8):
        e[str(d)] = C.per_d(h[str(d)]["items"], 3)
    r3 = HX.halt(e); r3["passes_minus_eff"] = {d: e[str(d)]["mean_passes_run"] - (d - 1) for d in range(1, 8)}
    out[run] = {"r2_miss_classes": miss, "r3": r3}
print(json.dumps({run: {"miss": v["r2_miss_classes"], "r3_label": v["r3"]["label"],
                        "r3": {d: (round(x["acc_k8"], 3), round(x["acc_halt"], 3), round(x["mean_khalt"], 2)) for d, x in v["r3"]["by_d"].items()},
                        "r3_passes_minus_eff": {d: round(x, 2) for d, x in v["r3"]["passes_minus_eff"].items()}} for run, v in out.items()}, indent=0))
json.dump(out, open("D:/Latent/theory/results/read_s259_c5_r3.json", "w", encoding="utf-8", newline="\n"), indent=1, default=str)
