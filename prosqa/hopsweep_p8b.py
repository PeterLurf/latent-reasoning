"""PROTOCOL8b follow-up (exploratory, run after P8b1 was scored): where should a written checkpoint point?
For each problem with answer depth in [--dmin, --dmax], write the oracle checkpoint "<root> is a <Y_k>." where Y_k is
the node k gold hops in (as in make_ckpt_data.py), for every k = 1..depth-1, at one random statement position (the
same position for all k), and run a fresh K'-thought episode. Also the plain episode (k = 0). Reports candidate
accuracy by (depth, k) and by hops remaining (depth - k).

    python hopsweep_p8b.py --ckpt <ckpt> --data p8data/p8_test.json --out <json> [--n_per_d 40] [--Kp 10] [--device cuda:2]
"""
import argparse
import json

import numpy as np
import torch

from reanchor_p8 import STMT, lp_of, split, with_checkpoint
from gpt2c import GPT2C, load_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_d", type=int, default=40); ap.add_argument("--dmin", type=int, default=7)
    ap.add_argument("--dmax", type=int, default=10); ap.add_argument("--Kp", type=int, default=10)
    ap.add_argument("--device", default="cuda:2"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_probe", action="store_true", help="exclude the problems the P8b1 probe used (reanchor_p8.py "
                    "defaults: first 100 with ext <= 2 for calibration, then the first 150 per ext) and print no accuracies")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT2C(load_state(a.ckpt, a.device)); model.mask_answer = True
    rng = np.random.default_rng(a.seed)
    data = json.load(open(a.data))
    probe = set()
    if a.skip_probe:
        cal = [i for i, x in enumerate(data) if x["ext"] <= 2][:100]; probe |= set(cal); per = {}
        for i, x in enumerate(data):
            if i in probe or per.get(x["ext"], 0) >= 150:
                continue
            per[x["ext"]] = per.get(x["ext"], 0) + 1; probe.add(i)
    seen, rows = {}, []
    for i, x in enumerate(data):
        d = len(x["steps"])
        if i in probe:
            continue
        if not a.dmin <= d <= a.dmax or seen.get(d, 0) >= a.n_per_d:
            continue
        seen[d] = seen.get(d, 0) + 1
        stmts, _ = split(x)
        pos = int(rng.integers(0, len(stmts) + 1))
        base = lp_of(model, x, a.Kp)[0]
        r = {"depth": d, "pos": pos, "k0": float(base[0] - base[1])}
        for k in range(1, d):
            y = STMT.match(x["steps"][k - 1]).group(3)
            lp = lp_of(model, with_checkpoint(x, y, pos), a.Kp)[0]
            r[f"k{k}"] = float(lp[0] - lp[1])
        rows.append(r)
        if len(rows) % 20 == 0:
            print(f"{len(rows)} problems", flush=True)
    by_dk = {d: {k: float(np.mean([r[f"k{k}"] > 0 for r in rows if r["depth"] == d])) for k in range(d)}
             for d in sorted(seen)}
    rem = {}
    for r in rows:
        for k in range(1, r["depth"]):
            rem.setdefault(r["depth"] - k, []).append(r[f"k{k}"] > 0)
    by_rem = {h: float(np.mean(v)) for h, v in sorted(rem.items())}
    base_acc = float(np.mean([r["k0"] > 0 for r in rows]))
    json.dump({"ckpt": a.ckpt, "Kp": a.Kp, "n": len(rows), "base": base_acc, "by_depth_k": by_dk,
               "by_hops_remaining": by_rem, "rows": rows}, open(a.out, "w"))
    if not a.skip_probe:
        print("base", round(base_acc, 3)); print("by hops remaining", {h: round(v, 3) for h, v in by_rem.items()})
    print("DONE", a.out)


if __name__ == "__main__":
    main()
