"""PROTOCOL7 T3: P7 at step checkpoints shortly after full-depth arrival predicts final extrapolation.
Uses the frozen far_sweep_frozen_I.measure unchanged; only its model loader is pointed at a step checkpoint.
Written 2026-09-28 ~23:42, before any checkpoint below was measured.
    PYTHONPATH=. python scratch/p7s_t3.py --device cuda:0
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "scratch")
import far_sweep_frozen_I as FS                                     # noqa: E402
from latent.graphs import set_shapes                                # noqa: E402
from latent.model import LatentReasoner                             # noqa: E402

OUT = "results/p7s_t3.json"
OFFSETS = (0, 2000, 5000)
FLOOR = 0.03


def ckpt_loader(path):
    def _load(run, device):
        ck = torch.load(path, map_location=device, weights_only=False)
        c = ck["config"]
        m = LatentReasoner(c["d"], c["layers"], c["heads"], k_max=max(c["K"], 8), mode=c["mode"],
                           emb=c.get("emb", "slots"), d_id=c.get("d_id", 64), init=c.get("init", "torch"),
                           feedback=c.get("feedback", "post_ln"), step_emb=not c.get("no_step_emb", False)).to(device)
        set_shapes(c.get("shapes", "default"), c.get("p_fork", 0.5))
        m.load_state_dict(ck["model"])
        return m.eval(), c
    return _load


def full_step(run):
    rows = [json.loads(l) for l in open(f"{run}/log.jsonl") if "curriculum_advance_to" in l]
    return next((x["step"] for x in rows if x["curriculum_advance_to"] is None), None)


def population():
    pop = {}
    for f in ("results/p30.json", "results/p_l.json"):
        d = json.load(open(f))
        for r in d["primary"]:
            if glob.glob(f"runs/{r}/ckpts/step_*.pt"):
                pop[f"runs/{r}"] = {"extrap": d["rows"][r]["extrap"], "P7_final": d["rows"][r]["P7"]}
    return pop


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    pop = population()
    for run, info in pop.items():
        s_full = full_step(run)
        ck = {int(re.search(r"step_(\d+)", p).group(1)): p for p in glob.glob(f"{run}/ckpts/step_*.pt")}
        steps = sorted(ck)
        row = res.get(run, {"full_step": s_full, **info, "early": {}})
        for off in OFFSETS:
            if str(off) in row["early"]:
                continue
            tgt = s_full + off
            s = min(steps, key=lambda z: (abs(z - tgt), z))
            FS.load = ckpt_loader(ck[s])
            r = FS.measure(run, a.device, seed=2033)
            row["early"][str(off)] = {"ckpt_step": s, "P7": r["P7"], "in_peak": r["in_peak"], "e7": r["e7"]}
            print(f"{run} full {s_full} +{off} -> step {s}: P7 {r['P7']:.3f} in_peak {r['in_peak']:.3f}", flush=True)
        res[run] = row
        json.dump(res, open(OUT, "w"), indent=1)
    # score (PROTOCOL7 T3 + amendment 2)
    runs = [r for r in pop if r in res]
    y = [res[r]["extrap"] for r in runs]
    summ = {}
    for off in OFFSETS:
        e = [res[r]["early"][str(off)] for r in runs]
        p = [x["P7"] if x["in_peak"] >= FLOOR else 0.0 for x in e]
        summ[str(off)] = {"auc": auc(p, y), "below_floor": sum(x["in_peak"] < FLOOR for x in e),
                          "class_agree_final": float(np.mean([(pp >= 0.10) == (res[r]["P7_final"] >= 0.10) for pp, r in zip(p, runs)]))}
    a2 = summ["2000"]["auc"]
    dec = "supported" if a2 is not None and a2 >= 0.80 else "not supported" if a2 is not None and a2 < 0.65 else "inconclusive"
    out = {"n": len(runs), "n_extrap": int(sum(y)), "by_offset": summ, "T3_decision": dec}
    json.dump({"rows": res, "summary": out}, open("results/p7s_t3_score.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
