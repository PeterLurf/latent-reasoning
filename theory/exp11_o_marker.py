"""THEORY §21: stage-3 retrieval marker on addendum-O runs (out of sample) + same-seed P30 comparison (exploratory).

Identical code path to exp7_dev / exp7_stage_features: checkpoints every 2000 steps + curriculum advances, stage by
stage_at(), marker = mean of the exp4 retrieval shares over steps 1..K at the LAST checkpoint whose stage is 3.
Measurements are written before any O label is read; labels are joined by theory/score_o_marker.py afterwards.
    python theory/exp11_o_marker.py --out theory/results/exp11_o_marker.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp4_shares import shares  # noqa: E402
from exp7_dev import load_ckpt, stage_at, sweep_head  # noqa: E402
from rollout import probe  # noqa: E402


def end_of_stage3(run, g, br, E):
    _, advs = stage_at(run, 0)
    adv_steps = {s for s, _ in advs}
    files = sorted(os.listdir(f"runs/{run}/ckpts"))
    pick = [f for f in files if int(f[5:10]) % 2000 == 0 or int(f[5:10]) in adv_steps]
    st3 = [f for f in pick if stage_at(run, int(f[5:10]))[0] == 3]
    if not st3:
        return None
    f = st3[-1]
    m, c, step = load_ckpt(f"runs/{run}/ckpts/{f}")
    K = c["K"]
    head, _ = sweep_head(m, K, g, br, E)
    S = shares(m, K, g, br, E, head, Kp=K + 2)
    retr = [x["groups"].get("retr", 0) / max(abs(x["amp"]), 1e-6) for x in S]
    return {"step": step, "share": float(np.mean(retr[:K])), "retr": retr, "ptr": [x["ptr"] for x in S]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp11_o_marker.json")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, E = probe(256)
    runs = sorted(r for r in os.listdir("runs") if (r.startswith("o_s") or r.startswith("o_dev6_s")) and "stage1fail" not in r
                  and os.path.isdir(f"runs/{r}/ckpts"))
    for r in runs:
        pair = r.replace("o_", "p30_", 1) if r.startswith("o_s") else r.replace("o_dev6", "dev6", 1)
        for name in (r, pair):
            if name in res or not os.path.isdir(f"runs/{name}/ckpts"):
                continue
            with torch.no_grad():
                res[name] = end_of_stage3(name, g, br, E)
            json.dump(res, open(a.out, "w"), indent=1)
            v = res[name]
            print(name, "no stage-3 checkpoint" if v is None else "step %d share %.3f ptr %s" % (v["step"], v["share"], v["ptr"]), flush=True)


if __name__ == "__main__":
    main()
