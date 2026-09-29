"""Exploratory exp7: when during training does the pointer become retrieved (vs local)?

For each model and each saved checkpoint (every 2000 steps + curriculum advances): curriculum stage, the sweep head
(largest layer-1 attention excess on chain A over steps 1..K), the pointer trajectory h_0..h_{K} (exp2 membership), and
the retrieval / local shares per step (exp4 split). Probe: long-chain graphs, seed 4247, n = 256.
    python theory/exp7_dev.py --out theory/results/exp7_dev.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp4_shares import shares  # noqa: E402
from rollout import probe, rollout  # noqa: E402
from latent.graphs import set_shapes  # noqa: E402
from latent.model import LatentReasoner  # noqa: E402


def load_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = LatentReasoner(c["d"], c["layers"], c["heads"], k_max=max(c["K"], 8), mode=c["mode"], emb=c.get("emb", "slots"),
                       d_id=c.get("d_id", 64), init=c.get("init", "torch"), feedback=c.get("feedback", "post_ln"),
                       step_emb=not c.get("no_step_emb", False))
    set_shapes(c.get("shapes", "default"), c.get("p_fork", 0.5))
    m.load_state_dict(ck["model"])
    return m.eval(), c, ck.get("step")


@torch.no_grad()
def sweep_head(m, K, g, br, E):
    T = lambda a: torch.as_tensor(a)
    out = rollout(m, K, T(g.src), T(g.dst), T(g.start), K)
    n = len(g); bi = np.arange(n)[:, None]
    reach = np.concatenate([E["A"][:, 1:], E["B"]], 1); decoy = np.concatenate([E["D"][:, 1:], E["C"]], 1)
    H = out["attn1"][0].shape[1]
    acc = np.zeros(H)
    for k in range(K):
        a = out["attn1"][k].numpy()
        for h in range(H):
            acc[h] += np.clip((a[bi, h, reach] - a[bi, h, decoy]).mean(0), 0, None).sum()
    return int(np.argmax(acc)), float(acc.max())


def stage_at(run, step):
    adv = [json.loads(l) for l in open(f"runs/{run}/log.jsonl") if "curriculum_advance_to" in l]
    st = 1
    for x in adv:
        if x["step"] <= step:
            st = "full" if x["curriculum_advance_to"] is None else x["curriculum_advance_to"]
    return st, [(x["step"], x["curriculum_advance_to"]) for x in adv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp7_dev.json")
    ap.add_argument("--runs", nargs="*")
    a = ap.parse_args()
    F = json.load(open("theory/results/share_features.json"))
    runs = a.runs or [n for n in F if os.path.isdir(f"runs/{n}/ckpts") and (
        (not F[n]["extrap"] and (F[n]["retr_max_tr"] >= 0.5 or F[n]["retr_K"] >= 0.5)) or (F[n]["extrap"] and F[n]["retr_max_tr"] < 0.5))]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, E = probe(256)
    for r in runs:
        if r in res:
            continue
        _, advs = stage_at(r, 0)
        adv_steps = {s for s, _ in advs}
        files = sorted(os.listdir(f"runs/{r}/ckpts"))
        pick = [f for f in files if int(f[5:10]) % 2000 == 0 or int(f[5:10]) in adv_steps]
        rows = []
        for f in pick:
            m, c, step = load_ckpt(f"runs/{r}/ckpts/{f}")
            K = c["K"]
            head, strength = sweep_head(m, K, g, br, E)
            S = shares(m, K, g, br, E, head, Kp=K + 2)
            rows.append({"step": step, "stage": stage_at(r, step)[0], "head": head, "sweep": strength,
                         "ptr": [x["ptr"] for x in S], "amp": [x["amp"] for x in S],
                         "retr": [x["groups"].get("retr", 0) / max(abs(x["amp"]), 1e-6) for x in S],
                         "local": [x["groups"].get("local", 0) / max(abs(x["amp"]), 1e-6) for x in S]})
        res[r] = {"extrap": F[r]["extrap"], "K": K, "advances": advs, "rows": rows}
        json.dump(res, open(a.out, "w"))
        print(r, "EXT" if F[r]["extrap"] else "   ", " | ".join("%s@%s r%.2f" % (x["step"], x["stage"], float(np.mean(x["retr"][:K]))) for x in rows), flush=True)


if __name__ == "__main__":
    main()
