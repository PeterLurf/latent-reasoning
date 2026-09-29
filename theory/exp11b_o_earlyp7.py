"""THEORY §21 PO3: T3-style early P7 on addendum-O runs (frozen far_sweep_frozen_I copy in theory/frozen/, unchanged;
only its loader is pointed at a step checkpoint, as in scratch/p7s_t3.py). Checkpoint nearest full-depth arrival + 2000.
    python theory/exp11b_o_earlyp7.py --out theory/results/exp11b_o_earlyp7.json
"""
import argparse
import glob
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import far_sweep_frozen_I as FS  # noqa: E402
from latent.graphs import set_shapes  # noqa: E402
from latent.model import LatentReasoner  # noqa: E402

torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp11b_o_earlyp7.json")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    runs = sorted(r for r in os.listdir("runs") if (r.startswith("o_s") or r.startswith("o_dev6_s")) and "stage1fail" not in r
                  and os.path.isdir(f"runs/{r}/ckpts"))
    for r in runs:
        if r in res:
            continue
        run = f"runs/{r}"
        s_full = full_step(run)
        if s_full is None:
            res[r] = None
            continue
        ck = {int(re.search(r"step_(\d+)", p).group(1)): p for p in glob.glob(f"{run}/ckpts/step_*.pt")}
        tgt = s_full + 2000
        s = min(sorted(ck), key=lambda z: (abs(z - tgt), z))
        FS.load = ckpt_loader(ck[s])
        with torch.no_grad():
            m = FS.measure(run, "cpu", seed=2033)
        p7 = m["P7"] if m["in_peak"] >= FLOOR else 0.0
        res[r] = {"full_step": s_full, "ckpt_step": s, "P7_early": p7, "in_peak": m["in_peak"], "e7": m["e7"]}
        json.dump(res, open(a.out, "w"), indent=1)
        print(r, "full", s_full, "ckpt", s, "P7_early %.3f in_peak %.4f" % (p7, m["in_peak"]), flush=True)


if __name__ == "__main__":
    main()
