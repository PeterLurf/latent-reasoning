"""THEORY §62 premise check (exploratory, CPU): are the label-free (T8) teacher's errors structured by depth?

For each Pilot L parent (runs/p30_s{H}), the T8 teacher of theory/distill_reanchor.py (episode 1 from START, then the
3 targets most attended by the parent's sweep head (exp12_reanchor.json) re-anchored, max-combined) labels 512 long-chain graphs drawn as in Pilot L's training
(latent.length_gen.long_chain_graphs, Pilot L's widths 2,4,6 by default, seed 2029). Columns: A0..A5 (distance 1-6), B0..B5 (7-12), C0..C5
(look-alikes of 7-12), D0..D5 (look-alikes of 1-6). Reported per parent: false-negative rate on A and on B, false-positive
rate on D and on C, and the seam contrast = error(d7-12) - error(d1-6) (mean of FN and FP).
    python theory/exp29_teacher_errors.py --out theory/results/exp29_teacher_errors.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
from distill_reanchor import teacher_targets  # noqa: E402
from latent.length_gen import long_chain_graphs  # noqa: E402
from rollout import load  # noqa: E402

H = [102, 109, 111, 112, 113, 115, 118, 119, 125, 126, 130, 131, 132, 135, 140, 145, 146, 147, 157, 158]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp29_teacher_errors.json")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--widths", default="2,4,6", help="Pilot L's --deep_widths default")
    ap.add_argument("--kind", default="t8", choices=["t8", "comp", "compv"], help="teacher (THEORY §63 adds comp/compv)")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    HEADS = json.load(open("theory/results/exp12_reanchor.json"))
    rng = np.random.default_rng(2029)
    g, br = long_chain_graphs(rng, a.n, [int(w) for w in a.widths.split(",")])
    bi = np.arange(a.n)
    q = torch.as_tensor(np.concatenate([g.nodes[bi, br[k]] for k in ("A", "B", "C", "D")], 1)).long()
    for s in H:
        name = f"p30_s{s}"
        if name in res:
            continue
        m, c = load(f"runs/{name}", "cpu"); m.eval()
        head = HEADS[name]["head"]                                         # the parent's sweep head, as Pilot L's --head
        y = teacher_targets(a.kind, m, c["K"], head, g, br, q, "cpu")
        w = None
        if isinstance(y, tuple):
            y, w = y
            w = w.numpy()
        y = y.numpy()
        A, B, C, D = y[:, 0:6], y[:, 6:12], y[:, 12:18], y[:, 18:24]
        o = {"fn_A": float(1 - A.mean()), "fn_B": float(1 - B.mean()), "fp_D": float(D.mean()), "fp_C": float(C.mean()),
             "fn_by_dist": [float(1 - v) for v in np.r_[A.mean(0), B.mean(0)]],
             "fp_by_dist": [float(v) for v in np.r_[D.mean(0), C.mean(0)]]}
        o["err_d1_6"] = 0.5 * (o["fn_A"] + o["fp_D"]); o["err_d7_12"] = 0.5 * (o["fn_B"] + o["fp_C"])
        o["seam_contrast"] = o["err_d7_12"] - o["err_d1_6"]
        if w is not None:                                                 # compv: error on kept labels, and coverage
            truth = np.r_[np.ones(12), np.zeros(12)][None].repeat(len(y), 0)
            near, far = np.r_[np.arange(6), np.arange(18, 24)], np.arange(6, 18)
            err = (y != truth)
            o["kept_d1_6"], o["kept_d7_12"] = float(w[:, near].mean()), float(w[:, far].mean())
            o["kept_err_d1_6"] = float((err * w)[:, near].sum() / max(w[:, near].sum(), 1))
            o["kept_err_d7_12"] = float((err * w)[:, far].sum() / max(w[:, far].sum(), 1))
        res[name] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        print("%-10s err d1-6 %.3f (FN %.3f FP %.3f) | d7-12 %.3f (FN %.3f FP %.3f) | seam %+.3f" % (
            name, o["err_d1_6"], o["fn_A"], o["fp_D"], o["err_d7_12"], o["fn_B"], o["fp_C"], o["seam_contrast"]), flush=True)


if __name__ == "__main__":
    main()
