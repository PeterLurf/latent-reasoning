"""Descriptive (Audit's IDEA 1, $0 first step): §110 s3's k = 1 transition accuracy on the (state, op) transitions its
R0 short problems labelled vs never labelled (labelled set = exact replay of self110's R0 sampler and draws for the seed).
k = 1 restart from the state's text, masked fork, greedy decode; all 14,400 transitions.
    python theory/s110_unlabelled.py --seed 3 --device cpu
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.environ["BOXES_NB"] = "5"
from boxes import s5  # noqa: E402
from boxes import gen97 as G  # noqa: E402
from boxes.self103 import draw, r0_items  # noqa: E402
from boxes.self110 import Gen110  # noqa: E402


def labelled_set(seed, Xtr, Ytr, r0_steps=1200, bs=64):
    rng = np.random.default_rng(11000 + seed); drng = np.random.default_rng(seed); g = r0_items(rng, Xtr, Ytr)[0]; seen = set()
    for _ in range(r0_steps):
        _, o_, _ = draw(g, drng, bs, "cpu"); o = o_.numpy()
        if o.shape[1] == 2:
            seen.update((int(s5.COMP[0, a]), int(b)) for a, b in o)
        else:
            seen.update((0, int(a)) for a in o[:, 0])
    return seen


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=3); ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=3); ap.add_argument("--bs", type=int, default=240)
    a = ap.parse_args(); torch.set_num_threads(a.threads); dev = a.device
    Xtr, Ytr = s5.load_csv(os.path.join(ROOT, "bench", "data", "s5", "S5=128.csv"))
    lab = labelled_set(a.seed, Xtr, Ytr)
    m = Gen110(s5.load_gpt2(dev)).to(dev)
    m.load_state_dict(torch.load(os.path.join(ROOT, "boxes", "runs_s5", "s110_s%d" % a.seed, "model.pt"), map_location=dev)); m.eval()
    S = np.repeat(np.arange(120), 120); O = np.tile(np.arange(120), 120); ok = np.zeros(len(S), bool)
    for i in range(0, len(S), a.bs):
        el, _, _ = G.chunks_dec(m, m.start_from(els=S[i:i + a.bs], dev=dev), O[i:i + a.bs, None], dev, None)
        ok[i:i + a.bs] = el[:, 0] == s5.COMP[S[i:i + a.bs], O[i:i + a.bs]]
    is_lab = np.array([(int(s), int(o)) in lab for s, o in zip(S, O)])
    res = {"seed": a.seed, "labelled_share": float(is_lab.mean()), "acc_labelled": float(ok[is_lab].mean()),
           "acc_unlabelled": float(ok[~is_lab].mean()), "acc_all": float(ok.mean()),
           "n_unlabelled_wrong": int((~ok & ~is_lab).sum()), "n_labelled_wrong": int((~ok & is_lab).sum())}
    print(res)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "s110_unlabelled_s%d.json" % a.seed), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
