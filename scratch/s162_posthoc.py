"""THEORY §162 post-hoc (descriptive; Theory 18:2x, Audit's question: is the 'first item kept' pattern the start segment's write persisting, or x_1?).
On a kept s162 delta model (<run>/model.pt), with s162's op_probe settings (near-OLS 120-way via deltas5._fit_cls: standardised dual ridge, 3-fold CV
lambda 1e-9..1e-6 edge-widened, one-hot argmax; 8k fit / 2k score, training-distribution chains):
    start_probe {p: acc}             the START state g_0 (120-way) from the flattened memory after item p (random-start chains)
    first_item_probe {p: {random, identity}}   x_1 (lag p - 1) from the memory after item p, fit and scored on random-start chains vs chains whose
                                     start is fixed to the identity
p in {4, 8} (+ 16 when l_max >= 32).   python scratch/s162_posthoc.py <run dir> [device]   -> <run dir>/eval_s162_posthoc.json"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltas5 as D5  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.fsq120 import Family, make_batch  # noqa: E402


def main():
    d = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"; torch.set_num_threads(3); t0 = time.time()
    a = json.load(open(os.path.join(d, "args.json"))); assert a.get("arm") == "delta", "delta runs only"
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    m = D5.DeltaS5(s5.load_gpt2(dev), a.get("mem_layer", 6), a.get("mem_write", "top")).to(dev); m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev)); m.eval()
    P = [p for p in (4, 8, 16) if p <= 8 or a["l_max"] >= 32]; Lp = max(P); bs = 250; nf, ns = 8000, 2000; fam = Family("s5")

    def feats(s0, X):
        return D5.rollout(m, s0, X, dev, amp, bs, every=False, keep=tuple(P))[1]
    rng = np.random.default_rng(16210); s0f, Xf, _ = make_batch(fam, rng, nf, Lp); s0s, Xs, _ = make_batch(fam, rng, ns, Lp)
    Kf, Ks = feats(s0f, Xf), feats(s0s, Xs)
    zf, zs = np.zeros(nf, np.int64), np.zeros(ns, np.int64); Kfi, Ksi = feats(zf, Xf), feats(zs, Xs)   # the same items, start fixed to the identity
    out = {"start_probe": {}, "first_item_probe": {}, "P": P, "n_fit": nf, "n_score": ns, "model": os.path.basename(d.rstrip("/")), "l_max": a["l_max"],
           "probe_def": "near-OLS 120-way (deltas5._fit_cls; s162 op_probe settings), training-distribution chains (uniform items; random start or identity start)"}
    for p in P:
        W, b, lam = D5._fit_cls(Kf[p], s0f, dev); out["start_probe"][str(p)] = {"acc": float((D5._score(W, b, Ks[p], dev).argmax(1) == s0s).mean()), "lambda": lam}
        r_ = {}
        for nm, A_, B_ in (("random", Kf, Ks), ("identity", Kfi, Ksi)):
            W, b, _ = D5._fit_cls(A_[p], Xf[:, 0], dev); r_[nm] = float((D5._score(W, b, B_[p], dev).argmax(1) == Xs[:, 0]).mean())
        out["first_item_probe"][str(p)] = r_
        print(p, out["start_probe"][str(p)], r_, flush=True)
    out["sec"] = round(time.time() - t0); out["device"] = dev
    json.dump(out, open(os.path.join(d, "eval_s162_posthoc.json"), "w", newline="\n"), indent=1); print("S162POST", json.dumps(out)[:400], flush=True)


if __name__ == "__main__":
    main()
