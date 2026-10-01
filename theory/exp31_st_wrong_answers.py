"""THEORY §64, descriptive (post hoc; Audit's request): what do S3 single loops answer past the trained length?
For gated S3 runs, on N = 10 and 12 inputs at K' = N (1024 each, seed 3000 + N): agreement (all items) of the loop's
answers with the arrangement after the first k operations (prefix_k, k = 0..N), after only the last k (suffix_k applied
to the init block), and the true final arrangement (= prefix_N).
    python theory/exp31_st_wrong_answers.py --out theory/results/exp31_st_wrong_answers.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_model import STReasoner, group_elems, sample, tt  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="theory/results/exp31_st_wrong_answers.json")
    a = ap.parse_args(); torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
    res = {}
    for path in sorted(glob.glob("theory/runs/st/s3_s*")):
        ev = json.load(open(os.path.join(path, "eval_st.json")))
        if np.mean([ev["single"][str(k)] for k in range(1, 9)]) < 0.90:
            continue
        ck = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=False)
        m = STReasoner(ck["args"]["group"]); m.load_state_dict(ck["model"]); m.eval()
        n, G = group_elems(ck["args"]["group"])
        name = os.path.basename(path); res[name] = {}
        for N in (10, 12):
            rng = np.random.default_rng(3000 + N)
            init, ops, pos, nops, final, _ = sample(rng, ck["args"]["group"], 1024, N, N, Nmax=24)
            pred = m.run(*tt("cpu", init, ops, pos, nops), N).argmax(-1).numpy()
            pre, suf = [], []
            for k in range(N + 1):
                sp, ss = init.copy(), init.copy()
                for b in range(len(init)):
                    s = init[b].copy()
                    for t in range(k):
                        s = np.array(G[ops[b, t]])[s]
                    sp[b] = s
                    s = init[b].copy()
                    for t in range(N - k, N):
                        s = np.array(G[ops[b, t]])[s]
                    ss[b] = s
                pre.append(float((pred == sp).all(1).mean())); suf.append(float((pred == ss).all(1).mean()))
            res[name][N] = {"prefix": pre, "suffix": suf}
            print(name, "N", N, "prefix_k", [round(v, 2) for v in pre], "| suffix_k", [round(v, 2) for v in suf], flush=True)
    json.dump(res, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
