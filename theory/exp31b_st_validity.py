"""THEORY §64, descriptive (post hoc): are S3 single-loop answers valid permutations past the trained length?
For gated S3 runs, N in {6, 9, 10, 12, 16} at K' = max(8, N) (512 each, seed 4000 + N): the share of answers that are
valid arrangements (no two items in one slot), and full-state accuracy among the valid ones.
Output: theory/results/exp31b_st_validity.txt (the printed lines).
    python theory/exp31b_st_validity.py > theory/results/exp31b_st_validity.txt
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
from st_model import STReasoner, sample, tt  # noqa: E402

for path in sorted(glob.glob("theory/runs/st/s3_s*")):
    ev = json.load(open(os.path.join(path, "eval_st.json")))
    if np.mean([ev["single"][str(k)] for k in range(1, 9)]) < 0.9:
        continue
    ck = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=False)
    m = STReasoner("S3"); m.load_state_dict(ck["model"]); m.eval()
    v = []
    with torch.no_grad():
        for N in (6, 9, 10, 12, 16):
            init, ops, pos, nops, final, _ = sample(np.random.default_rng(4000 + N), "S3", 512, N, N, Nmax=24)
            p = m.run(*tt("cpu", init, ops, pos, nops), max(8, N)).argmax(-1).numpy()
            valid = np.array([len(set(r)) == 3 for r in p])
            acc_valid = (p == final).all(1)[valid].mean() if valid.any() else np.nan
            v.append((N, valid.mean(), acc_valid))
    print(os.path.basename(path), " ".join("N%d valid %.2f acc|valid %.2f" % t for t in v), flush=True)
