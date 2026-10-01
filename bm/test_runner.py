"""Plumbing test for bm/run.py's batched runner (THEORY §60): replace the learned leaf encoder and merger by exact
Boolean operators that see ONLY the same inputs the model sees (slots, local ids, wiring, child tables). If the
runner, slot permutations, local ids and the canonical root remap are right, probe and entry accuracy are 1.000.
    python bm/test_runner.py [pool dir]"""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bm.data import MAXP, PATTERNS, row_mask  # noqa: E402
from bm.run import build_batch, forward  # noqa: E402

ROW = {p: r for r, p in enumerate(PATTERNS)}


class Exact:
    binarise = False

    def leaves(self, su, sv, rm, cm):
        out = torch.zeros(len(su), 16, MAXP)
        for i, (u, v) in enumerate(zip(su.tolist(), sv.tolist())):
            for r, pat in enumerate(PATTERNS):
                if not rm[i, r]:
                    continue
                for j in range(MAXP):
                    if cm[i, j]:
                        out[i, r, j] = float(pat[j] == 1 or (u < MAXP and pat[u] == 1 and j == v))
        return out

    def merge(self, TA, TB, idA, idB, idP, rmA, rmB, rmP, cmP, wire):
        out = torch.zeros(TA.shape[0], 16, MAXP)
        for i in range(TA.shape[0]):
            ia, ib, ip = idA[i].tolist(), idB[i].tolist(), idP[i].tolist()
            def app(T, ids, cur):
                pat = tuple(int(ids[s] in cur) if ids[s] != 8 else 0 for s in range(MAXP))
                return {ids[s] for s in range(MAXP) if ids[s] != 8 and T[ROW[pat], s] > 0.5}
            for r, pat in enumerate(PATTERNS):
                if not rmP[i, r]:
                    continue
                cur = {ip[s] for s in range(MAXP) if pat[s] and ip[s] != 8}
                while True:
                    new = cur | app(TA[i], ia, cur) | app(TB[i], ib, cur)
                    if new == cur:
                        break
                    cur = new
                for j in range(MAXP):
                    if cmP[i, j]:
                        out[i, r, j] = float(ip[j] in cur)
        return out


if __name__ == "__main__":
    pool = sys.argv[1] if len(sys.argv) > 1 else "bm/data_pool"
    rng = np.random.default_rng(0)
    for name in ("dev", "ev_orders"):
        tasks = pickle.load(open(os.path.join(pool, name + ".pkl"), "rb"))[:300]
        bb = build_batch(tasks, rng)
        R, _ = forward(Exact(), bb, "cpu")
        R = R.numpy()
        probe = np.mean([(R[i, t["row"], t["col"]] > 0.5) == (t["label"] > 0.5) for i, t in enumerate(tasks)])
        ent = np.mean([np.mean(((R[i] > 0.5) == (np.array(t["table"]) > 0.5))[row_mask(len(t["pinned"]))[0]][:, row_mask(len(t["pinned"]))[1]]) for i, t in enumerate(tasks)])
        print(f"{name}: runner with exact operators: probe {probe:.3f} entry {ent:.3f}")
