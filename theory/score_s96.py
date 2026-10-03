"""THEORY §96 scoring (rules fixed in §96; written before any §96 teacher was trained; Theory is the only scorer).
Teachers: theory/results/c1_s96/<arm>_s<seed>.json (c1_rl.py --save_table: hard table T (M x A -> M), decoder D (M -> element)).
P1: arm mixP0.05 certified (product-automaton certificate) in >= 4/5 seeds.
Selection (label-free, the deployable rule): a teacher is SELECTED iff its final-answer accuracy at N = 128 >= 0.999.
P2: every certified teacher's decoded per-prefix states D(code_t) agree with §92's training labels (bench/data/s5/S5=128.csv,
    all 100,000 x 128 positions) at 1.000. Agreement is also reported for every selected teacher, on S5=512 and S5=2048.
Index map (ground truth only, asserted exhaustively): boxes/s5 applies op o on the right (x o o); theory/c1_group on the left
(a o s). psi(x) = c1 index of the inverse permutation turns one into the other: COMP_c1[psi(x), psi(o)] = psi(COMP_s5[x, o]).
    python theory/score_s96.py
"""
import glob
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "theory"))
from c1_group import comp_table  # noqa: E402
from st_model import group_elems  # noqa: E402


def load_csv(path):
    X, Y = [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            _, x, y = line.rstrip("\n").split(",")
            X.append(np.array(x.split(), dtype=np.int64)); Y.append(np.array(y.split(), dtype=np.int64))
    return np.stack(X), np.stack(Y)


def index_map():
    S5E = list(itertools.permutations(range(1, 6))); S5I = {e: i for i, e in enumerate(S5E)}
    COMP5 = np.array([[S5I[tuple(a[b[i] - 1] for i in range(5))] for b in S5E] for a in S5E])
    C1, e, A = comp_table("S5"); _, G = group_elems("S5"); GI = {tuple(g): i for i, g in enumerate(G)}
    psi = np.array([GI[tuple(int(j) for j in np.argsort(np.array(S5E[x]) - 1))] for x in range(120)])
    for x in range(120):
        for o in range(120):
            assert C1[psi[x], psi[o]] == psi[COMP5[x, o]], (x, o)
    assert psi[0] == e
    return psi


def agreement(T, D, psi, X, Y, chunk=20000):
    agree, tot, seq_ok = 0, 0, 0
    for i in range(0, len(X), chunk):
        x, y = psi[X[i:i + chunk]], psi[Y[i:i + chunk]]
        c = np.zeros(len(x), dtype=np.int64); ok = np.ones(len(x), dtype=bool)
        for t in range(x.shape[1]):
            c = T[c, x[:, t]]; hit = D[c] == y[:, t]
            agree += int(hit.sum()); tot += hit.size; ok &= hit
        seq_ok += int(ok.sum())
    return {"positions": tot, "agree": agree / tot, "sequences_all_correct": seq_ok / len(X)}


def main():
    psi = index_map()
    data = {n: load_csv(os.path.join(ROOT, "bench", "data", "s5", "S5=%s.csv" % n)) for n in ("128", "512", "2048")}
    res = {"teachers": {}}
    for f in sorted(glob.glob(os.path.join(ROOT, "theory", "results", "c1_s96", "*_s*.json"))):
        arm, seed = os.path.basename(f)[:-5].rsplit("_s", 1)
        r = list(json.load(open(f)).values())[0]
        T, D = np.array(r["table"]), np.array(r["dec"])
        row = {"arm": arm, "seed": int(seed), "certified": r["certified"], "bad_pairs": r["bad_pairs"],
               "acc128": r["acc"]["128"], "selected": r["acc"]["128"] >= 0.999}
        if row["certified"] or row["selected"]:
            row["agree"] = {n: agreement(T, D, psi, *data[n]) for n in data}
        res["teachers"]["%s_s%s" % (arm, seed)] = row
        print(arm, seed, {k: v for k, v in row.items() if k not in ("arm", "seed")})
    by = {}
    for row in res["teachers"].values():
        by.setdefault(row["arm"], []).append(row)
    res["arms"] = {a: {"n": len(rs), "certified": sum(r["certified"] for r in rs), "selected": sum(r["selected"] for r in rs),
                       "selected_and_agree_1": sum(r["selected"] and r.get("agree", {}).get("128", {}).get("agree") == 1.0 for r in rs)}
                   for a, rs in by.items()}
    res["P1"] = res["arms"].get("mixP0.05", {}).get("certified", 0) >= 4
    cert = [r for r in res["teachers"].values() if r["certified"]]
    res["P2"] = bool(cert) and all(r["agree"]["128"]["agree"] == 1.0 for r in cert)
    print(res["arms"], "| P1", res["P1"], "| P2", res["P2"], "(%d certified teachers)" % len(cert))
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s96.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
