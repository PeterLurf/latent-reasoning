"""THEORY §57: where does Coconut GPT-2's ProsQA "adjacent option is wrong" cue come from? Data statistics only (CPU).

ProsQA problems name several persons ("Bob is a gwompus. Max is a boompus. ..."); only one is the root. For each problem:
  dt / dn = shortest distance from any person to the target / the negative option (person -> concept = 1 hop)
  adjacent = the option is one rule below a concept named in a person statement (distance 2)
Reported per file: where the negative is reachable from, P(dn < dt), adjacency rates, P(negative | exactly one option is
adjacent), and the expected accuracy of a person-agnostic proximity heuristic (answer = the option farther from every
person statement; ties = coin), overall and by answer depth.
    python theory/cue_origin.py --out theory/results/cue_origin.json prosqa/data/prosqa_train.json prosqa/p8data/p8_train.json ...
"""
import argparse
import collections
import json
from collections import deque

import numpy as np


def bfs(adj, srcs):
    d = {s: 0 for s in srcs}; q = deque(srcs)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


def stats(data):
    rows = []
    for x in data:
        persons = [i for i, s in enumerate(x["idx_to_symbol"]) if s[:1].isupper()]
        adj = collections.defaultdict(list)
        for a, b in x["edges"]:
            adj[a].append(b)
        r, t, n = x["root"], x["target"], x["neg_target"]
        dp = [bfs(adj, [p]) for p in persons]
        others = [d for p, d in zip(persons, dp) if p != r]
        rows.append((len(x["steps"]), min(d.get(t, 99) for d in dp), min(d.get(n, 99) for d in dp),
                     bfs(adj, [r]).get(n, 99) < 99, any(d.get(n, 99) < 99 for d in others)))
    A = np.array(rows, dtype=float)
    dep, dt, dn, n_root, n_other = A.T

    def summ(m):
        adjn, adjt = dn[m] == 2, dt[m] == 2
        one = adjn ^ adjt
        return {"n": int(m.sum()), "P_dn_lt_dt": float((dn[m] < dt[m]).mean()), "P_dn_eq_dt": float((dn[m] == dt[m]).mean()),
                "P_dn_gt_dt": float((dn[m] > dt[m]).mean()), "adjacent_neg": float(adjn.mean()), "adjacent_target": float(adjt.mean()),
                "P_neg_given_one_adjacent": float((adjn & ~adjt).sum() / max(one.sum(), 1)),
                "proximity_heuristic_acc": float(np.where(dn[m] < dt[m], 1.0, np.where(dn[m] > dt[m], 0.0, 0.5)).mean())}
    out = summ(np.ones(len(A), bool))
    out["neg_reachable_from_root"] = float(n_root.mean()); out["neg_reachable_from_other_person"] = float(n_other.mean())
    out["by_depth"] = {int(d): summ(dep == d) for d in sorted(set(dep))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--out", default="theory/results/cue_origin.json")
    a = ap.parse_args()
    res = {f: stats(json.load(open(f))) for f in a.files}
    json.dump(res, open(a.out, "w", newline="\n"), indent=1)
    for f, o in res.items():
        print("%-34s n %5d | neg from root %.2f, from another person %.2f | P(dn<dt) %.3f tie %.3f | adjacent neg %.3f target %.3f"
              " | P(neg | one adjacent) %.3f | proximity heuristic %.3f" % (
                  f, o["n"], o["neg_reachable_from_root"], o["neg_reachable_from_other_person"], o["P_dn_lt_dt"], o["P_dn_eq_dt"],
                  o["adjacent_neg"], o["adjacent_target"], o["P_neg_given_one_adjacent"], o["proximity_heuristic_acc"]))


if __name__ == "__main__":
    main()
