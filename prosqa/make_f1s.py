"""THEORY §61 (F1-S): question generator over ProsQA-NA.

    python prosqa/make_f1s.py            # writes prosqa/p8data/f1s_train.json, f1s_test_shallow.json; prints the depth mix
Training: p8_na_train problems of depth <= 6, split by a fixed seeded permutation into 1/3 shallow and 2/3 deep instances;
each instance contributes exactly ONE question (asserted: no instance id in both sets).
* Deep instance: its own NA question (target vs its negative), depth = len(steps).
* Shallow instance: "Is <root> a <A> or <B>?" with T a concept at BFS distance 1 or 2 from the root (half each) and
  N = the instance's unreachable negative, in random order; the steps are the root -> T path statements. Then
  theory/prosqa_na.py's neutralise() is re-run on (T, N), and the answer is asserted by BFS (T reachable, N not).
Test (shallow, held out): built the same way on p8_na_test instances, 100 per distance 1 and 2.
"""
import json
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "theory"))
from prosqa_na import neutralise  # noqa: E402


def bfs_tree(edges, root):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    d, par = {root: 0}, {root: None}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in d:
                d[v] = d[u] + 1; par[v] = u; q.append(v)
    return d, par


def shallow(x, dist, rng):
    """A shallow question on instance x with the target at BFS distance `dist`, neutralised; None if impossible."""
    sym = x["idx_to_symbol"]; root = x["root"]; neg = x["neg_target"]
    d, par = bfs_tree(x["edges"], root)
    cands = [v for v, dv in d.items() if dv == dist and sym[v][:1].islower() and v != neg]
    if not cands:
        return None
    T = int(rng.choice(cands))
    path = [T]
    while par[path[-1]] != root:
        path.append(par[path[-1]])
    path.append(root); path = path[::-1]                               # root -> ... -> T
    def stmt(a, b):
        art = "an" if sym[b][:1] in "aeiou" else "a"
        return f"{sym[a]} is {art} {sym[b]}." if sym[a][:1].isupper() else f"Every {sym[a]} is {art} {sym[b]}."
    body = x["question"].rsplit(". Is ", 1)[0] + "."
    sents = [s if s.endswith(".") else s + "." for s in body.split(". ")]
    steps = [stmt(path[i], path[i + 1]) for i in range(len(path) - 1)]
    if not all(s in sents for s in steps):
        return None
    A, B = (T, neg) if rng.random() < 0.5 else (neg, T)
    q = {**x, "target": T, "steps": steps, "answer": f"{sym[root]} is {'an' if sym[T][:1] in 'aeiou' else 'a'} {sym[T]}.",
         "question": " ".join(sents) + f" Is {sym[root]} a {sym[A]} or {sym[B]}?"}
    y, _ = neutralise(q, rng)
    if y is None:
        return None
    dd, _ = bfs_tree(y["edges"], root)
    assert T in dd and neg not in dd and dd[T] == dist, "shallow answer / depth wrong"
    y["f1s_kind"] = "shallow"; y["f1s_dist"] = dist
    return y


def main():
    rng = np.random.default_rng(61)
    tr = [x for x in json.load(open(os.path.join(HERE, "p8data", "p8_na_train.json"))) if len(x["steps"]) <= 6]
    perm = rng.permutation(len(tr)); ns = len(tr) // 3
    sh_ids, dp_ids = set(perm[:ns].tolist()), set(perm[ns:].tolist())
    assert not sh_ids & dp_ids and len(sh_ids) + len(dp_ids) == len(tr)
    out, fail = [], 0
    for k, i in enumerate(sorted(sh_ids)):
        y = shallow(tr[i], 1 + (k % 2), rng)
        if y is None:
            fail += 1; continue
        y["f1s_id"] = int(i); out.append(y)
    for i in sorted(dp_ids):
        out.append({**tr[i], "f1s_kind": "deep", "f1s_id": int(i)})
    ids = [y["f1s_id"] for y in out]
    assert len(ids) == len(set(ids)), "an instance contributes more than one question"
    mix = {}
    for y in out:
        mix[len(y["steps"])] = mix.get(len(y["steps"]), 0) + 1
    json.dump(out, open(os.path.join(HERE, "p8data", "f1s_train.json"), "w", newline="\n"))
    print("train questions", len(out), "shallow failures", fail, "depth mix", dict(sorted(mix.items())))
    te = json.load(open(os.path.join(HERE, "p8data", "p8_na_test.json")))
    rng2 = np.random.default_rng(6161); tst, k = [], 0
    for i in rng2.permutation(len(te)):
        if len(tst) >= 200:
            break
        y = shallow(te[int(i)], 1 + (k % 2), rng2)
        if y is not None:
            y["f1s_id"] = int(i); tst.append(y); k += 1
    json.dump(tst, open(os.path.join(HERE, "p8data", "f1s_test_shallow.json"), "w", newline="\n"))
    print("shallow test", len(tst), {d: sum(len(y["steps"]) == d for y in tst) for d in (1, 2)})


if __name__ == "__main__":
    main()
