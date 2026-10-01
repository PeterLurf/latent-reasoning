"""THEORY §59: ProsQA-NA — ProsQA with the proximity cue of §57 neutralised (no answer changes). CPU, data only.

§57: ProsQA's generator hangs the negative option off another person, so "the option one rule below a concept named in
a person statement is the distractor" is right 97 % of the time, and a person-agnostic proximity heuristic scores 0.91.
NA ("no adjacency cue") equalises the two options' distance from the nearest person statement, L = min(dt, dn, 2):
for each option farther than L it adds one statement "<P> is a <C>." with P a random non-root person and C = the option
itself (L = 1) or a random non-person parent of it (L = 2), inserted at a random sentence boundary before the question
(edge P -> C added). Problems where this is impossible are dropped and counted.
* Answers are unchanged: P is not the root, so the root's reachable set is unchanged (checked by BFS per problem).
* Then the two options' in-degrees are equalised: rules "Every <C> is a <option>." for the option with fewer parents,
  C a non-person concept the root cannot reach (neither the answer nor the target's depth changes; both asserted).
* Afterwards dt = dn and the in-degrees match in every problem, so the proximity heuristic is exactly at chance. Also
  reported, as residual-cue checks: a count heuristic ("the option with more persons at distance L is the distractor").
    python theory/prosqa_na.py prosqa/data/prosqa_test.json prosqa/p8data/prosqa_na_test.json [--seed 0]
"""
import argparse
import collections
import json
import re
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


def adjacency(edges):
    adj = collections.defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
    return adj


def person_dist(x, adj, node):
    persons = [i for i, s in enumerate(x["idx_to_symbol"]) if s[:1].isupper()]
    return min(bfs(adj, [p]).get(node, 99) for p in persons)


def neutralise(x, rng):
    names = x["idx_to_symbol"]
    persons = [i for i, s in enumerate(names) if s[:1].isupper()]
    others = [p for p in persons if p != x["root"]]
    adj = adjacency(x["edges"])
    dt, dn = person_dist(x, adj, x["target"]), person_dist(x, adj, x["neg_target"])
    L = min(dt, dn, 2)
    added = []
    for o, d in ((x["target"], dt), (x["neg_target"], dn)):
        if d == L:
            continue
        cands = [o] if L == 1 else sorted({a_ for a_, b_ in x["edges"] if b_ == o and a_ not in persons})
        if not cands or not others:
            return None, 0
        p, c = others[rng.integers(len(others))], cands[rng.integers(len(cands))]
        added.append((p, c)); adj[p].append(c)
    # in-degree balance: parent rules "Every C is a <option>" for the option with fewer parents, C a concept the root
    # cannot reach (so neither the answer nor the target's depth changes), never an option or a person
    edges = [tuple(e) for e in x["edges"]] + added
    reach = set(bfs(adjacency(edges), [x["root"]]))
    rules = []
    while True:
        it, i_n = (sum(b_ == o for _, b_ in edges) for o in (x["target"], x["neg_target"]))
        if it == i_n:
            break
        o = x["target"] if it < i_n else x["neg_target"]
        par = {a_ for a_, b_ in edges if b_ == o}
        cands = [c for c in range(len(names)) if c not in persons and c not in reach and c not in par
                 and c not in (x["target"], x["neg_target"])]
        if not cands:
            return None, 0
        c = cands[rng.integers(len(cands))]
        rules.append((c, o)); edges.append((c, o))
    y = dict(x)
    y["edges"] = [list(e) for e in edges]
    body, qtext = x["question"].rsplit(". Is ", 1)
    sents = [s + "." for s in body.split(". ")]
    for text in [f"{names[p]} is a {names[c]}." for p, c in added] + [f"Every {names[c]} is a {names[o]}." for c, o in rules]:
        k = int(rng.integers(len(sents) + 1))
        sents.insert(k, text)
    y["question"] = " ".join(sents) + " Is " + qtext
    y["na_added"] = len(added); y["na_rules"] = len(rules)
    r0 = bfs(adjacency(x["edges"]), [x["root"]]); r1 = bfs(adjacency(y["edges"]), [x["root"]])
    assert set(r0) == set(r1) and x["target"] in r1 and x["neg_target"] not in r1, "answer changed"
    assert r0[x["target"]] == r1[x["target"]], "target depth changed"
    return y, len(added)


def heuristic(data):
    """Expected accuracy of the proximity heuristic, and of the count heuristic (more persons at the minimum distance =
    the distractor; ties = coin)."""
    acc, cnt = [], []
    for x in data:
        adj = adjacency(x["edges"])
        persons = [i for i, s in enumerate(x["idx_to_symbol"]) if s[:1].isupper()]
        ds = [bfs(adj, [p_]) for p_ in persons]
        dt, dn = (min(d.get(o, 99) for d in ds) for o in (x["target"], x["neg_target"]))
        acc.append(1.0 if dn < dt else 0.0 if dn > dt else 0.5)
        L = min(dt, dn)
        ct, cn = (sum(d.get(o, 99) == L for d in ds) for o in (x["target"], x["neg_target"]))
        cnt.append(1.0 if cn > ct else 0.0 if cn < ct else 0.5)
    return float(np.mean(acc)), float(np.mean(cnt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    data = json.load(open(a.src))
    out, n_add, dropped = [], collections.Counter(), 0
    for x in data:
        y, k = neutralise(x, rng)
        if y is None:
            dropped += 1; continue
        out.append(y); n_add[k] += 1
    assert all(re.search(r"\. Is \w+ a \w+ or \w+\?$", y["question"]) for y in out)
    json.dump(out, open(a.dst, "w", newline="\n"))
    print("%s -> %s: %d problems (%d dropped) | added statements %s | proximity heuristic %.3f -> %.3f | count heuristic "
          "%.3f -> %.3f" % (a.src, a.dst, len(out), dropped, dict(sorted(n_add.items())), heuristic(data)[0], heuristic(out)[0],
                            heuristic(data)[1], heuristic(out)[1]))


if __name__ == "__main__":
    main()
