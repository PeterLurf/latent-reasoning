"""LANG-VB draft, $0 CPU check: the §59 shallow-heuristic panel by depth band, on option level (as §59) and on the
chain-ANCHOR level (new: p8 items hang m-hop unbranched chains off the original target / negative, so a model can chase
each chain back m hops and apply a base cue to the anchors). Reads the data read-only; writes JSON to argv[2].
Heuristic accuracies are direction-free: max(a, 1 - a) (a cue in either direction counts), ties = 0.5.
Validation: option-level values must reproduce §59's table (prosqa_test 0.91/0.66/0.66/0.59/0.55/0.52 in its own
direction; p8_na_test ~0.50 everywhere).
    python -B heur_panel.py <data.json> <out.json>
"""
import collections
import json
import re
import sys
from collections import deque


def bfs(adj, srcs):
    d = {s: 0 for s in srcs}; q = deque(srcs)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


def panel_for(x, a, b):
    """signed scores s(o) for 'o is the TARGET' under each heuristic, for options a (true) and b (false) -> accuracy terms
    (1 / 0 / 0.5) in the heuristic's own §59 direction."""
    names = x["idx_to_symbol"]; persons = [i for i, s in enumerate(names) if s[:1].isupper()]
    adj = collections.defaultdict(list)
    for u, v in x["edges"]:
        adj[u].append(v)
    ds = [bfs(adj, [p]) for p in persons]
    dist = lambda o: min(d.get(o, 99) for d in ds)  # noqa: E731
    par = lambda o: [u for u, v in x["edges"] if v == o]  # noqa: E731
    indeg = collections.Counter(v for _, v in x["edges"])
    q = x["question"]; body = q.rsplit(" Is ", 1)[0]
    toks = re.findall(r"[A-Za-z]+", body)
    def ment(o):
        return sum(1 for t in toks if t == names[o])
    def first(o):
        return next((i for i, t in enumerate(toks) if t == names[o]), 10 ** 9)
    def last(o):
        return max((i for i, t in enumerate(toks) if t == names[o]), default=-1)
    def pin(o):
        p = par(o); return sum(indeg[u] for u in p) / len(p) if p else 0.0
    def cmp(sa, sb):            # 1 if the heuristic picks a (true), 0 if b, 0.5 tie
        return 1.0 if sa > sb else 0.0 if sa < sb else 0.5
    return {
        "proximity": cmp(dist(a), dist(b)),          # target = the option farther from every person
        "parents": cmp(len(par(a)), len(par(b))),     # target = more parents
        "mentions": cmp(ment(a), ment(b)),            # target = more mentions
        "first": cmp(-first(a), -first(b)),           # target = mentioned first
        "last": cmp(last(a), last(b)),                # target = mentioned last
        "parents_indeg": cmp(pin(a), pin(b)),         # target = parents with higher in-degree
    }


def anchors(x):
    """p8 chains: the last 2m symbols are c_1..c_m (target side) then d_1..d_m; anchors = the parents of c_1 / d_1 on the
    chain edges (edges [bt, c1], [bn, d1])."""
    m = int(x.get("ext", 0))
    if m == 0:
        return x["target"], x["neg_target"]
    L = len(x["idx_to_symbol"]); c1, d1 = L - 2 * m, L - m
    bt = [u for u, v in x["edges"] if v == c1]; bn = [u for u, v in x["edges"] if v == d1]
    bt = [u for u in bt if u < L - 2 * m]; bn = [u for u in bn if u < L - 2 * m]
    return (bt[0] if bt else None), (bn[0] if bn else None)


def main():
    data = json.load(open(sys.argv[1]))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in data:
        depth = len(x["steps"])
        r = panel_for(x, x["target"], x["neg_target"])
        for k, v in r.items():
            by[depth]["opt_" + k].append(v)
        at, an = anchors(x)
        if at is not None and an is not None and int(x.get("ext", 0)) > 0:
            ra = panel_for(x, at, an)
            for k, v in ra.items():
                by[depth]["anc_" + k].append(v)
    out = {}
    for depth in sorted(by):
        row = {"n": len(by[depth]["opt_proximity"])}
        for k, v in sorted(by[depth].items()):
            a_ = sum(v) / len(v); row[k] = round(a_, 4); row[k + "_dirfree"] = round(max(a_, 1 - a_), 4); row[k + "_n"] = len(v)
        out[depth] = row
    allrow = collections.defaultdict(list)
    for depth in by:
        for k, v in by[depth].items():
            allrow[k] += v
    out["pooled"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(allrow.items())}
    json.dump(out, open(sys.argv[2], "w", newline="\n"), indent=1)
    for depth in sorted(by):
        r = out[depth]
        print(depth, r["n"], " ".join("%s=%.3f" % (k, r[k + "_dirfree"]) for k in sorted(by[depth]) if k.startswith("opt_")))
        if any(k.startswith("anc_") for k in by[depth]):
            print("   anc", " ".join("%s=%.3f" % (k[4:], r[k + "_dirfree"]) for k in sorted(by[depth]) if k.startswith("anc_")))
    print("pooled", out["pooled"])


if __name__ == "__main__":
    main()
