"""PROOFS_LANG item 1: independent re-implementation of the §57 proximity heuristic and the §59 residual-cue panel,
at the option level and at the p8 chain-ANCHOR level. Written from §57 / §59's text and make_long.py / prosqa_na.py's
data layout only; shares no code with heur_panel.py.

Graph is re-parsed from the premise TEXT (not x["edges"]); edges are cross-checked against x["edges"].
Anchors are found two independent ways and must agree:
  (A) edge-list position: make_long appends the 2m chain edges right after the original edges, and neutralise()
      appends na_added person edges and na_rules rules after that, so chain edges sit at
      E0 = len(edges) - 2m - na_added - na_rules ... E0 + 2m - 1, with edges[E0] = [bt, c_1], edges[E0+m] = [bn, d_1].
  (B) text structure: walk back from each option along concept parents that are chain nodes (last 2m symbols); the
      first non-chain concept parent of c_1 / d_1 that is not a person (m >= 2: unique; m = 1: the root-reachable one
      for the target side, edge order for the negative side is NOT used -> reported as ambiguous count).
Heuristics (direction as §57 / §59 state it; also reported direction-free max(a, 1-a)); ties = 0.5:
  proximity: target = the option FARTHER from every person (min BFS distance over all persons) [§57]
  parents: target = more parents; mentions: target = more mentions in the premises; first: target = first mentioned
  earlier; last: target = last mention later; last_first: target = first mention later (alt. reading of 'mentioned last');
  pindeg: target = parents with higher mean in-degree.
    python -B anchor_check.py <data.json> <out.json>
"""
import collections
import json
import re
import sys
from collections import deque

STMT = re.compile(r"^(?:Every (\w+)|([A-Z]\w*)) is an? (\w+)\.$")


def parse(x):
    body, q = x["question"].rsplit(" Is ", 1)
    sents = [s.strip() for s in re.split(r"(?<=\.)\s+", body.strip()) if s.strip()]
    sym = x["idx_to_symbol"]; ix = {s: i for i, s in enumerate(sym)}
    edges, words = [], []
    for s in sents:
        m = STMT.match(s)
        assert m, s
        a = m.group(1) or m.group(2); b = m.group(3)
        edges.append((ix[a], ix[b])); words += [a, b]
    qm = re.match(r"^(\w+) an? (\w+) or (\w+)\?$", q)
    assert qm, q
    return edges, words, qm.groups()


def bfs(adj, srcs):
    d = {s: 0 for s in srcs}; dq = deque(srcs)
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1; dq.append(v)
    return d


def heur(x, edges, words, a, b):
    sym = x["idx_to_symbol"]; persons = [i for i, s in enumerate(sym) if s[:1].isupper()]
    adj = collections.defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
    dist = {}
    for p in persons:
        for k, v in bfs(adj, [p]).items():
            dist[k] = min(dist.get(k, 10 ** 6), v)
    par = collections.defaultdict(list); indeg = collections.Counter()
    for u, v in edges:
        par[v].append(u); indeg[v] += 1
    pos = collections.defaultdict(list)
    for i, w in enumerate(words):
        pos[w].append(i)

    def c(sa, sb):
        return 1.0 if sa > sb else 0.0 if sa < sb else 0.5
    na, nb = sym[a], sym[b]
    pin = lambda o: (sum(indeg[u] for u in par[o]) / len(par[o])) if par[o] else 0.0  # noqa: E731
    return {
        "proximity": c(dist.get(a, 10 ** 6), dist.get(b, 10 ** 6)),
        "parents": c(len(par[a]), len(par[b])),
        "mentions": c(len(pos[na]), len(pos[nb])),
        "first": c(-min(pos[na], default=10 ** 9), -min(pos[nb], default=10 ** 9)),
        "last": c(max(pos[na], default=-1), max(pos[nb], default=-1)),
        "last_first": c(min(pos[na], default=10 ** 9), min(pos[nb], default=10 ** 9)),
        "pindeg": c(pin(a), pin(b)),
    }


def anchors(x, edges_txt):
    m = int(x.get("ext", 0)); sym = x["idx_to_symbol"]; L = len(sym)
    E = [tuple(e) for e in x["edges"]]
    E0 = len(E) - 2 * m - int(x.get("na_added", 0)) - int(x.get("na_rules", 0))
    c1, d1 = L - 2 * m, L - m
    A = (E[E0][0], E[E0 + m][0]); assert E[E0][1] == c1 and E[E0 + m][1] == d1, (E[E0], E[E0 + m], c1, d1)
    persons = {i for i, s in enumerate(sym) if s[:1].isupper()}; chain = set(range(L - 2 * m, L))
    adj = collections.defaultdict(list)
    for u, v in edges_txt:
        adj[u].append(v)
    reach = set(bfs(adj, [x["root"]]))
    par = collections.defaultdict(set)
    for u, v in edges_txt:
        par[v].add(u)
    bt = [u for u in par[c1] if u not in persons and u not in chain]
    bn = [u for u in par[d1] if u not in persons and u not in chain]
    if m == 1:
        bt = [u for u in bt if u in reach]
    amb = len(bt) != 1 or len(bn) != 1
    B = (bt[0] if len(bt) == 1 else None, bn[0] if len(bn) == 1 else None)
    return A, B, amb


def main():
    data = json.load(open(sys.argv[1]))
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    stats = collections.Counter(); strata = collections.defaultdict(collections.Counter)
    for x in data:
        edges, words, (qp, o1, o2) = parse(x)
        assert sorted(edges) == sorted(tuple(e) for e in x["edges"]), "text/edges mismatch"
        stats["items"] += 1
        D = len(x["steps"]); m = int(x.get("ext", 0))
        for k, v in heur(x, edges, words, x["target"], x["neg_target"]).items():
            rows[D]["opt_" + k].append(v)
        if m >= 1:
            A, B, amb = anchors(x, edges)
            stats["anchor_amb_m%d" % min(m, 2)] += amb
            if not amb:
                stats["anchor_agree"] += (A == B); stats["anchor_cmp"] += 1
            for k, v in heur(x, edges, words, A[0], A[1]).items():
                rows[D]["anc_" + k].append(v)
            if m >= 2:
                pv = heur(x, edges, words, A[0], A[1])["proximity"]
                strata[D]["pro" if pv == 1.0 else "anti" if pv == 0.0 else "tie"] += 1
    out = {"file": sys.argv[1], "stats": dict(stats), "by_depth": {}, "strata_m2": {d: dict(c) for d, c in sorted(strata.items())}}
    for D in sorted(rows):
        r = {}
        for k, v in sorted(rows[D].items()):
            a = sum(v) / len(v); r[k] = round(a, 4); r[k + "_df"] = round(max(a, 1 - a), 4)
        r["n_opt"] = len(rows[D]["opt_proximity"]); r["n_anc"] = len(rows[D].get("anc_proximity", []))
        out["by_depth"][D] = r
        print(D, "n", r["n_opt"], r["n_anc"], "opt prox %.3f" % r["opt_proximity"],
              ("anc prox %.3f ment_df %.3f par_df %.3f pindeg_df %.3f" % (r["anc_proximity"], r["anc_mentions_df"], r["anc_parents_df"], r["anc_pindeg_df"])) if r["n_anc"] else "",
              "| opt max_df %.3f" % max(r[k] for k in r if k.startswith("opt_") and k.endswith("_df")))
    pooled = collections.defaultdict(list)
    for D in rows:
        for k, v in rows[D].items():
            pooled[k] += v
    out["pooled"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(pooled.items())}
    tot = collections.Counter()
    for d, c in strata.items():
        if d >= 5:
            tot.update(c)
    out["strata_m2_depth_ge5"] = dict(tot)
    print("stats", dict(stats)); print("pooled", out["pooled"]); print("strata m>=2, depth>=5", dict(tot))
    json.dump(out, open(sys.argv[2], "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
