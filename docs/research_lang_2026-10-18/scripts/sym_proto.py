"""LANG-VB draft, $0 CPU prototype of ProsQA-SYM (§3.1): two worlds from one random process, symmetric cross edges into
leaf distractors, shuffled statements. Checks: invariants (dist = D both worlds; R does not reach o_P; P does not reach
o_R), the six §59 heuristics at the option level and at every path level k (a_k^R vs a_k^P), BIDIR(1) / BIDIR(2), and the
size law (statements, words, concepts). Not the implementation (Experiments writes boxes/prosqa_sym.py); a feasibility
and exchangeability check only.
    python -B sym_proto.py <out.json> [--n 1000] [--D 3,4,...]
"""
import argparse
import collections
import json
import random
import string
import sys
from collections import deque

sys.path.insert(0, "C:/Users/peter/AppData/Local/Temp/claude/D--/2f0034d5-4d85-51eb-880b-bab41adb0dff/scratchpad/research7/scripts")
from heur_panel import panel_for  # noqa: E402


def bfs(adj, srcs, maxd=10 ** 9):
    d = {s: 0 for s in srcs}; q = deque(srcs)
    while q:
        u = q.popleft()
        if d[u] >= maxd:
            continue
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


def world(rng, D, nid):
    """-> (nodes, edges, path, leaves, direct) with fresh ids from nid[0]; the person is added by the caller."""
    def new():
        nid[0] += 1; return nid[0]
    path = [new() for _ in range(D)]; edges = [(path[i], path[i + 1]) for i in range(D - 1)]
    depth = {a: i + 1 for i, a in enumerate(path)}; anchor = {}; nodes = list(path); children = collections.defaultdict(list)
    for k in range(D - 1):                                        # distractor subtrees off a_1..a_{D-1}
        for _ in range(sum(rng.random() < 0.5 for _ in range(2))):
            c = new(); nodes.append(c); edges.append((path[k], c)); children[path[k]].append(c); depth[c] = k + 2; anchor[c] = k
            frontier = [c]
            for _lvl in range(min(3, D - k - 1) - 1):
                nxt = []
                for u in frontier:
                    for _ in range(sum(rng.random() < 0.3 for _ in range(2))):
                        v = new(); nodes.append(v); edges.append((u, v)); children[u].append(v); depth[v] = depth[u] + 1; anchor[v] = k; nxt.append(v)
                frontier = nxt
    for u in [n for n in nodes if n in anchor]:                   # within-world edges u -> a_k, keeping a_k's depth (k <= depth(u) + 1) and no cycle (k > anchor)
        if rng.random() < 0.3:
            lo, hi = anchor[u] + 2, min(D, depth[u] + 1)          # path index k is 1-based: a_k = path[k-1]
            if lo <= hi:
                k = rng.randint(lo, hi); edges.append((u, path[k - 1]))
    direct = [path[0]]
    for _ in range(rng.randint(1, 4) - 1):                        # extra direct statements -> fresh small subtrees (no path edges)
        c = new(); nodes.append(c); direct.append(c)
        for _ in range(rng.randint(0, 2)):
            v = new(); nodes.append(v); edges.append((c, v))
    outdeg = collections.Counter(a for a, _ in edges)
    leaves = [n for n in nodes if outdeg[n] == 0 and n != path[-1]]
    return nodes, edges, path, leaves, direct


NAMES = None


def name_pool(rng, n=300):
    out = set(); syl = ["bo", "ra", "zu", "mi", "te", "lo", "ka", "ne", "si", "du", "ve", "pa", "go", "wi", "fe"]
    while len(out) < n:
        out.add("".join(rng.choice(syl) for _ in range(rng.randint(2, 3))) + "pus")
    return sorted(out)


def make(rng, D):
    nid = [0]
    WR = world(rng, D, nid); WP = world(rng, D, nid)
    edges = list(WR[1]) + list(WP[1])
    anc = lambda W: set(bfs(rev_adj(W[1]), [W[2][-1]]))  # noqa: E731  ancestors of the option (incl.)
    aR, aP = anc(WR), anc(WP)
    for src_w, dst_w, dst_anc in ((WP, WR, aR), (WR, WP, aP)):    # symmetric cross edges into leaf distractors only
        tgts = [n for n in dst_w[3] if n not in dst_anc]
        for _ in range(rng.randint(0, 2)):
            if tgts:
                edges.append((rng.choice(src_w[0]), rng.choice(tgts)))
    persons = ["R", "P"] + ["X%d" % i for i in range(rng.randint(0, 2))]
    pid = {p: -(i + 1) for i, p in enumerate(persons)}
    edges += [(pid["R"], c) for c in WR[4]] + [(pid["P"], c) for c in WP[4]]
    allnodes = WR[0] + WP[0]
    for p in persons[2:]:                                          # extra persons: small disjoint trees
        c = nid[0] = nid[0] + 1; allnodes.append(c); edges.append((pid[p], c))
        for _ in range(rng.randint(1, 3)):
            v = nid[0] = nid[0] + 1; allnodes.append(v); edges.append((c, v))
    # names and indices
    order = list(range(len(persons))) + [None] * 0
    ids = [pid[p] for p in persons] + allnodes; idx = {u: i for i, u in enumerate(ids)}
    pn = rng.sample(["Tom", "Sam", "Ann", "Bob", "Max", "Eve", "Kim", "Lou", "Ivy", "Rex", "Zoe", "Ned", "Amy", "Jay", "Liz", "Ted", "Una"], len(persons))
    cn = rng.sample(NAMES, len(allnodes)); sym = pn + cn
    E = [[idx[a], idx[b]] for a, b in edges]
    stm = [("%s is a %s." if sym[a][:1].isupper() else "Every %s is a %s.") % (sym[a], sym[b]) for a, b in E]
    rng.shuffle(stm)
    oR, oP = idx[WR[2][-1]], idx[WP[2][-1]]
    o1, o2 = (oR, oP) if rng.random() < 0.5 else (oP, oR)
    x = {"question": " ".join(stm) + " Is %s a %s or %s?" % (sym[0], sym[o1], sym[o2]), "idx_to_symbol": sym, "edges": E, "root": 0, "target": oR, "neg_target": oP}
    adj = collections.defaultdict(list)
    for a, b in E:
        adj[a].append(b)
    dR, dP = bfs(adj, [0]), bfs(adj, [1])
    ok = dR.get(oR) == D and dP.get(oP) == D and oP not in dR and oR not in dP
    levels = [(idx[WR[2][k]], idx[WP[2][k]]) for k in range(D)]
    return x, ok, levels, adj


def rev_adj(edges):
    r = collections.defaultdict(list)
    for a, b in edges:
        r[b].append(a)
    return r


def bidir(x, adj, r, rng):
    radj = collections.defaultdict(list)
    for a, b in x["edges"]:
        radj[b].append(a)
    F = set(bfs(adj, [x["root"]], r))
    hit = [bool(F & set(bfs(radj, [o], r))) for o in (x["target"], x["neg_target"])]
    return 1.0 if hit == [True, False] else 0.0 if hit == [False, True] else 0.5


def main():
    global NAMES
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--D", default="3,4,5,6,8,10,12,14,16"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed); NAMES = name_pool(random.Random(12345))
    out = {}
    for D in [int(v) for v in a.D.split(",")]:
        acc = collections.defaultdict(list); bad = 0; words = []; nst = []; ncon = []
        for _ in range(a.n):
            x, ok, levels, adj = make(rng, D)
            bad += not ok
            for k, (u, v) in enumerate(levels):
                for h, val in panel_for(x, u, v).items():
                    acc["L%d_%s" % (k + 1, h)].append(val)
            for r in (1, 2):
                acc["bidir%d" % r].append(bidir(x, adj, r, rng))
            words.append(len(x["question"].split())); nst.append(len(x["edges"])); ncon.append(sum(1 for s in x["idx_to_symbol"] if s[:1].islower()))
        row = {"n": a.n, "invariant_fail": bad, "words_mean": sum(words) / len(words), "words_max": max(words), "stmts_mean": sum(nst) / len(nst), "concepts_mean": sum(ncon) / len(ncon), "concepts_max": max(ncon)}
        worst = (0, None)
        for k, v in acc.items():
            m = sum(v) / len(v); df = max(m, 1 - m)
            if k.startswith("bidir"):
                row[k] = round(m, 4); continue
            row[k] = round(df, 4)
            if df > worst[0]:
                worst = (df, k)
        row["max_heuristic"], row["max_which"] = round(worst[0], 4), worst[1]
        out[D] = row
        print("D", D, "inv_fail", bad, "max_heur %.3f (%s)" % worst, "bidir1 %.3f bidir2 %.3f" % (row["bidir1"], row["bidir2"]), "words %.0f (max %d) concepts %.1f (max %d)" % (row["words_mean"], row["words_max"], row["concepts_mean"], row["concepts_max"]), flush=True)
    json.dump(out, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
