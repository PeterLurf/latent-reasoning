"""REDTEAM_LANG $0 CPU: the F2-1 fix prototype. SYM + ANCESTRAL DECOYS: per world, n_dec ~ U{1..2} decoy persons X, each with a fresh chain
X -> b_1 -> ... -> b_j (j ~ U{1..3}) and an edge b_j -> a_k (k ~ U{2..D}) into that world's path. Same law in both worlds (sigma-symmetric).
Checks: invariants (dist(R,o_R) = dist(P,o_P) = D, o_P not in Reach(R), o_R not in Reach(P)); the six §59 heuristics at every path level
(heur_panel.panel_for); BIDIR(1) / BIDIR(2); H-ROOT (first / random parent); H-ROOT-SOFT (full ancestor set contains R); UDIST; size.
Reuses sym_proto.world / bfs / rev_adj / name_pool (read-only import); the make() body is copied with the decoy block added.
    python -B sym_decoy.py <out.json> [--n 1000]
"""
import argparse
import collections
import json
import random
import sys
from collections import deque

sys.path.insert(0, "D:/Latent/docs/research_lang_2026-10-18/scripts")
import sym_proto as SP  # noqa: E402
from heur_panel import panel_for  # noqa: E402

NDEC = (1, 2); LEAVES = False
PN = ["Tom", "Sam", "Ann", "Bob", "Max", "Eve", "Kim", "Lou", "Ivy", "Rex", "Zoe", "Ned", "Amy", "Jay", "Liz", "Ted", "Una"]


def make_decoy(rng, D, decoy=True):
    nid = [0]
    WR = SP.world(rng, D, nid); WP = SP.world(rng, D, nid)
    edges = list(WR[1]) + list(WP[1])
    anc = lambda W: set(SP.bfs(SP.rev_adj(W[1]), [W[2][-1]]))  # noqa: E731
    aR, aP = anc(WR), anc(WP)
    for src_w, dst_w, dst_anc in ((WP, WR, aR), (WR, WP, aP)):
        tgts = [n for n in dst_w[3] if n not in dst_anc]
        for _ in range(rng.randint(0, 2)):
            if tgts:
                edges.append((rng.choice(src_w[0]), rng.choice(tgts)))
    n_extra = rng.randint(0, 2)
    dec = []                                                       # (world path, chain nodes)
    if decoy:
        for W in (WR, WP):
            for _ in range(rng.randint(NDEC[0], NDEC[1])):
                j = rng.randint(1, 3); ch = []
                for _ in range(j):
                    nid[0] += 1; ch.append(nid[0])
                k = rng.randint(2, D); dec.append((W, ch, k))
    persons = ["R", "P"] + ["X%d" % i for i in range(n_extra)] + ["Y%d" % i for i in range(len(dec))]
    pid = {p: -(i + 1) for i, p in enumerate(persons)}
    edges += [(pid["R"], c) for c in WR[4]] + [(pid["P"], c) for c in WP[4]]
    allnodes = WR[0] + WP[0]
    for p in persons[2:2 + n_extra]:
        c = nid[0] = nid[0] + 1; allnodes.append(c); edges.append((pid[p], c))
        for _ in range(rng.randint(1, 3)):
            v = nid[0] = nid[0] + 1; allnodes.append(v); edges.append((c, v))
    for i, (W, ch, k) in enumerate(dec):
        allnodes += ch; edges.append((pid["Y%d" % i], ch[0]))
        edges += [(ch[q], ch[q + 1]) for q in range(len(ch) - 1)]; edges.append((ch[-1], W[2][k - 1]))
        if LEAVES:                                                  # decoy chain nodes get the path nodes' distractor law (Binomial(2, 1/2) leaf children)
            for c_ in ch:
                for _ in range(sum(rng.random() < 0.5 for _ in range(2))):
                    v = nid[0] = nid[0] + 1; allnodes.append(v); edges.append((c_, v))
    ids = [pid[p] for p in persons] + allnodes; idx = {u: i for i, u in enumerate(ids)}
    pn = rng.sample(PN, len(persons)); cn = rng.sample(SP.NAMES, len(allnodes)); sym = pn + cn
    E = [[idx[a], idx[b]] for a, b in edges]
    stm = [("%s is a %s." if sym[a][:1].isupper() else "Every %s is a %s.") % (sym[a], sym[b]) for a, b in E]
    rng.shuffle(stm)
    oR, oP = idx[WR[2][-1]], idx[WP[2][-1]]
    o1, o2 = (oR, oP) if rng.random() < 0.5 else (oP, oR)
    x = {"question": " ".join(stm) + " Is %s a %s or %s?" % (sym[0], sym[o1], sym[o2]), "idx_to_symbol": sym, "edges": E, "root": 0, "target": oR, "neg_target": oP}
    adj = collections.defaultdict(list)
    for a, b in E:
        adj[a].append(b)
    dR, dP = SP.bfs(adj, [0]), SP.bfs(adj, [1])
    ok = dR.get(oR) == D and dP.get(oP) == D and oP not in dR and oR not in dP
    levels = [(idx[WR[2][k]], idx[WP[2][k]]) for k in range(D)]
    return x, ok, levels, adj


def h_root(x, rng):
    names = x["idx_to_symbol"]; nid = {s: i for i, s in enumerate(names)}
    par = collections.defaultdict(list)
    for s in [s.strip() for s in x["question"].rsplit(" Is ", 1)[0].split(".") if s.strip()]:
        w = s.split(); u, v = (nid[w[1]], nid[w[4]]) if w[0] == "Every" else (nid[w[0]], nid[w[3]]); par[v].append(u)
    out = {}
    od = collections.Counter(a_ for a_, _ in x["edges"])
    def maxdeg(L):
        m = max(od[u] for u in L); return rng.choice([u for u in L if od[u] == m])
    for tag, pick in (("first", lambda L: L[0]), ("random", lambda L: rng.choice(L)), ("maxdeg", maxdeg)):
        roots = []
        for o in (x["target"], x["neg_target"]):
            cur = o; k = 0
            while par[cur] and k < 500:
                cur = pick(par[cur]); k += 1
            roots.append(cur)
        hit = [roots[0] == x["root"], roots[1] == x["root"]]
        out[tag] = 1.0 if hit == [True, False] else 0.0 if hit == [False, True] else 0.5
    radj = collections.defaultdict(list)
    for a, b in x["edges"]:
        radj[b].append(a)
    hit = [x["root"] in SP.bfs(radj, [o]) for o in (x["target"], x["neg_target"])]
    out["soft"] = 1.0 if hit == [True, False] else 0.0 if hit == [False, True] else 0.5
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--D", default="3,5,6,8,10,12,16"); ap.add_argument("--ndec", default="1,2"); ap.add_argument("--leaves", type=int, default=0)
    a = ap.parse_args(); global NDEC, LEAVES; NDEC = tuple(int(v) for v in a.ndec.split(",")); LEAVES = bool(a.leaves)
    SP.NAMES = SP.name_pool(random.Random(12345)); rng = random.Random(991); hr = random.Random(992); out = {}
    for D in [int(v) for v in a.D.split(",")]:
        acc = collections.defaultdict(list); bad = 0; words = []
        for _ in range(a.n):
            x, ok, levels, adj = make_decoy(rng, D)
            bad += not ok
            for k, (u, v) in enumerate(levels):
                for h, val in panel_for(x, u, v).items():
                    acc["L%d_%s" % (k + 1, h)].append(val)
            for r in (1, 2):
                acc["bidir%d" % r].append(SP.bidir(x, adj, r, hr))
            for k_, v_ in h_root(x, hr).items():
                acc["H_ROOT_" + k_].append(v_)
            words.append(len(x["question"].split()))
        row = {"n": a.n, "invariant_fail": bad, "words_mean": round(sum(words) / len(words), 1), "words_max": max(words)}
        worst = (0, None)
        for k, v in acc.items():
            m = sum(v) / len(v); df = max(m, 1 - m); row[k] = round(df if k.startswith("L") else m, 4)
            if k.startswith("L") and df > worst[0]:
                worst = (df, k)
        row["max_heuristic"], row["max_which"] = round(worst[0], 4), worst[1]
        out[D] = row
        print("D", D, "inv_fail", bad, "max_heur %.3f (%s)" % worst, "bidir1/2 %.3f %.3f" % (row["bidir1"], row["bidir2"]),
              "H_ROOT first/random/maxdeg/soft %.3f %.3f %.3f %.3f" % (row["H_ROOT_first"], row["H_ROOT_random"], row["H_ROOT_maxdeg"], row["H_ROOT_soft"]), "words", row["words_mean"], flush=True)
    json.dump(out, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
