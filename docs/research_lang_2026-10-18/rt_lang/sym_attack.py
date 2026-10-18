"""REDTEAM_LANG $0 CPU: adversarial heuristics on the LANG draft's ProsQA-SYM prototype (docs/research_lang_2026-10-18/scripts/sym_proto.py,
imported read-only). Per D, on fresh draws (seed 777, disjoint from the draft's seed 0):
  H-ROOT   any-parent root chase: from each option follow ONE parent (first in statement order / a random one) until a node with no parent;
           pick the option whose root is the asked person R. Exact iff every ancestor path of o_R ends at R and of o_P at P.
  H-ROOT-SOFT  the soft (set-valued) backward chase: the full ancestor set of each option; pick the option whose ancestor set contains R.
           Reports the ancestor-set size by backward hop r (the breadth a soft-attention backward chase must carry) vs the forward
           frontier/reach size from R (the breadth of the supervised forward REACH computation).
  H-UBIDIR(r) undirected bidirectional ball heuristic (R's undirected r-ball meets the option's undirected r-ball), direction-free.
  H-UDIST  undirected distance from R: pick the nearer option (direction-free).
  backward chain-likeness: share of option ancestors (excluding persons) with exactly one parent; mean parents per ancestor.
  L-RR frontier sizes at D 9-12: max frontier |R_t \\ R_{t-1}| over t (the re-rooted input has that many 'R is a f.' statements; training
           had r ~ U{1..4} direct statements).
  depth / size cue: per-world concept count by D (with the draft's cap 2D+6 NOT applied in the prototype).
    python -B sym_attack.py <out.json> [--n 1000]
"""
import argparse
import collections
import json
import random
import sys
from collections import deque

sys.path.insert(0, "D:/Latent/docs/research_lang_2026-10-18/scripts")
import sym_proto as SP  # noqa: E402


def bfs_layers(adj, src):
    d = {src: 0}; q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


def undirected(E):
    u = collections.defaultdict(list)
    for a, b in E:
        u[a].append(b); u[b].append(a)
    return u


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--D", default="3,4,5,6,8,9,10,11,12,14,16"); a = ap.parse_args()
    SP.NAMES = SP.name_pool(random.Random(12345)); rng = random.Random(777); hr = random.Random(778); out = {}
    for D in [int(v) for v in a.D.split(",")]:
        acc = collections.defaultdict(list); anc_sz = collections.defaultdict(list); fwd_sz = collections.defaultdict(list)
        one_par = []; npar = []; chase_len = []; front_max = []; wsize = []
        for _ in range(a.n):
            x, ok, levels, adj = SP.make(rng, D)
            assert ok
            E = x["edges"]; R, oR, oP = x["root"], x["target"], x["neg_target"]
            persons = [i for i, s in enumerate(x["idx_to_symbol"]) if s[:1].isupper()]
            P = 1
            # statement order of edges: x["question"] is shuffled text; recover parent order from the text
            names = x["idx_to_symbol"]; body = x["question"].rsplit(" Is ", 1)[0]
            sents = [s.strip() for s in body.split(".") if s.strip()]
            nid = {s: i for i, s in enumerate(names)}
            par_order = collections.defaultdict(list)
            for s in sents:
                w = s.split()
                u, v = (nid[w[1]], nid[w[4]]) if w[0] == "Every" else (nid[w[0]], nid[w[3]])
                par_order[v].append(u)
            radj = collections.defaultdict(list)
            for u, v in E:
                radj[v].append(u)
            # H-ROOT (first parent in text order) and random parent
            def root(o, pick):
                cur, k = o, 0
                while par_order[cur] and k < 500:
                    cur = pick(par_order[cur]); k += 1
                return cur, k
            for tag, pick in (("first", lambda L: L[0]), ("random", lambda L: hr.choice(L))):
                (rR, kR), (rP, _) = root(oR, pick), root(oP, pick)
                hit = [rR == R, rP == R]
                acc["H_ROOT_" + tag].append(1.0 if hit == [True, False] else 0.0 if hit == [False, True] else 0.5)
                if tag == "first":
                    chase_len.append(kR)
            # H-ROOT-SOFT: full ancestor sets
            AR, AP = bfs_layers(radj, oR), bfs_layers(radj, oP)
            hit = [R in AR, R in AP]
            acc["H_ROOT_SOFT"].append(1.0 if hit == [True, False] else 0.0 if hit == [False, True] else 0.5)
            for r in range(1, D + 1):
                anc_sz[r].append(sum(1 for v, d in AR.items() if 1 <= d <= r))
            dR = bfs_layers(adj, R)
            for r in range(1, D + 1):
                fwd_sz[r].append(sum(1 for v, d in dR.items() if 1 <= d <= r))
            anc = [v for v in AR if v != oR and v not in persons]
            one_par.extend(1.0 if len(radj[v]) == 1 else 0.0 for v in anc); npar.extend(len(radj[v]) for v in anc)
            # undirected heuristics
            U = undirected(E); dU = bfs_layers(U, R); dUo = {o: bfs_layers(U, o) for o in (oR, oP)}
            for r in (1, 2, 3):
                ball = {v for v, d in dU.items() if d <= r}
                hit = [bool(ball & {v for v, d in dUo[o].items() if d <= r}) for o in (oR, oP)]
                acc["H_UBIDIR%d" % r].append(1.0 if hit == [True, False] else 0.0 if hit == [False, True] else 0.5)
            du = [dU.get(oR, 99), dU.get(oP, 99)]
            acc["H_UDIST_near"].append(1.0 if du[0] < du[1] else 0.0 if du[0] > du[1] else 0.5)
            # L-RR frontier
            layers = collections.Counter(d for v, d in dR.items() if d >= 1 and v not in persons)
            front_max.append(max(layers.values()) if layers else 0)
            wsize.append(sum(1 for s in names if s[:1].islower()) / 2)
        row = {"n": a.n}
        for k, v in acc.items():
            m = sum(v) / len(v); row[k] = round(m, 4); row[k + "_dirfree"] = round(max(m, 1 - m), 4)
        row["root_chase_len_mean"] = round(sum(chase_len) / len(chase_len), 2)
        row["anc_set_size_by_r"] = {r: round(sum(v) / len(v), 2) for r, v in anc_sz.items()}
        row["fwd_reach_size_by_r"] = {r: round(sum(v) / len(v), 2) for r, v in fwd_sz.items()}
        row["anc_one_parent_share"] = round(sum(one_par) / len(one_par), 4); row["anc_mean_parents"] = round(sum(npar) / len(npar), 3)
        row["frontier_max_mean"] = round(sum(front_max) / len(front_max), 2); row["frontier_max_gt4_share"] = round(sum(f > 4 for f in front_max) / len(front_max), 4)
        row["concepts_per_world_mean"] = round(sum(wsize) / len(wsize), 1)
        out[D] = row
        print("D", D, {k: row[k] for k in ("H_ROOT_first", "H_ROOT_random", "H_ROOT_SOFT", "H_UBIDIR1_dirfree", "H_UBIDIR2_dirfree", "H_UBIDIR3_dirfree",
                                           "H_UDIST_near_dirfree", "root_chase_len_mean", "anc_one_parent_share", "anc_mean_parents", "frontier_max_mean",
                                           "frontier_max_gt4_share")},
              "anc r=2/4/D", row["anc_set_size_by_r"].get(2), row["anc_set_size_by_r"].get(4), row["anc_set_size_by_r"][D],
              "fwd r=2/4/D", row["fwd_reach_size_by_r"].get(2), row["fwd_reach_size_by_r"].get(4), row["fwd_reach_size_by_r"][D], flush=True)
    json.dump(out, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
