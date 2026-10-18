"""PROOFS_LANG item 3 aid: in the 'direct' cross-edge variant, is every concept reachable from R within distance D
(so REACH saturates at t = D and HOLD-after-tau is exact)? Also the prototype kernel for comparison."""
import collections, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sym_variant as V
import sym_proto as SP
SP.NAMES = SP.name_pool(random.Random(12345))
res = {}
for mode in ("any", "direct"):
    rng = random.Random(31); res[mode] = {}
    for D in (3, 4, 5, 6, 8, 12, 16):
        gt = 0
        for _ in range(1000):
            x, lev = V.make(rng, D, mode)
            adj = collections.defaultdict(list)
            for a, b in x["edges"]:
                adj[a].append(b)
            d = V.bfs(adj, [0]); gt += max(d.values()) > D
        res[mode][D] = gt / 1000
    print(mode, res[mode], flush=True)
json.dump(res, open("direct_maxdist.json", "w", newline="\n"), indent=1)
