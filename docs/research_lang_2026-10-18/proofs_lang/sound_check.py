"""PROOFS_LANG item 3: the SOUND verifier (P1-P4) and Lemma S-L / Corollary on SYM prototype items.
(a) fuzz: random SOUND-accepted chains (each pass adds a random subset of the concepts with an in-edge from M_{t-1}, or
    R's direct concepts) -> check M_t subset of R_t, kept label = o_R, s >= D  (Lemma S-L).
(b) counterexample 1 (SOUND state != gold reach set): the gold-path-only chain M_t = {a_1..a_t} is KEPT.
(c) counterexample 2 (Corollary vs HOLD-after-tau gold): on an item with a concept at dist D+1, the chain
    M_t = R_t minus {o_R} for t <= D, M_{D+1} = R_{D+1} is KEPT with s = D+1 and marks a dist-(D+1) concept IN at pass
    D+1; gold with HOLD after tau = D has it OUT there (E_{D+1} = R_D); unheld gold (E_t = R_t) has it IN.
    python -B sound_check.py <out.json>
"""
import collections
import json
import os
import random
import sys
from collections import deque

sys.path.insert(0, "C:/Users/peter/AppData/Local/Temp/claude/D--/2f0034d5-4d85-51eb-880b-bab41adb0dff/scratchpad/research7/scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sym_proto as SP  # noqa: E402
from anchor_check import parse  # noqa: E402


def bfs(adj, s):
    d = {s: 0}; dq = deque([s])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1; dq.append(v)
    return d


def sound(chain, stmts, R, options, persons):
    """chain = [M_1, M_2, ...] (sets of concept ids). Returns (keep, s, label). P1-P4 exactly as §3.1."""
    direct = {c for a, c in stmts if a == R}
    par = collections.defaultdict(set)
    for a, c in stmts:
        par[c].add(a)
    prev = set()
    for t, M in enumerate(chain, 1):
        if any(c in persons for c in M):
            return False, None, None
        if t == 1 and not M <= direct:
            return False, None, None                       # P1
        if not prev <= M:
            return False, None, None                       # P3
        for c in M - prev:
            if not ((par[c] & prev) or c in direct):
                return False, None, None                   # P2
        hit = [o for o in options if o in M]
        if hit:
            return True, t, hit[0]                         # P4: first pass with an option
        prev = M
    return False, None, None


def main():
    SP.NAMES = SP.name_pool(random.Random(12345)); rng = random.Random(41); crng = random.Random(42)
    out = {"fuzz": collections.Counter(), "cex1": None, "cex2": None}
    for D in (3, 4, 5, 6, 8, 10, 12):
        for _ in range(300):
            x, ok, levels, _ = SP.make(rng, D)
            edges, words, _q = parse(x)
            sym = x["idx_to_symbol"]; persons = {j for j, s in enumerate(sym) if s[:1].isupper()}
            adj = collections.defaultdict(list)
            for a, b in edges:
                adj[a].append(b)
            dR = bfs(adj, 0); oR, oP = x["target"], x["neg_target"]
            par = collections.defaultdict(set)
            for a, c in edges:
                par[c].add(a)
            direct = {c for a, c in edges if a == 0}
            for _rep in range(20):                          # (a) fuzz random sound chains
                chain = []; M = set()
                for t in range(1, 17):
                    cand = sorted(c for c in set(dR) - {0} - M if (par[c] & M) or c in direct) if t > 1 else sorted(direct)
                    add = {c for c in cand if crng.random() < 0.6}
                    if _rep % 2:                            # half the chains also get arbitrary (possibly unsound) marks
                        add |= {c for c in range(len(sym)) if c not in persons and crng.random() < 0.01}
                    M = M | add; chain.append(set(M))
                keep, s, lab = sound(chain, edges, 0, (oR, oP), persons)
                out["fuzz"]["chains"] += 1
                if keep:
                    out["fuzz"]["kept"] += 1
                    out["fuzz"]["label_wrong"] += lab != oR
                    out["fuzz"]["s_lt_D"] += s < D
                    out["fuzz"]["subset_violation"] += any(not all(dR.get(c, 99) <= t for c in Mt) for t, Mt in enumerate(chain[:s], 1))
            if out["cex1"] is None and D == 5:              # (b) gold path only
                path = [levels[k][0] for k in range(D)]
                chain = [set(path[:t]) for t in range(1, D + 1)]
                keep, s, lab = sound(chain, edges, 0, (oR, oP), persons)
                R2 = {c for c, d in dR.items() if d <= 2 and c not in persons}
                out["cex1"] = {"D": D, "keep": keep, "s": s, "label_is_target": lab == oR, "M_2_size": len(chain[1]), "R_2_size": len(R2),
                               "omitted_at_t2": sorted(sym[c] for c in R2 - chain[1])}
            far = [c for c, d in dR.items() if d == D + 1 and any(dR.get(a, 99) == D and a != oR for a in par[c])]
            if out["cex2"] is None and far:                 # (c) late stop with a dist-(D+1) mark (parent at dist D other than o_R)
                Rt = lambda t: {c for c, d in dR.items() if 1 <= d <= t and c not in persons}  # noqa: E731
                chain = [Rt(t) - {oR} for t in range(1, D + 2)] + [Rt(D + 1)]
                keep, s, lab = sound(chain, edges, 0, (oR, oP), persons)
                c = far[0]
                out["cex2"] = {"D": D, "keep": keep, "s": s, "label_is_target": lab == oR, "marked_concept": sym[c], "dist": dR[c],
                               "self_target_at_pass_D+1": "IN", "gold_HOLD_after_tau_at_D+1": "IN" if dR[c] <= D else "OUT",
                               "gold_unheld_at_D+1": "IN" if dR[c] <= D + 1 else "OUT"}
    out["fuzz"] = dict(out["fuzz"])
    print(json.dumps(out, indent=1))
    json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
