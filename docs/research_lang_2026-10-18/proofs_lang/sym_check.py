"""PROOFS_LANG item 2: ProsQA-SYM prototype re-check on FRESH draws, with my own panel (anchor_check.heur, text-parsed).
Generator under test = research7/scripts/sym_proto.make (imported; its panel / bidir code is NOT used).
Per item:
  invariants: dist(R, a_k^R) = k and dist(P, a_k^P) = k for every path level k (so depth exactly D, no shortcut edge
  to any path node), o_P not in Reach(R), o_R not in Reach(P), extra persons reach no option, every concept mentioned.
  person-agnostic panel: six §59 heuristics + option out-degree, at every path level k = 1..D (direction-free).
  ROOT-AWARE shallow heuristics (not covered by the exchangeability theorem):
    dBIDIR(r): directed r hops from R vs r hops backward from each option (the draft's BIDIR).
    uDIST(r): the option with the smaller UNDIRECTED distance from R, distances capped at r + 1 (ties 0.5).
  HOLD check: share of items with concepts at directed distance > D from R (gold REACH keeps changing after tau = D).
    python -B sym_check.py <out.json> --seed S --n N [--D 3,4,5,6,8,10,12,14,16]
"""
import argparse
import collections
import json
import os
import random
import sys
from collections import deque

sys.path.insert(0, "C:/Users/peter/AppData/Local/Temp/claude/D--/2f0034d5-4d85-51eb-880b-bab41adb0dff/scratchpad/research7/scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sym_proto as SP  # noqa: E402
from anchor_check import parse, heur  # noqa: E402


def bfs(adj, srcs, maxd=10 ** 9):
    d = {s: 0 for s in srcs}; dq = deque(srcs)
    while dq:
        u = dq.popleft()
        if d[u] >= maxd:
            continue
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1; dq.append(v)
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--D", default="3,4,5,6,8,10,12,14,16")
    a = ap.parse_args()
    SP.NAMES = SP.name_pool(random.Random(12345)); rng = random.Random(a.seed)
    tok = None
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception as e:  # noqa: BLE001
        print("no tokenizer", type(e).__name__)
    out = {"seed": a.seed, "n": a.n, "bands": {}}
    for D in [int(v) for v in a.D.split(",")]:
        acc = collections.defaultdict(list); inv = collections.Counter(); beyond = 0; maxd_extra = collections.Counter(); T = []
        for i in range(a.n):
            x, ok, levels, _adj = SP.make(rng, D)
            edges, words, (qp, o1, o2) = parse(x)
            assert sorted(edges) == sorted(tuple(e) for e in x["edges"])
            sym = x["idx_to_symbol"]; persons = [j for j, s in enumerate(sym) if s[:1].isupper()]
            adj = collections.defaultdict(list); uadj = collections.defaultdict(list); radj = collections.defaultdict(list)
            for u, v in edges:
                adj[u].append(v); radj[v].append(u); uadj[u].append(v); uadj[v].append(u)
            oR, oP = x["target"], x["neg_target"]
            dR, dP = bfs(adj, [0]), bfs(adj, [1])
            bad = []
            if any(dR.get(levels[k][0]) != k + 1 for k in range(D)):
                bad.append("R_path_depth")
            if any(dP.get(levels[k][1]) != k + 1 for k in range(D)):
                bad.append("P_path_depth")
            if oP in dR:
                bad.append("R_reaches_oP")
            if oR in dP:
                bad.append("P_reaches_oR")
            for p in persons[2:]:
                if {oR, oP} & set(bfs(adj, [p])):
                    bad.append("extra_reaches_opt")
            ment = collections.Counter(words)
            if any(ment[s] == 0 for s in sym):
                bad.append("unmentioned")
            if sym[0] != qp:
                bad.append("question_person")
            for b in bad:
                inv[b] += 1
            inv["any"] += bool(bad); inv["proto_ok_false"] += (not ok)
            md = max(dR.values()); beyond += md > D; maxd_extra[md - D] += 1
            for k, (u, v) in enumerate(levels):
                h = heur(x, edges, words, u, v)
                for name, val in h.items():
                    if name != "last_first":
                        acc["L%d_%s" % (k + 1, name)].append(val)
                ou, ov = len(adj[u]), len(adj[v])
                acc["L%d_outdeg" % (k + 1)].append(1.0 if ou > ov else 0.0 if ou < ov else 0.5)
            # root-aware
            for r in (1, 2, 3, 4):
                F = set(bfs(adj, [0], r))
                hit = [bool(F & set(bfs(radj, [o], r))) for o in (oR, oP)]
                acc["dBIDIR%d" % r].append(1.0 if hit == [True, False] else 0.0 if hit == [False, True] else 0.5)
            ud = bfs(uadj, [0])
            for r in (2, 3, 4, 5, 6, 8, 99):
                a_, b_ = min(ud.get(oR, 10 ** 6), r + 1), min(ud.get(oP, 10 ** 6), r + 1)
                acc["uDIST%d" % r].append(1.0 if a_ < b_ else 0.0 if a_ > b_ else 0.5)
            acc["udist_R_oR_minus_D"].append(ud.get(oR, 10 ** 6) - D)
            acc["udist_R_oP"].append(ud.get(oP, 10 ** 6))
            if tok is not None and i < 200:
                T.append(len(tok(x["question"])["input_ids"]))
        row = {"invariants": dict(inv), "items_with_dist_gt_D": beyond / a.n, "maxdist_minus_D": dict(sorted(maxd_extra.items()))}
        worst = (0.0, None)
        for k, v in acc.items():
            mu = sum(v) / len(v)
            if k.startswith(("dBIDIR", "uDIST", "udist")):
                row[k] = round(mu, 4); continue
            df = max(mu, 1 - mu); row[k] = round(df, 4)
            if df > worst[0]:
                worst = (df, k)
        row["max_person_agnostic"] = round(worst[0], 4); row["max_which"] = worst[1]; row["n_cells"] = sum(1 for k in acc if k.startswith("L"))
        if T:
            Ts = sorted(T); row["gpt2_tokens_first200"] = {"mean": round(sum(Ts) / len(Ts), 1), "p90": Ts[int(0.9 * len(Ts))], "max": Ts[-1]}
        out["bands"][D] = row
        print("D", D, "inv", dict(inv), "max_pa %.3f (%s, %d cells)" % (worst[0], worst[1], row["n_cells"]),
              "dBIDIR1-4", [row["dBIDIR%d" % r] for r in (1, 2, 3, 4)], "uDIST2,3,4,5,6,8,full", [row["uDIST%d" % r] for r in (2, 3, 4, 5, 6, 8, 99)],
              "dist>D %.3f" % row["items_with_dist_gt_D"], "udist(R,oR)-D %.2f" % row["udist_R_oR_minus_D"], "tok", row.get("gpt2_tokens_first200"), flush=True)
        json.dump(out, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
