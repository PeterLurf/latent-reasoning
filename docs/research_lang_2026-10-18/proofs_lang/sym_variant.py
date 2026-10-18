"""PROOFS_LANG item 2b: cross-edge variants of the SYM prototype and the root-aware UNDIRECTED panel.
Worlds come from sym_proto.world (imported, unchanged); only the cross-edge kernel differs:
  any        : the prototype (source = ANY node of the source world, incl. path nodes and the option; target = a leaf)
  distractor : the draft spec, "between distractor nodes only" (source = a non-path node; target = a leaf)
  direct     : both ends in the persons' extra direct-statement subtrees (depth 1-2, joined to the world only via the
               person), so every undirected R -> o_P path has length >= D + 3
Kernel is the same in both directions (c ~ U{0..2}), so every variant stays sigma-symmetric.
Reported per D: invariants, person-agnostic panel max (my heur), dBIDIR(r), uBIDIR(r) = undirected r-balls of R and of
each option intersect (= undirected dist <= 2r) for r = 1..D, and GPT-2 tokens T(tpn) for concept names of tpn tokens.
    python -B sym_variant.py <out.json> --mode any|distractor|direct --seed S --n N --D 6,8,10,12
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


def make(rng, D, mode):
    nid = [0]
    WR = SP.world(rng, D, nid); WP = SP.world(rng, D, nid)
    edges = list(WR[1]) + list(WP[1])

    def direct_sub(W):
        dset = set(W[4][1:]); out = set(dset)
        for a, b in W[1]:
            if a in dset:
                out.add(b)
        return out
    for src_w, dst_w in ((WP, WR), (WR, WP)):
        if mode == "any":
            srcs = list(src_w[0]); tgts = list(dst_w[3])
        elif mode == "distractor":
            srcs = [n for n in src_w[0] if n not in set(src_w[2])]; tgts = list(dst_w[3])
        else:
            srcs = sorted(direct_sub(src_w)); tgts = sorted(direct_sub(dst_w))
        for _ in range(rng.randint(0, 2)):
            if srcs and tgts:
                edges.append((rng.choice(srcs), rng.choice(tgts)))
    persons = ["R", "P"] + ["X%d" % i for i in range(rng.randint(0, 2))]
    pid = {p: -(i + 1) for i, p in enumerate(persons)}
    edges += [(pid["R"], c) for c in WR[4]] + [(pid["P"], c) for c in WP[4]]
    allnodes = WR[0] + WP[0]
    for p in persons[2:]:
        c = nid[0] = nid[0] + 1; allnodes.append(c); edges.append((pid[p], c))
        for _ in range(rng.randint(1, 3)):
            v = nid[0] = nid[0] + 1; allnodes.append(v); edges.append((c, v))
    ids = [pid[p] for p in persons] + allnodes; idx = {u: i for i, u in enumerate(ids)}
    pn = rng.sample(["Tom", "Sam", "Ann", "Bob", "Max", "Eve", "Kim", "Lou", "Ivy", "Rex", "Zoe", "Ned", "Amy", "Jay", "Liz", "Ted", "Una"], len(persons))
    cn = rng.sample(SP.NAMES, len(allnodes)); sym = pn + cn
    E = [[idx[a], idx[b]] for a, b in edges]
    stm = [("%s is a %s." if sym[a][:1].isupper() else "Every %s is a %s.") % (sym[a], sym[b]) for a, b in E]
    rng.shuffle(stm)
    oR, oP = idx[WR[2][-1]], idx[WP[2][-1]]
    o1, o2 = (oR, oP) if rng.random() < 0.5 else (oP, oR)
    x = {"question": " ".join(stm) + " Is %s a %s or %s?" % (sym[0], sym[o1], sym[o2]), "idx_to_symbol": sym, "edges": E, "root": 0, "target": oR, "neg_target": oP}
    levels = [(idx[WR[2][k]], idx[WP[2][k]]) for k in range(D)]
    return x, levels


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--mode", default="any"); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--D", default="6,8,10,12")
    a = ap.parse_args()
    SP.NAMES = SP.name_pool(random.Random(12345)); rng = random.Random(a.seed)
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception:  # noqa: BLE001
        tok = None
    out = {"mode": a.mode, "seed": a.seed, "n": a.n, "bands": {}}
    for D in [int(v) for v in a.D.split(",")]:
        acc = collections.defaultdict(list); inv = collections.Counter(); Tb = []; men = []; Tp = []
        for i in range(a.n):
            x, levels = make(rng, D, a.mode)
            edges, words, _q = parse(x)
            adj = collections.defaultdict(list); radj = collections.defaultdict(list); uadj = collections.defaultdict(list)
            for u, v in edges:
                adj[u].append(v); radj[v].append(u); uadj[u].append(v); uadj[v].append(u)
            oR, oP = x["target"], x["neg_target"]; dR, dP = bfs(adj, [0]), bfs(adj, [1])
            ok = all(dR.get(levels[k][0]) == k + 1 and dP.get(levels[k][1]) == k + 1 for k in range(D)) and oP not in dR and oR not in dP
            inv["fail"] += not ok
            for k, (u, v) in enumerate(levels):
                for name, val in heur(x, edges, words, u, v).items():
                    if name != "last_first":
                        acc["L%d_%s" % (k + 1, name)].append(val)
            ud = bfs(uadj, [0])
            for r in range(1, D + 1):
                a_, b_ = ud.get(oR, 10 ** 6) <= 2 * r, ud.get(oP, 10 ** 6) <= 2 * r
                acc["uBIDIR%d" % r].append(1.0 if (a_ and not b_) else 0.0 if (b_ and not a_) else 0.5)
            for r in (1, 2, 3):
                F = set(bfs(adj, [0], r)); hit = [bool(F & set(bfs(radj, [o], r))) for o in (oR, oP)]
                acc["dBIDIR%d" % r].append(1.0 if hit == [True, False] else 0.0 if hit == [False, True] else 0.5)
            acc["ud_oR_lt_D"].append(float(ud.get(oR, 10 ** 6) < D)); acc["ud_oP_le_D"].append(float(ud.get(oP, 10 ** 6) <= D))
            if tok is not None and i < 300:
                q = x["question"]; m = 0
                for s in x["idx_to_symbol"]:
                    if s[:1].islower():
                        cnt = q.count(" " + s + " ") + q.count(" " + s + ".") + q.count(" " + s + "?")
                        m += cnt
                men.append(m)
                qq = " " + q
                for s in sorted((s for s in x["idx_to_symbol"] if s[:1].islower()), key=len, reverse=True):
                    qq = qq.replace(" " + s + " ", " cat ").replace(" " + s + ".", " cat.").replace(" " + s + "?", " cat?")
                Tb.append(len(tok(qq.strip())["input_ids"])); Tp.append(len(tok(q)["input_ids"]))
        row = {"invariant_fail": inv["fail"]}; worst = (0.0, None)
        for k, v in acc.items():
            mu = sum(v) / len(v)
            if not k.startswith("L"):
                row[k] = round(mu, 4); continue
            df = max(mu, 1 - mu); row[k] = round(df, 4)
            if df > worst[0]:
                worst = (df, k)
        row["max_person_agnostic"] = round(worst[0], 4); row["max_which"] = worst[1]
        ub = {r: row["uBIDIR%d" % r] for r in range(1, D + 1)}
        row["uBIDIR_max_dirfree_below_exact"] = round(max(max(v, 1 - v) for r, v in ub.items() if 2 * r < D), 4)
        if Tb:
            n_ = len(Tb); row["T_proto_mean"] = round(sum(Tp) / n_, 1); row["T_proto_max"] = max(Tp)
            row["concept_mentions_mean"] = round(sum(men) / n_, 1)
            for tpn in (2, 2.5, 3):
                Tt = sorted(b + mm * (tpn - 1) for b, mm in zip(Tb, men))
                row["T_tpn%s" % tpn] = {"mean": round(sum(Tt) / n_, 1), "p90": Tt[int(0.9 * n_)], "max": Tt[-1]}
        out["bands"][D] = row
        print(a.mode, "D", D, "inv", inv["fail"], "max_pa %.3f %s" % worst, "dBIDIR", [row["dBIDIR%d" % r] for r in (1, 2, 3)],
              "uBIDIR", [ub[r] for r in sorted(ub)], "ud(oR)<D %.3f ud(oP)<=D %.3f" % (row["ud_oR_lt_D"], row["ud_oP_le_D"]),
              "T", row.get("T_proto_mean"), row.get("T_tpn2.5"), flush=True)
        json.dump(out, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
