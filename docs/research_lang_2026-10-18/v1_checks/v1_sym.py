"""LANG v1 $0 CPU prototype of ProsQA-SYM v1 (the spec for boxes/prosqa_sym.py; a feasibility + gate-design check, not the implementation).
v1 changes over the draft's sym_proto.make (each from a review, see LANG_v1 §1):
  - cross edges ONLY between the two persons' extra direct-statement subtrees, targets = leaves of those subtrees (PROOFS 2b / §3 fix (b));
  - ANCESTRAL DECOYS per world (RT F2-1): n_dec ~ U{dlo..dhi} decoy persons, each with a chain b_1..b_j (j ~ U{1..3}) and an edge
    b_j -> a_k, k ~ U{klo(D)..D}; chain nodes get the path nodes' distractor-child law (Binomial(2, 1/2) leaves);
  - per-world concept cap cap(D) for D >= 9, applied by redrawing THAT world (independently per world: sigma-symmetric, PROOFS 2a(i));
  - single-token person names (PERSON-40) and NONCE-300 concept names with distinct final tokens (v1_names.json; RT F2-4 / F2-5);
  - item-level rejection on T <= 1000 GPT-2 tokens (sigma-invariant: person names are all 1 token).
Measured per D: invariants; person-agnostic panel (heur_panel.panel_for at the option and every path level); H-ROOT (first / random /
max-outdegree single-parent backward chase: GATED), H-ROOT-SOFT (full backward set; exact), H-ROOT-LA(h) (backward chase with h-hop
lookahead; descriptive); uBIDIR(r) for 2r < D and UDIST; concepts beyond D; |S|; GPT-2 T; BFS-layer widths (L-RR chunks of <= 4);
PREMISE-SWAP slot-local (i') exposure at D 4..8 with token-position identity; FD-MIRROR class sizes at D 6..8.
    python -B v1_sym.py <out.json> --n 1000 --D 3,4,...,16 --dec 3,5 --klo half --cap 2D+10
"""
import argparse
import collections
import json
import math
import os
import random
import sys
from collections import deque

import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from heur_panel import panel_for  # noqa: E402  (draft's validated §59 panel; read-only import)

ENC = tiktoken.get_encoding("gpt2")
NM = json.load(open(os.path.join(HERE, "v1_names.json")))
PERSONS, NONCE = NM["persons"], NM["nonce"]
LEAD = ""                                                           # --lead_space: the rendered text starts with " " (every name in its leading-space form)


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


def world(rng, D, nid, P):
    """one world (same law for both persons). -> dict(nodes, edges, path, direct, dsub, dleaves, decoys=[(chain, k)])"""
    def new():
        nid[0] += 1; return nid[0]
    path = [new() for _ in range(D)]; edges = [(path[i], path[i + 1]) for i in range(D - 1)]
    depth = {a: i + 1 for i, a in enumerate(path)}; anchor = {}; nodes = list(path)
    for k in range(D - 1):                                        # distractor subtrees off a_1..a_{D-1} (draft law)
        for _ in range(sum(rng.random() < 0.5 for _ in range(2))):
            c = new(); nodes.append(c); edges.append((path[k], c)); depth[c] = k + 2; anchor[c] = k
            frontier = [c]
            for _lvl in range(min(3, D - k - 1) - 1):
                nxt = []
                for u in frontier:
                    for _ in range(sum(rng.random() < 0.3 for _ in range(2))):
                        v = new(); nodes.append(v); edges.append((u, v)); depth[v] = depth[u] + 1; anchor[v] = k; nxt.append(v)
                frontier = nxt
    for u in [n for n in nodes if n in anchor]:                   # within-world edges u -> a_k keeping a_k's depth (draft law)
        if rng.random() < P["pextra"]:                            # draft 0.3; v1 lowers it (fewer R-leading alternative parents)
            lo, hi = anchor[u] + 2, min(D, depth[u] + 1)
            if lo <= hi:
                k = rng.randint(lo, hi); edges.append((u, path[k - 1]))
    direct = [path[0]]; dsub = []; dleaves = []
    for _ in range(rng.randint(1, P["rmax"]) - 1):                # extra direct statements -> fresh small subtrees (no path edges)
        c = new(); nodes.append(c); direct.append(c); dsub.append(c); kids = []
        for _ in range(rng.randint(0, 2)):
            v = new(); nodes.append(v); edges.append((c, v)); dsub.append(v); kids.append(v)
        dleaves += kids if kids else [c]
    cap = P["cap"](D)                                              # per-world cap on non-decoy concepts: thin whole distractor subtrees
    if cap is not None and len(nodes) > cap:                      # (uniformly at random, the same rule in each world: sigma-symmetric)
        roots = [c for c in anchor if depth[c] == anchor[c] + 2]; rng.shuffle(roots)
        kids = collections.defaultdict(list)
        for u, v in edges:
            kids[u].append(v)
        drop = set()
        for rt in roots:
            if len(nodes) - len(drop) <= cap:
                break
            st = [rt]
            while st:
                u = st.pop()
                if u in anchor and u not in drop:
                    drop.add(u); st += [v for v in kids[u] if v in anchor]
        nodes = [n for n in nodes if n not in drop]; edges = [(u, v) for u, v in edges if u not in drop and v not in drop]
    decoys = []
    klo = 2 if P["klo"] == "2" else max(2, math.ceil(D / 2)) if P["klo"] == "half" else max(2, D - int(P["klo"][3:]))   # "top2": k in {D-2..D}
    ks = [rng.randint(klo, D) for _ in range(rng.randint(P["dlo"], P["dhi"]))]
    if P["rr"] > 0:                                               # v1 law: rr decoys per world, levels D, D-1, ... round-robin, wrapping at 2
        nrr = P["rr_small"] if D <= 4 else P["rr"]
        lv_ = list(range(D, 1, -1)); ks += [lv_[i % len(lv_)] for i in range(nrr)]
    if P["toplev"] > 0:                                           # (alternative law): m_k ~ U{mlo..mhi} decoys at each of the top `toplev` levels (k >= 2)
        ks += [k for k in range(max(2, D - P["toplev"] + 1), D + 1) for _ in range(rng.randint(P["mlo"], P["mhi"]))]
    for k in ks:                                                  # ancestral decoys (RT F2-1), the path nodes' child law on the competing node
        ch = [new() for _ in range(rng.randint(1, P["jmax"]))]; nodes += ch
        edges += [(ch[q], ch[q + 1]) for q in range(len(ch) - 1)]; edges.append((ch[-1], path[k - 1]))
        for c_ in (ch if P["leaves"] == "all" else ch[-1:]):
            for _ in range(sum(rng.random() < 0.5 for _ in range(2))):
                v = new(); nodes.append(v); edges.append((c_, v))
        decoys.append((ch, k))
    return {"nodes": nodes, "edges": edges, "path": path, "direct": direct, "dsub": dsub, "dleaves": dleaves, "decoys": decoys}


def capped_world(rng, D, nid, P):
    return world(rng, D, nid, P)


def make(rng, D, P):
    for _try in range(200):
        nid = [0]
        WR = capped_world(rng, D, nid, P); WP = capped_world(rng, D, nid, P)
        edges = list(WR["edges"]) + list(WP["edges"])
        for S, T in ((WP, WR), (WR, WP)):                          # direct-subtree kernel, leaf targets, c ~ U{0..2} each way
            for _ in range(rng.randint(0, 2)):
                if S["dsub"] and T["dleaves"]:
                    edges.append((rng.choice(S["dsub"]), rng.choice(T["dleaves"])))
        n_extra = rng.randint(0, 2); n_dec = len(WR["decoys"]) + len(WP["decoys"])
        persons = ["R", "P"] + ["X%d" % i for i in range(n_extra)] + ["Y%d" % i for i in range(n_dec)]
        pid = {p: -(i + 1) for i, p in enumerate(persons)}
        edges += [(pid["R"], c) for c in WR["direct"]] + [(pid["P"], c) for c in WP["direct"]]
        allnodes = WR["nodes"] + WP["nodes"]
        for p in persons[2:2 + n_extra]:                           # extra persons: small disjoint trees
            c = nid[0] = nid[0] + 1; allnodes.append(c); edges.append((pid[p], c))
            for _ in range(rng.randint(1, 3)):
                v = nid[0] = nid[0] + 1; allnodes.append(v); edges.append((c, v))
        for i, (ch, k) in enumerate(WR["decoys"] + WP["decoys"]):
            edges.append((pid["Y%d" % i], ch[0]))
        ids = [pid[p] for p in persons] + allnodes; idx = {u: i for i, u in enumerate(ids)}
        if len(persons) > len(PERSONS) or len(allnodes) > len(NONCE):
            continue
        pn = rng.sample(PERSONS, len(persons)); cn = rng.sample(NONCE, len(allnodes)); sym = pn + cn
        E = [[idx[a], idx[b]] for a, b in edges]
        order = list(range(len(E))); rng.shuffle(order)
        stm = [("%s is a %s." if sym[E[j][0]][:1].isupper() else "Every %s is a %s.") % (sym[E[j][0]], sym[E[j][1]]) for j in order]
        oR, oP = idx[WR["path"][-1]], idx[WP["path"][-1]]
        o1, o2 = (oR, oP) if rng.random() < 0.5 else (oP, oR)
        body = " ".join(stm); q = LEAD + body + " Is %s a %s or %s?" % (sym[0], sym[o1], sym[o2])
        T = len(ENC.encode(q)) + 1
        if T > 1000:
            continue
        x = {"question": q, "body": body, "idx_to_symbol": sym, "edges": E, "order": order, "root": 0, "target": oR, "neg_target": oP,
             "o1": o1, "o2": o2, "T": T, "n_persons": len(persons), "n_concepts": len(allnodes),
             "levels": [(idx[WR["path"][k]], idx[WP["path"][k]]) for k in range(D)]}
        return x
    raise RuntimeError("T > 1000 in 200 draws at D %d" % D)


def chase(x, rng, pick, h=0):
    names = x["idx_to_symbol"]; par = collections.defaultdict(list); od = collections.Counter(a for a, _ in x["edges"])
    for j in x["order"]:                                            # parents in text order (the 'first' variant reads them in this order)
        a, b = x["edges"][j]; par[b].append(a)
    ispers = lambda u: names[u][:1].isupper()  # noqa: E731

    def score(u):                                                   # h-hop backward lookahead: +1 sees R, -1 sees another person only
        seen = bfs(par, [u], h - 1) if h > 0 else {u: 0}
        if x["root"] in seen:
            return 1
        return -1 if any(ispers(v) for v in seen) else 0
    roots = []
    for o in (x["target"], x["neg_target"]):
        cur, k = o, 0
        while par[cur] and k < 500:
            L = par[cur]
            if pick == "first":
                cur = L[0]
            elif pick == "random":
                cur = rng.choice(L)
            elif pick == "maxdeg":
                m = max(od[u] for u in L); cur = rng.choice([u for u in L if od[u] == m])
            else:
                m = max(score(u) for u in L); cur = rng.choice([u for u in L if score(u) == m])
            k += 1
        roots.append(cur)
    hit = [roots[0] == x["root"], roots[1] == x["root"]]
    return 1.0 if hit == [True, False] else 0.0 if hit == [False, True] else 0.5


def first_mention_last_tok(text, names, idx_list):
    """token index of the last BPE token of each name's first mention (names are whole words; the text is ' '-joined statements)"""
    out = {}; ids = ENC.encode(text); pos = 0; offs = []
    for t in ids:                                                   # char offset where each token ends
        pos += len(ENC.decode([t])); offs.append(pos)
    for i in idx_list:
        w = names[i]; cands = [" " + w + " ", " " + w + ".", " " + w + "?"]
        c = min((p for p in ((" " + text).find(s) for s in cands) if p >= 0), default=-1)
        if c < 0:
            out[i] = None; continue
        end = c + len(w)                                            # (" " + text) offset c -> text offset c - 1 + 1 + len(w)
        out[i] = next(j for j, e in enumerate(offs) if e >= end)
    return out


def premise_swap(x):
    """donor = recipient with the two asked/foil person names exchanged in the premises only; the question is unchanged"""
    s = x["idx_to_symbol"]; R, P = s[0], s[1]; body = x["body"]
    tmp = body.replace(R + " is a ", "\x00 is a ").replace(P + " is a ", R + " is a ").replace("\x00 is a ", P + " is a ")
    return LEAD + tmp + " Is %s a %s or %s?" % (R, s[x["o1"]], s[x["o2"]])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--D", default="3,4,5,6,7,8,9,10,11,12,13,14,15,16"); ap.add_argument("--seed", type=int, default=24000)
    ap.add_argument("--dec", default="3,5"); ap.add_argument("--klo", default="half"); ap.add_argument("--rmax", type=int, default=4)
    ap.add_argument("--cap", default="2D+6"); ap.add_argument("--expo_n", type=int, default=300)
    ap.add_argument("--lead_space", type=int, default=0); ap.add_argument("--jmax", type=int, default=2); ap.add_argument("--leaves", default="last"); ap.add_argument("--pextra", type=float, default=0.3)
    ap.add_argument("--toplev", type=int, default=0); ap.add_argument("--rr", type=int, default=0); ap.add_argument("--rr_small", type=int, default=0); ap.add_argument("--mper", default="1,1")
    a = ap.parse_args(); global LEAD; LEAD = " " if a.lead_space else ""
    lo, hi = (int(v) for v in a.dec.split(","))
    cap = None if a.cap == "none" else (lambda D, c=a.cap: None if D < 9 else 2 * D + int(c.split("+")[1]))
    P = {"dlo": lo, "dhi": hi, "klo": a.klo, "rmax": a.rmax, "cap": cap, "jmax": a.jmax, "leaves": a.leaves, "pextra": a.pextra, "toplev": a.toplev, "mlo": int(a.mper.split(",")[0]), "mhi": int(a.mper.split(",")[1]), "rr": a.rr, "rr_small": a.rr_small}
    rng = random.Random(a.seed); hr = random.Random(a.seed + 1)
    out = {"params": {"dec": a.dec, "klo": a.klo, "rmax": a.rmax, "cap": a.cap, "jmax": a.jmax, "leaves": a.leaves, "pextra": a.pextra, "toplev": a.toplev, "mper": a.mper, "rr": a.rr, "rr_small": a.rr_small, "lead_space": a.lead_space, "seed": a.seed, "n": a.n}, "bands": {}}
    for D in [int(v) for v in a.D.split(",")]:
        acc = collections.defaultdict(list); inv = collections.Counter(); Ts = []; S = []; P_ = []; widths = []; chunks = []; beyond = 0
        expo = collections.Counter(); Lsz = []; Ldiff = []; far = []; fdm = collections.Counter()
        for i in range(a.n):
            x = make(rng, D, P)
            adj = collections.defaultdict(list); uadj = collections.defaultdict(list)
            for u, v in x["edges"]:
                adj[u].append(v); uadj[u].append(v); uadj[v].append(u)
            oR, oP = x["target"], x["neg_target"]; dR, dP = bfs(adj, [0]), bfs(adj, [1])
            lv = x["levels"]
            ok = all(dR.get(lv[k][0]) == k + 1 and dP.get(lv[k][1]) == k + 1 for k in range(D)) and oP not in dR and oR not in dP
            names = x["idx_to_symbol"]; men = set()
            for w in x["question"].replace(".", " ").replace("?", " ").split():
                men.add(w)
            ok = ok and all(s in men for s in names)
            inv["fail"] += not ok
            beyond += max(dR.values()) > D
            for k, (u, v) in enumerate(lv):
                for hname, val in panel_for(x, u, v).items():
                    acc["L%d_%s" % (k + 1, hname)].append(val)
            for pick in ("first", "random", "maxdeg"):
                acc["HROOT_" + pick].append(chase(x, hr, pick))
            for h in (1, 2, 3, 4):
                acc["HROOT_LA%d" % h].append(chase(x, hr, "la", h))
            radj = collections.defaultdict(list)
            for u, v in x["edges"]:
                radj[v].append(u)
            hs = [0 in bfs(radj, [o]) for o in (oR, oP)]; acc["HROOT_soft"].append(1.0 if hs == [True, False] else 0.0 if hs == [False, True] else 0.5)
            ud = bfs(uadj, [0])
            for r in range(1, (D + 1) // 2 + 1):
                if 2 * r < D:
                    a_, b_ = ud.get(oR, 10 ** 6) <= 2 * r, ud.get(oP, 10 ** 6) <= 2 * r
                    acc["uBIDIR%d" % r].append(1.0 if (a_ and not b_) else 0.0 if (b_ and not a_) else 0.5)
            uR, uP = ud.get(oR, 10 ** 6), ud.get(oP, 10 ** 6); acc["UDIST"].append(1.0 if uR < uP else 0.0 if uP < uR else 0.5)
            Ts.append(x["T"]); S.append(x["n_concepts"]); P_.append(x["n_persons"])
            lay = collections.Counter(dv for c, dv in dR.items() if c != 0 and not names[c][:1].isupper())
            widths.append(max(lay.values())); chunks.append(sum(math.ceil(w / 4) for w in lay.values()))
            if 4 <= D <= 8 and i < a.expo_n:                        # PREMISE-SWAP x slot-local (i') exposure
                t = hr.randint(3, D - 1)
                conc = [c for c in range(len(names)) if not names[c][:1].isupper()]
                L = [c for c in conc if c not in (oR, oP) and min(dR.get(c, 99), dP.get(c, 99)) <= t + 1]
                diff = [c for c in L if (dR.get(c, 99) <= t) != (dP.get(c, 99) <= t)]
                farc = [c for c in conc if c not in L and c not in (oR, oP) and (dR.get(c, 99) <= D) != (dP.get(c, 99) <= D)]
                don = premise_swap(x)
                pr = first_mention_last_tok(x["question"], names, conc); pd = first_mention_last_tok(don, names, conc)
                same_pos = pr == pd and all(v is not None for v in pr.values()) and len(ENC.encode(don)) == len(ENC.encode(x["question"]))
                expo["items"] += 1; expo["same_pos"] += same_pos; expo["frontier_exposed"] += len(diff) > 0
                expo["options_outside"] += (oR not in L and oP not in L); expo["far_diff_outside"] += len(farc) > 0
                expo["exposed"] += same_pos and len(diff) > 0 and (oR not in L and oP not in L)
                Lsz.append(len(L)); Ldiff.append(len(diff)); far.append(len(farc))
            if 6 <= D <= 8:
                for t in (1, 2):
                    for k in range(3, D - t + 1):
                        fdm["t%d_k%d_pos" % (t, k)] += sum(1 for c, dv in dR.items() if dv == t + k and not names[c][:1].isupper())
                        fdm["t%d_k%d_neg" % (t, k)] += sum(1 for c, dv in dP.items() if dv == t + k and not names[c][:1].isupper())
        row = {"n": a.n, "invariant_fail": inv["fail"], "beyond_D_share": round(beyond / a.n, 4)}
        worst = (0.0, None)
        for k, v in acc.items():
            m = sum(v) / len(v)
            if k.startswith("L"):
                df = max(m, 1 - m); row[k] = round(df, 4)
                if df > worst[0]:
                    worst = (df, k)
            else:
                row[k] = round(max(m, 1 - m), 4)
        row["max_person_agnostic"], row["max_which"] = round(worst[0], 4), worst[1]
        row["HROOT_gated_max"] = max(row["HROOT_first"], row["HROOT_random"], row["HROOT_maxdeg"])
        ub = [row[k] for k in row if k.startswith("uBIDIR")]; row["uBIDIR_max"] = max(ub) if ub else None
        Ts.sort(); S.sort()
        row.update({"T_mean": round(sum(Ts) / len(Ts), 1), "T_p95": Ts[int(0.95 * len(Ts))], "T_max": Ts[-1],
                    "S_mean": round(sum(S) / len(S), 1), "S_max": S[-1], "persons_mean": round(sum(P_) / len(P_), 2), "persons_max": max(P_),
                    "layer_width_max_mean": round(sum(widths) / len(widths), 2), "share_layer_gt4": round(sum(w > 4 for w in widths) / len(widths), 3),
                    "LRR_chunks_mean": round(sum(chunks) / len(chunks), 2)})
        if expo:
            row["expo"] = dict(expo); row["L_size_mean"] = round(sum(Lsz) / len(Lsz), 1); row["L_diff_mean"] = round(sum(Ldiff) / len(Ldiff), 1)
            row["far_diff_mean"] = round(sum(far) / len(far), 1)
        if fdm:
            row["fd_mirror_slots_per_item"] = {k: round(v / a.n, 2) for k, v in fdm.items()}
        out["bands"][D] = row
        print("D", D, "inv", inv["fail"], "beyond", row["beyond_D_share"], "PA %.3f %s" % worst, "HROOT f/r/m %.3f %.3f %.3f" % (row["HROOT_first"], row["HROOT_random"], row["HROOT_maxdeg"]),
              "LA1-4", [row["HROOT_LA%d" % h] for h in (1, 2, 3, 4)], "soft", row["HROOT_soft"], "uBIDIR", row["uBIDIR_max"], "UDIST", row["UDIST"],
              "T %.0f/%d/%d" % (row["T_mean"], row["T_p95"], row["T_max"]), "S %.0f/%d" % (row["S_mean"], row["S_max"]), "pers", row["persons_max"],
              "w>4 %.2f chunks %.1f" % (row["share_layer_gt4"], row["LRR_chunks_mean"]), row.get("expo", ""), flush=True)
        json.dump(out, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
