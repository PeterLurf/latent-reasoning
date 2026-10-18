"""ProsQA-SYM v1 (§218 LANG-VB; the registered law is LANG_v1 §2.2 items 1-10): generator, REACH targets, role-symmetric signature,
streams / fixed sets with signature exclusion, PREMISE-SWAP / PERSON-SWAP / CONSUMPTION-FRONT pair makers (§2.5, §3.6) and the G-L0 panel
(person-agnostic §59 cues x levels, gated single-parent chase set, uBIDIR(r), UDIST, beyond-D; §3.3, §7). CPU only, no torch.

The world / item construction is a port of the measured prototype docs/research_lang_2026-10-18/v1_checks/v1_sym.py run with the v1 law
parameters of v1_cf.py (dec 0,0; rr 4 / rr_small 5; jmax 2; leaves last; pextra 0.1; rmax 4; cap 2D+6 at D >= 9; lead space). The rng is
consumed in the prototype's order, so make_item(random.Random(s), D) renders the same item as v1_sym.make(random.Random(s), D, P_v1)
(checked by scratch/gate_s218_sym.py parity).

API
  make_item(rng, D) -> item dict (invariants of §2.2 item 8 asserted; raises SymInvariantError). Keys:
    text      rendered statements + question, no leading space; the model input is ids = [EOT] + enc(" " + text)
    ids, T    GPT-2 ids of [EOT] + " " + text; T = len(ids) <= 1000
    D, names  names[i] = the surface name of node i; persons are nodes 0..n_p-1 (R = 0, P = 1, then extra X, then decoy Y persons),
              concepts are nodes n_p..n-1
    R, P, o_R, o_P, o1, o2   node ids (question person, foil person, R's option, P's option, displayed option order)
    answer, answer_foil      "### <o_R>." and "### <o_P>."
    persons, concepts        node id lists;  edges: [[a, b], ...] node ids ("a is a b"); order: text order of the edge indices
    levels    [(a_k^R, a_k^P) for k = 1..D] (a_D = the option)
    slot_idx  {concept id: index in ids of the LAST BPE token of its FIRST mention}
    dist_R, dist_P   BFS distances {node id: d} over reachable nodes (unreachable = absent)
    sig       role-symmetric contamination signature (§2.2 item 9);  tag: per-node structure tags used by sig_of
    n_cross, n_decoys, n_extra   (C_RP, C_PR) cross-edge counts, decoys per world, extra persons
  reach_targets(item, t_max) -> E[t][i] in {0, 1} for t = 0..t_max, i over item["concepts"]: 1 (IN) iff dist(R, c) <= t; E[0] all OUT
  Stream(seed, exclude=set()).draw(D)  band-D draws = the accepted items of random.Random(seed * 1000 + D) in order; an item whose sig is
              in `exclude` is redrawn (logged in .collisions)
  test_set(n_per_D=1000, D_range=range(3, 17), seed=22200) -> {D: [items]} = the first n_per_D draws of Stream(22200) per band
  gate_set(n_per_D=2000, D_range=range(9, 13), seed=22400);  fixed_sigs(seed, n_per_D, D_range) -> set of sigs (memory-light)
  exclusion_sigs() = fixed_sigs(test) | fixed_sigs(GATE): the training / pool / probe streams' exclusion set (§2.2 item 10)
  premise_swap(item) -> donor (R's and P's names exchanged in every premise; question still asks R; donor["o_R"] = the recipient's o_P;
              T and slot_idx asserted identical; donor invariants asserted)
  person_swap(item) -> donor (question person R -> P; answer o_P; T and slot_idx asserted identical)
  cf_donor(item, j) -> None (hop-j edge not unique) or {j, k, flip, same_len, same_slots, exposed, donor}: R's hop-j statement gets object
              a_j^P (§2.5 CF); exposed iff flip (in x' R reaches o_P at distance D and not o_R) and equal T and identical first-mention slots
  cf_select(items, rng=Random(22630), max_per_cell=400, js=None) -> {(D, j): [(item index, cf_donor result)]}, exposed pairs only (asserted)
  panel_levels(item) -> {"L<k>_<cue>": 1 / 0 / 0.5} for k = 0..D (L0 = (R, P), Lk = (a_k^R, a_k^P)); the six §59 cues of heur_panel.panel_for
  chase_set(item, rng) -> {first, random, maxdeg, la1, la2, la3, la4, soft}: single-parent backward chases (1 / 0 / 0.5)
  ubidir(item) -> {r: value} for 2r < D;  udist(item);  beyond_D(item) -> bool;  name_checks() -> dict;  set_sha(items)

AMBIGUITY (each resolved to the measured prototype's behaviour):
  A1 cross-edge source "a node of the source person's direct subtrees" = the r - 1 fresh direct nodes and their children (a_1 is a path node,
     not a direct subtree); target = a leaf of the other person's direct subtrees (a childless fresh direct node is its own leaf). If the
     source person has r = 1 (no direct subtree) the drawn c edges are dropped. A repeated (source, target) draw keeps a duplicate statement.
  A2 within-world extra edge u -> a_k: k ~ U{anchor(u) + 1 .. min(D, depth(u) + 1)} (strictly below u's anchor path node, so acyclic);
     the spec states only depth(u) + 1 >= k.
  A3 the D >= 9 cap counts path + distractor + direct-subtree nodes (the non-decoy concepts); decoys are attached after the cap; whole
     distractor subtrees (rooted at a path node's child) are dropped in a uniformly shuffled order until the count is <= 2D + 6.
  A4 extra person = one concept + U{1..3} children (a 2-4-node star).
  A5 T <= 1000 rejection retries the whole item (both worlds) up to 200 times, then RuntimeError (never observed).
  A6 signature: the name-free canonical form uses generator-internal WORLD-LOCAL creation ids (not an isomorphism-canonical form): per world
     the sorted edge list (own person, own concepts and decoy persons, cross edges out of the world, targets tagged self / other world);
     the two world strings sorted; extra-person trees sorted; D included; statement order, option coin and names excluded.
  A7 per-band rngs random.Random(seed * 1000 + D) for every stream and fixed set (test = first 1000 per band; the G-L0 "4,000 more of the same
     law" = the next 4000 of the same band rng; GATE = the first 2000 per band D 9-12 of seed 22400).
  A8 "6 cues x all D + 1 levels": the prototype reads D levels (a_1..a_D, a_D = the option); L0 = (R, P) is added and reported separately.
  A9 the prototype draws U{0..0} extra decoys (value unused) per world; the draw is kept for rng parity with the measured prototype.
"""
import collections
import hashlib
import json
import random
import re
from collections import deque

import tiktoken

ENC = tiktoken.get_encoding("gpt2"); EOT = ENC.eot_token
T_MAX, D_MIN, D_MAX = 1000, 3, 16
P_EXTRA, R_MAX, CHAIN_MAX = 0.1, 4, 2                               # §2.2 items 1, 3
SEED_TRAIN, SEED_TEST, SEED_POOL, SEED_GATE, SEED_PROBE = 22100, 22200, 22300, 22400, 22500   # §2.2 item 10
SEED_SWAP, SEED_CF, SEED_PLANT = 22610, 22630, 22550


def cap_of(D):
    """§2.2 item 2: per-world cap on non-decoy concepts (D >= 9)."""
    return None if D < 9 else 2 * D + 6


def n_decoys(D):
    """§2.2 item 3: decoys per world."""
    return 5 if D <= 4 else 4


# §2.2 item 6: PERSON-40 (single GPT-2 token in leading-space form; ENT17's 15 one-token names first) and NONCE-300 (two tokens, pairwise
# distinct final tokens), frozen from docs/research_lang_2026-10-18/v1_checks/v1_names.json.
PERSON_40 = tuple("Alex Bob Carol Davis Eva Gabriel Jack Max Oliver Polly Rex Sally Sam Stella Tom Anna Ben Chris Dan Emma Frank George Henry "
                  "Ian Jane Kate Leo Mark Nick Paul Ray Rose Ruth Sara Tim Victor Will Adam Amy Brian".split())
NONCE_300 = tuple("""
mostden duplour parlo priron memgrain walumen shirtur berju hutroo bemol gifiar nabount sanok thousbre trialtrain flara higak wokeesp volpor burvi hookli ropek pontnon loterk woolus
chimka bourfour tonor bruntbil presti flavrex losha giri rodol diachi gunpool trochi huntro loopru hogland gruzan miabee treesta shavees touwa keenmi naistan feesum mosment cheesee
brinia baitround vonva digamboo panjo mootust flavoust haisse mailwi gigim rookti trooji grooveest slursa sito ziste libain nifa humpu drumix febial dunrees therweek peelli furhai
misint buflo mimthin hiro pandhe podrous rovre bargork chimda pagound stashoos tendme susker brasirk mathoux zeji mosfin mantjar gemsnu giantskus doomne sukos flungisp hegar demsten
trumpox bouthex binbal ventrox fallom treemi cheeskees gleedar drohi toobe fliprou lotos darklin tobo fako fistook prochar junmis belrem milni jograx snolar sadax belzhou despnia
troba murbia mentpe boslar chastren posslia miallo zuba hiades boxko migai stemsnian dramween pontpri globund stiami libres sidrend shelgil mecha timir bindront waiveem diarrunt dimthen
quarper welfous langam snonis malze vaime wonthem fendfle doupast lummar quafa vimwan bropen parsna tugran dredia steja fendfun loslast suchon paprar nage flekai sleekdust soulske
hothel begas somthia bropi dehak beero delrast slumpril gamstand gunlee brebee nonpa koche burshi nitis patrum bofal soufo slira gisthend weerror gemple plainsle thama cheeklund
worthasp haloul deerhair duprek kuple groumust westlem nasram senvu thenvas rantdi storno vanrus vonho queerbes solra plube lolin hinos weesthes vexrust glekes gerte profian ninhor
naigon pestthal markskind floorsark priwant flourwar drullor snemia somer nitand rentbek wikgam navrint dontdam boufail sadrak pestgram lostfa thucha jooras thusbe kehas gistbound lemount prailler
plefest jova respial tamham piso mimhin junbol fixjun paigan winnum gimo nilpler tagost vexster tholon bagri peekpend bobria thouslust toupeer bronvin glaja traisu memhol fendtri
jamge mantloo baique queeral reeflees gratree chemkar leje prepro bootoon withul visvol pelger pluli sleeklain dramthe wobroom snego souwik koka taskik womma kaifle sneber preellan
""".split())
SHA_PERSONS = "7fed438610c420a99d14ff7a92ed2b7100feb1f30c5121328fe721bee2723700"   # sha256("\n".join(PERSON_40)), = v1_names.json
SHA_NONCE = "846094a5b889e87851d9bd7f0ea65a13b4125a9b030e8d0ee3a01a5c618ceb25"     # sha256("\n".join(NONCE_300)), = v1_names.json
assert hashlib.sha256("\n".join(PERSON_40).encode()).hexdigest() == SHA_PERSONS and len(PERSON_40) == 40
assert hashlib.sha256("\n".join(NONCE_300).encode()).hexdigest() == SHA_NONCE and len(NONCE_300) == 300

_TOK = {}


def _tk(s):
    """cached GPT-2 encoding of one pre-token chunk (every chunk here is a whole GPT-2 pre-token, so concatenation = the full encoding)."""
    t = _TOK.get(s)
    if t is None:
        t = _TOK[s] = ENC.encode(s)
    return t


class SymInvariantError(AssertionError):
    """an item violates §2.2 item 8 (a code bug, never a law event)."""


def name_checks():
    """§7 G-L0 names: PERSON-40 one token each, NONCE-300 two tokens with pairwise-distinct finals, shas."""
    pt = [_tk(" " + p) for p in PERSON_40]; nt = [_tk(" " + w) for w in NONCE_300]
    fin = [t[-1] for t in nt]; nonfin = {x for t in nt for x in t[:-1]}; ptok = {t[0] for t in pt}
    return {"n_persons": len(PERSON_40), "persons_distinct": len(set(PERSON_40)) == 40, "persons_one_token": all(len(t) == 1 for t in pt),
            "persons_tokens_distinct": len(ptok) == 40, "n_nonce": len(NONCE_300), "nonce_distinct": len(set(NONCE_300)) == 300,
            "nonce_two_tokens": all(len(t) == 2 for t in nt), "nonce_finals_distinct": len(set(fin)) == 300,
            "finals_disjoint_from_nonfinals": not (set(fin) & nonfin), "finals_disjoint_from_person_tokens": not (set(fin) & ptok),
            "nonce_lowercase": all(w.islower() and w.isalpha() for w in NONCE_300), "persons_capitalised": all(p[:1].isupper() for p in PERSON_40),
            "sha256_persons": SHA_PERSONS, "sha256_nonce": SHA_NONCE}


def bfs(adj, srcs, maxd=10 ** 9):
    """BFS distances from srcs over adjacency lists (nodes beyond maxd not expanded)."""
    d = {s: 0 for s in srcs}; q = deque(srcs)
    while q:
        u = q.popleft()
        if d[u] >= maxd:
            continue
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


def _adj(n, E):
    a = [[] for _ in range(n)]
    for u, v in E:
        a[u].append(v)
    return a


def _world(rng, D, nid):
    """§2.2 item 1-3: one world of law G(D) (the prototype's world() at the v1 parameters; rng order kept)."""
    base = nid[0]

    def new():
        nid[0] += 1; return nid[0]
    path = [new() for _ in range(D)]; edges = [(path[i], path[i + 1]) for i in range(D - 1)]
    depth = {a: i + 1 for i, a in enumerate(path)}; anchor = {}; nodes = list(path)
    for k in range(D - 1):                                          # a_{k+1}: Binomial(2, 1/2) distractor children, subtree depth <= min(3, D - k - 1)
        for _ in range(sum(rng.random() < 0.5 for _ in range(2))):
            c = new(); nodes.append(c); edges.append((path[k], c)); depth[c] = k + 2; anchor[c] = k; frontier = [c]
            for _lvl in range(min(3, D - k - 1) - 1):               # branching Binomial(2, 0.3)
                nxt = []
                for u in frontier:
                    for _ in range(sum(rng.random() < 0.3 for _ in range(2))):
                        v = new(); nodes.append(v); edges.append((u, v)); depth[v] = depth[u] + 1; anchor[v] = k; nxt.append(v)
                frontier = nxt
    for u in [n for n in nodes if n in anchor]:                     # extra edge u -> a_k w.p. 0.1 per distractor node (AMBIGUITY A2)
        if rng.random() < P_EXTRA:
            lo, hi = anchor[u] + 2, min(D, depth[u] + 1)
            if lo <= hi:
                edges.append((u, path[rng.randint(lo, hi) - 1]))
    direct = [path[0]]; dsub = []; dleaves = []
    for _ in range(rng.randint(1, R_MAX) - 1):                      # r ~ U{1..4}: a_1 + r - 1 fresh direct nodes with 0-2 children each
        c = new(); nodes.append(c); direct.append(c); dsub.append(c); kids = []
        for _ in range(rng.randint(0, 2)):
            v = new(); nodes.append(v); edges.append((c, v)); dsub.append(v); kids.append(v)
        dleaves += kids if kids else [c]
    cap = cap_of(D)
    if cap is not None and len(nodes) > cap:                        # item 2: drop whole distractor subtrees, shuffled order (AMBIGUITY A3)
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
    rng.randint(0, 0)                                               # AMBIGUITY A9: the prototype's U{0..0} extra-decoy count draw (rng parity)
    lv = list(range(D, 1, -1)); decoys = []
    for k in [lv[i % len(lv)] for i in range(n_decoys(D))]:          # item 3: round-robin a_D, a_{D-1}, .., a_2, again from a_D
        ch = [new() for _ in range(rng.randint(1, CHAIN_MAX))]; nodes += ch
        edges += [(ch[q], ch[q + 1]) for q in range(len(ch) - 1)]; edges.append((ch[-1], path[k - 1]))
        for _ in range(sum(rng.random() < 0.5 for _ in range(2))):  # b_last: Binomial(2, 1/2) leaf children
            v = new(); nodes.append(v); edges.append((ch[-1], v))
        decoys.append((ch, k))
    return {"nodes": nodes, "edges": edges, "path": path, "direct": direct, "dsub": dsub, "dleaves": dleaves, "decoys": decoys, "base": base}


def render(names, E, order, q, o1, o2):
    """§2.2 item 7: text, ids = [EOT] + enc(" " + text), first-mention last-token index of every lowercase (concept) name; ids asserted equal to
    the full tiktoken encoding."""
    stm = []; ids = [EOT]; slot = {}
    for j in order:
        a, b = E[j]; na, nb = names[a], names[b]
        if na[:1].isupper():
            stm.append("%s is a %s." % (na, nb)); parts = ((" " + na, a), (" is", None), (" a", None), (" " + nb, b), (".", None))
        else:
            stm.append("Every %s is a %s." % (na, nb)); parts = ((" Every", None), (" " + na, a), (" is", None), (" a", None), (" " + nb, b), (".", None))
        for s, c in parts:
            ids += _tk(s)
            if c is not None and c not in slot and not names[c][:1].isupper():
                slot[c] = len(ids) - 1
    text = " ".join(stm) + " Is %s a %s or %s?" % (names[q], names[o1], names[o2]); q_start = len(ids)
    for s, c in ((" Is", None), (" " + names[q], q), (" a", None), (" " + names[o1], o1), (" or", None), (" " + names[o2], o2), ("?", None)):
        ids += _tk(s)
        if c is not None and c not in slot and not names[c][:1].isupper():
            slot[c] = len(ids) - 1
    if ids[1:] != ENC.encode(" " + text):
        raise SymInvariantError("chunked ids != full encoding")
    return text, ids, slot, q_start


def _finish(it):
    """render + BFS fields for an item skeleton (names, edges, order, R, P, o_R, o_P, o1, o2, D, levels, tag, ...)."""
    names = it["names"]; n = len(names)
    it["text"], it["ids"], it["slot_idx"], it["q_start"] = render(names, it["edges"], it["order"], it["R"], it["o1"], it["o2"]); it["T"] = len(it["ids"])
    adj = _adj(n, it["edges"]); it["dist_R"] = bfs(adj, [it["R"]]); it["dist_P"] = bfs(adj, [it["P"]])
    it["persons"] = [i for i in range(n) if names[i][:1].isupper()]; it["concepts"] = [i for i in range(n) if not names[i][:1].isupper()]
    it["answer"] = "### %s." % names[it["o_R"]]; it["answer_foil"] = "### %s." % names[it["o_P"]]
    return it


def check(it):
    """§2.2 item 8 invariants -> list of failures (empty = ok)."""
    f = []; D = it["D"]; dR, dP = it["dist_R"], it["dist_P"]
    for k, (aR, aP) in enumerate(it["levels"]):
        if dR.get(aR) != k + 1 or dP.get(aP) != k + 1:
            f.append("level %d distance" % (k + 1))
    if it["levels"][-1] != (it["o_R"], it["o_P"]):
        f.append("options != a_D")
    if it["o_P"] in dR:
        f.append("o_P in Reach(R)")
    if it["o_R"] in dP:
        f.append("o_R in Reach(P)")
    if any(c not in it["slot_idx"] or it["slot_idx"][c] >= it["q_start"] for c in it["concepts"]):
        f.append("concept not mentioned in the premises")
    if it["T"] > T_MAX or it["T"] != len(it["ids"]):
        f.append("T")
    if any(it["names"][p] not in _PSET or len(_tk(" " + it["names"][p])) != 1 for p in it["persons"]):
        f.append("person name not one PERSON-40 token")
    if any(it["names"][c] not in _NSET for c in it["concepts"]) or len(set(it["names"])) != len(it["names"]):
        f.append("names not distinct NONCE-300 / PERSON-40")
    if max(dR.values()) > D:
        f.append("max dist(R, c) > D")
    return f


_PSET, _NSET = set(PERSON_40), set(NONCE_300)


def make_item(rng, D):
    """§2.2 SYM(D, rng): one item (invariants asserted). The prototype's make() at the v1 law, rng order kept."""
    assert D_MIN <= D <= D_MAX
    for _try in range(200):
        nid = [0]
        WR = _world(rng, D, nid); WP = _world(rng, D, nid)
        edges = list(WR["edges"]) + list(WP["edges"]); ncross = []
        for S, Tw in ((WP, WR), (WR, WP)):                           # item 4: c ~ U{0..2} each direction, direct-subtree source -> leaf target
            m = 0
            for _ in range(rng.randint(0, 2)):
                if S["dsub"] and Tw["dleaves"]:
                    edges.append((rng.choice(S["dsub"]), rng.choice(Tw["dleaves"]))); m += 1
            ncross.append(m)
        n_extra = rng.randint(0, 2); n_dec = len(WR["decoys"]) + len(WP["decoys"])  # item 5
        persons = ["R", "P"] + ["X%d" % i for i in range(n_extra)] + ["Y%d" % i for i in range(n_dec)]
        pid = {p: -(i + 1) for i, p in enumerate(persons)}
        edges += [(pid["R"], c) for c in WR["direct"]] + [(pid["P"], c) for c in WP["direct"]]
        allnodes = WR["nodes"] + WP["nodes"]; xtree = {}
        for xi, p in enumerate(persons[2:2 + n_extra]):
            c = nid[0] = nid[0] + 1; allnodes.append(c); edges.append((pid[p], c)); xtree[c] = (xi, 0); xtree[pid[p]] = (xi, -1)
            for q in range(rng.randint(1, 3)):
                v = nid[0] = nid[0] + 1; allnodes.append(v); edges.append((c, v)); xtree[v] = (xi, q + 1)
        dec = WR["decoys"] + WP["decoys"]
        for i, (ch, k) in enumerate(dec):
            edges.append((pid["Y%d" % i], ch[0]))
        gid = [pid[p] for p in persons] + allnodes; idx = {u: i for i, u in enumerate(gid)}
        if len(persons) > len(PERSON_40) or len(allnodes) > len(NONCE_300):
            continue
        names = rng.sample(PERSON_40, len(persons)) + rng.sample(NONCE_300, len(allnodes))   # item 6
        E = [[idx[a], idx[b]] for a, b in edges]
        order = list(range(len(E))); rng.shuffle(order)             # item 7
        oR, oP = idx[WR["path"][-1]], idx[WP["path"][-1]]
        o1, o2 = (oR, oP) if rng.random() < 0.5 else (oP, oR)
        it = {"D": D, "names": names, "edges": E, "order": order, "R": 0, "P": 1, "o_R": oR, "o_P": oP, "o1": o1, "o2": o2,
              "levels": [(idx[WR["path"][k]], idx[WP["path"][k]]) for k in range(D)], "n_cross": ncross,
              "n_decoys": [len(WR["decoys"]), len(WP["decoys"])], "n_extra": n_extra}
        _finish(it)
        if it["T"] > T_MAX:
            continue
        wset = {g: 0 for g in WR["nodes"]}; wset.update({g: 1 for g in WP["nodes"]})
        tag = []; ndec_R = len(WR["decoys"])
        for pos, g in enumerate(gid):
            if g in wset:
                tag.append(("c", wset[g], g - (WR if wset[g] == 0 else WP)["base"]))
            elif g in xtree:
                tag.append(("x", xtree[g][0], xtree[g][1]))
            elif pos < 2:
                tag.append(("p",))
            else:                                                   # decoy person Y_i: its world and its chain head's local id
                i = pos - 2 - n_extra; w = 0 if i < ndec_R else 1
                tag.append(("y", w, dec[i][0][0] - (WR if w == 0 else WP)["base"]))
        it["tag"] = tag; it["sig"] = sig_of(it)
        f = check(it)
        if f:
            raise SymInvariantError("D %d: %s" % (D, f))
        return it
    raise RuntimeError("T > %d in 200 draws at D %d" % (T_MAX, D))


def sig_of(it):
    """§2.2 item 9 (S-sigma2): sha256 of the name-free canonical form with the two worlds' strings sorted (AMBIGUITY A6). The world of R / P is
    read from its statements, so a PREMISE-SWAP donor has the recipient's signature."""
    tag, E = it["tag"], it["edges"]; pw = {}
    for a, b in E:
        if tag[a][0] == "p":
            if pw.setdefault(a, tag[b][1]) != tag[b][1]:
                raise SymInvariantError("person states into both worlds")

    def own(a):
        t = tag[a]; return pw[a] if t[0] == "p" else t[1] if t[0] in ("c", "y") else None

    def rep(i, w):
        t = tag[i]
        return ["p"] if t[0] == "p" else [t[0], int(t[1] != w), t[2]] if t[0] in ("c", "y") else [t[0], t[2]]
    ws = {0: [], 1: []}; xs = collections.defaultdict(list)
    for a, b in E:
        w = own(a)
        if w is None:
            xs[tag[a][1]].append(json.dumps([rep(a, None), rep(b, None)]))
        else:
            ws[w].append(json.dumps([rep(a, w), rep(b, w)]))
    form = [it["D"], sorted(json.dumps(sorted(ws[w])) for w in (0, 1)), sorted(json.dumps(sorted(v)) for v in xs.values())]
    return hashlib.sha256(json.dumps(form).encode()).hexdigest()


def reach_targets(it, t_max):
    """§2.3 REACH: E[t][i] = 1 (IN) iff dist(R, concepts[i]) <= t, t = 0..t_max; E[0] all OUT (R is a person, not a slot)."""
    big = 10 ** 9; d = [it["dist_R"].get(c, big) for c in it["concepts"]]
    return [[1 if (t > 0 and x <= t) else 0 for x in d] for t in range(t_max + 1)]


class Stream:
    """§2.2 item 10 streams with signature exclusion (AMBIGUITY A7): band D's draws come from random.Random(seed * 1000 + D) in order."""

    def __init__(self, seed, exclude=None):
        self.seed = seed; self.exclude = exclude if exclude is not None else set(); self.rngs = {}; self.n = collections.Counter(); self.collisions = []

    def draw(self, D):
        if D not in self.rngs:
            self.rngs[D] = random.Random(self.seed * 1000 + D)
        while True:
            it = make_item(self.rngs[D], D); self.n[D] += 1
            if it["sig"] not in self.exclude:
                return it
            self.collisions.append({"seed": self.seed, "D": D, "draw": self.n[D], "sig": it["sig"]})


def test_set(n_per_D=1000, D_range=range(3, 17), seed=SEED_TEST):
    """§2.2 item 10: the SYM v1 test set {D: [items]} (1,000 per D, D 3..16)."""
    s = Stream(seed); return {D: [s.draw(D) for _ in range(n_per_D)] for D in D_range}


def gate_set(n_per_D=2000, D_range=range(9, 13), seed=SEED_GATE):
    """§2.4 GATE items (2,000 per D 9-12, seed 22400)."""
    return test_set(n_per_D, D_range, seed)


def fixed_sigs(seed, n_per_D, D_range):
    """signatures of a fixed set without keeping the items."""
    s = Stream(seed); return {s.draw(D)["sig"] for D in D_range for _ in range(n_per_D)}


def exclusion_sigs():
    """§2.2 item 10: every test and GATE item's signature (the training / pool / probe exclusion set)."""
    return fixed_sigs(SEED_TEST, 1000, range(3, 17)) | fixed_sigs(SEED_GATE, 2000, range(9, 13))


def set_sha(items):
    """sha256 of an item list's rendered texts (the scorer's items_sha)."""
    return hashlib.sha256("\n".join(x["text"] for x in items).encode()).hexdigest()


def _derive(it, **kw):
    d = {k: it[k] for k in ("D", "names", "edges", "order", "R", "P", "o_R", "o_P", "o1", "o2", "levels", "tag", "n_cross", "n_decoys", "n_extra")}
    d.update(kw); return _finish(d)


def premise_swap(it):
    """§2.5 / §3.6 PREMISE-SWAP donor: R's and P's names exchanged in every premise, question still asks R; answer o_P. PS-0 asserted."""
    R, P = it["R"], it["P"]; sw = {R: P, P: R}
    don = _derive(it, edges=[[sw.get(a, a), b] for a, b in it["edges"]], o_R=it["o_P"], o_P=it["o_R"], levels=[(b, a) for a, b in it["levels"]])
    if don["T"] != it["T"] or don["slot_idx"] != it["slot_idx"]:
        raise SymInvariantError("PREMISE-SWAP positions")
    f = check(don)
    if f:
        raise SymInvariantError("PREMISE-SWAP donor: %s" % f)
    don["sig"] = sig_of(don)
    if don["sig"] != it["sig"]:
        raise SymInvariantError("PREMISE-SWAP signature not role-symmetric")
    return don


def person_swap(it):
    """§2.5 PERSON-SWAP: the question's person R -> P (answer o_P); positions asserted identical."""
    don = _derive(it, R=it["P"], P=it["R"], o_R=it["o_P"], o_P=it["o_R"], levels=[(b, a) for a, b in it["levels"]])
    if don["T"] != it["T"] or don["slot_idx"] != it["slot_idx"]:
        raise SymInvariantError("PERSON-SWAP positions")
    f = check(don)
    if f:
        raise SymInvariantError("PERSON-SWAP donor: %s" % f)
    don["sig"] = it["sig"]; return don


def cf_donor(it, j):
    """§2.5 CONSUMPTION-FRONT one-fact donor x': R's hop-j statement ("Every a_{j-1} is a a_j^R"; j = 1: "<R> is a a_1^R") gets the object
    a_j^P (same token length: every NONCE-300 name is two tokens). None if that statement is not unique."""
    D = it["D"]; lv = it["levels"]; src = it["R"] if j == 1 else lv[j - 2][0]; aR, aP = lv[j - 1]
    ks = [i for i, e in enumerate(it["edges"]) if e[0] == src and e[1] == aR]
    if len(ks) != 1:
        return None
    E2 = [list(e) for e in it["edges"]]; E2[ks[0]] = [src, aP]
    don = _derive(it, edges=E2, o_R=it["o_P"], o_P=it["o_R"]); don["levels"] = None; don["tag"] = None
    flip = don["dist_R"].get(it["o_P"]) == D and it["o_R"] not in don["dist_R"]
    same_len = don["T"] == it["T"]; same_slots = don["slot_idx"] == it["slot_idx"]
    return {"j": j, "k": ks[0], "flip": flip, "same_len": same_len, "same_slots": same_slots, "exposed": flip and same_len and same_slots, "donor": don}


def cf_select(items, rng=None, max_per_cell=400, js=None):
    """§2.5 CF pairs: up to max_per_cell exposed recipients per (D, j) (rng 22630); identity asserted per returned pair."""
    rng = rng if rng is not None else random.Random(SEED_CF); cells = collections.defaultdict(list)
    for i, it in enumerate(items):
        for j in (js(it["D"]) if js else range(1, it["D"] + 1)):
            r = cf_donor(it, j)
            if r is not None and r["exposed"]:
                if r["donor"]["T"] != it["T"] or r["donor"]["slot_idx"] != it["slot_idx"]:
                    raise SymInvariantError("CF positions")
                cells[(it["D"], j)].append((i, r))
    out = {}
    for key in sorted(cells):
        v = cells[key]; rng.shuffle(v); out[key] = v[:max_per_cell]
    return out


CUES = ("proximity", "parents", "mentions", "first", "last", "parents_indeg")


def panel_levels(it):
    """§59 person-agnostic panel (heur_panel.panel_for's six cues, same definitions and tie rule) at L0 = (R, P) and Lk = (a_k^R, a_k^P)."""
    names, E = it["names"], it["edges"]; n = len(names)
    dmin = bfs(_adj(n, E), [i for i in range(n) if names[i][:1].isupper()])
    par = [[] for _ in range(n)]; indeg = [0] * n
    for u, v in E:
        par[v].append(u); indeg[v] += 1
    toks = re.findall(r"[A-Za-z]+", it["text"].rsplit(" Is ", 1)[0]); cnt = collections.Counter(toks); fst = {}; lst = {}
    for i, t in enumerate(toks):
        fst.setdefault(t, i); lst[t] = i

    def cmp(sa, sb):
        return 1.0 if sa > sb else 0.0 if sa < sb else 0.5

    def pin(o):
        p = par[o]; return sum(indeg[u] for u in p) / len(p) if p else 0.0
    out = {}
    for k, (a, b) in enumerate([(it["R"], it["P"])] + list(it["levels"])):
        na, nb = names[a], names[b]
        out["L%d_proximity" % k] = cmp(dmin.get(a, 99), dmin.get(b, 99)); out["L%d_parents" % k] = cmp(len(par[a]), len(par[b]))
        out["L%d_mentions" % k] = cmp(cnt[na], cnt[nb]); out["L%d_first" % k] = cmp(-fst.get(na, 10 ** 9), -fst.get(nb, 10 ** 9))
        out["L%d_last" % k] = cmp(lst.get(na, -1), lst.get(nb, -1)); out["L%d_parents_indeg" % k] = cmp(pin(a), pin(b))
    return out


def chase_set(it, rng):
    """§3.3(e) single-parent backward chases from each option to a root (1 = R from o_R only, 0 = R from o_P only, 0.5 tie): first (text
    order) / random / maxdeg (max out-degree) / la<h> (h-hop backward lookahead: prefer parents that see R, avoid ones seeing another person
    only); soft = the full backward set. The prototype's chase(), rng order kept."""
    names, E = it["names"], it["edges"]; R = it["R"]
    par = collections.defaultdict(list); od = collections.Counter(a for a, _ in E)
    for j in it["order"]:
        a, b = E[j]; par[b].append(a)

    def score(u, h):
        seen = bfs(par, [u], h - 1) if h > 0 else {u: 0}
        if R in seen:
            return 1
        return -1 if any(names[v][:1].isupper() for v in seen) else 0

    def one(pick, h=0):
        roots = []
        for o in (it["o_R"], it["o_P"]):
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
                    sc = [score(u, h) for u in L]; m = max(sc); cur = rng.choice([u for u, s in zip(L, sc) if s == m])
                k += 1
            roots.append(cur)
        hit = [roots[0] == R, roots[1] == R]
        return 1.0 if hit == [True, False] else 0.0 if hit == [False, True] else 0.5
    out = {p: one(p) for p in ("first", "random", "maxdeg")}
    for h in (1, 2, 3, 4):
        out["la%d" % h] = one("la", h)
    hs = [R in bfs(par, [o]) for o in (it["o_R"], it["o_P"])]
    out["soft"] = 1.0 if hs == [True, False] else 0.0 if hs == [False, True] else 0.5
    return out


def _ud(it):
    n = len(it["names"]); a = [[] for _ in range(n)]
    for u, v in it["edges"]:
        a[u].append(v); a[v].append(u)
    return bfs(a, [it["R"]])


def ubidir(it, ud=None):
    """§3.3(d) uBIDIR(r), 2r < D: undirected r-balls of R and of each option meet iff ud(R, o) <= 2r; 1 / 0 / 0.5 (tie). Proved 0.5."""
    ud = ud if ud is not None else _ud(it); D = it["D"]; big = 10 ** 6; out = {}
    for r in range(1, (D + 1) // 2 + 1):
        if 2 * r < D:
            a, b = ud.get(it["o_R"], big) <= 2 * r, ud.get(it["o_P"], big) <= 2 * r
            out[r] = 1.0 if (a and not b) else 0.0 if (b and not a) else 0.5
    return out


def udist(it, ud=None):
    """§3.3(d) UDIST: the option undirected-nearer to R (1 / 0 / 0.5). Proved 1."""
    ud = ud if ud is not None else _ud(it); big = 10 ** 6; uR, uP = ud.get(it["o_R"], big), ud.get(it["o_P"], big)
    return 1.0 if uR < uP else 0.0 if uP < uR else 0.5


def beyond_D(it):
    """§3.3(c): True iff some concept has dist(R, c) > D (proved never)."""
    return max(it["dist_R"].values()) > it["D"]
