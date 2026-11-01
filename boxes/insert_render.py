"""Note 255-3 / 255-5: irrelevant op sentences about NEW names inserted into OW items (answers unchanged; OW.solve asserted). Shared by the INSERT eval set builder
(scratch/s255_insert.py's rule, moved here unchanged; equality gated) and the trainer's --distract_p (Note 255-5).
A pair = a new GIVEN "Y has v u." (NW.GIVEN) + an op FRAMES["add_c"][f] "X ... c ... Y" (B = X, A = Y), f from OW.FRAME_TRAIN, v 1..9, c 1..10 - v (no wrap), names not in
the item's spine / givens / distractors. Every new given sits at the end of the given block (right before S1's sentence); the new op sentences go where the caller says.
Seed families: INSERT eval default_rng((25190, cond index, item index)); training (--distract_p) coin / k default_rng((seed, 25196, step, k)) and the draw
default_rng((seed, 25195, step, k)) - disjoint."""
import hashlib

import numpy as np

import boxes.nested_words as NW
import boxes.ow_words as OW

SEED_INSERT_EVAL, SEED_DISTRACT, SEED_DISTRACT_COIN = 25190, 25195, 25196   # unused elsewhere in boxes/*.py (grep 2026-11-01)
CONDS = ["clean", "before_k1", "before_k2", "inside_k1", "inside_k2", "after_k1", "after_k2"]


def statements(it):
    """the item's sentences in it['order'] (NW._statements rendered as NW._finish does) + its question; ' '.join(sents + [q]) == it['text'] (asserted by callers)."""
    nm = it["names"]; st, _ = NW._statements(it["ops"], it["operands"], it["src_j"], it["values"], nm["spine"], nm["givens"], nm["distractors"], it["frames"],
                                             it["distractor_values"])
    byid = {x["id"]: x for x in st}; u = it["unit"]

    def render(x):
        if x["kind"] in ("given", "distractor"):
            return NW.GIVEN.format(A=x["name"], v=x["v"], u=u)
        return NW.FRAMES[x["op"]][x["frame"]].format(B=x["B"], A=x["A"], C=x["C"], c=x["c"], u=u)
    return [render(byid[i]) for i in it["order"]], NW.QUESTION.format(u=u, A=nm["spine"][-1])


def draw_pairs(it, k, rng):
    """k (given, op) pairs on new names; -> ([("NG<j>", text)], [("NO<j>", text)], meta)."""
    used = set(it["names"]["spine"]) | set(it["names"]["givens"]) | set(it["names"]["distractors"])
    free = [n for n in OW.NAMES if n not in used]; pick = rng.choice(len(free), size=2 * k, replace=False); giv = []; ops = []; meta = []
    for j in range(k):
        Y, X = free[pick[2 * j]], free[pick[2 * j + 1]]; v = int(rng.integers(1, 10)); c = int(rng.integers(1, 11 - v)); f = int(OW.FRAME_TRAIN[int(rng.integers(0, len(OW.FRAME_TRAIN)))])
        giv.append(("NG%d" % j, NW.GIVEN.format(A=Y, v=v, u=it["unit"]))); ops.append(("NO%d" % j, NW.FRAMES["add_c"][f].format(B=X, A=Y, C=None, c=c, u=it["unit"])))
        meta.append({"Y": Y, "X": X, "v": v, "c": c, "frame": f})
    return giv, ops, meta


def assemble(it, giv, ops_at, meta):
    """the new item: givens at the end of the given block; ops_at = [((side, label), ("NO<j>", text))], side 'before' / 'after' the op sentence `label`; ops sharing a gap
    go in as one block in their given order (the INSERT eval's rule); solve asserted; indices recorded in meta."""
    sents, q = statements(it); assert " ".join(sents + [q]) == it["text"], "insert_render: clean rebuild"
    seq = list(zip(it["order"], sents)); s1 = it["order"].index("S1"); seq = seq[:s1] + giv + seq[s1:]
    blocks = {}
    for gap, op in ops_at:
        blocks.setdefault(gap, []).append(op)
    for (side, label), blk in blocks.items():
        lab = [x[0] for x in seq]; at = lab.index(label) + (1 if side == "after" else 0); seq = seq[:at] + blk + seq[at:]
    lab = [x[0] for x in seq]
    for j, m_ in enumerate(meta):
        m_["given_idx"] = lab.index("NG%d" % j); m_["op_idx"] = lab.index("NO%d" % j)
    o = dict(it); o["text"] = " ".join([x[1] for x in seq] + [q]); o["insert_labels"] = lab   # the final sentence labels (S* / G* / D* / NG* / NO*), for the gates
    assert OW.solve(o["text"]) == it["answer"], ("insert_render: solve != answer", it["sig"]); return o


def insert_eval(it, cond, idx):
    """the INSERT eval rule (Note 255-3 + Audit's G-D fix): k pairs, ops before S1 / after S2 (before S3) / after S4."""
    if cond == "clean":
        sents, q = statements(it); assert " ".join(sents + [q]) == it["text"], "insert_render: clean rebuild"; return dict(it), []
    pos, k = cond.split("_k"); k = int(k); giv, ops, meta = draw_pairs(it, k, np.random.default_rng((SEED_INSERT_EVAL, CONDS.index(cond), idx)))
    gap = {"before": ("before", "S1"), "inside": ("after", "S2"), "after": ("after", "S4")}[pos]   # the INSERT eval's rule (Note 255-3)
    o = assemble(it, giv, [(gap, op) for op in ops], meta); o["sig"] = it["sig"] + ":" + cond; return o, meta


def insert_eval_sets(raw):
    sets = {}; metas = {}
    for c in CONDS:
        its = []; ms = []
        for i, it in enumerate(raw):
            o, m = insert_eval(it, c, i); its.append(o); ms.append(m)
        sets[c] = its; metas[c] = ms
    return sets, metas


SEED_HO = 25200                                                             # §256 held-out families (DS_v1 §3): default_rng((25200, family index, position index, item))
HO_FAMS = {"H-OP-sub_c": ["before", "inside", "after"], "H-OP-mul_c": ["before", "inside", "after"], "H-OP-add_v": ["before", "inside", "after"],
           "H-CHAINSRC": ["1", "2", "3"], "H-GIVPOS": ["start", "before_op", "after_Sd"], "H-EXTEND": ["after_Sd"], "H-LIVE": ["before", "inside", "after"], "K3": ["mixed"]}
GAP = {"before": ("before", "S1"), "inside": ("after", "S2"), "after": ("after", "S4")}


def _names(it, rng, n):
    used = set(it["names"]["spine"]) | set(it["names"]["givens"]) | set(it["names"]["distractors"]); free = [x for x in OW.NAMES if x not in used]
    return [free[i] for i in rng.choice(len(free), size=n, replace=False)]


def _frame(rng):
    return int(OW.FRAME_TRAIN[int(rng.integers(0, len(OW.FRAME_TRAIN)))])


def _seq(it):
    sents, q = statements(it); assert " ".join(sents + [q]) == it["text"], "insert_render: clean rebuild"; return list(zip(it["order"], sents)), q


def _put(seq, x, side, label):
    lab = [y[0] for y in seq]; at = lab.index(label) + (1 if side == "after" else 0); return seq[:at] + x + seq[at:]


def _finish_ho(it, seq, q, meta, tag):
    lab = [x[0] for x in seq]; o = dict(it); o["text"] = " ".join([x[1] for x in seq] + [q]); o["insert_labels"] = lab; o["sig"] = it["sig"] + ":" + tag
    assert OW.solve(o["text"]) == it["answer"], ("held-out insert: solve != answer", tag, it["sig"]); meta["labels"] = lab
    meta["op_idx"] = [lab.index(x) for x in sorted(y for y in lab if y.startswith("NO"))]; return o   # §257: the insert ops' sentence indices


def held_out(it, fam, pos, idx):
    """§256 DS_v1 §3 held-out insert families on a d = 4 item (answers unchanged; solve asserted). New givens go at the end of the given block unless the family moves them."""
    fi = list(HO_FAMS).index(fam); pi = HO_FAMS[fam].index(pos); rng = np.random.default_rng((SEED_HO, fi, pi, idx)); u = it["unit"]; sp = it["names"]["spine"]
    seq, q = _seq(it); s1 = [x[0] for x in seq].index("S1"); meta = {"fam": fam, "pos": pos}
    if fam.startswith("H-OP"):
        op = fam[len("H-OP-"):]; f = _frame(rng)
        if op == "add_v":
            Y, Z, X = _names(it, rng, 3); vy, vz = int(rng.integers(1, 10)), int(rng.integers(1, 10))
            giv = [("NG0", NW.GIVEN.format(A=Y, v=vy, u=u)), ("NG1", NW.GIVEN.format(A=Z, v=vz, u=u))]; txt = NW.FRAMES["add_v"][f].format(B=X, A=Y, C=Z, u=u)
        else:
            Y, X = _names(it, rng, 2); vy = int(rng.integers(1, 10)); c = int(rng.integers(2, 10)) if op == "mul_c" else int(rng.integers(1, 10))
            giv = [("NG0", NW.GIVEN.format(A=Y, v=vy, u=u))]; txt = NW.FRAMES[op][f].format(B=X, A=Y, C=None, c=c, u=u)
        seq = seq[:s1] + giv + seq[s1:]; seq = _put(seq, [("NO0", txt)], *GAP[pos])
        meta.update(frame=f, op=op, v=vy, c=None if op == "add_v" else c, **({"v2": vz} if op == "add_v" else {}))   # §257: meta only
    elif fam == "H-CHAINSRC":
        j = int(pos); (X,) = _names(it, rng, 1); f = _frame(rng); c = int(rng.integers(1, 10))
        seq = _put(seq, [("NO0", NW.FRAMES["add_c"][f].format(B=X, A=sp[j], C=None, c=c, u=u))], "after", "S%d" % j); meta.update(frame=f, j=j, op="add_c", c=c, v=int(it["values"][j]))
    elif fam == "H-GIVPOS":
        Y, X = _names(it, rng, 2); f = _frame(rng); vy = int(rng.integers(1, 10)); c = int(rng.integers(1, 10))
        g = [("NG0", NW.GIVEN.format(A=Y, v=vy, u=u))]; op = [("NO0", NW.FRAMES["add_c"][f].format(B=X, A=Y, C=None, c=c, u=u))]
        seq = _put(seq, op, *GAP["inside"])                                # the op is fixed after S2; only the given moves
        if pos == "start":
            seq = g + seq
        elif pos == "before_op":
            seq = _put(seq, g, "before", "NO0")
        else:
            seq = _put(seq, g, "after", "S4")
        meta.update(frame=f, op="add_c", c=c, v=vy)
    elif fam == "H-EXTEND":
        X1, X2 = _names(it, rng, 2); f1, f2 = _frame(rng), _frame(rng); c1, c2 = int(rng.integers(1, 10)), int(rng.integers(1, 10))
        ops = [("NO0", NW.FRAMES["add_c"][f1].format(B=X1, A=sp[4], C=None, c=c1, u=u)), ("NO1", NW.FRAMES["add_c"][f2].format(B=X2, A=X1, C=None, c=c2, u=u))]
        seq = _put(seq, ops, "after", "S4"); meta.update(frames=[f1, f2], op="add_c", j=4, c=[c1, c2], v=int(it["values"][4]))
    elif fam == "H-LIVE":
        Y, X1, X2 = _names(it, rng, 3); f1, f2 = _frame(rng), _frame(rng); vy = int(rng.integers(1, 10)); c1, c2 = int(rng.integers(1, 10)), int(rng.integers(1, 10))
        seq = seq[:s1] + [("NG0", NW.GIVEN.format(A=Y, v=vy, u=u))] + seq[s1:]
        ops = [("NO0", NW.FRAMES["add_c"][f1].format(B=X1, A=Y, C=None, c=c1, u=u)), ("NO1", NW.FRAMES["add_c"][f2].format(B=X2, A=X1, C=None, c=c2, u=u))]
        seq = _put(seq, ops, *GAP[pos]); meta.update(frames=[f1, f2], op="add_c", c=[c1, c2], v=vy)
    elif fam == "K3":
        giv, ops, m3 = draw_pairs(it, 3, rng); gaps = [("before", "S1")] + [("after", "S%d" % t) for t in range(1, 5)]
        seq = seq[:s1] + giv + seq[s1:]
        for op in ops:
            seq = _put(seq, [op], *gaps[int(rng.integers(0, len(gaps)))])
        meta.update(pairs=m3, op="add_c")
    return _finish_ho(it, seq, q, meta, "%s:%s" % (fam, pos)), meta


def held_out_sets(raw):
    """{fam: {pos: [items]}} + metas on TEST2 d 4 (1,000 per (family, position))."""
    sets = {}; metas = {}
    for fam, poss in HO_FAMS.items():
        for p in poss:
            its = []; ms = []
            for i, it in enumerate(raw):
                o, m = held_out(it, fam, p, i); its.append(o); ms.append(m)
            sets.setdefault(fam, {})[p] = its; metas.setdefault(fam, {})[p] = ms
    return sets, metas


def set_shas(sets):
    """sha256 over the texts of each (family, position) set (DS_v1 §5: committed from a clean build, asserted on the host)."""
    return {fam: {p: hashlib.sha256("\n".join(x["text"] for x in its).encode()).hexdigest() for p, its in d.items()} for fam, d in sets.items()}


def distract(it, seed, step, k):
    """--distract_p's training re-render of item k at step (the caller draws the p coin): n ~ U{0, 1, 2} pairs (default_rng((seed, 25195, step, k))); each new op at a
    uniformly drawn gap of the op block (before S1, between S_t and S_t+1, or after S_d; chain order of S_1..S_d kept). -> (item, n)."""
    rng = np.random.default_rng((seed, SEED_DISTRACT, step, k)); n = int(rng.integers(0, 3))
    if n == 0:
        return it, 0
    giv, ops, meta = draw_pairs(it, n, rng); d = it["d"]; gaps = [("before", "S1")] + [("after", "S%d" % t) for t in range(1, d + 1)]   # the d + 1 gaps of the op block
    o = assemble(it, giv, [(gaps[int(rng.integers(0, len(gaps)))], op) for op in ops], meta); o["distract_n"] = n; return o, n


SEED_MIX, SEED_HO2 = 25205, 25210                                           # §257 (DC2_v1): --distract_mix draws; the §257 held-out sets (unused elsewhere in boxes/ + scratch/, grep 11-01)
MIX_TYPES = ("s256", "chain", "givop")


def distract_mix(it, seed, step, k, types=MIX_TYPES):
    """§257 --distract_mix: the training re-render of item k at step (the caller draws the p coin): n ~ U{0, 1, 2} inserts (default_rng((seed, 25205, step, k))), each of a
    type drawn uniformly from `types` ('chain' only when d >= 2):
      s256   a new given Y at the end of the given block + "X has c more than Y" (add_c) at a uniform gap of the op block (the §256 rule: v 1..9, c 1..10 - v);
      chain  "X has c more than S_j" (add_c, c 1..9, no new given) directly after S_j's sentence, j ~ U{1..d-1} (placed last, so nothing lands between S_j and it);
      givop  a new given Y directly before its "X has c more than Y" (v 1..9, c 1..10 - v), the pair at a uniform gap of the op block.
    Names new and distinct; frames FRAME_TRAIN; S_1..S_d keep their order; solve asserted. -> (item, n); item["distract_types"] / ["distract_meta"] / ["insert_labels"]."""
    rng = np.random.default_rng((seed, SEED_MIX, step, k)); n = int(rng.integers(0, 3))
    if n == 0:
        return it, 0
    d = it["d"]; u = it["unit"]; sp = it["names"]["spine"]; av = [t for t in types if t != "chain" or d >= 2]
    ty = [av[int(rng.integers(0, len(av)))] for _ in range(n)]; nm = _names(it, rng, 2 * n); gaps = [("before", "S1")] + [("after", "S%d" % t) for t in range(1, d + 1)]
    seq, q = _seq(it); meta = []; mid = []; late = []
    for i, t in enumerate(ty):
        X, Y = nm[2 * i], nm[2 * i + 1]; f = _frame(rng)
        if t == "chain":
            j = int(rng.integers(1, d)); c = int(rng.integers(1, 10))
            late.append(([("NO%d" % i, NW.FRAMES["add_c"][f].format(B=X, A=sp[j], C=None, c=c, u=u))], "after", "S%d" % j))
            meta.append({"type": t, "X": X, "j": j, "c": c, "v": int(it["values"][j]), "frame": f})
        else:
            v = int(rng.integers(1, 10)); c = int(rng.integers(1, 11 - v)); gap = gaps[int(rng.integers(0, len(gaps)))]
            g = ("NG%d" % i, NW.GIVEN.format(A=Y, v=v, u=u)); op = ("NO%d" % i, NW.FRAMES["add_c"][f].format(B=X, A=Y, C=None, c=c, u=u))
            if t == "s256":
                s1 = [x[0] for x in seq].index("S1"); seq = seq[:s1] + [g] + seq[s1:]; mid.append(([op], gap[0], gap[1]))   # givens first, ops after all givens are in
            else:
                mid.append(([g, op], gap[0], gap[1]))
            meta.append({"type": t, "X": X, "Y": Y, "v": v, "c": c, "frame": f, "gap": list(gap)})
    for x, side, lab in mid + late:
        seq = _put(seq, x, side, lab)
    lab = [x[0] for x in seq]
    for i, m_ in enumerate(meta):
        m_["op_idx"] = lab.index("NO%d" % i); m_["given_idx"] = lab.index("NG%d" % i) if "NG%d" % i in lab else None
    o = dict(it); o["text"] = " ".join([x[1] for x in seq] + [q]); o["insert_labels"] = lab; o["distract_n"] = n; o["distract_types"] = ty; o["distract_meta"] = meta
    assert OW.solve(o["text"]) == it["answer"], ("distract_mix: solve != answer", it["sig"]); return o, n


HO2_FAMS = {"H-CHAINSRC-OP-sub_c": ["1", "2", "3"], "H-CHAINSRC-OP-mul_c": ["1", "2", "3"], "H-CHAINSRC-LIVE": ["1", "2", "3"]}   # §257 DC2_v1 §3 (+ Theory 11-01: LIVE)


def held_out2(it, fam, pos, idx):
    """§257 held-out chain-reading families on a d = 4 item (answers unchanged; solve asserted); position = j (the insert directly after S_j's sentence, no new given):
      H-CHAINSRC-OP-sub_c / -mul_c  "X has c fewer than S_j" (c 1..9) / "X has c times as many as S_j" (c 2..9);
      H-CHAINSRC-LIVE               "X1 has c more than S_j", then "X2 has c' more than X1" right after it (c, c' 1..9).
    default_rng((25210, family index, position index, item)); frames FRAME_TRAIN. meta: c, v (= S_j's value), j, op, frame(s), labels, op_idx."""
    fi = list(HO2_FAMS).index(fam); pi = HO2_FAMS[fam].index(pos); rng = np.random.default_rng((SEED_HO2, fi, pi, idx)); u = it["unit"]; sp = it["names"]["spine"]
    j = int(pos); seq, q = _seq(it); meta = {"fam": fam, "pos": pos, "j": j, "v": int(it["values"][j])}
    if fam.startswith("H-CHAINSRC-OP-"):
        op = fam[len("H-CHAINSRC-OP-"):]; (X,) = _names(it, rng, 1); f = _frame(rng); c = int(rng.integers(2, 10)) if op == "mul_c" else int(rng.integers(1, 10))
        seq = _put(seq, [("NO0", NW.FRAMES[op][f].format(B=X, A=sp[j], C=None, c=c, u=u))], "after", "S%d" % j); meta.update(op=op, frame=f, c=c)
    else:
        X1, X2 = _names(it, rng, 2); f1, f2 = _frame(rng), _frame(rng); c1, c2 = int(rng.integers(1, 10)), int(rng.integers(1, 10))
        ops = [("NO0", NW.FRAMES["add_c"][f1].format(B=X1, A=sp[j], C=None, c=c1, u=u)), ("NO1", NW.FRAMES["add_c"][f2].format(B=X2, A=X1, C=None, c=c2, u=u))]
        seq = _put(seq, ops, "after", "S%d" % j); meta.update(op="add_c", frames=[f1, f2], c=[c1, c2])
    return _finish_ho(it, seq, q, meta, "%s:%s" % (fam, pos)), meta


def held_out2_sets(raw):
    """{fam: {pos: [items]}} + metas on TEST2 d 4 (1,000 per (family, position))."""
    sets = {}; metas = {}
    for fam, poss in HO2_FAMS.items():
        for p in poss:
            its = []; ms = []
            for i, it in enumerate(raw):
                o, m = held_out2(it, fam, p, i); its.append(o); ms.append(m)
            sets.setdefault(fam, {})[p] = its; metas.setdefault(fam, {})[p] = ms
    return sets, metas
