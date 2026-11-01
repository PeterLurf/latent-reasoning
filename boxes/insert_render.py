"""Note 255-3 / 255-5: irrelevant op sentences about NEW names inserted into OW items (answers unchanged; OW.solve asserted). Shared by the INSERT eval set builder
(scratch/s255_insert.py's rule, moved here unchanged; equality gated) and the trainer's --distract_p (Note 255-5).
A pair = a new GIVEN "Y has v u." (NW.GIVEN) + an op FRAMES["add_c"][f] "X ... c ... Y" (B = X, A = Y), f from OW.FRAME_TRAIN, v 1..9, c 1..10 - v (no wrap), names not in
the item's spine / givens / distractors. Every new given sits at the end of the given block (right before S1's sentence); the new op sentences go where the caller says.
Seed families: INSERT eval default_rng((25190, cond index, item index)); training (--distract_p) coin / k default_rng((seed, 25196, step, k)) and the draw
default_rng((seed, 25195, step, k)) - disjoint."""
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


def distract(it, seed, step, k):
    """--distract_p's training re-render of item k at step (the caller draws the p coin): n ~ U{0, 1, 2} pairs (default_rng((seed, 25195, step, k))); each new op at a
    uniformly drawn gap of the op block (before S1, between S_t and S_t+1, or after S_d; chain order of S_1..S_d kept). -> (item, n)."""
    rng = np.random.default_rng((seed, SEED_DISTRACT, step, k)); n = int(rng.integers(0, 3))
    if n == 0:
        return it, 0
    giv, ops, meta = draw_pairs(it, n, rng); d = it["d"]; gaps = [("before", "S1")] + [("after", "S%d" % t) for t in range(1, d + 1)]   # the d + 1 gaps of the op block
    o = assemble(it, giv, [(gaps[int(rng.integers(0, len(gaps)))], op) for op in ops], meta); o["distract_n"] = n; return o, n
