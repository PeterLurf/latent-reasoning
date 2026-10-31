"""§252 UNSCAFFOLDED (NS_v1 §2; Note 252-1): dependency-violating re-renders of OW items, shared by the trainer (--perm_p) and the eval (TEST-PERM).
Moved from scratch/s251ns_s0.py (622fb717, Audit G-D PASS; render / permuted unchanged). An item's op sentences S_1..S_d are reassigned over their own
slots in it["order"] by a non-identity permutation (>= 1 inversion; d >= 2 only); S_0, the givens G_t and the distractors D_k keep their slots; the text is
re-rendered by NW._finish from the item's own fields; OW.solve(text) == answer is asserted by the callers. Seed families (disjoint by construction):
  TEST-PERM (eval): default_rng(SEED_PERM_EVAL + d) per d, items in TEST2 order (622fb717's);
  training (--perm_p): default_rng((seed, SEED_PERM_TRAIN, step, k)) per item k of the step's batch (no draws when perm_p = 0)."""
import numpy as np

import boxes.nested_words as NW
import boxes.ow_words as OW

SEED_PERM_EVAL, SEED_PERM_TRAIN = 25150, 25160                             # unused elsewhere in boxes/*.py (grep 2026-10-31)


def render(it, order):
    nm = it["names"]
    o = NW._finish(it["d"], it["ops"], it["operands"], it["src_j"], it["values"], it["mul_s_t"], nm["spine"], nm["givens"], nm["distractors"], it["unit"],
                   it["frames"], it["distractor_values"], order, it["n_distractors"])
    return OW._decorate(o, it["order_mode"], None, it.get("dag", True))


def permuted(it, rng):
    """-> (the re-rendered item, its order, the number of inversions); d < 2: the item re-rendered in its own order, 0."""
    d = it["d"]; order = list(it["order"])
    if d < 2:
        return render(it, order), order, 0
    slots = [i for i, x in enumerate(order) if x in {"S%d" % t for t in range(1, d + 1)}]; assert [order[i] for i in slots] == ["S%d" % t for t in range(1, d + 1)], "REG chain order"
    while True:
        p = rng.permutation(d)
        if (p != np.arange(d)).any():
            break
    new = list(order)
    for k, i in enumerate(slots):
        new[i] = "S%d" % (p[k] + 1)
    inv = sum(int(p[i] > p[j]) for i in range(d) for j in range(i + 1, d)); return render(it, new), new, inv


def check(it, pit):
    for k_ in ("values", "answer", "names", "ops", "operands", "d", "question"):
        assert pit[k_] == it[k_], ("perm_render changed", k_)
    assert OW.solve(pit["text"]) == it["answer"], "perm_render: OW.solve != answer"


def test_perm_sets(raw, ds):
    """TEST-PERM from TEST2 (622fb717): {d: [items]} + per-d info; sig + ':perm'; item['perm'] = 1 for d >= 2."""
    out = {}; info = {}
    for d in ds:
        rng = np.random.default_rng(SEED_PERM_EVAL + d); its = []; invs = []
        for it in raw[d]:
            pit, order, inv = permuted(it, rng); check(it, pit)
            assert sorted(pit["text"].split(". ")) == sorted(it["text"].split(". ")) or d == 1, "TEST-PERM: the same sentences"
            pit["sig"] = it["sig"] + ":perm"; pit["perm"] = int(d >= 2); its.append(pit); invs.append(inv)
        out[d] = its; info[str(d)] = {"n": len(its), "min_inversions": int(min(invs)), "mean_inversions": float(np.mean(invs)),
                                      "changed_text_share": float(np.mean([a_["text"] != b_["text"] for a_, b_ in zip(its, raw[d])]))}
    return out, info


def train_perm(it, seed, step, k):
    """--perm_p's re-render of training item k at step (d >= 2; the caller draws the coin): the training seed family, solve asserted."""
    pit, _, inv = permuted(it, np.random.default_rng((seed, SEED_PERM_TRAIN, step, k))); check(it, pit); assert inv >= 1
    pit["perm"] = 1; return pit
