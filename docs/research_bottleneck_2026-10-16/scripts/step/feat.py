"""Per-step structural features of Feng's leftmost-ready reductions (nested170 order), for the step-capability diagnosis.
Run with PYTHONPATH unset; imports D:/Latent/boxes/nested170.py read-only via importlib. CPU only."""
import importlib.util
import json
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("n170", "D:/Latent/boxes/nested170.py")
N = importlib.util.module_from_spec(spec); spec.loader.exec_module(N)


def height(nd):
    return 0 if nd.op is None else 1 + max(height(nd.l), height(nd.r))


def steps_features(root):
    """-> list over steps t = 1..n of dicts: prov (0/1/2 computed operands), age_max, age_min, held (computed values alive in E_{t-1} not consumed),
    alive (live symbols in E_{t-1}), hh (handle subtree height in the original tree), bdepth (enclosing brackets of the handle in E_{t-1}),
    lc/rc (left / right operand computed), r (ops left incl. this)."""
    nd = N._copy(root)
    # annotate original heights: map copy nodes -> height via a parallel walk
    H = {}

    def ann(a):
        H[id(a)] = height(a)
        if a.op is not None:
            ann(a.l); ann(a.r)
    ann(nd)
    keep = list(iter_ops(nd)) + leaves(nd)   # hold references: ids stay unique
    comp = {}          # id(node) -> step it was computed at (for reduced nodes, now leaves)
    n_ops = sum(1 for _ in iter_ops(nd)); out = []; t = 0
    while nd.op is not None:
        t += 1
        path = N._leftmost_ready(nd); h = nd; par = []
        for p_ in path:
            par.append(h); h = getattr(h, p_)
        # bracket depth: count enclosing ancestors whose rendering adds brackets around their subtree containing h
        bd = 0; anc = [nd] + par[1:] if par else [nd]
        chain = par + [h]
        for k in range(1, len(chain)):
            child, parent, right = chain[k], chain[k - 1], path[k - 1] == "r"
            if child.op is not None and (N.PREC[child.op] < N.PREC[parent.op] or (right and N.PREC[child.op] == N.PREC[parent.op])):
                bd += 1
        lc = id(h.l) in comp; rc = id(h.r) in comp
        ages = [t - comp[id(c)] for c in (h.l, h.r) if id(c) in comp]
        alive_comp = [k for k in leaves(nd) if id(k) in comp]
        held = len([k for k in alive_comp if k is not h.l and k is not h.r])
        out.append({"t": t, "prov": int(lc) + int(rc), "lc": int(lc), "rc": int(rc), "age_max": max(ages) if ages else 0, "age_min": min(ages) if ages else 0,
                    "held": held, "alive": len(N.render(nd)), "hh": H[id(h)], "bdepth": bd, "r": n_ops - t + 1, "op": h.op})
        v = N.apply(h.op, h.l.v, h.r.v); h.v = v; h.op = h.l = h.r = None; comp[id(h)] = t
    return out


def iter_ops(nd):
    if nd.op is not None:
        yield nd; yield from iter_ops(nd.l); yield from iter_ops(nd.r)


def leaves(nd):
    if nd.op is None:
        return [nd]
    return leaves(nd.l) + leaves(nd.r)


if __name__ == "__main__":
    TE = N.test_set(1000)
    feats = [steps_features(it["root"]) for it in TE]
    json.dump({"n": [it["n"] for it in TE], "feats": feats}, open(sys.argv[1], "w"))
    print("ok", len(feats))
