"""THEORY §270 FULL-PERM eval part (FP_v1 §3; registered in inject_aw_eval's run(); the run's own tokenization, fp32, no_grad) + the --perm_scope all disjointness sets:
  full_perm  (binding d 2-4, descriptive d 5-6; score_fp)  REG TEST2 d 2-6, all items: item i's order = default_rng((SEED_FP, d, i)).permutation(len(order)) over ALL
             statements (S_0, G_t, S_t, D_k), redrawn from the same rng while it is the identity; the question stays last (NW._finish appends it); PR.render + PR.check
             (fields, OW.solve == answer) -> by_d {d: {acc (K' = think_k), acc_k2 (K' 2), n, changed, by_flag}}; per-item sig / pred / pred_k2 / ans / order / layout flags.
  Layout flags (per item, from the new order): s0_pos (S_0's index); n3 (an op sentence before S_0); n4 (the last pre-question sentence is S_0 or a base given G_t);
  given_after_consumer (some G_t after its S_t); n_viol = n1 + n2 + n3 + n4 with n1 = some S_t before S_{t-1} (an op before its producer), n2 = given_after_consumer.
Helpers for inject_aw --perm_scope all (FP_v1 §2 disjointness): fp_texts (this part's texts, d 2-6), sg_texts (of_parts.part_shuffle_graded's two families' texts, d 2-4,
its exact order law; of_parts.py stays byte-identical), msig (the item's statement multiset, question excluded).
Seed 27010 is unused elsewhere in boxes / theory / scratch *.py (grep 2026-11-04)."""
import numpy as np
import torch

SEED_FP = 27010
FP_DS = (2, 3, 4, 5, 6)


def _ops(d):
    return ["S%d" % t for t in range(1, d + 1)]


def fp_order(it, i):
    """item i of TEST2 depth it["d"]: a non-identity permutation of all of it["order"] from default_rng((SEED_FP, d, i))."""
    o = list(it["order"]); r = np.random.default_rng((SEED_FP, it["d"], i))
    while True:
        p = r.permutation(len(o))
        if (p != np.arange(len(o))).any():
            return [o[q] for q in p]


def fp_items(raw, d):
    """-> the full_perm items of depth d (sig + ':fp'; perm = 1); fields and OW.solve == answer asserted; the question last."""
    import boxes.perm_render as PR
    out = []
    for i, it in enumerate(raw):
        new = fp_order(it, i); pit = PR.render(it, new); PR.check(it, pit)
        assert pit["text"].endswith(" " + pit["question"]) and msig(pit) == msig(it), "full_perm: question last, same statements"
        pit["sig"] = it["sig"] + ":fp"; pit["perm"] = 1; out.append(pit)
    return out


def fp_texts(test2):
    return {p["text"] for d in FP_DS for p in fp_items(test2[d], d)}


def sg_texts(test2):
    """of_parts.part_shuffle_graded's rendered texts (independent: default_rng((26310, d, i)) over the non-op slots; full: default_rng((26311, d, i)) over all), d 2-4."""
    import boxes.of_parts as OFP
    import boxes.perm_render as PR
    out = set()
    for d in (2, 3, 4):
        ops = _ops(d)
        for i, it in enumerate(test2[d]):
            o = list(it["order"]); sl = [k for k, x in enumerate(o) if x not in ops]; pm = np.random.default_rng((OFP.SEED_SG, d, i)).permutation(len(sl)); ind = list(o)
            for k, q in zip(sl, pm):
                ind[k] = o[sl[q]]
            full = [o[q] for q in np.random.default_rng((OFP.SEED_SG + 1, d, i)).permutation(len(o))]
            out |= {PR.render(it, ind)["text"], PR.render(it, full)["text"]}
    return out


def msig(it):
    """the item's statement multiset (the question excluded): tuple(sorted(statements)); statements are '. '-joined, each ending '.'."""
    t, q = it["text"], it["question"]; assert t.endswith(" " + q), "msig: the question last"
    return tuple(sorted(x.rstrip(".") for x in t[:-len(q) - 1].split(". ")))      # the last statement keeps its '.': normalised


def flags(order, d):
    pos = {x: k for k, x in enumerate(order)}; ops = _ops(d)
    gac = any(pos["S%d" % t] < pos["G%d" % t] for t in range(1, d + 1) if "G%d" % t in pos)
    n1 = any(pos["S%d" % t] < pos["S%d" % (t - 1)] for t in range(2, d + 1))
    n3 = any(pos[s] < pos["S0"] for s in ops)
    last = order[-1]; n4 = last == "S0" or (last.startswith("G") and last[1:].isdigit())
    return {"s0_pos": pos["S0"], "n3": int(n3), "n4": int(n4), "given_after_consumer": int(gac), "n_viol": int(n1) + int(gac) + int(n3) + int(n4)}


@torch.no_grad()
def part_full_perm(m, g, sets, a, dev, bs, ds=FP_DS):
    import boxes.inject_aw_eval as EV
    K = a["think_k"]; out = {"by_d": {}, "items": {}}
    for d in ds:
        raw = sets["test2"][d]; its = fp_items(raw, d); T = EV.tokd({"test2": {d: its}}, a, (d,))[d]
        A, _ = EV.passes(m, T, dev, bs, {K, 2}); y = np.array([it["ans"] for it in T]); fl = [flags(it["raw"]["order"], d) for it in T]
        ok, ok2 = (A[K] == y), (A[2] == y); byf = {}
        for key in ("n3", "n4", "given_after_consumer", "n_viol", "s0_pos"):
            for v in sorted({f[key] for f in fl}):
                sel = np.array([f[key] == v for f in fl]); byf.setdefault(key, {})[str(v)] = {"acc": float(ok[sel].mean()), "n": int(sel.sum())}
        out["by_d"][str(d)] = {"acc": float(ok.mean()), "acc_k2": float(ok2.mean()), "n": len(T),
                               "changed": float(np.mean([p["text"] != q["text"] for p, q in zip(its, raw)])), "by_flag": byf}
        out["items"][str(d)] = {"sig": [it["sig"] for it in T], "pred": A[K].tolist(), "pred_k2": A[2].tolist(), "ans": y.tolist(),
                                "order": [it["raw"]["order"] for it in T], "flags": fl}
    out["def"] = ("§270 FULL-PERM (FP_v1 §3): REG TEST2 d 2-6, all items, every statement (S_0, G_t, S_t, D_k) permuted by default_rng((27010, d, i)) (non-identity, "
                  "redrawn), question last; OW.solve == answer asserted; acc at K' = think_k and K' 2; d 2-4 binding, d 5-6 descriptive")
    return out
