"""THEORY §263 ORDER-FREE eval parts (registered in inject_aw_eval's run() by scratch/s270_build/edits_s270e.py; the run's own tokenization, fp32, no_grad):
  late_premise    (binding, score_of)  REG TEST2 d 2-4 (all items) re-rendered with S_1's op sentence LAST among the op sentences, the others in chain order
                  (S_2..S_d, S_1 over the op slots; givens / distractors keep their slots; OW.solve == answer asserted) -> by_d {d: {acc, n}} (acc at K' = think_k).
  name_swap       (descriptive, Audit L1b)  REG TEST2 d 3-4: pass 1 as the run; at pass 2 the cue's entity token (S_2, 6 before ':') is replaced by (a) another chain
                  entity S_j, j in {1..d} minus {2}, drawn per item from default_rng((SEED_NS, d, i)), and (b) a random name absent from the item (the floor); the
                  pass-2 ':' readout (argmax over ' 0'..' 10') is compared with v_j: follow = P(swap readout == v_j), floor = P(floor readout == v_j); + the intact
                  pass-2 readout acc (== v_2) and follow by j.
  shuffle_graded  (descriptive)  REG TEST2 d 2-4: (a) independent-only shuffle - the non-op sentences (S_0, givens, distractors) permuted over their own slots
                  (default_rng((SEED_SG, d, i))), the op sentences S_1..S_d keep their slots; (b) full shuffle - every sentence permuted (default_rng((SEED_SG + 1, d, i)));
                  solve asserted -> by_d {d: {independent: {acc, changed}, full: {acc, changed, op_chain_order}}}.
Seeds 26300 / 26310 / 26311 are unused elsewhere in boxes/*.py and scratch/*.py (grep 2026-11-03)."""
import numpy as np
import torch

SEED_NS, SEED_SG = 26300, 26310
CUE_NAME_OFF = 6                                                            # ':' - 6 = QUESTION's A (A, ' have', '?', ' The', ' answer', ' is', ':')


def _ops(d):
    return ["S%d" % t for t in range(1, d + 1)]


def late_order(it):
    d = it["d"]; o = list(it["order"]); ops = _ops(d); slots = [i for i, x in enumerate(o) if x in ops]
    assert [o[i] for i in slots] == ops, "late_premise: REG chain order"
    for i, x in zip(slots, ops[1:] + ops[:1]):
        o[i] = x
    return o


def _render(it, order, tag):
    import boxes.perm_render as PR
    pit = PR.render(it, order); PR.check(it, pit); pit["sig"] = it["sig"] + ":" + tag; pit["perm"] = 1; return pit


@torch.no_grad()
def part_late_premise(m, g, sets, a, dev, bs):
    import boxes.inject_aw_eval as EV
    K = a["think_k"]; out = {"by_d": {}, "items": {}}
    for d in (2, 3, 4):
        its = [_render(it, late_order(it), "late") for it in sets["test2"][d]]; T = EV.tokd({"test2": {d: its}}, a, (d,))[d]
        A, _ = EV.passes(m, T, dev, bs, {K}); y = np.array([it["ans"] for it in T])
        out["by_d"][str(d)] = {"acc": float((A[K] == y).mean()), "n": len(T), "changed_text_share": float(np.mean([p["text"] != q["text"] for p, q in zip(its, sets["test2"][d])]))}
        out["items"][str(d)] = {"sig": [it["sig"] for it in T], "pred": A[K].tolist(), "ans": y.tolist(), "order": [it["raw"]["order"] for it in T]}
    out["def"] = ("§263 LATE-PREMISE (Audit L1a): REG TEST2 d 2-4 with S_1's op sentence last among the op sentences (S_2..S_d, S_1 over the op slots), the rest unchanged; "
                  "acc at K' = think_k; OW.solve == answer asserted")
    return out


@torch.no_grad()
def part_name_swap(m, g, sets, a, dev, bs):
    import boxes.inject_aw as AW
    import boxes.inject_aw_eval as EV
    import boxes.nested_words as NW
    from boxes.inject_ow import ENC
    from boxes.nested170 import NUM_IDS
    tok = {n: ENC.encode(" " + n)[0] for n in NW.NAMES}; assert all(len(ENC.encode(" " + n)) == 1 for n in NW.NAMES); num = torch.tensor(NUM_IDS, device=dev)
    out = {"by_d": {}, "items": {}}
    for d in (3, 4):
        T = EV.tokd({"test2": {d: sets["test2"][d]}}, a, (d,))[d]; js, fl = [], []
        for i, it in enumerate(T):
            r = np.random.default_rng((SEED_NS, d, i)); nm = it["raw"]["names"]
            js.append(int(r.choice([j for j in range(1, d + 1) if j != 2])))
            used = set(nm["spine"]) | set(nm["givens"]) | set(nm["distractors"]); fl.append(str(r.choice(sorted(n for n in NW.NAMES if n not in used))))
        res = {"intact": [], "swap": [], "floor": []}
        for b0 in range(0, len(T), bs):
            ch = T[b0:b0 + bs]; xs, msk = AW.cue_inputs(m, ch, dev); ar = torch.arange(len(ch), device=dev); pos = msk.long().sum(1) - 1
            S1, _, _ = m.pass_aw(m.mem.zero_state(len(ch), dev), xs[0], msk, m.lower(xs[0])); x2 = xs[1]
            want = torch.tensor([tok[it["raw"]["names"]["spine"][2]] for it in ch], device=dev); assert bool((x2[ar, pos - CUE_NAME_OFF] == want).all()), "name_swap: the pass-2 cue name"
            for key, names in (("intact", None), ("swap", [it["raw"]["names"]["spine"][js[b0 + k]] for k, it in enumerate(ch)]), ("floor", fl[b0:b0 + len(ch)])):
                x = x2.clone()
                if names is not None:
                    x[ar, pos - CUE_NAME_OFF] = torch.tensor([tok[n] for n in names], device=dev)
                _, h, p_ = m.pass_aw(S1, x, msk, m.lower(x)); res[key] += m.colon_logits(h, p_)[:, num].argmax(-1).tolist()
        v2 = [int(it["raw"]["values"][2]) for it in T]; vj = [int(it["raw"]["values"][j]) for it, j in zip(T, js)]
        f = [int(p == v) for p, v in zip(res["swap"], vj)]; fo = [int(p == v) for p, v in zip(res["floor"], vj)]
        out["by_d"][str(d)] = {"n": len(T), "intact_acc": float(np.mean([p == v for p, v in zip(res["intact"], v2)])), "follow": float(np.mean(f)), "floor": float(np.mean(fo)),
                               "follow_by_j": {str(j): float(np.mean([x for x, jj in zip(f, js) if jj == j])) for j in sorted(set(js))},
                               "floor_by_j": {str(j): float(np.mean([x for x, jj in zip(fo, js) if jj == j])) for j in sorted(set(js))}}
        out["items"][str(d)] = {"sig": [it["sig"] for it in T], "j": js, "floor_name": fl, "pred": res, "v2": v2, "vj": vj}
    out["def"] = ("§263 NAME-SWAP (Audit L1b, descriptive): REG TEST2 d 3-4; pass 1 as the run; at pass 2 the cue's entity token (S_2) replaced by S_j (j != 2, seeded) or by a "
                  "name absent from the item (floor); follow = P(pass-2 ':' readout == v_j), floor = P(floor readout == v_j); intact_acc = the unswapped pass-2 readout vs v_2")
    return out


@torch.no_grad()
def part_shuffle_graded(m, g, sets, a, dev, bs):
    import boxes.inject_aw_eval as EV
    K = a["think_k"]; out = {"by_d": {}, "items": {}}
    for d in (2, 3, 4):
        raw = sets["test2"][d]; ops = _ops(d); rec = {}
        for key in ("independent", "full"):
            its = []
            for i, it in enumerate(raw):
                o = list(it["order"])
                if key == "independent":
                    sl = [k for k, x in enumerate(o) if x not in ops]; pm = np.random.default_rng((SEED_SG, d, i)).permutation(len(sl)); new = list(o)
                    for k, q in zip(sl, pm):
                        new[k] = o[sl[q]]
                else:
                    new = [o[q] for q in np.random.default_rng((SEED_SG + 1, d, i)).permutation(len(o))]
                its.append(_render(it, new, key))
            T = EV.tokd({"test2": {d: its}}, a, (d,))[d]; A, _ = EV.passes(m, T, dev, bs, {K}); y = np.array([it["ans"] for it in T])
            r_ = {"acc": float((A[K] == y).mean()), "n": len(T), "changed": float(np.mean([p["text"] != q["text"] for p, q in zip(its, raw)]))}
            if key == "full":
                r_["op_chain_order"] = float(np.mean([[x for x in p["order"] if x in ops] == ops for p in its]))
            rec[key] = r_; out["items"].setdefault(str(d), {})[key] = {"sig": [it["sig"] for it in T], "pred": A[K].tolist(), "ans": y.tolist()}
        out["by_d"][str(d)] = rec
    out["def"] = ("§263 SHUFFLE-GRADED (descriptive): REG TEST2 d 2-4; independent = the non-op sentences (S_0, givens, distractors) permuted over their own slots, op "
                  "sentences in place; full = every sentence permuted; acc at K' = think_k; solve asserted")
    return out
