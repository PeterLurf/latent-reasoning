

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
