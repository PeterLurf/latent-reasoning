"""Note 255-5 DISTRACT: main() edits for boxes/inject_aw.py, applied by scratch/s234_build/mk_inject_aw.py AFTER E252 (each asserted once).
--distract_p p (default 0 = off: no draws, the loop bitwise unchanged): each training item gets, with probability p (coin default_rng((seed, 25196, step, k))), n ~ U{0, 1, 2}
irrelevant (given, add_c op) pairs about new names (boxes.insert_render.distract: givens at the end of the given block, each op at a uniform gap of the op block, chain order
kept; solve asserted). Audit (Note 255-5): (a) every distracted item's aux targets and pass-cue tails are asserted == the undistracted item's (the targets stay the chain's
S_t / v_t); (b) names disjoint, solve asserted (insert_render); every training text asserted absent from the INSERT eval texts (REG TEST2 d 4, 7 conditions); (d) with
--distract_p the --mem_check / --auto_micro items carry 2 pairs (ops after the last op sentence), so the memory check measures the d_max + 2-insert shape.
Logged per 1k bin: distract_n = {n: count} over the bin's training items. Not combinable with --perm_p (asserted)."""
E255 = [
(r'''    ap.add_argument("--perm_p", type=float, default=0.0,''',
 r'''    ap.add_argument("--distract_p", type=float, default=0.0, help="Note 255-5: P(an item gets U{0,1,2} irrelevant new-name op sentences) (0 = off, unchanged)")
    ap.add_argument("--perm_p", type=float, default=0.0,'''),
(r'''        if a.perm_p:                                                        # §252 --perm_p: per-item seeded coin / permutation (d >= 2), solve + disjointness asserted''',
 r'''        if a.distract_p:                                                    # Note 255-5 --distract_p: per-item seeded coin / n / draw; targets, solve, disjointness asserted
            raws = [stream.draw(d) for _ in range(a.bs)]; out_ = []
            for k_, r_ in enumerate(raws):
                n_ = 0
                if np.random.default_rng((a.seed, IR.SEED_DISTRACT_COIN, step, k_)).random() < a.distract_p:
                    r2_, n_ = IR.distract(r_, a.seed, step, k_)
                    if n_:
                        t_new, t_old = tok_aw(r2_, a.aux_target, a.cue, a.write_src), tok_aw(r_, a.aux_target, a.cue, a.write_src)
                        assert t_new["aux"] == t_old["aux"] and t_new["ans"] == t_old["ans"] and len(t_new["think_cues"]) == len(t_old["think_cues"]) and all(
                            q1[len(t_new["think"]):] == q0[len(t_old["think"]):] for q1, q0 in zip(t_new["think_cues"], t_old["think_cues"])), "DISTRACT: targets / cues changed"
                        r_ = r2_
                DSTAT[n_] = DSTAT.get(n_, 0) + 1; out_.append(r_)
            assert not any(r_["text"] in INS_TEXTS for r_ in out_), "DISTRACT: a training text equals an INSERT eval text"
            return [tok_aw(r_, a.aux_target, a.cue, a.write_src) for r_ in out_]
        if a.perm_p:                                                        # §252 --perm_p: per-item seeded coin / permutation (d >= 2), solve + disjointness asserted'''),
(r'''    TP_TEXTS = {it["text"] for d_, its_ in PR.test_perm_sets(''',
 r'''    assert not (a.distract_p and a.perm_p), "--distract_p with --perm_p: not built"
    INS_TEXTS = {it["text"] for its_ in IR.insert_eval_sets(SETS["test2"][4])[0].values() for it in its_} if (a.distract_p and a.d_max >= 4) else set()   # Note 255-5 (b)
    DSTAT = {}                                                              # Note 255-5: {n: count} in the open 1k bin
    TP_TEXTS = {it["text"] for d_, its_ in PR.test_perm_sets('''),
(r'''    MC_TR = sorted([tok_aw(s_mc.draw(a.d_max), a.aux_target, a.cue, a.write_src) for _ in range(256)], key=lambda it: -st_len(it)) if s_mc else None''',
 r'''    if a.distract_p and s_mc:                                               # Note 255-5 (d): the memory-check items carry 2 insert pairs (ops after the last op sentence)
        def _mc2(i_):
            r_ = s_mc.draw(a.d_max); g_, o_, m_ = IR.draw_pairs(r_, 2, np.random.default_rng((a.seed, IR.SEED_DISTRACT + 2, i_)))
            return IR.assemble(r_, g_, [(("after", "S%d" % r_["d"]), x_) for x_ in o_], m_)
        MC_TR = sorted([tok_aw(_mc2(i_), a.aux_target, a.cue, a.write_src) for i_ in range(256)], key=lambda it: -st_len(it))
    else:
        MC_TR = sorted([tok_aw(s_mc.draw(a.d_max), a.aux_target, a.cue, a.write_src) for _ in range(256)], key=lambda it: -st_len(it)) if s_mc else None'''),
(r'''            for k_ in ("perm", "chain") if a.perm_p else ():''',
 r'''            if a.distract_p:                                                # Note 255-5: the bin's insert-count distribution
                br["distract_n"] = {str(k_): v_ for k_, v_ in sorted(DSTAT.items())}; DSTAT.clear()
            for k_ in ("perm", "chain") if a.perm_p else ():'''),
]
