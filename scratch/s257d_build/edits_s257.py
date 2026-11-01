"""§257 DISTRACT-CHAIN (DC2_v1 §2): main() edits for boxes/inject_aw.py, applied by scratch/s234_build/mk_inject_aw.py AFTER E255 (each asserted once).
--distract_mix T1,T2,.. (default '' = off: the §256 --distract_p rule, bitwise unchanged): with --distract_p, each coin-selected item is re-rendered by
boxes.insert_render.distract_mix (n ~ U{0,1,2} inserts, each type uniform over the listed s256 / chain / givop) instead of insert_render.distract. The E255 asserts stand
(aux targets, answer and pass-cue tails == the undistracted item's; solve inside); INS_TEXTS also holds the §257 held-out texts (held_out2_sets). Logged per 1k bin:
distract_types = {type: count} over the bin's inserts."""
E257 = [
(r'''    ap.add_argument("--distract_p", type=float, default=0.0,''',
 r'''    ap.add_argument("--distract_mix", default="", help="§257 DC2_v1: insert types drawn uniformly per insert (s256,chain,givop) under --distract_p ('' = §256 rule)")
    ap.add_argument("--distract_p", type=float, default=0.0,'''),
(r'''                    r2_, n_ = IR.distract(r_, a.seed, step, k_)''',
 r'''                    r2_, n_ = IR.distract_mix(r_, a.seed, step, k_, MIXT) if MIXT else IR.distract(r_, a.seed, step, k_)   # §257 --distract_mix'''),
(r'''                        r_ = r2_
''',
 r'''                        r_ = r2_
                        for t_ in r2_.get("distract_types", ()):                # §257: per-type insert counts
                            DTYPE[t_] = DTYPE.get(t_, 0) + 1
'''),
(r'''                 | {it["text"] for d_ in IR.held_out_sets(SETS["test2"][4])[0].values() for its_ in d_.values() for it in its_}) if (a.distract_p and a.d_max >= 4) else set()''',
 r'''                 | {it["text"] for d_ in IR.held_out_sets(SETS["test2"][4])[0].values() for its_ in d_.values() for it in its_}
                 | {it["text"] for d_ in IR.held_out2_sets(SETS["test2"][4])[0].values() for its_ in d_.values() for it in its_}) if (a.distract_p and a.d_max >= 4) else set()'''),
(r'''    DSTAT = {}                                                              # Note 255-5: {n: count} in the open 1k bin''',
 r'''    DSTAT = {}                                                              # Note 255-5: {n: count} in the open 1k bin
    MIXT = tuple(a.distract_mix.split(",")) if a.distract_mix else (); DTYPE = {}   # §257: the insert types; {type: count} in the open 1k bin
    assert not MIXT or (a.distract_p > 0 and set(MIXT) <= set(IR.MIX_TYPES)), ("--distract_mix needs --distract_p and known types", MIXT)'''),
(r'''                br["distract_n"] = {str(k_): v_ for k_, v_ in sorted(DSTAT.items())}; DSTAT.clear()''',
 r'''                br["distract_n"] = {str(k_): v_ for k_, v_ in sorted(DSTAT.items())}; DSTAT.clear()
                if MIXT:                                                    # §257: the bin's insert-type distribution
                    br["distract_types"] = dict(sorted(DTYPE.items())); DTYPE.clear()'''),
]
