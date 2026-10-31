"""§252 UNSCAFFOLDED (NS_v1 §2-§3; Note 252-1): main() edits for boxes/inject_aw.py, applied by scratch/s234_build/mk_inject_aw.py AFTER EAC (each asserted once).
  --perm_p p (default 0 = off: no draws, the loop bitwise unchanged): each training item of a d >= 2 step is re-rendered by boxes.perm_render.train_perm (op sentences in
    a dependency-violating order, >= 1 inversion; seeded per item from (seed, 25160, step, k); the coin from (seed, 25161, step, k)) with probability p; OW.solve ==
    answer asserted per item; every training text (permuted or not) asserted absent from the TEST-PERM texts of this run's TEST2 d 1..d_max (Note 252-1 disjointness).
    Logged per 1k bin: ws_readout_perm / ws_readout_chain = {d: {pass: acc}} on the step's own training items (the trunk's argmax over the 11 number tokens at
    each pass's ':' vs v_min(t, d), no_grad, from the same pass_aw call whose S the loss uses: pass_aw()[0] is m.write's S), split by permuted / chain-order, + _n.
  --cue generic: tok_aw's plan-free cue (inject_aw head) accepted wherever --cue step is (LAT-AW asserts, HOLD), its own format string."""
E252 = [
(r'''    ap.add_argument("--cue", default="none", choices=["none", "step", "all"],''',
 r'''    ap.add_argument("--perm_p", type=float, default=0.0, help="§252: P(a d >= 2 training item is re-rendered dependency-violating) (0 = off, unchanged)")
    ap.add_argument("--cue", default="none", choices=["none", "step", "all", "generic"],'''),
(r'''    assert a.cue != "step" or (a.arm == "inject" and not a.ckpt_passes and a.src == "ow"), "--cue step: LAT-ST (inject, OW, no ckpt_passes)"''',
 r'''    assert a.cue not in ("step", "generic") or (a.arm == "inject" and not a.ckpt_passes and a.src == "ow"), "--cue step / generic: LAT-ST (inject, OW, no ckpt_passes)"
    assert not a.perm_p or (a.src == "ow" and a.arm == "inject" and a.write_src == "answer" and not a.ckpt_passes), "--perm_p: LAT-AW (inject, OW, write_src answer)"'''),
(r'''    assert a.write_src != "answer" or (a.arm == "inject" and a.cue == "step"), "--write_src answer: LAT-AW (inject, --cue step)"''',
 r'''    assert a.write_src != "answer" or (a.arm == "inject" and a.cue in ("step", "generic")), "--write_src answer: LAT-AW (inject, --cue step / generic)"'''),
(r'''    assert not HOLD or (a.arm == "inject" and a.cue == "step" and''',
 r'''    assert not HOLD or (a.arm == "inject" and a.cue in ("step", "generic") and'''),
(r'''        return [tok_aw(stream.draw(d), a.aux_target, a.cue, a.write_src) for _ in range(a.bs)]''',
 r'''        if a.perm_p:                                                        # §252 --perm_p: per-item seeded coin / permutation (d >= 2), solve + disjointness asserted
            raws = [stream.draw(d) for _ in range(a.bs)]
            raws = [PR.train_perm(r_, a.seed, step, k_) if (d >= 2 and np.random.default_rng((a.seed, PR.SEED_PERM_TRAIN + 1, step, k_)).random() < a.perm_p) else r_
                    for k_, r_ in enumerate(raws)]
            assert not any(r_["text"] in TP_TEXTS for r_ in raws), "§252: a training text equals a TEST-PERM text"
            return [tok_aw(r_, a.aux_target, a.cue, a.write_src) for r_ in raws]
        return [tok_aw(stream.draw(d), a.aux_target, a.cue, a.write_src) for _ in range(a.bs)]'''),
(r'''    def draw_items(d):''',
 r'''    TP_TEXTS = {it["text"] for d_, its_ in PR.test_perm_sets({d_: SETS["test2"][d_] for d_ in range(1, a.d_max + 1)}, range(1, a.d_max + 1))[0].items()
                for it in its_} if a.perm_p else set()                     # §252 Note 252-1: the TEST-PERM texts (622fb717's family) for the disjointness assert

    def draw_items(d):'''),
(r'''                    else:
                        S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False) if (a.ckpt_passes and torch.is_grad_enabled()) else m.write(S, x, msk, hl)''',
 r'''                    elif a.perm_p:                                          # §252: pass_aw (S = m.write's) keeps h for the split WS readout
                        S, h_c, pos_c = m.pass_aw(S, x, msk, hl)
                    else:
                        S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False) if (a.ckpt_passes and torch.is_grad_enabled()) else m.write(S, x, msk, hl)
                    if a.perm_p and stat is not None:                       # §252 ws_readout_perm / _chain on the step's own training items (no_grad, never fed back)
                        with torch.no_grad():
                            pw_ = m.colon_logits(h_c.detach(), pos_c)[:, NUM_IDS].argmax(-1).tolist()
                        for i_, it_ in enumerate(items):
                            stat.setdefault("wsp", []).append((int(it_["raw"].get("perm", 0)), str(it_["d"]), str(t), int(pw_[i_] == int(it_["raw"]["values"][min(t, it_["d"])]))))'''),
(r'''        for dd_, c_, n_, s_ in st_.get("acue", []):''',
 r'''        for pp_, dd_, tt_, ok_ in st_.get("wsp", []):                      # §252: the split WS readout (per 1k bin)
            wb_ = wsbin.setdefault("perm" if pp_ else "chain", {}).setdefault(dd_, {}).setdefault(tt_, [0, 0]); wb_[0] += ok_; wb_[1] += 1
        for dd_, c_, n_, s_ in st_.get("acue", []):'''),
(r'''            if acbin:''',
 r'''            for k_ in ("perm", "chain") if a.perm_p else ():                # §252 Note 252-1: ws_readout_perm / ws_readout_chain = {d: {pass: acc}} (+ _n)
                wv_ = wsbin.get(k_, {})
                br["ws_readout_" + k_] = {d_: {t_: v_[0] / v_[1] for t_, v_ in sorted(tv.items(), key=lambda kv: int(kv[0]))} for d_, tv in sorted(wv_.items())}
                br["ws_readout_" + k_ + "_n"] = {d_: {t_: v_[1] for t_, v_ in sorted(tv.items(), key=lambda kv: int(kv[0]))} for d_, tv in sorted(wv_.items())}
            if acbin:'''),
(r'''hbin_d = {}; dseq = []; iseq = []; acbin = {}''',
 r'''hbin_d = {}; dseq = []; iseq = []; acbin = {}; wsbin = {}'''),
(r'''    acbin = {}                                                              # Note 251-10:''',
 r'''    wsbin = {}                                                              # §252: {"perm" / "chain": {d: {t: [correct, n]}}} (the open 1k bin; not in the resume ckpt)
    acbin = {}                                                              # Note 251-10:'''),
(r'''+ CUE_FMT[a.cue] + AW_FMT[a.write_src]}''',
 r'''+ (CUE_FMT[a.cue] if a.cue != "generic" else " | §252 GP: think_t = [EOT] + enc(' ' + text + ' What is the next value?') (every pass)") + AW_FMT[a.write_src]}'''),
]
