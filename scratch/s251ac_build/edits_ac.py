"""Note 251-10 ANSWER-AT-CUE (AC_draft.md; REVIEW_AC.md #7): main() edits for boxes/inject_aw.py, applied by scratch/s234_build/mk_inject_aw.py AFTER E251 (each asserted once).
--cue_ans λ (default 0 = off, the loop bitwise unchanged): at pass t == the item's d, the pass runs through pass_aw (m.write = pass_aw()[0], the same S) and its ':' hidden goes
through colon_logits (ln_f + the tied LM head, full vocab); loss += λ · mean over those items of CE(logits, the answer token NUM_IDS[ans]). Logged per 1k bin:
acue_by_d {d: {acc (full-vocab argmax == the answer token), ce, n}}."""
EAC = [
(r'''    ap.add_argument("--aux_d_max", type=int, default=0,''',
 r'''    ap.add_argument("--cue_ans", type=float, default=0.0, help="Note 251-10: + λ CE(full-vocab LM logits at pass d's ':', the answer token) (0 = off, unchanged)")
    ap.add_argument("--aux_d_max", type=int, default=0,'''),
(r'''            xs, msk = cue_inputs(m, items, dev); per = [[] for _ in items]; vok = {}''',
 r'''            xs, msk = cue_inputs(m, items, dev); per = [[] for _ in items]; vok = {}; per_c = {}   # Note 251-10: per_c = {item: the ACUE CE}'''),
(r'''                    S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False) if (a.ckpt_passes and torch.is_grad_enabled()) else m.write(S, x, msk, hl)''',
 r'''                    cidx = [i for i, it in enumerate(items) if it["d"] == t] if a.cue_ans else []   # Note 251-10 --cue_ans: pass t == d -> the ':' LM logits vs the answer
                    if cidx:
                        assert not a.ckpt_passes and a.write_src == "answer", "--cue_ans: needs --write_src answer (pass_aw), no --ckpt_passes"
                        S, h_c, pos_c = m.pass_aw(S, x, msk, hl); lg_c = m.colon_logits(h_c[cidx], pos_c[cidx])
                        y_c = torch.tensor(NUM_IDS[[items[i]["ans"] for i in cidx]], device=dev); ce_c = Fn.cross_entropy(lg_c, y_c, reduction="none")
                        for r_, i in enumerate(cidx):
                            per_c[i] = ce_c[r_]
                        if stat is not None:
                            stat.setdefault("acue", []).append((str(t), int((lg_c.detach().argmax(-1) == y_c).sum()), len(cidx), float(ce_c.detach().sum())))
                    else:
                        S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False) if (a.ckpt_passes and torch.is_grad_enabled()) else m.write(S, x, msk, hl)'''),
(r'''            loss = la + lam_now[0] * aux
''',
 r'''            loss = la + lam_now[0] * aux
            if a.cue_ans:                                                   # Note 251-10: + λ · mean CE over the items whose pass d ran (an in-graph zero if none)
                acue = torch.stack([per_c[i] for i in sorted(per_c)]).mean() if per_c else la * 0.0; loss = loss + a.cue_ans * acue
'''),
(r'''        for vok, _n in st_.get("vacc", []):''',
 r'''        for dd_, c_, n_, s_ in st_.get("acue", []):                        # Note 251-10: ACUE acc / CE by d (per 1k bin)
            ab_ = acbin.setdefault(dd_, [0, 0, 0.0]); ab_[0] += c_; ab_[1] += n_; ab_[2] += s_
        for vok, _n in st_.get("vacc", []):'''),
(r'''            if WS_ITEMS is not None:                                        # §234 WS-READOUT''',
 r'''            if acbin:                                                       # Note 251-10 --cue_ans
                br["acue_by_d"] = {k_: {"acc": v_[0] / v_[1], "ce": v_[2] / v_[1], "n": v_[1]} for k_, v_ in sorted(acbin.items())}
            if WS_ITEMS is not None:                                        # §234 WS-READOUT'''),
(r'''            binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; hbin = {}; hbin_d = {}; dseq = []; iseq = []''',
 r'''            binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; hbin = {}; hbin_d = {}; dseq = []; iseq = []; acbin = {}'''),
(r'''    expo = {}; binacc = {}; vbin = {}; vbin_d = {};''',
 r'''    acbin = {}                                                              # Note 251-10: {d: [correct, n, ce sum]} (the open 1k bin; not in the resume ckpt)
    expo = {}; binacc = {}; vbin = {}; vbin_d = {};'''),
]
