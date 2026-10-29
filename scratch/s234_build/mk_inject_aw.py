"""builds boxes/inject_aw.py = scratch/s234_build/inject_aw_head.py + inject_st.main() copied with explicit edits (each asserted to apply exactly once)."""
src = open("boxes/inject_st.py", encoding="utf-8").read()
i0 = src.index("def main(argv=None):"); i1 = src.index('if __name__ == "__main__":')
main = src[i0:i1]
E = [
# flags + checks
('    ap.add_argument("--cue", default="none", choices=["none", "step", "all"], help="§233: step = LAT-ST (the S_min(t, d) cue at pass t), all = CUED-DIRECT (cues S_1..S_d on the decoder); none = inject_ow")',
 '    ap.add_argument("--cue", default="none", choices=["none", "step", "all"], help="§233: step = LAT-ST (the S_min(t, d) cue at pass t), all = CUED-DIRECT (cues S_1..S_d on the decoder); none = inject_ow")\n'
 '    ap.add_argument("--write_src", default="all", choices=["all", "answer"], help="§234: answer = + \' The answer is:\' on every pass input, only the \':\' state writes")\n'
 '    ap.add_argument("--init_trunk", default="", help="§234: GPT-2 loaded strict from this file before DeltaMem is built"); ap.add_argument("--init_sha", default="", help="its sha256 prefix (asserted)")\n'
 '    ap.add_argument("--mul_s_p3", type=float, default=1.0, help="§235: P(forced mul_s | d 3) on the TRAINING stream only (1.0 = unchanged)")'),
('    assert a.cue != "all" or (a.arm == "direct" and a.src == "ow"), "--cue all: CUED-DIRECT (direct, OW)"',
 '    assert a.cue != "all" or (a.arm == "direct" and a.src == "ow"), "--cue all: CUED-DIRECT (direct, OW)"\n'
 '    assert a.write_src != "answer" or (a.arm == "inject" and a.cue == "step"), "--write_src answer: LAT-AW (inject, --cue step)"\n'
 '    assert bool(a.init_trunk) == bool(a.init_sha), "--init_trunk needs --init_sha (G12)"'),
# the model (init trunk loaded before DeltaMem; InjectAW)
('    m, g, params = build_model(a, dev)', '    m, g, params = build_aw(a, dev)'),
# items
('        return [tok_st(stream.draw(d), a.aux_target, a.cue) for _ in range(a.bs)]', '        return [tok_aw(stream.draw(d), a.aux_target, a.cue, a.write_src) for _ in range(a.bs)]'),
# memory-check items (+ the WS-READOUT items: TEST2's first 200 of d 1 and 2)
('    MC_TR = sorted([tok_st(s_mc.draw(a.d_max), a.aux_target, a.cue) for _ in range(256)], key=lambda it: -st_len(it)) if s_mc else None\n'
 '    MC_EV = sorted([tok_st(it, a.aux_target, a.cue) for it in SETS["test2"][12]], key=lambda it: -st_len(it))[:1] if s_mc else None   # §233: TEST2\'s own longest d 12 (cued)',
 '    MC_TR = sorted([tok_aw(s_mc.draw(a.d_max), a.aux_target, a.cue, a.write_src) for _ in range(256)], key=lambda it: -st_len(it)) if s_mc else None\n'
 '    MC_EV = sorted([tok_aw(it, a.aux_target, a.cue, a.write_src) for it in SETS["test2"][12]], key=lambda it: -st_len(it))[:1] if s_mc else None   # §233: TEST2\'s own longest d 12 (cued)\n'
 '    WS_ITEMS = {d_: [tok_aw(it, a.aux_target, a.cue, a.write_src) for it in SETS["test2"][d_][:200]] for d_ in range(1, a.d_max + 1)} if a.write_src == "answer" else None   # §234 WS-READOUT (Note 234-5 c: d 1..d_max)'),
('        from boxes import inject_st_eval as EV\n        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); EV.answers_by_kprime(m, g, MC_EV * ebs, 14 if K else 0, dev, ebs, None)',
 '        from boxes import inject_aw_eval as EV\n        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); EV.answers_by_kprime(m, g, MC_EV * ebs, 14 if K else 0, dev, ebs, None)'),
# info
('"inject_ow_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_ow.py")),',
 '"inject_ow_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_ow.py")), "inject_st_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_st.py")),'),
('"eval_module_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_st_eval.py")),', '"eval_module_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_aw_eval.py")),'),
('[a.arm] + CUE_FMT[a.cue]}', '[a.arm] + CUE_FMT[a.cue] + AW_FMT[a.write_src]}'),
# calibration on the AW inputs
('its = [tok_st(s_.draw(1 + i % a.d_max), a.aux_target, a.cue) for i in range(64)]', 'its = [tok_aw(s_.draw(1 + i % a.d_max), a.aux_target, a.cue, a.write_src) for i in range(64)]'),
# 1k bins: WS-READOUT
('                br["aux_vacc_by_d_pass_n"] = {d_: {k_: v_[1] for k_, v_ in sorted(tv.items(), key=lambda kv: int(kv[0]))} for d_, tv in sorted(vbin_d.items())}',
 '                br["aux_vacc_by_d_pass_n"] = {d_: {k_: v_[1] for k_, v_ in sorted(tv.items(), key=lambda kv: int(kv[0]))} for d_, tv in sorted(vbin_d.items())}\n'
 '            if WS_ITEMS is not None:                                        # §234 WS-READOUT: TEST2 d 1..d_max (first 200 each), fp32, no_grad, never fed back\n'
 '                br["ws_readout"] = ws_readout(m, WS_ITEMS, dev, K)'),
# the eval module
('    from boxes import inject_st_eval as EV\n    EV.run(a.out,', '    from boxes import inject_aw_eval as EV\n    EV.run(a.out,'),
# §235 D4-DECOR (D4_v1 §1): --mul_s_p3 on the TRAINING stream only (never OWKW: calibration / memcheck / every eval stream unchanged) + the realised d 3 shares in the bins
('    m, g, params = build_aw(a, dev)', '    assert a.mul_s_p3 == 1.0 or a.src == "ow", "--mul_s_p3: OW only"\n    m, g, params = build_aw(a, dev)'),
('        SETS = build_sets(); EXCL = SETS["excl"]; stream = OW.Stream(OW.SEED_TRAIN, EXCL, frames=OW.FRAME_TRAIN, **OWKW)',
 '        SETS = build_sets(); EXCL = SETS["excl"]; stream = OW.Stream(OW.SEED_TRAIN, EXCL, frames=OW.FRAME_TRAIN, mul_s_p3=a.mul_s_p3, **OWKW)   # §235: training stream only'),
('                br["ws_readout"] = ws_readout(m, WS_ITEMS, dev, K)',
 '                br["ws_readout"] = ws_readout(m, WS_ITEMS, dev, K)\n'
 '            if a.mul_s_p3 < 1.0 and stream is not None:                    # §235: cumulative accepted d 3 training items with a mul_s step / all d 3 items\n'
 '                br["stream_d3_mul_s"] = [stream.stats.get("d3_mul_s", 0), stream.stats.get("d3_items", 0)]'),
# §236 REG-CHAIN (RC_v1 §1): --mul_s_p 0 = the REG law on the training / calibration / memcheck streams and every eval set (new seeds); the training EXCL += the eval
# module's own REG sets (swap / probe); default 1 = unchanged
('    ap.add_argument("--mul_s_p3", type=float, default=1.0, help="§235: P(forced mul_s | d 3) on the TRAINING stream only (1.0 = unchanged)")',
 '    ap.add_argument("--mul_s_p3", type=float, default=1.0, help="§235: P(forced mul_s | d 3) on the TRAINING stream only (1.0 = unchanged)")\n'
 '    ap.add_argument("--mul_s_p", type=int, default=1, choices=[0, 1], help="§236: 0 = REG law (no forced mul_s) on every stream / set (new eval seeds); 1 = unchanged")'),
('    assert a.mul_s_p3 == 1.0 or a.src == "ow", "--mul_s_p3: OW only"',
 '    assert a.mul_s_p3 == 1.0 or a.src == "ow", "--mul_s_p3: OW only"\n'
 '    assert a.mul_s_p == 1 or (a.mul_s_p3 == 1.0 and a.src == "ow"), "--mul_s_p 0: OW only, mul_s_p3 must be 1"'),
('        SETS = build_sets(); EXCL = SETS["excl"]; stream = OW.Stream(OW.SEED_TRAIN, EXCL, frames=OW.FRAME_TRAIN, mul_s_p3=a.mul_s_p3, **OWKW)   # §235: training stream only',
 '        SETS = build_sets(mul_s_p=a.mul_s_p); EXCL = SETS["excl"] | (reg_excl(a, SETS["excl"]) if a.mul_s_p == 0 else set())   # §236: + the eval-own REG sets\n'
 '        stream = OW.Stream(OW.SEED_TRAIN, EXCL, frames=OW.FRAME_TRAIN, mul_s_p3=a.mul_s_p3, mul_s_p=a.mul_s_p, **OWKW)   # §235 / §236: training stream'),
('    s_mc = OW.Stream(SEED_MEMCHECK, frames=OW.FRAME_TRAIN, **OWKW) if a.src == "ow" else None',
 '    s_mc = OW.Stream(SEED_MEMCHECK, frames=OW.FRAME_TRAIN, mul_s_p=a.mul_s_p, **OWKW) if a.src == "ow" else None   # §236: the run\'s law'),
('            s_ = OW.Stream(SEED_CAL, frames=OW.FRAME_TRAIN, **OWKW); its =', '            s_ = OW.Stream(SEED_CAL, frames=OW.FRAME_TRAIN, mul_s_p=a.mul_s_p, **OWKW); its ='),
# §237 REG-BOOT: --init_model (the full params state loaded strict after the build; calibration skipped; fresh optimizer); default '' = unchanged
('    ap.add_argument("--mul_s_p", type=int, default=1, choices=[0, 1], help="§236: 0 = REG law (no forced mul_s) on every stream / set (new eval seeds); 1 = unchanged")',
 '    ap.add_argument("--mul_s_p", type=int, default=1, choices=[0, 1], help="§236: 0 = REG law (no forced mul_s) on every stream / set (new eval seeds); 1 = unchanged")\n'
 '    ap.add_argument("--init_model", default="", help="§237: the full params state (trunk + DeltaMem + W_o + gains) loaded strict after the build; calibration skipped")\n'
 '    ap.add_argument("--init_model_sha", default="", help="§237: its sha256 prefix (asserted)")'),
('    m, g, params = build_aw(a, dev)',
 '    m, g, params = build_aw(a, dev)\n'
 '    assert bool(a.init_model) == bool(a.init_model_sha), "--init_model needs --init_model_sha"\n'
 '    if a.init_model:                                                        # §237: the full model state, strict; the sha logged (init_model_sha256)\n'
 '        sha_m = hashlib.sha256(open(a.init_model, "rb").read()).hexdigest(); assert sha_m.startswith(a.init_model_sha), ("init_model sha", sha_m, a.init_model_sha)\n'
 '        params.load_state_dict(torch.load(a.init_model, map_location=dev, weights_only=True), strict=True); a.init_model_sha256 = sha_m'),
('    elif m is not None and K > 0:                                           # note 147-2: g_add calibration at the think hand-offs (nested170\'s)',
 '    elif m is not None and K > 0 and not a.init_model:                      # note 147-2: g_add calibration (§237: skipped under --init_model, the loaded s_add kept)'),
# §237 (Note 237-1 (2)): the pre-clip grad norm per d in the 1k bins (emitted under --init_model only, so the default logs are unchanged); checkpointed
('    expo = {}; binacc = {}; vbin = {}; vbin_d = {}; dseq = []; iseq = [];', '    expo = {}; binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; dseq = []; iseq = [];'),
('        vbin_d = ck.get("vbin_d", {})', '        vbin_d = ck.get("vbin_d", {}); gnbin = ck.get("gnbin", {})'),
('        b_ = binacc.setdefault(dk, [0, 0.0, 0.0, 0]); b_[0] += 1; b_[1] += acc_; b_[2] += la_; b_[3] += float(gn) > 1.0',
 '        b_ = binacc.setdefault(dk, [0, 0.0, 0.0, 0]); b_[0] += 1; b_[1] += acc_; b_[2] += la_; b_[3] += float(gn) > 1.0\n'
 '        gb_ = gnbin.setdefault(dk, [0.0, 0]); gb_[0] += float(gn); gb_[1] += 1   # §237: pre-clip grad norm per d (emitted under --init_model)'),
('                br["stream_d3_mul_s"] = [stream.stats.get("d3_mul_s", 0), stream.stats.get("d3_items", 0)]',
 '                br["stream_d3_mul_s"] = [stream.stats.get("d3_mul_s", 0), stream.stats.get("d3_items", 0)]\n'
 '            if a.init_model and gnbin:                                       # §237 (Note 237-1): the mean pre-clip grad norm per d over the bin\n'
 '                br["gn_preclip_by_d"] = {k_: v_[0] / v_[1] for k_, v_ in sorted(gnbin.items())}'),
('            binacc = {}; vbin = {}; vbin_d = {}; dseq = []; iseq = []', '            binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; dseq = []; iseq = []'),
('"vbin": vbin, "vbin_d": vbin_d, "dseq": dseq,', '"vbin": vbin, "vbin_d": vbin_d, "gnbin": gnbin, "dseq": dseq,'),
# §238 DAG-FREE: --mul_s_p 2 (DF); the eval-own EXCL for any non-default law
('choices=[0, 1], help="§236: 0 = REG law (no forced mul_s) on every stream / set (new eval seeds); 1 = unchanged")',
 'choices=[0, 1, 2], help="§236: 0 = REG law (no forced mul_s); §238: 2 = DF (per-step mul_s, q_t); 1 = unchanged - on every stream / set (new eval seeds)")'),
('(reg_excl(a, SETS["excl"]) if a.mul_s_p == 0 else set())   # §236: + the eval-own REG sets',
 '(reg_excl(a, SETS["excl"]) if a.mul_s_p != 1 else set())   # §236 / §238: + the eval-own sets of the run law'),
]
for o, n in E:
    assert main.count(o) == 1, ("edit not applied exactly once", main.count(o), o[:90]); main = main.replace(o, n)
import runpy  # noqa: E402
for o, n in runpy.run_path("scratch/s240_build/edits_s240.py")["E240"]:   # §240 LANG-HOLD main() edits (raw strings; each asserted once)
    assert main.count(o) == 1, ("§240 edit not applied exactly once", main.count(o), o[:90]); main = main.replace(o, n)
for o, n in runpy.run_path("scratch/s243_build/edits_s243.py")["E243"]:   # §243 DEPTH-CURRIC: the per-d norm guard (stats only)
    assert main.count(o) == 1, ("§243 edit not applied exactly once", main.count(o), o[:90]); main = main.replace(o, n)
assert "tok_st(" not in main and "inject_st_eval" not in main and "build_model(" not in main, "leftovers"
hdr = open("scratch/s234_build/inject_aw_head.py", encoding="utf-8").read() + open("scratch/s240_build/head_s240.py", encoding="utf-8").read()   # + §240 helpers
open("boxes/inject_aw.py", "w", encoding="utf-8", newline="\n").write(hdr + "\n\n" + main + 'if __name__ == "__main__":\n    sys.exit(main())\n')
print("written", len(E), "edits")
