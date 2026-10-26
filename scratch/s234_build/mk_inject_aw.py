"""builds boxes/inject_aw.py = scratch/s234_build/inject_aw_head.py + inject_st.main() copied with explicit edits (each asserted to apply exactly once)."""
src = open("boxes/inject_st.py", encoding="utf-8").read()
i0 = src.index("def main(argv=None):"); i1 = src.index('if __name__ == "__main__":')
main = src[i0:i1]
E = [
# flags + checks
('    ap.add_argument("--cue", default="none", choices=["none", "step", "all"], help="§233: step = LAT-ST (the S_min(t, d) cue at pass t), all = CUED-DIRECT (cues S_1..S_d on the decoder); none = inject_ow")',
 '    ap.add_argument("--cue", default="none", choices=["none", "step", "all"], help="§233: step = LAT-ST (the S_min(t, d) cue at pass t), all = CUED-DIRECT (cues S_1..S_d on the decoder); none = inject_ow")\n'
 '    ap.add_argument("--write_src", default="all", choices=["all", "answer"], help="§234: answer = + \' The answer is:\' on every pass input, only the \':\' state writes")\n'
 '    ap.add_argument("--init_trunk", default="", help="§234: GPT-2 loaded strict from this file before DeltaMem is built"); ap.add_argument("--init_sha", default="", help="its sha256 prefix (asserted)")'),
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
 '    WS_ITEMS = {d_: [tok_aw(it, a.aux_target, a.cue, a.write_src) for it in SETS["test2"][d_][:200]] for d_ in (1, 2)} if a.write_src == "answer" else None   # §234 WS-READOUT'),
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
 '            if WS_ITEMS is not None:                                        # §234 WS-READOUT: TEST2 d 1 / 2 (first 200 each), fp32, no_grad, never fed back\n'
 '                br["ws_readout"] = ws_readout(m, WS_ITEMS, dev, K)'),
# the eval module
('    from boxes import inject_st_eval as EV\n    EV.run(a.out,', '    from boxes import inject_aw_eval as EV\n    EV.run(a.out,'),
]
for o, n in E:
    assert main.count(o) == 1, ("edit not applied exactly once", main.count(o), o[:90]); main = main.replace(o, n)
assert "tok_st(" not in main and "inject_st_eval" not in main and "build_model(" not in main, "leftovers"
hdr = open("scratch/s234_build/inject_aw_head.py", encoding="utf-8").read()
open("boxes/inject_aw.py", "w", encoding="utf-8", newline="\n").write(hdr + "\n\n" + main + 'if __name__ == "__main__":\n    sys.exit(main())\n')
print("written", len(E), "edits")
