"""builds boxes/inject_st.py = a header + cue helpers + inject_ow.main() copied with explicit edits (each asserted to apply exactly once)."""
src = open("boxes/inject_ow.py", encoding="utf-8").read()
i0 = src.index("def main(argv=None):"); i1 = src.index('if __name__ == "__main__":')
main = src[i0:i1]
E = [
# --cue flag + checks
('    ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--eval_parts", default="acc"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)',
 '    ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--eval_parts", default="acc"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)\n'
 '    ap.add_argument("--cue", default="none", choices=["none", "step", "all"], help="§233: step = LAT-ST (the S_min(t, d) cue at pass t), all = CUED-DIRECT (cues S_1..S_d on the decoder); none = inject_ow")'),
('    assert a.src == "ow" or a.d_max == 8, "--src nested: n ~ U{1..n_max_train} per item (nested170\'s law)"',
 '    assert a.src == "ow" or a.d_max == 8, "--src nested: n ~ U{1..n_max_train} per item (nested170\'s law)"\n'
 '    assert a.cue != "step" or (a.arm == "inject" and not a.ckpt_passes and a.src == "ow"), "--cue step: LAT-ST (inject, OW, no ckpt_passes)"\n'
 '    assert a.cue != "all" or (a.arm == "direct" and a.src == "ow"), "--cue all: CUED-DIRECT (direct, OW)"'),
# items
('        return [tok_ow(stream.draw(d), a.aux_target) for _ in range(a.bs)]', '        return [tok_st(stream.draw(d), a.aux_target, a.cue) for _ in range(a.bs)]'),
# the per-pass inputs in the loss
('            x, msk = m.think_inputs(items, dev); per = [[] for _ in items]; vok = {}\n            with ac():\n                S = m.mem.zero_state(len(items), dev); hl = m.lower(x) if a.cache_lower else None\n                for t in range(1, K + 1):\n                    S = checkpoint(',
 '            xs, msk = cue_inputs(m, items, dev); per = [[] for _ in items]; vok = {}   # §233: pass t reads xs[min(t, len(xs)) - 1] (cue none: one input)\n'
 '            with ac():\n                S = m.mem.zero_state(len(items), dev); hls = [m.lower(x_) for x_ in xs] if a.cache_lower else None   # the lower cache per distinct input\n'
 '                for t in range(1, K + 1):\n                    i_ = min(t, len(xs)) - 1; x = xs[i_]; hl = hls[i_] if hls is not None else None\n                    S = checkpoint('),
('                stat["aux"] = stat.get("aux", 0.0) + float(aux.detach()) * len(items); stat.setdefault("vacc", []).append((vok, len(items)))',
 '                stat["aux"] = stat.get("aux", 0.0) + float(aux.detach()) * len(items); stat.setdefault("vacc", []).append((vok, len(items)))\n'
 '                stat.setdefault("vacc_d", []).append((str(items[0]["d"]), vok, len(items)))   # §233 aux_vacc_by_d_pass (one d per step)'),
# memory-check items: the cued shapes; the eval probe = TEST2's longest cued d 12 item
('    MC_TR = sorted([tok_ow(s_mc.draw(a.d_max), a.aux_target) for _ in range(256)], key=lambda it: -len(it["dec"])) if s_mc else None\n'
 '    MC_EV = sorted([tok_ow(s_mc.draw(12)) for _ in range(256)], key=lambda it: -len(it["dec"]))[:1] if s_mc else None',
 '    MC_TR = sorted([tok_st(s_mc.draw(a.d_max), a.aux_target, a.cue) for _ in range(256)], key=lambda it: -st_len(it)) if s_mc else None\n'
 '    MC_EV = sorted([tok_st(it, a.aux_target, a.cue) for it in SETS["test2"][12]], key=lambda it: -st_len(it))[:1] if s_mc else None   # §233: TEST2\'s own longest d 12 (cued)'),
('        from boxes import inject_ow_eval as EV\n        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); EV.answers_by_kprime(m, g, MC_EV * ebs, 14 if K else 0, dev, ebs, None)',
 '        from boxes import inject_st_eval as EV\n        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); EV.answers_by_kprime(m, g, MC_EV * ebs, 14 if K else 0, dev, ebs, None)'),
# info
('"hostprobe_sps": os.environ.get("HOSTPROBE_SPS"), "module_sha256": sha_file(os.path.abspath(__file__)),',
 '"hostprobe_sps": os.environ.get("HOSTPROBE_SPS"), "module_sha256": sha_file(os.path.abspath(__file__)), "inject_ow_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_ow.py")),'),
('"eval_module_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_ow_eval.py")),', '"eval_module_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_st_eval.py")),'),
('                                 "enc(\' \' + R_step[t]) + EOT, read-only on S_t, t <= d"}[a.arm]}',
 '                                 "enc(\' \' + R_step[t]) + EOT, read-only on S_t, t <= d"}[a.arm] + CUE_FMT[a.cue]}'),
# calibration on the cued inputs
('            s_ = OW.Stream(SEED_CAL, frames=OW.FRAME_TRAIN, **OWKW); its = [tok_ow(s_.draw(1 + i % a.d_max), a.aux_target) for i in range(64)]\n'
 '        x, msk = m.think_inputs(its, dev); m.mem.norms = []\n        with torch.no_grad(), ac():\n            m.think(m.mem.zero_state(64, dev), x, msk, min(K, 4))',
 '            s_ = OW.Stream(SEED_CAL, frames=OW.FRAME_TRAIN, **OWKW); its = [tok_st(s_.draw(1 + i % a.d_max), a.aux_target, a.cue) for i in range(64)]\n'
 '        xs, msk = cue_inputs(m, its, dev); m.mem.norms = []\n        with torch.no_grad(), ac():\n            S_c = m.mem.zero_state(64, dev)\n'
 '            for t_ in range(1, min(K, 4) + 1):                              # InjectN.think\'s write sequence, the pass-t input xs[min(t, len) - 1]\n'
 '                S_c = m.write(S_c, xs[min(t_, len(xs)) - 1], msk)'),
# 1k bins: aux_vacc_by_d_pass (+ n); checkpointed
('    expo = {}; binacc = {}; vbin = {}; dseq = []', '    expo = {}; binacc = {}; vbin = {}; vbin_d = {}; dseq = []'),
('        expo, binacc, vbin, dseq, iseq, d2_bins, ent, fut, wlast = ck["expo"], ck["binacc"], ck["vbin"], ck["dseq"], ck["iseq"], ck["d2_bins"], ck["ent"], ck["fut"], ck["wlast"]',
 '        expo, binacc, vbin, dseq, iseq, d2_bins, ent, fut, wlast = ck["expo"], ck["binacc"], ck["vbin"], ck["dseq"], ck["iseq"], ck["d2_bins"], ck["ent"], ck["fut"], ck["wlast"]\n'
 '        vbin_d = ck.get("vbin_d", {})'),
('        for vok, _n in st_.get("vacc", []):\n            for t_, v_ in vok.items():\n                vb_ = vbin.setdefault(str(t_), [0.0, 0]); vb_[0] += v_; vb_[1] += 1',
 '        for vok, _n in st_.get("vacc", []):\n            for t_, v_ in vok.items():\n                vb_ = vbin.setdefault(str(t_), [0.0, 0]); vb_[0] += v_; vb_[1] += 1\n'
 '        for dd_, vok, _n in (st_.get("vacc_d", []) if a.cue != "none" else []):\n            for t_, v_ in vok.items():\n'
 '                vb_ = vbin_d.setdefault(dd_, {}).setdefault(str(t_), [0.0, 0]); vb_[0] += v_; vb_[1] += 1'),
('                br["aux_vacc_by_pass"] = {k_: v_[0] / v_[1] for k_, v_ in sorted(vbin.items(), key=lambda kv: int(kv[0]))}',
 '                br["aux_vacc_by_pass"] = {k_: v_[0] / v_[1] for k_, v_ in sorted(vbin.items(), key=lambda kv: int(kv[0]))}\n'
 '            if vbin_d:                                                      # §233 LP2 key (+ the per-cell counts: G10 pools them back to aux_vacc_by_pass)\n'
 '                br["aux_vacc_by_d_pass"] = {d_: {k_: v_[0] / v_[1] for k_, v_ in sorted(tv.items(), key=lambda kv: int(kv[0]))} for d_, tv in sorted(vbin_d.items())}\n'
 '                br["aux_vacc_by_d_pass_n"] = {d_: {k_: v_[1] for k_, v_ in sorted(tv.items(), key=lambda kv: int(kv[0]))} for d_, tv in sorted(vbin_d.items())}'),
('            binacc = {}; vbin = {}; dseq = []; iseq = []', '            binacc = {}; vbin = {}; vbin_d = {}; dseq = []; iseq = []'),
('"vbin": vbin, "dseq": dseq,', '"vbin": vbin, "vbin_d": vbin_d, "dseq": dseq,'),
# the eval module
('    from boxes import inject_ow_eval as EV\n    EV.run(a.out,', '    from boxes import inject_st_eval as EV\n    EV.run(a.out,'),
]
for o, n in E:
    assert main.count(o) == 1, ("edit not applied exactly once", main.count(o), o[:90]); main = main.replace(o, n)
assert "tok_ow(" not in main and "inject_ow_eval" not in main and "think_inputs(" not in main, "leftovers"
hdr = open("C:/Users/peter/AppData/Local/Temp/claude/D--/09263851-6ec3-5138-b59d-fa8edabd53a4/scratchpad/inject_st_head.py", encoding="utf-8").read()
open("boxes/inject_st.py", "w", encoding="utf-8", newline="\n").write(hdr + "\n\n" + main + 'if __name__ == "__main__":\n    sys.exit(main())\n')
print("written", len(E), "edits")
