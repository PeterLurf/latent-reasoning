"""§237 REG-BOOT (THEORY §237, 69fc9e4e): adds --init_model / --init_model_sha to scratch/s234_build/mk_inject_aw.py (appended to its E list; each asserted once there).
--init_model P: after the model is built (trunk incl. --init_trunk, DeltaMem), the FULL params state (trunk + DeltaMem + W_o + norms / gains; inject: InjectAW, direct:
the trunk) is loaded strict from P (sha256 prefix asserted; the full sha logged as init_model_sha256); the g_add norm calibration is skipped (the loaded s_add is kept);
the optimizer / schedule start fresh at step 0. Default '' = unchanged (flags-off gate)."""
p = "scratch/s234_build/mk_inject_aw.py"; s = open(p, encoding="utf-8").read()
o = "]\nfor o, n in E:"
add = '''# §237 REG-BOOT: --init_model (the full params state loaded strict after the build; calibration skipped; fresh optimizer); default '' = unchanged
(\'    ap.add_argument("--mul_s_p", type=int, default=1, choices=[0, 1], help="§236: 0 = REG law (no forced mul_s) on every stream / set (new eval seeds); 1 = unchanged")\',
 \'    ap.add_argument("--mul_s_p", type=int, default=1, choices=[0, 1], help="§236: 0 = REG law (no forced mul_s) on every stream / set (new eval seeds); 1 = unchanged")\\n\'
 \'    ap.add_argument("--init_model", default="", help="§237: the full params state (trunk + DeltaMem + W_o + gains) loaded strict after the build; calibration skipped")\\n\'
 \'    ap.add_argument("--init_model_sha", default="", help="§237: its sha256 prefix (asserted)")\'),
(\'    m, g, params = build_aw(a, dev)\',
 \'    m, g, params = build_aw(a, dev)\\n\'
 \'    assert bool(a.init_model) == bool(a.init_model_sha), "--init_model needs --init_model_sha"\\n\'
 \'    if a.init_model:                                                        # §237: the full model state, strict; the sha logged (init_model_sha256)\\n\'
 \'        sha_m = hashlib.sha256(open(a.init_model, "rb").read()).hexdigest(); assert sha_m.startswith(a.init_model_sha), ("init_model sha", sha_m, a.init_model_sha)\\n\'
 \'        params.load_state_dict(torch.load(a.init_model, map_location=dev, weights_only=True), strict=True); a.init_model_sha256 = sha_m\'),
(\'    elif m is not None and K > 0:                                           # note 147-2: g_add calibration at the think hand-offs (nested170\\\'s)\',
 \'    elif m is not None and K > 0 and not a.init_model:                      # note 147-2: g_add calibration (§237: skipped under --init_model, the loaded s_add kept)\'),
# §237 (Note 237-1 (2)): the pre-clip grad norm per d in the 1k bins (emitted under --init_model only, so the default logs are unchanged); checkpointed
(\'    expo = {}; binacc = {}; vbin = {}; vbin_d = {}; dseq = []; iseq = [];\', \'    expo = {}; binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; dseq = []; iseq = [];\'),
(\'        vbin_d = ck.get("vbin_d", {})\', \'        vbin_d = ck.get("vbin_d", {}); gnbin = ck.get("gnbin", {})\'),
(\'        b_ = binacc.setdefault(dk, [0, 0.0, 0.0, 0]); b_[0] += 1; b_[1] += acc_; b_[2] += la_; b_[3] += float(gn) > 1.0\',
 \'        b_ = binacc.setdefault(dk, [0, 0.0, 0.0, 0]); b_[0] += 1; b_[1] += acc_; b_[2] += la_; b_[3] += float(gn) > 1.0\\n\'
 \'        gb_ = gnbin.setdefault(dk, [0.0, 0]); gb_[0] += float(gn); gb_[1] += 1   # §237: pre-clip grad norm per d (emitted under --init_model)\'),
(\'                br["stream_d3_mul_s"] = [stream.stats.get("d3_mul_s", 0), stream.stats.get("d3_items", 0)]\',
 \'                br["stream_d3_mul_s"] = [stream.stats.get("d3_mul_s", 0), stream.stats.get("d3_items", 0)]\\n\'
 \'            if a.init_model and gnbin:                                       # §237 (Note 237-1): the mean pre-clip grad norm per d over the bin\\n\'
 \'                br["gn_preclip_by_d"] = {k_: v_[0] / v_[1] for k_, v_ in sorted(gnbin.items())}\'),
(\'            binacc = {}; vbin = {}; vbin_d = {}; dseq = []; iseq = []\', \'            binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; dseq = []; iseq = []\'),
(\'"vbin": vbin, "vbin_d": vbin_d, "dseq": dseq,\', \'"vbin": vbin, "vbin_d": vbin_d, "gnbin": gnbin, "dseq": dseq,\'),
]
for o, n in E:'''
assert s.count(o) == 1; s = s.replace(o, add); open(p, "w", encoding="utf-8", newline="\n").write(s); print("mk_inject_aw: +3 edits (§237)")
