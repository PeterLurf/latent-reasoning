"""§243 DEPTH-CURRIC (DC_v1 §3 fix 2): main() edits for boxes/inject_aw.py, applied by scratch/s234_build/mk_inject_aw.py AFTER E240 (each asserted once).
The per-d norm guard: each step's hold stats s8 / sd2 (one d per step) are also accumulated per d (hbin_d) and written per bin as hold.norm_guard_by_d {d: {s8, sd2}}.
Only under HOLD (--hold_inc > 0 or --hold_log); the training computation is unchanged (stats only; no rng draw)."""
E243 = [
(r'''    expo = {}; binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; hbin = {}; gn_plain_log = []; dseq = []; iseq = [];''',
 r'''    expo = {}; binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; hbin = {}; hbin_d = {}; gn_plain_log = []; dseq = []; iseq = [];'''),
(r'''hbin = ck.get("hbin", {}); gn_plain_log = list(ck.get("gn_plain_log", []))''',
 r'''hbin = ck.get("hbin", {}); hbin_d = ck.get("hbin_d", {}); gn_plain_log = list(ck.get("gn_plain_log", []))'''),
(r'''T_far, K, a.bs, inc_on, a.hold_inc, ac, dev); hold_acc(hbin, hs_)''',
 r'''T_far, K, a.bs, inc_on, a.hold_inc, ac, dev); hold_acc(hbin, hs_)
            hold_acc(hbin_d.setdefault(str(its[0]["d"]), {}), {k_: hs_[k_] for k_ in ("s8", "sd2")})   # §243 DC_v1 fix 2: the per-d norm guard (one d per step)'''),
(r'''                                                     "norm_guard": {"s8": hm_.get("s8"), "sd2": hm_.get("sd2")}}''',
 r'''                                                     "norm_guard": {"s8": hm_.get("s8"), "sd2": hm_.get("sd2")}}
                br["hold"]["norm_guard_by_d"] = {k_: hold_mean(v_) for k_, v_ in sorted(hbin_d.items())}   # §243: {d: {s8, sd2}} (s8 = ||S_K||, K = think_k)'''),
(r'''            binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; hbin = {}; dseq = []; iseq = []''',
 r'''            binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; hbin = {}; hbin_d = {}; dseq = []; iseq = []'''),
(r'''"gnbin": gnbin, "hbin": hbin, "gn_plain_log": gn_plain_log,''',
 r'''"gnbin": gnbin, "hbin": hbin, "hbin_d": hbin_d, "gn_plain_log": gn_plain_log,'''),
]
