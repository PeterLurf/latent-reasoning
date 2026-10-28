"""§240 LANG-HOLD (LH_v1 §2-3 + Note 240-1): main() edits for boxes/inject_aw.py, applied by scratch/s234_build/mk_inject_aw.py after its own E list (each asserted
to apply exactly once there). Default (--hold_inc 0, no --hold_log) = the §238 path bitwise (no state kept, no extra rng draw, no extra log key)."""
E240 = [
# flags
(r'''    ap.add_argument("--init_model", default="",''',
 r'''    ap.add_argument("--hold_inc", type=float, default=0.0, help="§240: lambda of L_inc (HOLD-INC; 0 = off)"); ap.add_argument("--hold_start", type=int, default=500)
    ap.add_argument("--hold_far_n", type=int, default=16); ap.add_argument("--hold_far_lo", type=int, default=16); ap.add_argument("--hold_far_hi", type=int, default=40)
    ap.add_argument("--hold_log", action="store_true", help="§240: the hold stats + gn_plain without the penalty (CTRL arms); implied by --hold_inc > 0")
    ap.add_argument("--init_model", default="",'''),
# checks
(r'''    assert a.mul_s_p == 1 or (a.mul_s_p3 == 1.0 and a.src == "ow"), "--mul_s_p 0: OW only, mul_s_p3 must be 1"''',
 r'''    assert a.mul_s_p == 1 or (a.mul_s_p3 == 1.0 and a.src == "ow"), "--mul_s_p 0: OW only, mul_s_p3 must be 1"
    HOLD = a.hold_inc > 0 or a.hold_log                                     # §240: keep the detached states (penalty and / or the hold stats)
    assert not HOLD or (a.arm == "inject" and a.cue == "step" and not a.ckpt_passes and a.src == "ow" and a.think_k >= 3), "--hold_*: LAT (inject, cue step)"'''),
# the far-pass rng (its own stream; drawn once per step only under HOLD)
(r'''    torch.manual_seed(a.seed); rng_d = np.random.default_rng(a.seed + SEED_D); rng = np.random.default_rng(a.seed)''',
 r'''    torch.manual_seed(a.seed); rng_d = np.random.default_rng(a.seed + SEED_D); rng = np.random.default_rng(a.seed)
    rng_far = np.random.default_rng(a.seed + 24800)                         # §240: T_far ~ U{hold_far_lo..hold_far_hi}, one draw per step (HOLD only)'''),
# keep the detached states in the loss
(r'''                S = m.mem.zero_state(len(items), dev); hls = [m.lower(x_) for x_ in xs] if a.cache_lower else None   # the lower cache per distinct input''',
 r'''                S = m.mem.zero_state(len(items), dev); hls = [m.lower(x_) for x_ in xs] if a.cache_lower else None   # the lower cache per distinct input
                Ssv = [S.detach()] if (HOLD and stat is not None) else None   # §240: S_0..S_K detached (the penalty's inputs / the hold stats)'''),
(r'''                    S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False) if (a.ckpt_passes and torch.is_grad_enabled()) else m.write(S, x, msk, hl)''',
 r'''                    S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False) if (a.ckpt_passes and torch.is_grad_enabled()) else m.write(S, x, msk, hl)
                    if Ssv is not None:
                        Ssv.append(S.detach())'''),
(r'''                stat.setdefault("vacc_d", []).append((str(items[0]["d"]), vok, len(items)))   # §233 aux_vacc_by_d_pass (one d per step)''',
 r'''                stat.setdefault("vacc_d", []).append((str(items[0]["d"]), vok, len(items)))   # §233 aux_vacc_by_d_pass (one d per step)
                if Ssv is not None:
                    stat.setdefault("_keep", []).append((Ssv, xs, msk, [it["d"] for it in items]))'''),
# memory check: the penalty + the longest far pass inside the measured peak
(r'''        torch.cuda.reset_peak_memory_stats(dev); lam_now[0] = a.aux_step; l_ = loss_items(MC_TR[:a.bs // micro]); l_.backward(); params.zero_grad(set_to_none=True)''',
 r'''        torch.cuda.reset_peak_memory_stats(dev); lam_now[0] = a.aux_step; st_m = {} if HOLD else None; l_ = loss_items(MC_TR[:a.bs // micro], st_m); l_.backward()
        if HOLD:                                                            # §240: the penalty (on) and the far pass at hold_far_hi inside the peak
            hold_step(m, st_m.pop("_keep"), MC_TR[:min(a.hold_far_n, a.bs // micro)], a.hold_far_hi, K, a.bs, a.hold_inc > 0, a.hold_inc, ac, dev)
        params.zero_grad(set_to_none=True)'''),
# state
(r'''    expo = {}; binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; dseq = []; iseq = [];''',
 r'''    expo = {}; binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; hbin = {}; gn_plain_log = []; dseq = []; iseq = [];'''),
(r'''        vbin_d = ck.get("vbin_d", {}); gnbin = ck.get("gnbin", {})''',
 r'''        vbin_d = ck.get("vbin_d", {}); gnbin = ck.get("gnbin", {}); hbin = ck.get("hbin", {}); gn_plain_log = list(ck.get("gn_plain_log", []))
        if ck.get("rng_far") is not None:
            rng_far.bit_generator.state = ck["rng_far"]'''),
# the step: plain backward per chunk (unchanged), then gn_plain, then the penalty (disjoint graph) + the far pass, then the total clip
(r'''            lk = loss_items(its[k_ * ch:(k_ + 1) * ch], st_); (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
''',
 r'''            lk = loss_items(its[k_ * ch:(k_ + 1) * ch], st_); (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
        if HOLD:                                                            # §240 (LH_v1 §2, Note 240-1): the plain grads are complete -> gn_plain (exact)
            gn_plain_log.append(float(torch.sqrt(sum((p_.grad.float() ** 2).sum() for p_ in params.parameters() if p_.grad is not None))))
            T_far = int(rng_far.integers(a.hold_far_lo, a.hold_far_hi + 1)); inc_on = a.hold_inc > 0 and step >= a.hold_start * sc
            hs_ = hold_step(m, st_.pop("_keep"), its[:a.hold_far_n], T_far, K, a.bs, inc_on, a.hold_inc, ac, dev); hold_acc(hbin, hs_)
            if step == step0 and not a.resume:                              # the norm guard at step 0 (this step's forward, before any update)
                log.write(json.dumps({"step": 0, "norm_guard": {"s8": hs_["s8"][0] / hs_["s8"][1], "sd2": hs_["sd2"][0] / hs_["sd2"][1]},
                                      "hold_ratios": {k_: (v_[0] / v_[1] if v_[1] else None) for k_, v_ in hs_.items() if k_ in ("comp", "hold")}}) + "\n"); log.flush()
'''),
# 100-step record
(r'''            rec.update({"gn_preclip": gn_log[-1], "gn_preclip_mean100": float(np.mean(gn_log[-100:])),''',
 r'''            if HOLD and gn_plain_log:                                       # §240 LP0-W: the plain-loss grad norm (exact, every step)
                rec["gn_plain_mean100"] = float(np.mean(gn_plain_log[-100:]))
            rec.update({"gn_preclip": gn_log[-1], "gn_preclip_mean100": float(np.mean(gn_log[-100:])),'''),
# 1k-bin record
(r'''                br["gn_preclip_by_d"] = {k_: v_[0] / v_[1] for k_, v_ in sorted(gnbin.items())}''',
 r'''                br["gn_preclip_by_d"] = {k_: v_[0] / v_[1] for k_, v_ in sorted(gnbin.items())}
            if HOLD and hbin:                                               # §240: inc (the penalised squared ratio), far, comp / hold ratios, the norm guard
                hm_ = hold_mean(hbin); br["hold"] = {"inc_ratio": hm_.get("inc"), "far_ratio": hm_.get("far"), "comp_ratio": hm_.get("comp"), "hold_ratio": hm_.get("hold"),
                                                     "norm_guard": {"s8": hm_.get("s8"), "sd2": hm_.get("sd2")}}'''),
(r'''            binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; dseq = []; iseq = []''',
 r'''            binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; hbin = {}; dseq = []; iseq = []'''),
# checkpoint
(r'''"vbin": vbin, "vbin_d": vbin_d, "gnbin": gnbin, "dseq": dseq,''',
 r'''"vbin": vbin, "vbin_d": vbin_d, "gnbin": gnbin, "hbin": hbin, "gn_plain_log": gn_plain_log, "rng_far": rng_far.bit_generator.state, "dseq": dseq,'''),
]
