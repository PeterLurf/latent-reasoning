"""§234 P-2 build gates (AW_draft §2.4 G1-G12 + AW_v1 §4 FX-LEAK; for Audit's G-D). -> scratch/gate_s234/<part>.json ("pass"). Runs under scratch/_tmp/s234;
the 4060 (cuda:0) unless fp64 CPU. LAT-AW = the P-2 command (--write_src answer, init askname, gamma = 1 from step 1); CD-AN = --arm direct --cue all + init.
  g1       FLAGS-OFF: boxes.inject_st vs boxes.inject_aw (--write_src all, no init), det wrapper, fp32, micro 8, sched_scale 0.02, stop_at 30 (ckpt 30):
           (a) §233 LAT-ST flags (--cue step), (b) CD (--arm direct --cue all): ckpt model + opt + streams + bins torch.equal, logs (minus sec) equal.
  g2       the gathered single-token write (InjectAW.write) vs the one-hot-masked full write (deltamem.gpt_forward_mem_r, mask = one-hot at ':'), fp64 CPU
           (Tensor.float -> double), perturbed model (gate_s228.perturbed, class InjectAW), gamma 1 (the treatment) and 0.5: the full LAT-AW loss (answer CE
           + AUX_W aux, K 8, lower cache) on d 1 / 2 / 4 / 8 (2 items each): loss, every S_t and every grad, max rel diff <= 1e-10.
  g2b      batch-of-one (fp64 CPU, gamma 1): each item's cached lower rows and its S after passes 1..d (+1 hold) in a mixed-d batch == the item alone, uncached.
  g3       per-term gradflow (fp64 CPU, perturbed, gamma 1, 4 d 2 items): answer CE and aux CE each nonzero on all 148 GPT-2 + 12 memory tensors.
  g4_g10   smokes on the 4060 (stop_at 50, log 10, micro 8 -> rc 9): plain inject_aw (no new flags: --cue none inject), LAT-AW, CD-AN. G10: in LAT-AW's
           step-50 bin aux_vacc_by_d_pass pooled with _n == aux_vacc_by_pass (|diff| <= 1e-12), and the bin carries ws_readout (d 1 / 2 x passes 1..8).
  g5       memory / timing at the pilot shapes on the 4060 (budget 6.0): LAT-AW and CD-AN d <= 2, --auto_micro --time_steps 30 (the 3090 numbers come from
           the queue's K7 / G-TIME at admission).
  args     G6: --args_only LAT-AW vs scratch/s233_args_frozen_st_plst2_s1.json and CD-AN vs scratch/s233_args_frozen_st_pcd2_s1.json (s228_args_check CONTROL;
           the new module-sha key inject_st_sha256 listed apart): LAT-AW == {write_src, init_trunk, init_sha, init_sha256, write_mix_start, write_mix_end,
           format, eval_parts}; CD-AN == {write_src, init_trunk, init_sha, init_sha256}.
  g7       FX-LEAK (AW_v1 §4) through the REAL entry points with hooks on InjectAW.write / dec_logits (class-level): (L) inject_aw.main (the training loss,
           LAT-AW on the 4060, stop_at 2, micro 8) - every dec_logits call reads the S object the K-th write of its micro-batch returned, on dec tokens ==
           tok_ow's dec; (E) inject_aw_eval.passes (every eval part's runner), fp64 CPU perturbed gamma 1, 8 d 2 items x 3 think variants (real / wrong
           cue / another item's text, same dec): (a) S_K hooked to one fixed state -> dec logits bitwise equal across variants; (b) PLANTED LEAK (a value
           stashed on the module: each write stores its ':' argmax token on self, dec_logits appends it to dec) -> detected (logits differ); (c) unhooked:
           S_K and logits differ across variants (live channel); (d) S_K bitwise equal with the aux computed (aux_ts 1, 2) and not; (e) batch composition:
           each item's dec logits alone == its row in the batch (rel <= 1e-10).
  g9       decoded inputs, 3 items per d (1, 2, 4, 8): every pass t = 1..d + 1 -> " " + text + " " + QUESTION(u, S_min(t, d)) + " The answer is:", the
           decoder -> " " + text + " The answer is:", CUE-ONLY / CUE-WRONG pass-1 inputs end in the suffix.
  g11      ast: every part_* in boxes/inject_aw_eval.py is decorated @torch.no_grad() (233-4).
  g12      strict init load: build_aw on CPU with the askname init: sha256 starts c00f7ec0bd7977fe, every trunk tensor == the file's, mem.qkv == block 6
           c_attn^T, mem.ln == block 6 ln_1, mem.ln_w == block 11 ln_1 (all torch.equal); a wrong --init_sha asserts.
  evalsmoke  every inject_aw_eval part at the REAL eval_bs (250) on a 20-step LAT-AW model (4060; TEST2 cut to 250 per d incl. each d's longest item;
           swap_k / swap_1 / probe in full) and acc on a 20-step CD-AN model (eval_bs 100: CD's 3090 eval peak 12.1 GiB > the 4060): files, per-item counts.
    python -B scratch/gate_s234.py g11 g12 g9 args g1 g2 g2b g3 g7 g4_g10 g5 evalsmoke"""
import json
import math
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch")
OUT = ROOT + "/scratch/gate_s234"; TMP = ROOT + "/scratch/_tmp/s234"; DEV = "cuda:0"
INIT = "boxes/runs_fsq/s233_askname_s1/model.pt"; INIT_SHA = "c00f7ec0bd7977fe"
import gate_s228 as G8  # noqa: E402

LAT2 = ["--arm", "inject", "--think_k", "8", "--aux_step", "1", "--aux_fade", "48000,48000", "--aux_reweight", "--fast_add", "--cache_lower", "--d_max", "2", "--steps", "12000"]
DIR2 = ["--arm", "direct", "--d_max", "2", "--steps", "12000"]
AWF = ["--cue", "step", "--write_src", "answer", "--init_trunk", INIT, "--init_sha", INIT_SHA, "--write_mix_start", "0", "--write_mix_end", "0"]
LAT_AW = LAT2 + AWF; CD_AN = DIR2 + ["--cue", "all", "--init_trunk", INIT, "--init_sha", INIT_SHA]


def det(mod, fl, d):
    G8.fresh(d); return G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", mod] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")


def g1():
    o = {}
    for tag, fl in (("lat_st", LAT2 + ["--cue", "step"]), ("cd", DIR2 + ["--cue", "all"])):
        com = fl + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
        da, db = TMP + "/g1_st_" + tag, TMP + "/g1_aw_" + tag
        ra = det("boxes/inject_st.py", com, da); rb = det("boxes/inject_aw.py", com + ["--write_src", "all"], db); r = {"rc": [ra, rb]}
        if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
            r.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt", "rng_d", "stream_rng", "expo", "binacc", "vbin", "vbin_d", "gn_log")))
            strip = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731
            r["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
        r["pass"] = r["rc"] == [9, 9] and r.get("n_mismatch") == 0 and bool(r.get("log_equal")); o[tag] = r
    o["pass"] = all(v["pass"] for v in o.values()); return o


def aw_items(ds, n, seed=22997, write_src="answer"):
    import boxes.inject_ow as IO
    import boxes.inject_aw as AW
    import boxes.ow_words as OW
    s = OW.Stream(seed, frames=OW.FRAME_TRAIN, **IO.OWKW); return {d: [AW.tok_aw(s.draw(d), "value", "step", write_src) for _ in range(n)] for d in ds}


def aw_model(dt, gamma=1.0):
    import torch
    import boxes.inject_aw as AW
    m, g = G8.perturbed(dt); m.__class__ = AW.InjectAW
    with torch.no_grad():
        m.mem.gamma.fill_(gamma)
    return m, g


class OneHot:
    """the reference write: deltamem's full DELTA-R pass with the write mask one-hot at each item's last real token (':')."""
    def __init__(self, m):
        self.m = m

    def __call__(self, S, x, mask, h_low=None):
        import torch
        import boxes.deltamem as DM
        pos = mask.long().sum(1) - 1; oh = torch.zeros_like(mask); oh[torch.arange(x.shape[0]), pos] = True
        return DM.gpt_forward_mem_r(self.m.g, self.m.g.embed(x), self.m.mem, S, mask=oh, h_low=h_low)[1]


def aw_loss(m, items, K, cached, write=None):
    """inject_aw's LAT loss (answer CE + AUX_W-weighted aux) on the AW inputs; write = m.write (gathered) or OneHot(m)."""
    import torch
    import torch.nn.functional as Fn
    import boxes.inject_ow as IO
    import boxes.inject_aw as AW
    write = write or m.write; xs, msk = AW.cue_inputs(m, items, "cpu"); hls = [m.lower(x) for x in xs] if cached else None; dt = next(m.parameters()).dtype
    S = m.mem.zero_state(len(items), "cpu").to(dt); per = [[] for _ in items]; Ss = []
    for t in range(1, K + 1):
        i_ = min(t, len(xs)) - 1; S = write(S, xs[i_], msk, hls[i_] if hls else None); Ss.append(S)
        idx = [i for i, it in enumerate(items) if len(it["aux"]) >= t]
        if idx:
            lg_a, y_a = m.aux_logits(S[idx], [items[i] for i in idx], t, "cpu")
            ce = Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100, reduction="none").sum(1) / (y_a != -100).sum(1)
            for r_, i in enumerate(idx):
                per[i].append(ce[r_] * IO.AUX_W[t - 1])
    la = Fn.cross_entropy(m.dec_logits(S, items, "cpu"), torch.tensor(IO.NUM_IDS[[it["ans"] for it in items]]))
    aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean(); return la, aux, Ss


def rel(u, v):
    return float((u - v).abs().max() / v.abs().max().clamp_min(1e-300))


def g2():
    with G8.F64():
        import torch
        torch.set_num_threads(2); its = aw_items((1, 2, 4, 8), 2); o = {}; ok = True
        for gm in (1.0, 0.5):
            m, g = aw_model(torch.float64, gm); og = {}
            for d, it_ in its.items():
                res = []
                for wr in (None, OneHot(m)):
                    m.zero_grad(set_to_none=True); la, aux, Ss = aw_loss(m, it_, 8, True, wr); (la + aux).backward()
                    res.append((float(la + aux), [s.detach() for s in Ss], {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}))
                (l0, S0, G0), (l1, S1, G1) = res
                r = {"loss_rel": abs(l1 - l0) / abs(l0), "S_rel_max": max(rel(u, v) for u, v in zip(S0, S1)),
                     "grad_rel_max": max(rel(G0[n], G1[n]) for n in G1 if G1[n].abs().max() > 0), "grad_keys_equal": sorted(G0) == sorted(G1), "n_grads": len(G1)}
                r["pass"] = max(r["loss_rel"], r["S_rel_max"], r["grad_rel_max"]) <= 1e-10 and r["grad_keys_equal"]; og[str(d)] = r; ok = ok and r["pass"]
            o["gamma_%s" % gm] = og
        o["pass"] = ok; return o


def g2b():
    with G8.F64():
        import torch
        import boxes.inject_aw as AW
        torch.set_num_threads(2); m, g = aw_model(torch.float64); its = sum(aw_items((1, 2, 3, 4), 2).values(), [])
        xs, msk = AW.cue_inputs(m, its, "cpu"); hls = [m.lower(x) for x in xs]; S = m.mem.zero_state(len(its), "cpu").double(); Sb = []
        for t in range(1, len(xs) + 2):
            i_ = min(t, len(xs)) - 1; S = m.write(S, xs[i_], msk, hls[i_]); Sb.append(S)
        worst_low = worst_S = 0.0
        for r, it in enumerate(its):
            xs1, m1 = AW.cue_inputs(m, [it], "cpu"); L = int(m1.sum()); S1 = m.mem.zero_state(1, "cpu").double()
            for t in range(1, len(xs) + 2):
                i_ = min(t, len(xs)) - 1; j_ = min(t, len(xs1)) - 1
                lo_b = hls[i_][r, :L]; lo_1 = m.lower(xs1[j_])[0, :L]; worst_low = max(worst_low, float((lo_b - lo_1).abs().max() / lo_1.abs().max()))
                S1 = m.write(S1, xs1[j_], m1, None); worst_S = max(worst_S, float((Sb[t - 1][r] - S1[0]).abs().max() / S1.abs().max().clamp_min(1e-300)))
        return {"n_items": len(its), "ds": sorted({it["d"] for it in its}), "lower_rel_max": worst_low, "S_rel_max": worst_S, "pass": worst_low <= 1e-10 and worst_S <= 1e-10}


def g3():
    with G8.F64():
        import torch
        torch.set_num_threads(2); m, g = aw_model(torch.float64); its = aw_items((2,), 4)[2]; o = {}
        for term in ("answer_ce", "aux_ce"):
            m.zero_grad(set_to_none=True); la, aux, _ = aw_loss(m, its, 8, True); (la if term == "answer_ce" else aux).backward()
            gr = {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in m.named_parameters()}
            okf = lambda v: v is not None and v > 0 and math.isfinite(v)  # noqa: E731
            o[term] = {"gpt2_nonzero": sum(okf(v) for n, v in gr.items() if n.startswith("g.")), "mem_nonzero": sum(okf(v) for n, v in gr.items() if n.startswith("mem.")),
                       "mem_grad_norms": {n: v for n, v in gr.items() if n.startswith("mem.")}, "zero_or_none": sorted(n for n, v in gr.items() if not okf(v))[:20]}
        o["pass"] = all(o[t]["gpt2_nonzero"] == 148 and o[t]["mem_nonzero"] == 12 for t in ("answer_ce", "aux_ce")); return o


def g4_g10():
    o = {}
    for tag, fl in (("plain_inject", LAT2 + ["--cue", "none"]), ("lat_aw", LAT_AW), ("cd_an", CD_AN)):
        d = G8.fresh(TMP + "/g4_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--micro", "8", "--stop_at", "50", "--ckpt_every", "0", "--log_every", "10",
                                                                                                          "--device", DEV, "--out", d], d + "/stdout.log")
        o[tag] = {"rc": rc}
        if tag == "lat_aw" and os.path.exists(d + "/log.jsonl"):
            br = [json.loads(l_) for l_ in open(d + "/log.jsonl") if '"by_d"' in l_][-1]; p = br.get("aux_vacc_by_pass", {}); bd = br.get("aux_vacc_by_d_pass", {}); bn = br.get("aux_vacc_by_d_pass_n", {})
            pooled = {t: sum(bd[dd][t] * bn[dd][t] for dd in bd if t in bd[dd]) / sum(bn[dd][t] for dd in bn if t in bn[dd]) for t in p}
            ws = br.get("ws_readout", {}); rows = [json.loads(l_) for l_ in open(d + "/log.jsonl") if '"gamma"' in l_]
            o["g10"] = {"aux_vacc_by_pass": p, "pooled_from_by_d": pooled, "max_abs_diff": max(abs(pooled[t] - p[t]) for t in p) if p else None, "d_keys": sorted(bd),
                        "ws_readout": ws, "gamma_logged": sorted({r_["gamma"] for r_ in rows})}
    g10 = o.get("g10", {}); ws = g10.get("ws_readout", {})
    o["pass"] = (all(v.get("rc") == 9 for k, v in o.items() if k != "g10") and g10.get("max_abs_diff") is not None and g10["max_abs_diff"] <= 1e-12
                 and sorted(ws) == ["1", "2"] and all(sorted(ws[k], key=int) == [str(t) for t in range(1, 9)] for k in ws) and g10["gamma_logged"] == [1.0]); return o


def g5():
    o = {}
    for tag, fl in (("lat_aw_d2", LAT_AW), ("cd_an_d2", CD_AN)):
        d = G8.fresh(TMP + "/g5_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--auto_micro", "--mem_budget", "6.0", "--time_steps", "30", "--log_every", "10",
                                                                                                        "--device", DEV, "--out", d], d + "/stdout.log")
        r = {"rc": rc}
        for f_ in ("mem_check.json", "timing.json"):
            if os.path.exists(os.path.join(d, f_)):
                r[f_[:-5]] = json.load(open(os.path.join(d, f_)))
        o[tag] = r
    o["pass"] = all(v.get("rc") == 0 and "timing" in v for v in o.values()); return o


def args_part():
    import s228_args_check as C
    o = {}
    exp = {"lat_aw": {"write_src", "init_trunk", "init_sha", "init_sha256", "write_mix_start", "write_mix_end", "format", "eval_parts"},
           "cd_an": {"write_src", "init_trunk", "init_sha", "init_sha256"}}
    for tag, fl, ref in (("lat_aw", LAT_AW + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", "acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws"],
                          "scratch/s233_args_frozen_st_plst2_s1.json"),
                         ("cd_an", CD_AN + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", "acc"], "scratch/s233_args_frozen_st_pcd2_s1.json")):
        d = G8.fresh(TMP + "/args_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        new = json.load(open(d + "/args.json")); dif = C.diff(json.load(open(ROOT + "/" + ref)), {k: v for k, v in new.items() if k != "inject_st_sha256"})
        o[tag] = {"rc": rc, "diff_keys": sorted(dif), "diff": {k: v for k, v in dif.items() if k != "format"}, "format_new": new.get("format"), "args_file": d + "/args.json",
                  "inject_st_sha256": new.get("inject_st_sha256"), "inject_ow_sha256_equal": new.get("inject_ow_sha256") == json.load(open(ROOT + "/" + ref)).get("inject_ow_sha256")}
        o[tag]["pass"] = rc == 0 and set(dif) == exp[tag] and o[tag]["inject_ow_sha256_equal"]
    o["pass"] = all(v["pass"] for v in o.values()); return o


def g7():
    import torch
    import numpy as np
    import boxes.inject_ow as IO
    import boxes.inject_aw as AW
    import boxes.inject_aw_eval as AE
    import boxes.ow_words as OW
    W0, D0 = AW.InjectAW.write, AW.InjectAW.dec_logits; H = {"fix": None, "leak": False, "log": []}

    def write_h(self, S, x, mask, h_low=None):
        S2, h, pos = self.pass_aw(S, x, mask, h_low)
        if H["leak"]:                                                       # the planted leak: a value stashed on the module, not an argument
            with torch.no_grad():
                self._leak = self.colon_logits(h, pos).argmax(-1).tolist()
        H["log"].append(("w", id(S2))); return S2

    def dec_h(self, S, items, dev):
        sid = id(S); decs = [list(it["dec"]) for it in items]
        if H["fix"] is not None:                                            # S_K hooked: one fixed state for every item
            S = H["fix"].to(S.dtype)[None].expand(S.shape[0], *H["fix"].shape).contiguous()
        if H["leak"] and getattr(self, "_leak", None) is not None:          # the planted leak reaches the decoder through the module stash
            items = [{**it, "dec": it["dec"][:-1] + [self._leak[i]] + it["dec"][-1:]} for i, it in enumerate(items)]
        lg = D0(self, S, items, dev); H["log"].append(("d", sid, decs, lg.detach().clone())); return lg
    AW.InjectAW.write, AW.InjectAW.dec_logits = write_h, dec_h; o = {}
    try:
        # (L) the training loss entry point (inject_aw.main, LAT-AW, 4060)
        d = G8.fresh(TMP + "/g7_loss"); H["log"] = []
        rc = AW.main(LAT_AW + ["--micro", "8", "--stop_at", "2", "--ckpt_every", "0", "--log_every", "1", "--device", DEV, "--out", d])
        seq = H["log"]; bad = []; nd = 0; raw_dec_ok = True
        for i, e in enumerate(seq):
            if e[0] == "d":
                nd += 1; ws_ = [x for x in seq[max(0, i - 8):i] if x[0] == "w"]
                if len(ws_) != 8 or seq[i - 1][0] != "w" or seq[i - 1][1] != e[1]:
                    bad.append(i)
        sfx = all(dd[-4:] == AW.SFX for e in seq if e[0] == "d" for dd in e[2])
        o["loss_path"] = {"rc": rc, "n_dec_calls": nd, "n_write_calls": sum(e[0] == "w" for e in seq), "dec_reads_last_write_S": not bad, "bad_idx": bad[:5],
                          "dec_ends_in_suffix": sfx, "note": "dec tokens == tok_ow's dec is asserted in tok_aw for every item (G9 re-checks)"}
        o["loss_path"]["pass"] = rc == 9 and nd == 16 and not bad and sfx
        # (E) the eval runner (inject_aw_eval.passes, every eval part's), fp64 CPU, perturbed, gamma 1, batch of one per item (variants differ in length)
        with G8.F64():
            torch.set_num_threads(2); m, g = aw_model(torch.float64); s = OW.Stream(23999, frames=OW.FRAME_TRAIN, **IO.OWKW); raw = [s.draw(2) for _ in range(9)]
            real = [AW.tok_aw(it, "value", "step", "answer") for it in raw[:8]]

            def variant(kind):
                out = []
                for i, it in enumerate(real):
                    r = it["raw"]; o_ = dict(it)
                    if kind == "wrong_cue":                                 # pass 1 asks S_2, pass 2 asks S_1
                        o_["think_cues"] = [[IO.EOT] + IO.ENC.encode(" " + r["text"] + " " + AW.cue_str(r, 3 - c)) + IO.SFX for c in (1, 2)]
                    elif kind == "other_text":
                        o_["think_cues"] = [[IO.EOT] + IO.ENC.encode(" " + raw[(i + 1) % 9]["text"] + " " + AW.cue_str(r, c)) + IO.SFX for c in (1, 2)]
                    out.append(o_)
                return out
            V = {k: variant(k) for k in ("real", "wrong_cue", "other_text")}

            def run_v(kind, fix=None, leak=False, aux_ts=()):
                """-> per item (S_K, answer at K' 8, the dec tokens the decoder got, its logits)."""
                res = []
                for it in V[kind]:
                    H["fix"], H["leak"], H["log"] = fix, leak, []; m._leak = None; SK = {}

                    def grab(t, S, sl):
                        if t == 8:
                            SK["S"] = S.detach().clone()
                    A, X = AE.passes(m, [it], "cpu", 1, {8}, aux_ts=aux_ts, on_state=grab); dl = [e for e in H["log"] if e[0] == "d"][-1]
                    res.append((SK["S"], int(A[8][0]), dl[2], dl[3]))
                H["fix"], H["leak"] = None, False; return res
            with torch.no_grad():
                R0 = run_v("real"); Sfix = R0[0][0][0].clone(); Lg = {}
                for leak in (False, True):
                    for kind in V:
                        Lg[(leak, kind)] = torch.stack([r_[3] for r_ in run_v(kind, fix=Sfix, leak=leak)])
                inv = {k: bool(torch.equal(Lg[(False, "real")], Lg[(False, k)])) for k in ("wrong_cue", "other_text")}
                leak_det = {k: not bool(torch.equal(Lg[(True, "real")], Lg[(True, k)])) for k in ("wrong_cue", "other_text")}
                live = {}
                for k in ("wrong_cue", "other_text"):
                    b_ = run_v(k); live[k] = {"S_K_differ_all": all(not torch.equal(x[0], y[0]) for x, y in zip(R0, b_)),
                                              "logits_differ_any": any(not torch.equal(x[3], y[3]) for x, y in zip(R0, b_)),
                                              "answers_real": [x[1] for x in R0], "answers_var": [y[1] for y in b_]}
                Ra = run_v("real", aux_ts=(1, 2)); aux_eq = all(torch.equal(x[0], y[0]) for x, y in zip(R0, Ra))
                dec_eq = all(r_[2] == [IO.tok_ow(it["raw"])["dec"]] for r_, it in zip(R0, real))
                # (e) batch composition: the 8 real items as one batch (the runner, bs 8) vs each alone: S_K and dec logits
                H["log"] = []; SB = {}

                def grabb(t, S, sl):
                    if t == 8:
                        SB["S"] = S.detach().clone()
                AE.passes(m, real, "cpu", 8, {8}, on_state=grabb); lb = [e for e in H["log"] if e[0] == "d"][-1][3]
                comp = max(rel(lb[i], R0[i][3]) for i in range(8)); Sb = max(rel(SB["S"][i], R0[i][0][0]) for i in range(8))
            o["eval_path"] = {"a_invariant_S_fixed": inv, "b_planted_leak_detected": leak_det, "c_live": live, "d_SK_equal_aux_on_off": aux_eq,
                              "e_dec_tokens_eq_tok_ow": dec_eq, "e_batch_composition_logits_rel": comp, "e_batch_composition_SK_rel": Sb}
            o["eval_path"]["pass"] = (all(inv.values()) and any(leak_det.values()) and all(v["S_K_differ_all"] and v["logits_differ_any"] for v in live.values())
                                      and aux_eq and dec_eq and comp <= 1e-10 and Sb <= 1e-10)
    finally:
        AW.InjectAW.write, AW.InjectAW.dec_logits = W0, D0; H["fix"] = None; H["leak"] = False
    o["pass"] = o["loss_path"]["pass"] and o["eval_path"]["pass"]; return o


def g9():
    import boxes.inject_ow as IO
    import boxes.inject_aw as AW
    import boxes.ow_words as OW
    s = OW.Stream(22996, frames=OW.FRAME_TRAIN, **IO.OWKW); bad = []; fx = []
    for d in (1, 2, 4, 8):
        for _ in range(3):
            it = s.draw(d); o = AW.tok_aw(it, "value", "step", "answer"); sp = it["names"]["spine"]; q = lambda k: OW.QUESTION.format(u=it["unit"], A=sp[k])  # noqa: E731
            for t in range(1, d + 2):
                got = IO.ENC.decode(o["think_cues"][min(t, d) - 1][1:]); want = " " + it["text"] + " " + q(min(t, d)) + " The answer is:"
                if got != want:
                    bad.append(("think", d, t))
            if IO.ENC.decode(o["dec"][1:]) != " " + it["text"] + " The answer is:" or o["dec"] != IO.tok_ow(it)["dec"]:
                bad.append(("dec", d))
            fx.append({"d": d, "pass1": IO.ENC.decode(o["think_cues"][0][1:])[-120:], "pass_hold": IO.ENC.decode(o["think_cues"][-1][1:])[-70:], "dec_tail": IO.ENC.decode(o["dec"])[-60:]})
    co = [IO.EOT] + IO.ENC.encode(" " + AW.cue_str(it, 1)) + IO.SFX
    fx.append({"cue_only_pass1": IO.ENC.decode(co[1:])})
    src = open(ROOT + "/boxes/inject_aw_eval.py", encoding="utf-8").read()
    n_sfx = src.count('(IO.SFX if a.get("write_src") == "answer" else [])')
    return {"bad": bad, "fixtures": fx, "cue_only_cue_wrong_suffix_sites": n_sfx, "pass": not bad and n_sfx == 2}


def g11():
    import ast
    t = ast.parse(open(ROOT + "/boxes/inject_aw_eval.py", encoding="utf-8").read()); r = {}
    for n in t.body:
        if isinstance(n, ast.FunctionDef) and n.name.startswith("part_"):
            r[n.name] = [ast.unparse(d) for d in n.decorator_list]
    return {"parts": r, "pass": len(r) == 9 and all(v == ["torch.no_grad()"] for v in r.values())}


def g12():
    import argparse
    import hashlib
    import torch
    import boxes.inject_aw as AW
    ns = lambda **k: argparse.Namespace(**{"arm": "inject", "mem_write": "top", "write_src": "answer", "init_trunk": INIT, "init_sha": INIT_SHA, **k})  # noqa: E731
    torch.manual_seed(1); m, g, _ = AW.build_aw(ns(), "cpu"); sd = torch.load(ROOT + "/" + INIT, map_location="cpu", weights_only=True)
    gs = g.state_dict(); trunk_eq = set(gs) == set(sd) and all(torch.equal(gs[k], sd[k]) for k in sd)
    mem = m.mem; w = lambda k: g.w(k).detach()  # noqa: E731   (the loaded trunk; == the file by trunk_eq)
    cp = {"qkv_w": torch.equal(mem.qkv.weight, w("h.6.attn.c_attn.weight").float().T), "qkv_b": torch.equal(mem.qkv.bias, w("h.6.attn.c_attn.bias").float()),
          "ln": torch.equal(mem.ln.weight, w("h.6.ln_1.weight").float()) and torch.equal(mem.ln.bias, w("h.6.ln_1.bias").float()),
          "ln_w": torch.equal(mem.ln_w.weight, w("h.11.ln_1.weight").float()) and torch.equal(mem.ln_w.bias, w("h.11.ln_1.bias").float())}
    sha = hashlib.sha256(open(ROOT + "/" + INIT, "rb").read()).hexdigest()
    try:
        AW.build_aw(ns(init_sha="0000000000000000"), "cpu"); wrong = False
    except AssertionError:
        wrong = True
    base = {k: v for k, v in AW.load_trunk("cpu", {"scratch": ""}).state_dict().items()}; differs = sum(not torch.equal(base[k], sd[k]) for k in sd)
    return {"sha256": sha, "class": type(m).__name__, "init_sha256_on_args": getattr(ns(), "init_sha256", None), "trunk_equal_file": trunk_eq, "copies": cp,
            "wrong_sha_asserts": wrong, "tensors_differing_from_base_gpt2": differs, "n_tensors": len(sd),
            "pass": sha.startswith(INIT_SHA) and trunk_eq and all(cp.values()) and wrong and type(m).__name__ == "InjectAW" and differs > 0}


def evalsmoke():
    import boxes.inject_ow as IO
    import boxes.inject_aw_eval as AE
    o = {}
    for tag, fl, ebs, parts in (("lat_aw", LAT_AW, 250, list(AE.PARTS)), ("cd_an", CD_AN, 100, ["acc"])):
        d = G8.fresh(TMP + "/evsmoke_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--steps", "20", "--micro", "8", "--eval_bs", str(ebs), "--eval_parts", "",
                                                                                                                  "--ckpt_every", "0", "--log_every", "10", "--device", DEV, "--out", d], d + "/stdout.log")
        S = IO.build_sets(); L = lambda it: len(IO.ENC.encode(it["text"]))  # noqa: E731
        S["test2"] = {k: sorted(v, key=lambda it: -L(it))[:1] + v[:249] if max(v, key=L) not in v[:249] else v[:250] for k, v in S["test2"].items()}
        import torch
        torch.zeros(1, device=DEV); torch.cuda.reset_peak_memory_stats(DEV); AE.run(d, parts, DEV, ebs, sets=S); pk = torch.cuda.max_memory_allocated(DEV) / 2 ** 30
        fs = sorted(f_ for f_ in os.listdir(d) if f_.startswith("eval")); o[tag] = {"rc": rc, "files": fs, "eval_bs": ebs, "peak_gib": pk,
                                                                                  "n_items_acc_d2": len(json.load(open(d + "/eval_acc.json"))["items"]["2"]["ans"])}
        if tag == "lat_aw":
            sk = json.load(open(d + "/eval_swap_k.json")); pr = json.load(open(d + "/eval_probe.json")); ws = json.load(open(d + "/eval_ws.json")); s1 = json.load(open(d + "/eval_swap_1.json"))
            o[tag].update({"swap_k_n": sk["n_pairs"], "swap_k_tried": sk["tried"], "swap_k_items": len(sk["items"]["swap"]), "swap_1_ks": sorted(s1["by_k"]),
                           "probe_keys": sorted(pr["probes"]), "probe_n": pr["n"], "read_vec_check_rel": pr["read_vec_check_rel"], "ws_ds": sorted(ws["by_d"]),
                           "ws_items_d2": len(ws["items"]["2"]["pred_by_pass"]["1"])})
    lw = o["lat_aw"]
    o["pass"] = (lw["rc"] == 0 and o["cd_an"]["rc"] == 0 and set(lw["files"]) == {"eval.json"} | {"eval_%s.json" % p_ for p_ in AE.PARTS} and o["cd_an"]["files"] == ["eval.json", "eval_acc.json"]
                 and lw["n_items_acc_d2"] == 250 and lw["swap_k_n"] == 1000 and lw["swap_k_items"] == 1000 and lw["probe_n"] == 3000 and lw["ws_items_d2"] == 250 and lw["peak_gib"] < 7.5)
    return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"g1": g1, "g2": g2, "g2b": g2b, "g3": g3, "g4_g10": g4_g10, "g5": g5, "args": args_part, "g7": g7, "g9": g9, "g11": g11, "g12": g12, "evalsmoke": evalsmoke}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
