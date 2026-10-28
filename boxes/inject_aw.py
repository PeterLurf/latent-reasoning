"""THEORY §234 ANSWER-WRITE (AW_v1.md + AW_draft §2.3 / §2.4; Notes 234-1 / 234-2): the write source = the ':' of " The answer is:" appended to each
§233 LAT-ST pass input; only that last token writes the INJECT state (gamma = 1 from step 1 via --write_mix_start 0 --write_mix_end 0); init = the
s233_askname_s1 trunk. A NEW module: inject_st / inject_st_eval / inject_ow / inject_ow_eval / nested170 / deltamem stay byte-identical. main() below is
inject_st.main() copied with listed edits (scratch/s234_build/mk_inject_aw.py, each asserted to apply once), so `--write_src all --init_trunk ''` is
inject_st exactly (gate G1). Eval: boxes/inject_aw_eval.py (inject_st_eval's parts on the AW inputs + swap_k / swap_1 / probe / ws).
  --write_src answer  pass t's input = [EOT] + enc(" " + text + " " + QUESTION(u, S_min(t, d))) + enc(" The answer is:"); InjectAW.write runs the trunk
                      over every position (the read at block 6, blocks 6-11) and write_seq on the gathered ':' rows only (aw_checks write_at; equal to
                      the one-hot-masked full write in fp64, gate G2). Decoder and aux unchanged (§228 / §233). No think-pass token is decoded or fed.
  --init_trunk P      GPT-2 loaded strict from P before DeltaMem is built (DeltaMem copies block 6's ln_1 / c_attn and block 11's ln_1 from the LOADED
                      trunk; g_add calibrated on it); --init_sha the expected sha256 prefix (asserted). CD-AN = --arm direct --cue all --init_trunk P.
  WS-READOUT          every 1k-bin record (LAT-AW): the trunk's argmax over the 11 number tokens at ':' of pass t vs v_min(t, d) on the first 200 TEST2
                      items of d 1..d_max (Note 234-5 c; held out by construction; no EXCL change, so the training stream is st_plst2_s1's), no_grad, never fed back.
    python -m boxes.inject_aw --arm inject --cue step --write_src answer --init_trunk boxes/runs_fsq/s233_askname_s1/model.pt --init_sha c00f7ec0bd7977fe
        --think_k 8 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add --cache_lower --write_mix_start 0 --write_mix_end 0 --d_max 2 --steps 12000 --out ..."""
import os
os.environ.setdefault("BOXES_NB", "5")
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N7  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import AUX_W, EOT, NUM_IDS, pad  # noqa: E402
from boxes.inject_ow import (ENC, SFX, SEED_D, SEED_CAL, SEED_MEMCHECK, OWKW, META, tok_ow, tok_nested, direct_logits, sha_file, build_sets,  # noqa: E402, F401
                             InjectOW, load_trunk)
from boxes.inject_st import CUE_FMT, cue_str, tok_st, cue_inputs, st_len  # noqa: E402, F401

COLON = SFX[-1]
AW_FMT = {"all": "", "answer": " | §234 AW: + ' The answer is:' on every pass input; only the ':' state writes (gathered single-token write)"}


class InjectAW(InjectOW):
    """InjectOW whose think write comes from the last real token only (':'). pass_aw also returns that token's logits (WS-READOUT; eval only)."""
    def pass_aw(self, S, x, mask, h_low=None):
        g, mem = self.g, self.mem; B, T = x.shape; D = g.w("wte.weight").shape[1]; hd = D // g.H; assert mem.write_mode == "top", "AW: h_w = block 11's output"
        h = (g.embed(x) + DM.wpe_slice(g, T)[None]) if h_low is None else h_low; h6 = None
        for i in range(0 if h_low is None else mem.layer, g.L):
            if i == mem.layer:
                h6 = h; h = DM.read_fixed(mem, h, S)
            h = DM._block(g, i, h, hd)
        pos = (mask.long().sum(1) - 1) if mask is not None else torch.full((B,), T - 1, device=x.device); ar = torch.arange(B, device=x.device)
        assert bool((x[ar, pos] == COLON).all()), "AW: the write position must be ':'"
        return DM.write_seq(mem, h6[ar, pos][:, None], h[ar, pos][:, None], S, None), h, pos

    def write(self, S, x, mask, h_low=None):
        return self.pass_aw(S, x, mask, h_low)[0]

    def colon_logits(self, h, pos):
        D = h.shape[-1]; hf = Fn.layer_norm(h, (D,), self.g.w("ln_f.weight"), self.g.w("ln_f.bias"))
        return self.g.logits(hf[torch.arange(h.shape[0], device=h.device), pos]).float()


def tok_aw(it, aux_target="value", cue="none", write_src="all"):
    """tok_st, then (write_src answer) " The answer is:" appended to every pass input; the decoder / aux untouched."""
    o = tok_st(it, aux_target, cue)
    if write_src == "answer":
        assert cue == "step" and "think_cues" in o, "AW: --write_src answer needs --cue step"
        o["think_cues"] = [q + SFX for q in o["think_cues"]]
        assert all(q[-1] == COLON for q in o["think_cues"]) and o["dec"] == o["think"] + SFX, "AW: inputs / decoder"
    return o


def build_aw(a, dev):
    """inject_ow.build_model with --init_trunk: GPT-2 loaded strict (sha asserted) BEFORE DeltaMem is built; InjectAW for --write_src answer."""
    g = load_trunk(dev, {"scratch": ""})
    if getattr(a, "init_trunk", ""):
        sha = hashlib.sha256(open(a.init_trunk, "rb").read()).hexdigest(); assert not a.init_sha or sha.startswith(a.init_sha), ("init sha", sha, a.init_sha)
        g.load_state_dict(torch.load(a.init_trunk, map_location=dev, weights_only=True), strict=True); a.init_sha256 = sha
    if a.arm == "direct":
        return None, g, g
    cls = InjectAW if getattr(a, "write_src", "all") == "answer" else InjectOW
    m = cls(g, 6, a.mem_write, 6, 6, None).to(dev); m.nopass = False
    assert torch.equal(m.mem.qkv.weight.detach(), g.w("h.6.attn.c_attn.weight").detach().float().T), "G12: DeltaMem qkv = the loaded block-6 c_attn"
    return m, g, m


def reg_excl(a, excl):
    """§236: the union of the eval module's own REG sets' sigs (swap_k d 2, swap_kd d 3..d_max + d 3 t 2, probe) for the training EXCL (0 overlap by construction)."""
    from boxes import inject_aw_eval as AE
    return set().union(*AE.reg_eval_sigs(vars(a), excl).values())


@torch.no_grad()
def ws_readout(m, items_by_d, dev, K, bs=50, per_item=False):
    """WS-READOUT: per d, per pass t, the trunk's argmax over the 11 number tokens at ':' vs v_min(t, d); never fed back. per_item: also
    {d: {t: [argmax value per item]}} (the eval part)."""
    out = {}; preds = {}; num = torch.tensor(NUM_IDS, device=dev); was = m.training; m.eval()
    try:
        for d, its in items_by_d.items():
            ok = {t: [] for t in range(1, K + 1)}; pv = {t: [] for t in range(1, K + 1)}
            for i in range(0, len(its), bs):
                ch = its[i:i + bs]; xs, msk = cue_inputs(m, ch, dev); hls = [m.lower(x_) for x_ in xs]; S = m.mem.zero_state(len(ch), dev)
                for t in range(1, K + 1):
                    j = min(t, len(xs)) - 1; S, h, pos = m.pass_aw(S, xs[j], msk, hls[j])
                    y = torch.tensor([int(it["raw"]["values"][min(t, it["d"])]) for it in ch], device=dev)
                    p_ = m.colon_logits(h, pos)[:, num].argmax(-1); ok[t].append((p_ == y).float()); pv[t].append(p_)
            out[str(d)] = {str(t): float(torch.cat(v).mean()) for t, v in ok.items()}; preds[str(d)] = {str(t): torch.cat(v).tolist() for t, v in pv.items()}
    finally:
        m.train(was)
    return (out, preds) if per_item else out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["direct", "inject"]); ap.add_argument("--think_k", type=int, default=0); ap.add_argument("--d_max", type=int, default=8)
    ap.add_argument("--steps", type=int, default=48000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--micro", type=int, default=1); ap.add_argument("--ckpt_passes", action="store_true"); ap.add_argument("--mem_write", default="top", choices=["top"])
    ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000); ap.add_argument("--cache_lower", action="store_true")
    ap.add_argument("--aux_step", type=float, default=0.0); ap.add_argument("--aux_fade", default="48000,48000"); ap.add_argument("--aux_reweight", action="store_true")
    ap.add_argument("--aux_target", default="value", choices=["value", "cumulative"], help="§229: cumulative = the re-rooted program R_t per pass (default value = §228)")
    ap.add_argument("--fast_add", action="store_true"); ap.add_argument("--fp32", action="store_true"); ap.add_argument("--sched_scale", type=float, default=1.0, help="smokes / gates only")
    ap.add_argument("--src", default="ow", choices=["ow", "nested"], help="nested = GATE ONLY (port-kernel identity vs nested170)"); ap.add_argument("--n_max_train", type=int, default=8, help="--src nested only")
    ap.add_argument("--futility", default="", help="LAT_v1 §4: 'step1,step2' (24000,32000): stop if the d = 2 train acc of the last 1k bin < 0.50 and the OLS slope of the "
                    "last 4 d = 2 bins <= +0.02 per bin (a still-rising first check defers to the second); '' = off")
    ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", default=None); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--stop_at", type=int, default=0, help="train to this step under the --steps schedule (ckpt at the stop step), then exit 9 (no eval)")
    ap.add_argument("--time_steps", type=int, default=0); ap.add_argument("--args_only", action="store_true"); ap.add_argument("--args_ref", default="")
    ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--auto_micro", action="store_true"); ap.add_argument("--mem_budget", type=float, default=21.0)
    ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--eval_parts", default="acc"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    ap.add_argument("--cue", default="none", choices=["none", "step", "all"], help="§233: step = LAT-ST (the S_min(t, d) cue at pass t), all = CUED-DIRECT (cues S_1..S_d on the decoder); none = inject_ow")
    ap.add_argument("--write_src", default="all", choices=["all", "answer"], help="§234: answer = + ' The answer is:' on every pass input, only the ':' state writes")
    ap.add_argument("--init_trunk", default="", help="§234: GPT-2 loaded strict from this file before DeltaMem is built"); ap.add_argument("--init_sha", default="", help="its sha256 prefix (asserted)")
    ap.add_argument("--mul_s_p3", type=float, default=1.0, help="§235: P(forced mul_s | d 3) on the TRAINING stream only (1.0 = unchanged)")
    ap.add_argument("--mul_s_p", type=int, default=1, choices=[0, 1, 2], help="§236: 0 = REG law (no forced mul_s); §238: 2 = DF (per-step mul_s, q_t); 1 = unchanged - on every stream / set (new eval seeds)")
    ap.add_argument("--init_model", default="", help="§237: the full params state (trunk + DeltaMem + W_o + gains) loaded strict after the build; calibration skipped")
    ap.add_argument("--init_model_sha", default="", help="§237: its sha256 prefix (asserted)")
    a = ap.parse_args(argv); a.task = "ow_topo" if a.src == "ow" else "nested"
    import boxes.deltamem as DM_
    from boxes.fsq120 import make_opt, lr_at, driver_version, env_info
    from boxes.deltas5 import write_grad_norm
    DM_.FAST_ADD = bool(a.fast_add); os.makedirs(a.out, exist_ok=True); dev = a.device; sc = a.sched_scale
    if a.arm == "direct":
        assert not (a.think_k or a.aux_step or a.aux_reweight or a.cache_lower or a.fast_add or a.futility or a.aux_target != "value"), "DIRECT has no latent flags"
        a.mem_write = a.write_mix_start = a.write_mix_end = None
    else:
        assert a.think_k >= 1 and a.aux_step > 0, "§228 inject: K >= 1 with the step aux"
    assert a.src == "ow" or a.d_max == 8, "--src nested: n ~ U{1..n_max_train} per item (nested170's law)"
    assert a.cue != "step" or (a.arm == "inject" and not a.ckpt_passes and a.src == "ow"), "--cue step: LAT-ST (inject, OW, no ckpt_passes)"
    assert a.cue != "all" or (a.arm == "direct" and a.src == "ow"), "--cue all: CUED-DIRECT (direct, OW)"
    assert a.write_src != "answer" or (a.arm == "inject" and a.cue == "step"), "--write_src answer: LAT-AW (inject, --cue step)"
    assert bool(a.init_trunk) == bool(a.init_sha), "--init_trunk needs --init_sha (G12)"
    assert 1 <= a.d_max <= 8 and a.bs % a.micro == 0
    K = a.think_k if a.arm == "inject" else 0
    torch.manual_seed(a.seed); rng_d = np.random.default_rng(a.seed + SEED_D); rng = np.random.default_rng(a.seed)
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    t_sets = time.time()
    if a.src == "ow":
        SETS = build_sets(mul_s_p=a.mul_s_p); EXCL = SETS["excl"] | (reg_excl(a, SETS["excl"]) if a.mul_s_p != 1 else set())   # §236 / §238: + the eval-own sets of the run law
        stream = OW.Stream(OW.SEED_TRAIN, EXCL, frames=OW.FRAME_TRAIN, mul_s_p3=a.mul_s_p3, mul_s_p=a.mul_s_p, **OWKW)   # §235 / §236: training stream
    else:                                                                   # gate only: nested170's TE / SPOOL exclusion and item law
        TE = N7.test_set(1000); EXCL = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)
        for _ in range(2200):
            r__, _t = N7.generate(r_sp, 8); EXCL.add(tuple(N7.render(r__)))
        SETS = None; stream = None
    set_sec = round(time.time() - t_sets)
    assert a.mul_s_p3 == 1.0 or a.src == "ow", "--mul_s_p3: OW only"
    assert a.mul_s_p == 1 or (a.mul_s_p3 == 1.0 and a.src == "ow"), "--mul_s_p 0: OW only, mul_s_p3 must be 1"
    m, g, params = build_aw(a, dev)
    assert bool(a.init_model) == bool(a.init_model_sha), "--init_model needs --init_model_sha"
    if a.init_model:                                                        # §237: the full model state, strict; the sha logged (init_model_sha256)
        sha_m = hashlib.sha256(open(a.init_model, "rb").read()).hexdigest(); assert sha_m.startswith(a.init_model_sha), ("init_model sha", sha_m, a.init_model_sha)
        params.load_state_dict(torch.load(a.init_model, map_location=dev, weights_only=True), strict=True); a.init_model_sha256 = sha_m
    f0_, f1_ = [float(x_) for x_ in a.aux_fade.split(",")]

    def lam_at(step):
        if not a.aux_step:
            return 0.0
        return a.aux_step if step <= f0_ * sc else a.aux_step * max(0.0, (f1_ * sc - step) / max(1.0, (f1_ - f0_) * sc))
    lam_now = [lam_at(1)]

    def draw_items(d):
        if a.src == "nested":
            return [tok_nested(it) for it in N7.train_items(rng, a.bs, a.n_max_train, EXCL)]
        return [tok_aw(stream.draw(d), a.aux_target, a.cue, a.write_src) for _ in range(a.bs)]

    def loss_items(items, stat=None):
        """nested170.loss_items (direct; inject with the aux on) on token-list items; stat (dict) collects the log fields (no effect on the graph)."""
        tgt = torch.tensor(NUM_IDS[[it["ans"] for it in items]], device=dev)
        if a.arm == "direct":
            lg = direct_logits(g, items, dev, ac); la = Fn.cross_entropy(lg, tgt); loss = la
        else:
            from torch.utils.checkpoint import checkpoint
            xs, msk = cue_inputs(m, items, dev); per = [[] for _ in items]; vok = {}   # §233: pass t reads xs[min(t, len(xs)) - 1] (cue none: one input)
            with ac():
                S = m.mem.zero_state(len(items), dev); hls = [m.lower(x_) for x_ in xs] if a.cache_lower else None   # the lower cache per distinct input
                for t in range(1, K + 1):
                    i_ = min(t, len(xs)) - 1; x = xs[i_]; hl = hls[i_] if hls is not None else None
                    S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False) if (a.ckpt_passes and torch.is_grad_enabled()) else m.write(S, x, msk, hl)
                    idx = [i for i, it in enumerate(items) if len(it["aux"]) >= t]
                    if idx:
                        lg_a, y_a = m.aux_logits(S[idx], [items[i] for i in idx], t, dev)
                        ce = Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100, reduction="none").sum(1) / (y_a != -100).sum(1)
                        for r_, i in enumerate(idx):
                            per[i].append(ce[r_] * AUX_W[t - 1] if a.aux_reweight else ce[r_])
                        if stat is not None and items[idx[0]]["vidx"] is not None:   # aux value-token acc (teacher-forced argmax at the value position)
                            with torch.no_grad():
                                pr_ = lg_a.detach().argmax(-1); vi = torch.tensor([items[i]["vidx"][t - 1] for i in idx], device=dev); ar = torch.arange(len(idx), device=dev)
                                vok[t] = (pr_[ar, vi] == y_a[ar, vi]).float().mean().item()
                lg = m.dec_logits(S, items, dev)
            la = Fn.cross_entropy(lg, tgt)
            aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean(); loss = la + lam_now[0] * aux
            if stat is not None:
                stat["aux"] = stat.get("aux", 0.0) + float(aux.detach()) * len(items); stat.setdefault("vacc", []).append((vok, len(items)))
                stat.setdefault("vacc_d", []).append((str(items[0]["d"]), vok, len(items)))   # §233 aux_vacc_by_d_pass (one d per step)
        if stat is not None:
            with torch.no_grad():
                stat["acc"] = stat.get("acc", 0.0) + float((lg.detach()[:, NUM_IDS].argmax(-1).cpu().numpy() == np.array([it["ans"] for it in items])).sum())
            stat["la"] = stat.get("la", 0.0) + float(la.detach()) * len(items); stat["n"] = stat.get("n", 0) + len(items)
        return loss

    s_mc = OW.Stream(SEED_MEMCHECK, frames=OW.FRAME_TRAIN, mul_s_p=a.mul_s_p, **OWKW) if a.src == "ow" else None   # §236: the run's law
    MC_TR = sorted([tok_aw(s_mc.draw(a.d_max), a.aux_target, a.cue, a.write_src) for _ in range(256)], key=lambda it: -st_len(it)) if s_mc else None
    MC_EV = sorted([tok_aw(it, a.aux_target, a.cue, a.write_src) for it in SETS["test2"][12]], key=lambda it: -st_len(it))[:1] if s_mc else None   # §233: TEST2's own longest d 12 (cued)
    WS_ITEMS = {d_: [tok_aw(it, a.aux_target, a.cue, a.write_src) for it in SETS["test2"][d_][:200]] for d_ in range(1, a.d_max + 1)} if a.write_src == "answer" else None   # §234 WS-READOUT (Note 234-5 c: d 1..d_max)

    def mem_train(micro):
        """the training step's peak at bs / micro of the longest d_max items + AdamW's two fp32 states (nested170's rule)."""
        torch.cuda.reset_peak_memory_stats(dev); lam_now[0] = a.aux_step; l_ = loss_items(MC_TR[:a.bs // micro]); l_.backward(); params.zero_grad(set_to_none=True)
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30 + 2 * sum(p_.numel() * 4 for p_ in params.parameters() if p_.requires_grad) / 2 ** 30

    def mem_eval(ebs):
        """the eval's peak (the optimiser state is cleared before the in-process eval): the longest d 12 item x ebs at K' 0..14, fp32."""
        from boxes import inject_aw_eval as EV
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); EV.answers_by_kprime(m, g, MC_EV * ebs, 14 if K else 0, dev, ebs, None)
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30
    if a.src == "ow" and not a.args_only and (a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.resume)):
        mc = []
        for kind, fn in (("train", mem_train), ("eval", mem_eval)):         # micro from the training peak, then eval_bs from the eval peak (separately)
            while True:
                try:
                    pk = fn(a.micro if kind == "train" else a.eval_bs)
                except torch.OutOfMemoryError:
                    pk = float("inf")
                params.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                mc.append({"kind": kind, "micro": a.micro, "eval_bs": a.eval_bs, "peak_gib": pk}); print("MEM", kind, a.micro, a.eval_bs, pk, flush=True)
                if pk <= a.mem_budget or a.mem_check:
                    break
                if kind == "train" and a.micro < a.bs:
                    a.micro *= 2
                elif kind == "eval" and a.eval_bs > 10:
                    a.eval_bs //= 2
                else:
                    break
        json.dump({"checks": mc, "micro": a.micro, "eval_bs": a.eval_bs, "peak_gib_train": [c for c in mc if c["kind"] == "train"][-1]["peak_gib"],
                   "peak_gib_eval": [c for c in mc if c["kind"] == "eval"][-1]["peak_gib"],
                   "train": "the longest bs / micro of 256 d = %d items (SEED_MEMCHECK) at K = %d, + AdamW states" % (a.d_max, K),
                   "eval": "the longest of 256 d = 12 items x eval_bs at K' 0..14 (fp32, no optimiser state)"}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return 0
        assert all(c["peak_gib"] <= a.mem_budget for c in mc[-1:]) and [c for c in mc if c["kind"] == "train"][-1]["peak_gib"] <= a.mem_budget, mc
    info = {**vars(a), "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info(),
            "hostprobe_sps": os.environ.get("HOSTPROBE_SPS"), "module_sha256": sha_file(os.path.abspath(__file__)), "inject_ow_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_ow.py")), "inject_st_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_st.py")),
            "eval_module_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_aw_eval.py")), "nested170_sha256": sha_file(os.path.join(ROOT, "boxes", "nested170.py")),
            "deltamem_sha256": sha_file(os.path.join(ROOT, "boxes", "deltamem.py")), "ow_words_sha256": sha_file(os.path.join(ROOT, "boxes", "ow_words.py")),
            "nested_words_sha256": sha_file(os.path.join(ROOT, "boxes", "nested_words.py")), "set_build_sec": set_sec, "excl_n": len(EXCL), "d_law": "d = default_rng(seed + 22331).integers(1, d_max + 1) per step",
            "format": {"direct": "dec = [EOT] + enc(' ' + text) + enc(' The answer is:') -> the number token (plain trunk, one pass)",
                       "inject": "think [EOT] + enc(' ' + text) (padded writes masked), K passes; dec as direct, read-only on S_K; aux [EOT, ' Step', ' t', ' :'] + "
                                 "enc(' ' + R_step[t]) + EOT, read-only on S_t, t <= d"}[a.arm] + CUE_FMT[a.cue] + AW_FMT[a.write_src]}
    if SETS is not None:
        info.update(SETS["shas"]); info["pair_stats"] = SETS["pair_stats"]
    if a.args_ref:
        ref = json.load(open(a.args_ref)); dif = {k: [ref.get(k), info.get(k)] for k in sorted(set(ref) | set(info)) if k not in META and ref.get(k) != info.get(k)}
        json.dump({"ref": a.args_ref, "diff": dif}, open(os.path.join(a.out, "args_diff.json"), "w", newline="\n"), indent=1); print("ARGS-DIFF", json.dumps(dif), flush=True)
    json.dump(info, open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    if a.args_only:
        print("ARGS_ONLY", a.out, flush=True); return 0
    opt = make_opt(params, a.lr); step0 = 1; n_nonfinite = 0; hist = []; wg_hist = []; wwin = []; wlast = []; gn_log = []; aux_hist = []
    expo = {}; binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; dseq = []; iseq = []; d2_bins = []; ent = {"buf": [], "entry_d1": None}; fut = {"checked": [], "stop": None}
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time()
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); params.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rng_d.bit_generator.state = ck["rng_d"]; rng.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"].cpu())
        if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        if stream is not None:
            stream.rng.bit_generator.state = ck["stream_rng"]; stream.stats = dict(ck["stream_stats"])
        step0 = ck["step"] + 1; n_nonfinite = ck["n_nonfinite"]; wg_hist = ck["wg_hist"]; gn_log = list(ck["gn_log"]); t0 = time.time() - ck["elapsed"]
        expo, binacc, vbin, dseq, iseq, d2_bins, ent, fut, wlast = ck["expo"], ck["binacc"], ck["vbin"], ck["dseq"], ck["iseq"], ck["d2_bins"], ck["ent"], ck["fut"], ck["wlast"]
        vbin_d = ck.get("vbin_d", {}); gnbin = ck.get("gnbin", {})
        log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    elif m is not None and K > 0 and not a.init_model:                      # note 147-2: g_add calibration (§237: skipped under --init_model, the loaded s_add kept)
        if a.src == "nested":
            its = [tok_nested(it) for it in N7.train_items(np.random.default_rng(17001), 64, a.n_max_train, set())]
        else:
            s_ = OW.Stream(SEED_CAL, frames=OW.FRAME_TRAIN, mul_s_p=a.mul_s_p, **OWKW); its = [tok_aw(s_.draw(1 + i % a.d_max), a.aux_target, a.cue, a.write_src) for i in range(64)]
        xs, msk = cue_inputs(m, its, dev); m.mem.norms = []
        with torch.no_grad(), ac():
            S_c = m.mem.zero_state(64, dev)
            for t_ in range(1, min(K, 4) + 1):                              # InjectN.think's write sequence, the pass-t input xs[min(t, len) - 1]
                S_c = m.write(S_c, xs[min(t_, len(xs)) - 1], msk)
        nr = m.mem.norms[1:]; m.mem.norms = None; r0 = float(torch.stack([d_ / a_.clamp_min(1e-12) for d_, a_ in nr]).median()) if nr else 1.0
        m.mem.s_add.data.fill_(math.log(max(r0, 1e-12))); json.dump({"g_add_init": r0, "items": "64 items, d = 1 + i % d_max (SEED_CAL)" if a.src == "ow" else "nested170 17001"},
                                                                    open(os.path.join(a.out, "s228_init.json"), "w", newline="\n"))
    FUT = [int(x_) for x_ in a.futility.split(",")] if a.futility else []
    last = (a.stop_at or a.steps) if not a.time_steps else min(a.stop_at or a.steps, step0 + a.time_steps - 1); tt0 = time.time(); step = step0 - 1
    for step in range(step0, last + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        params.train()
        if m is not None:
            m.mem.fade.fill_(min(1.0, step / max(1, int(500 * sc))))
            m.mem.gamma.fill_(0.0 if step <= a.write_mix_start * sc else min(1.0, (step - a.write_mix_start * sc) / max(1.0, (a.write_mix_end - a.write_mix_start) * sc)))
        lam_now[0] = lam_at(step)
        d = int(rng_d.integers(1, a.d_max + 1)) if a.src == "ow" else 0
        its = draw_items(d); opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro; st_ = {}
        for k_ in range(a.micro):
            lk = loss_items(its[k_ * ch:(k_ + 1) * ch], st_); (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
        if m is not None:
            wwin.append(write_grad_norm(m))
        gn = torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0); gn_log.append(float(gn))
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            n_nonfinite += 1; opt.zero_grad(); log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush()
        else:
            opt.step(); hist.append(tot)
        la_ = st_["la"] / st_["n"]; acc_ = st_["acc"] / st_["n"]; aux_hist.append((la_, st_["aux"] / st_["n"] if "aux" in st_ else None))
        dk = str(its[0]["d"]) if a.src == "ow" else "mix"; expo[dk] = expo.get(dk, 0) + 1
        b_ = binacc.setdefault(dk, [0, 0.0, 0.0, 0]); b_[0] += 1; b_[1] += acc_; b_[2] += la_; b_[3] += float(gn) > 1.0
        gb_ = gnbin.setdefault(dk, [0.0, 0]); gb_[0] += float(gn); gb_[1] += 1   # §237: pre-clip grad norm per d (emitted under --init_model)
        for vok, _n in st_.get("vacc", []):
            for t_, v_ in vok.items():
                vb_ = vbin.setdefault(str(t_), [0.0, 0]); vb_[0] += v_; vb_[1] += 1
        for dd_, vok, _n in (st_.get("vacc_d", []) if a.cue != "none" else []):
            for t_, v_ in vok.items():
                vb_ = vbin_d.setdefault(dd_, {}).setdefault(str(t_), [0.0, 0]); vb_[0] += v_; vb_[1] += 1
        dseq.append(dk if a.src == "ow" else "0"); iseq.append(dk + ":" + ",".join(it["sig"] for it in its))
        if dk == "1" and ent["entry_d1"] is None:                           # entry_d1: the first 100 consecutive d = 1 steps with mean answer CE < 0.5
            ent["buf"].append(la_)
            if len(ent["buf"]) == 100:
                if float(np.mean(ent["buf"])) < 0.5:
                    ent["entry_d1"] = step; log.write(json.dumps({"step": step, "entry_d1": step}) + "\n"); log.flush()
                ent["buf"] = []
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "sec": round(time.time() - t0), "aux_lambda": lam_now[0],
                   "answer_ce": float(np.mean([x_[0] for x_ in aux_hist])) if aux_hist else None}
            if m is not None:
                rec["aux_ce"] = float(np.mean([x_[1] for x_ in aux_hist])) if aux_hist else None
            hist = []; aux_hist = []
            if m is not None and wwin:
                md_ = float(np.median(wlast)) if wlast else 0.0; mx_ = float(np.max(wwin)); late = step > a.write_mix_start * sc
                w_rec = {"step": step, "mean": float(np.mean(wwin)), "max": mx_, "median_prev_1k": md_, "spike_100x_floor": bool(late and mx_ > 100 * md_ > 0 and mx_ > 1.0)}
                wlast.extend(wwin); del wlast[:-int(1000 * sc) or None]; wwin = []; wg_hist.append(w_rec)
                rec.update({"gamma": float(m.mem.gamma), "fade": float(m.mem.fade), "write_grad_norm": w_rec["mean"], "write_grad_spike_100x_floor": w_rec["spike_100x_floor"]})
            rec.update({"gn_preclip": gn_log[-1], "gn_preclip_mean100": float(np.mean(gn_log[-100:])), "clip_frac_100": float(np.mean([v_ > 1.0 for v_ in gn_log[-100:]]))})
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if step % 1000 == 0 or step == last:                                # the 1k-bin record (a partial bin at a stop / the end)
            bd = {k_: {"steps": v_[0], "acc": v_[1] / v_[0], "ce": v_[2] / v_[0], "clip": v_[3] / v_[0]} for k_, v_ in sorted(binacc.items())}
            br = {"step": step, "bin": (step - 1) // 1000, "bin_steps": len(dseq), "by_d": bd, "exposure_by_d": dict(sorted(expo.items())), "dseq": "".join(dseq),
                  "ihash": hashlib.sha1("\n".join(iseq).encode()).hexdigest()[:16]}
            if vbin:
                br["aux_vacc_by_pass"] = {k_: v_[0] / v_[1] for k_, v_ in sorted(vbin.items(), key=lambda kv: int(kv[0]))}
            if vbin_d:                                                      # §233 LP2 key (+ the per-cell counts: G10 pools them back to aux_vacc_by_pass)
                br["aux_vacc_by_d_pass"] = {d_: {k_: v_[0] / v_[1] for k_, v_ in sorted(tv.items(), key=lambda kv: int(kv[0]))} for d_, tv in sorted(vbin_d.items())}
                br["aux_vacc_by_d_pass_n"] = {d_: {k_: v_[1] for k_, v_ in sorted(tv.items(), key=lambda kv: int(kv[0]))} for d_, tv in sorted(vbin_d.items())}
            if WS_ITEMS is not None:                                        # §234 WS-READOUT: TEST2 d 1..d_max (first 200 each), fp32, no_grad, never fed back
                br["ws_readout"] = ws_readout(m, WS_ITEMS, dev, K)
            if a.mul_s_p3 < 1.0 and stream is not None:                    # §235: cumulative accepted d 3 training items with a mul_s step / all d 3 items
                br["stream_d3_mul_s"] = [stream.stats.get("d3_mul_s", 0), stream.stats.get("d3_items", 0)]
            if a.init_model and gnbin:                                       # §237 (Note 237-1): the mean pre-clip grad norm per d over the bin
                br["gn_preclip_by_d"] = {k_: v_[0] / v_[1] for k_, v_ in sorted(gnbin.items())}
            log.write(json.dumps(br) + "\n"); log.flush()
            if "2" in bd and step % 1000 == 0:
                d2_bins.append([step, bd["2"]["acc"]])
            binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; dseq = []; iseq = []
        if FUT and step in FUT and fut["stop"] is None:                     # LAT_v1 §4 futility (the d = 2 train acc of complete 1k bins)
            l4 = [v_ for _, v_ in d2_bins[-4:]]; sl = float(np.polyfit(np.arange(len(l4)), l4, 1)[0]) if len(l4) >= 2 else None
            accl = l4[-1] if l4 else None; rising = sl is not None and sl > 0.02
            frec = {"step": step, "d2_last_bin_acc": accl, "slope_last4": sl, "bins": d2_bins[-4:], "rising": rising,
                    "stop": bool(accl is not None and accl < 0.50 and not rising)}
            fut["checked"].append(frec); json.dump(fut, open(os.path.join(a.out, "futility.json"), "w", newline="\n"), indent=1)
            log.write(json.dumps({"futility": frec}) + "\n"); log.flush()
            if frec["stop"]:
                fut["stop"] = step; json.dump(fut, open(os.path.join(a.out, "futility.json"), "w", newline="\n"), indent=1); break
        if a.ckpt_every and (step % a.ckpt_every == 0 or step == a.stop_at) and step < a.steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": params.state_dict(), "opt": opt.state_dict(), "rng_d": rng_d.bit_generator.state, "np_rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None,
                        "stream_rng": stream.rng.bit_generator.state if stream is not None else None, "stream_stats": dict(stream.stats) if stream is not None else None,
                        "step": step, "n_nonfinite": n_nonfinite, "wg_hist": wg_hist, "gn_log": gn_log, "elapsed": time.time() - t0, "expo": expo, "binacc": binacc,
                        "vbin": vbin, "vbin_d": vbin_d, "gnbin": gnbin, "dseq": dseq, "iseq": iseq, "d2_bins": d2_bins, "ent": ent, "fut": fut, "wlast": wlast}, cp)
    if a.time_steps:
        sps = (time.time() - tt0) / max(1, last - step0 + 1)
        tj = {"arm": a.arm, "think_k": K, "d_max": a.d_max, "steps_timed": last - step0 + 1, "sec_per_step": sps, "est_hours_steps": sps * a.steps / 3600,
              "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "micro": a.micro, "peak_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else None}
        for f_ in ("timing.json", "eval.json"):
            json.dump(tj, open(os.path.join(a.out, f_), "w", newline="\n"), indent=1)
        print("TIMING", sps, flush=True); return 0
    if a.stop_at and step == a.stop_at and a.stop_at < a.steps:
        print("STOP_AT", step, flush=True); return 9
    train_sec = round(time.time() - t0); torch.save(params.state_dict(), os.path.join(a.out, "model.pt"))
    opt.zero_grad(set_to_none=True); opt.state.clear()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    gl = np.array(gn_log) if gn_log else np.array([np.nan])
    train = {"train_sec": train_sec, "last_step": step, "n_nonfinite": n_nonfinite, "entry_d1": ent["entry_d1"], "futility": fut if FUT else None,
             "clip_rate": float((gl > 1.0).mean()), "grad_norm_preclip": {"median": float(np.median(gl)), "p90": float(np.percentile(gl, 90))},
             "write_grad_spikes_floor": [x_ for x_ in wg_hist if x_["spike_100x_floor"]], "micro": a.micro, "train_gpu": info["train_gpu"]}
    json.dump(train, open(os.path.join(a.out, "train_summary.json"), "w", newline="\n"), indent=1)
    if a.src == "nested":
        return 0
    from boxes import inject_aw_eval as EV
    EV.run(a.out, [p_ for p_ in a.eval_parts.split(",") if p_], dev, a.eval_bs, model=(m, g), sets=SETS, train=train)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("DONE", a.out, flush=True); return 0


if __name__ == "__main__":
    sys.exit(main())
