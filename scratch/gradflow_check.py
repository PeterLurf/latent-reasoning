"""Audit 2026-10-05 (after the §116 copy_loss no_grad bug): gradient-flow assertion per loss term, before any run of a
code path. For each term a run uses, backward on that term ALONE and assert a nonzero grad norm on the trainable
parameters; norms are logged per term. Terms computed inside one chunk are isolated by differencing the loss with the
term on and off on identical inputs (the main part cancels exactly). Noise terms (dither) are checked as a change in the
loss under fixed inputs plus a nonzero gradient through the noisy carry.
    python scratch/gradflow_check.py --paths accel99 mk108 fsq120 self110_copy --device cpu
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
from boxes import s5  # noqa: E402


def gnorm(m):
    return float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in m.parameters() if p.grad is not None)))


def term(m, f):
    m.zero_grad(set_to_none=True); loss = f(); assert loss.requires_grad, "term has no graph"; loss.backward(); n = gnorm(m)
    return n


def check_accel99(dev, out):
    from boxes.accel99 import MultGen99, latent_batch, textmix_batch
    from boxes.mult98 import load
    D = load(os.path.join(ROOT, "bench", "data", "mult_train", "4x4_train.txt"), limit=64); g = s5.load_gpt2(dev)
    m = MultGen99(g, "genm", std=False).to(dev); m.markov = True; m.train()
    rng = np.random.default_rng(0); pre, st_, ops, tg, ans = latent_batch(D, rng, 4, dev, 4)
    def L(reader, anchor):
        torch.manual_seed(0); m.reader, m.anchor = reader, anchor; return m.chunk(pre, st_, ops, step_tgt=tg, ans_tgt=ans)[0]
    out["accel99.main"] = term(m, lambda: L(False, False))
    out["accel99.reader"] = term(m, lambda: L(True, False) - L(False, False))
    out["accel99.anchor"] = term(m, lambda: L(False, True) - L(False, False))
    tb = textmix_batch(D, rng, 4, dev, 2); m.reader = m.anchor = False
    out["accel99.textmix"] = term(m, lambda: m.chunk(*tb[:3], step_tgt=tb[3], ans_tgt=tb[4])[0])


def check_mk108(dev, out):
    from boxes.gen97 import true_chain
    from boxes.mk108 import MK108
    m = MK108(s5.load_gpt2(dev)).to(dev); m.train(); rng = np.random.default_rng(0)
    S = rng.integers(0, 120, 4); O = rng.integers(0, 120, (4, 4)); tc = true_chain(S, O)
    st = m.start_from(els=S, dev=dev); o = torch.tensor(O, device=dev); t = torch.tensor(tc, device=dev)
    def L(reader, anchor):
        torch.manual_seed(0); m.reader, m.anchor = reader, anchor; return m.chunk(st, o, t, decode=None)[0]
    out["mk108.main"] = term(m, lambda: L(False, False))
    out["mk108.reader"] = term(m, lambda: L(True, False) - L(False, False))
    out["mk108.anchor"] = term(m, lambda: L(False, True) - L(False, False))


def check_fsq120(dev, out):
    from boxes.fsq120 import FSQ120, Family, make_batch
    for fam_name in ("s5", "ff"):
        fam = Family(fam_name); m = FSQ120(s5.load_gpt2(dev), fam).to(dev); m.train(); rng = np.random.default_rng(0)
        s0, x, lab = make_batch(fam, rng, 4, 4); X = torch.tensor(x, device=dev)
        st = lambda: m.start(4, dev, s0 if fam_name == "s5" else None)  # noqa: E731
        for mode in ("fsq", "no_round"):
            m.no_round = mode == "no_round"; m.p_hard = 1.0
            def L(noise):
                m.noise = noise; torch.manual_seed(0); return m.chain(st(), X, lab)[0]
            out[f"fsq120.{fam_name}.{mode}.main"] = term(m, lambda: L(0.0))
            out[f"fsq120.{fam_name}.{mode}.dither_grad"] = term(m, lambda: L(0.25))
            with torch.no_grad():
                out[f"fsq120.{fam_name}.{mode}.dither_loss_delta"] = abs(float(L(0.25)) - float(L(0.0)))
            gd = m.down.weight.grad; out[f"fsq120.{fam_name}.{mode}.down_grad"] = float(gd.norm()) if gd is not None else 0.0
        m.noise = 0.0


def check_self110_copy(dev, out):
    from boxes.gen97 import STTOK
    from boxes.self110 import Gen110
    m = Gen110(s5.load_gpt2(dev)).to(dev); m.train(); cs = np.random.default_rng(0).integers(0, 120, 4)
    out["self110.copy"] = term(m, lambda: m.copy_loss(torch.tensor(STTOK[cs], device=dev), torch.tensor(cs, device=dev)))
    out["self110.expose"] = term(m, lambda: m.expose_loss(torch.tensor(STTOK[cs], device=dev)))   # §124 TEXT-EXPOSE


def check_self110_resume(dev, out):
    """note 112-5: the resume path trains the masked-chunk loss from a LOADED finished model (s112_f010_s22)."""
    from boxes.gen97 import STTOK, true_chain
    from boxes.self110 import Gen110
    m = Gen110(s5.load_gpt2(dev)).to(dev); src = os.path.join(ROOT, "boxes", "runs_s5", "s112_f010_s22", "model.pt")
    m.load_state_dict(torch.load(src, map_location=dev)); m.train(); rng = np.random.default_rng(0)
    S = np.zeros(4, np.int64); O = rng.integers(0, 120, (4, 4)); tc = true_chain(S, O)
    out["self110.resume_main"] = term(m, lambda: m.chunk_masked(torch.tensor(STTOK[S], device=dev), torch.tensor(O, device=dev), torch.tensor(tc, device=dev)))


def check_fsq120_std(dev, out):
    """§125 STD: the standardised carried embedding (map inside embed_code) passes gradient to W_up / W_down."""
    from boxes.fsq120 import FSQ120, Family, make_batch
    for fam_name in ("s5", "ff"):
        fam = Family(fam_name); m = FSQ120(s5.load_gpt2(dev), fam).to(dev); m.enable_std(); m.train(); m.p_hard = 1.0
        s0, x, lab = make_batch(fam, np.random.default_rng(0), 4, 4); X = torch.tensor(x, device=dev)
        out[f"fsq120.{fam_name}.std.main"] = term(m, lambda: m.chain(m.start(4, dev, s0 if fam_name == "s5" else None), X, lab)[0])
        out[f"fsq120.{fam_name}.std.up_grad"] = float(m.up.weight.grad.norm()); out[f"fsq120.{fam_name}.std.down_grad"] = float(m.down.weight.grad.norm())
        m.update_stats(); out[f"fsq120.{fam_name}.std.stats_seen"] = float(int(m.seen))


def check_fsq120_rawh(dev, out):
    """§126 RAW-H: carrier = std(last hidden); gradient reaches GPT-2 through the carried path."""
    from boxes.fsq120 import FSQ120, Family, make_batch
    fam = Family("s5"); m = FSQ120(s5.load_gpt2(dev), fam).to(dev); m.raw_h = True; m.enable_std(); m.train()
    s0, x, lab = make_batch(fam, np.random.default_rng(0), 4, 4); X = torch.tensor(x, device=dev)
    out["fsq120.s5.rawh.main"] = term(m, lambda: m.chain(m.start(4, dev, s0), X, lab)[0])
    m.update_stats(); out["fsq120.s5.rawh.stats_seen"] = float(int(m.seen))


def check_fsq120_anneal(dev, out):
    """§126 ANNEAL (ramp, lam 0.5): the QBN distillation term trains the bottleneck (W_up / W_down)."""
    from boxes.fsq120 import FSQ120, Family, make_batch
    fam = Family("s5"); m = FSQ120(s5.load_gpt2(dev), fam).to(dev); m.enable_std(); m.std = False; m.anneal = True
    m.lam = 0.5; m.distill_on = True; m.p_hard = 0.0; m.train()
    s0, x, lab = make_batch(fam, np.random.default_rng(0), 4, 4); X = torch.tensor(x, device=dev)
    def dist():
        m._dist = []; m.chain(m.start(4, dev, s0), X, lab); d = torch.stack(m._dist).mean(); m._dist = []; return d
    out["fsq120.s5.anneal.distill"] = term(m, dist); out["fsq120.s5.anneal.up_grad"] = float(m.up.weight.grad.norm())


def check_fsq120_s127(dev, out):
    """§127 (note 127-1, before Stage 1): VAR's and SKIP's gradients reach W_down at step 1; LR10's group LRs; STD2's row norm."""
    from boxes.fsq120 import FSQ120, Family, lr_at, make_batch, make_opt
    fam = Family("s5"); s0, x, lab = make_batch(fam, np.random.default_rng(0), 8, 2); X = torch.tensor(x, device=dev)
    m = FSQ120(s5.load_gpt2(dev), fam).to(dev); m.enable_var(fam.n_items); m.train(); m.p_hard = 0.0; m.var_on = True; m._var = []
    m.chain(m.start(8, dev, s0), X, lab); vl = m.var_loss()
    out["fsq120.s127.var.grad_down"] = float(torch.autograd.grad(vl, m.down.weight)[0].norm()); out["fsq120.s127.var.term"] = float(vl)
    m = FSQ120(s5.load_gpt2(dev), fam).to(dev); m.enable_skip(2); m.train(); m.p_hard = 0.0
    _, e, _, _ = m.chain(m.start(8, dev, s0), X); ans = torch.tensor(fam.answer_tokens(lab[:, -1]), device=dev)
    Ls = m.answer_ce(e.detach(), ans, m.last_qs)                           # the answer loss through the skip maps only
    out["fsq120.s127.skip.grad_down"] = float(torch.autograd.grad(Ls, m.down.weight)[0].norm())
    out["fsq120.s127.skip.grad_maps"] = float(torch.autograd.grad(m.answer_ce(e.detach(), ans, m.last_qs), m.skip_w)[0].norm())
    m = FSQ120(s5.load_gpt2(dev), fam).to(dev); opt = make_opt(m, 1e-4, 10.0); lrs = [lr_at(1e-4, gr, 1, 24000) for gr in opt.param_groups]
    out["fsq120.s127.lr10.ratio"] = lrs[1] / lrs[0]; out["fsq120.s127.lr10.n_new"] = float(sum(p_.numel() for p_ in opt.param_groups[1]["params"]))
    m = FSQ120(s5.load_gpt2(dev), fam).to(dev); m.enable_std2(); m.train(); m.p_hard = 0.0
    _, _, _, embs = m.chain(m.start(8, dev, s0), X, record=True); rn = float(torch.cat(embs).float().norm(dim=-1).mean())
    wn = float(m.g.w("wte.weight").float().norm(dim=-1).mean()); out["fsq120.s127.std2.row_norm_ratio"] = rn / wn
    out["fsq120.s127.std2.in_band"] = float(0.5 <= rn / wn <= 2.0)
    out["fsq120.s127.lr10.ratio_ok"] = float(abs(lrs[1] / lrs[0] / 10 - 1) <= 0.01)


def check_fsq120_s127_varbatch(dev, out):
    """Note 127-3: VAR with the forward's own group means reaches W_down; a forward without a group of >= 2 rows gives 0."""
    from boxes.fsq120 import FSQ120, Family, make_batch
    fam = Family("s5"); s0, x, lab = make_batch(fam, np.random.default_rng(0), 64, 2); X = torch.tensor(x, device=dev)
    m = FSQ120(s5.load_gpt2(dev), fam).to(dev); m.enable_var(fam.n_items); m.var_est = "batch"; m.train(); m.p_hard = 0.0
    m.var_on = True; m._var = []; m.chain(m.start(64, dev, s0), X, lab); vl = m.var_loss()
    out["fsq120.s127.varbatch.grad_down"] = float(torch.autograd.grad(vl, m.down.weight)[0].norm()); out["fsq120.s127.varbatch.term"] = float(vl.detach())
    m._var = [(torch.randn(3, 8, device=dev), torch.tensor([1, 2, 3], device=dev), torch.zeros(3, dtype=torch.long, device=dev))]
    out["fsq120.s127.varbatch.singletons_zero"] = float(m.var_loss() == 0)


def check_fsq120_s128(dev, out):
    """§128: the main answer loss reaches the bottleneck and GPT-2 on each held-out family (code-zero start, digit answers)."""
    from boxes.fsq120 import FSQ120, Family, make_batch
    for name in ("cyc", "mod", "rdfa"):
        fam = Family(name); s0, x, lab = make_batch(fam, np.random.default_rng(0), 8, 3)
        m = FSQ120(s5.load_gpt2(dev), fam).to(dev); m.train(); m.p_hard = 0.0
        out["fsq120.s128.%s.main" % name] = term(m, lambda: m.chain(m.start(8, dev, None), torch.tensor(x, device=dev), lab)[0])
        out["fsq120.s128.%s.down" % name] = float(m.down.weight.grad.norm())


def check_fsq120_s129(dev, out):
    """§129 (run on the host, CUDA bf16 as training): after 3 training steps, the head re-draw drops ln / W_down / W_up's AdamW
    state and leaves the body's untouched; at the first phase-2 step the re-drawn head gets gradient."""
    from boxes.fsq120 import FSQ120, Family, make_batch, make_opt, s129_redraw
    fam = Family("s5"); amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    torch.manual_seed(2); m = FSQ120(s5.load_gpt2(dev), fam).to(dev); opt = make_opt(m, 1e-4); m.train(); m.p_hard = 0.0; rng = np.random.default_rng(2)
    def step():
        s0, x, lab = make_batch(fam, rng, 8, 2)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            loss = m.chain(m.start(8, dev, s0), torch.tensor(x, device=dev), lab)[0]
        opt.zero_grad(); loss.backward(); return loss
    for _ in range(3):
        step(); opt.step()
    head = list(m.ln.parameters()) + list(m.down.parameters()) + list(m.up.parameters()); hid = {id(p_) for p_ in head}
    body = [p_ for p_ in m.parameters() if id(p_) not in hid and p_ in opt.state]
    snap = [(opt.state[p_]["exp_avg"].clone(), opt.state[p_]["exp_avg_sq"].clone()) for p_ in body]
    w0 = m.down.weight.detach().clone(); s129_redraw(m, opt, 2)
    out["fsq120.s129.amp_bf16"] = float(amp is not None)
    out["fsq120.s129.head_state_dropped"] = float(all(p_ not in opt.state for p_ in head))
    out["fsq120.s129.body_state_unchanged"] = float(all(torch.equal(opt.state[p_]["exp_avg"], a_) and torch.equal(opt.state[p_]["exp_avg_sq"], b_) for p_, (a_, b_) in zip(body, snap)))
    out["fsq120.s129.head_redrawn"] = float(not torch.equal(w0, m.down.weight.detach()))
    step()
    out["fsq120.s129.phase2_grad_down"] = float(m.down.weight.grad.float().norm()); out["fsq120.s129.phase2_grad_up"] = float(m.up.weight.grad.float().norm())
    out["fsq120.s129.phase2_grad_ln"] = float(m.ln.weight.grad.float().norm())


def check_table132(dev, out):
    """§132: the answer loss reaches the bottleneck (W_down / W_up / LN) and GPT-2 through the table-prefixed segments; CUDA bf16 when
    available (as training)."""
    from boxes.table132 import Splits, Table132, TableFam, problems
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    for k, mm in ((4, 2), (6, 3)):
        sp = Splits(k, mm); rng = np.random.default_rng(0); tabs = sp.train[:4] if sp.train is not None else sp.held[:4]
        c, x, lab = problems(rng, tabs, 3, mm)
        m = Table132(s5.load_gpt2(dev), TableFam(k, mm)).to(dev); m.train(); m.p_hard = 0.0; m.ctx = torch.tensor(c, device=dev)
        def L():
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                return m.chain(m.start(4, dev), torch.tensor(x, device=dev), lab)[0]
        out["table132.k%d.main" % k] = term(m, L)
        for nm, p_ in (("down", m.down.weight), ("up", m.up.weight), ("ln", m.ln.weight)):
            out["table132.k%d.%s" % (k, nm)] = float(p_.grad.float().norm())
    out["table132.amp_bf16"] = float(amp is not None)


def _b119(method, dev):
    from boxes.base119 import Base119, load_gpt2x
    from boxes.gen97 import STTOK
    from boxes.latent86 import batch
    if method == "codi":
        m = Base119(load_gpt2x(dev, lora_r=128), "codi").to(dev)
        with torch.no_grad():                                               # LoRA B = 0 at init: nudge it so dL/dA is testable
            for p_ in m.g.lb.values():
                p_.normal_(0, 1e-3)
    else:
        m = Base119(load_gpt2x(dev), method, c=2 if method == "simcot" else 1, dec=load_gpt2x(dev) if method == "simcot" else None).to(dev)
    m.train(); X = np.random.default_rng(0).integers(0, 120, (8, 128)); Y = np.zeros_like(X); w = np.zeros(8, dtype=np.int64)
    for t in range(128):
        w = s5.COMP[w, X[:, t]]; Y[:, t] = w
    st, o, tg = batch(np.random.default_rng(1), X, Y, 4, 4)
    return m, (torch.tensor(STTOK[st], device=dev), torch.tensor(o, device=dev), torch.tensor(tg, device=dev))


def gnorm_of(params):
    return float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))


def check_base119_coconut(dev, out):
    """§119 Coconut: CE on the remaining text at stage 0 and at the final stage (gradient through the thoughts)."""
    m, b = _b119("coconut", dev)
    for s in (0, 3):
        out[f"base119.coconut.stage{s}.ce"] = term(m, lambda: m.loss(*b, stage=s)[0])
    out["base119.coconut.special"] = float(m.special.grad.norm())


def check_base119_simcot(dev, out):
    """§119 SIM-CoT: the main CE, and the step decoder term alone; the decoder term must reach theta (through the thoughts)."""
    m, b = _b119("simcot", dev)
    out["base119.simcot.stage3.ce"] = term(m, lambda: m.loss(*b, stage=3)[1]["ce"])
    out["base119.simcot.stage3.step"] = term(m, lambda: m.loss(*b, stage=3)[1]["step_ce"])
    out["base119.simcot.step.theta"] = gnorm_of(m.g.parameters()); out["base119.simcot.step.phi"] = gnorm_of(m.dec.parameters())


def check_base119_codi(dev, out):
    """§119 CODI: student CE, teacher CE and KD each alone; KD and the student CE must reach the LoRA and the projection."""
    m, b = _b119("codi", dev)
    for t in ("student", "teacher", "kd"):
        out[f"base119.codi.{t}"] = term(m, lambda: m.loss(*b)[1][t])
        out[f"base119.codi.{t}.lora"] = gnorm_of(list(m.g.la.values()) + list(m.g.lb.values()))
        if t != "teacher":
            out[f"base119.codi.{t}.prj"] = gnorm_of(m.prj.parameters())
    out["base119.codi.base_frozen"] = float(all(not p.requires_grad for p in m.g.p.values()))


def check_s133(dev, out):
    """§133 (Theory deb9d7fa): every stage-1 arm, chain loss under CUDA bf16 autocast at p_hard 1 (the hard regime): nonzero grad on the
    total and on W_down / W_up (the carrier); NOISE also as a loss change under dither."""
    import math
    import boxes.fsq120 as F120
    arms = {"cont64": dict(d=64, no_round=True), "fsq16": dict(d=16), "fsq64": dict(d=64), "soft16": dict(d=16, soft=0.1),
            "soft64": dict(d=64, soft=0.1), "noise64": dict(d=64, no_round=True, noise=0.25), "pin64": dict(d=64, no_round=True, ln=True)}
    saved = (F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT); amp = torch.bfloat16 if str(dev).startswith("cuda") else None
    for name, c in arms.items():
        F120.D_Q = c["d"]; F120.LEV = 8; F120.HALF = (F120.LEV - 1) / 2; F120.SHIFT = math.atanh(F120.OFF / F120.HALF)
        fam = F120.Family("s5"); m = F120.FSQ120(s5.load_gpt2(dev), fam).to(dev); m.train(); rng = np.random.default_rng(0)
        m.no_round = c.get("no_round", False); m.soft_beta = c.get("soft", 0.0); m.carrier_ln = c.get("ln", False); m.p_hard = 1.0
        assert m.down.out_features == c["d"] and m.up.in_features == c["d"]
        s0, x, lab = F120.make_batch(fam, rng, 4, 4); X = torch.tensor(x, device=dev)
        def L(noise=0.0):
            m.noise = noise; torch.manual_seed(0)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                return m.chain(m.start(4, dev, s0), X, lab)[0]
        out[f"s133.{name}.main"] = term(m, lambda: L(c.get("noise", 0.0)))
        for nm_, w_ in (("down", m.down.weight), ("up", m.up.weight)):
            out[f"s133.{name}.{nm_}_grad"] = float(w_.grad.norm()) if w_.grad is not None else 0.0
        if c.get("noise"):
            with torch.no_grad():
                out[f"s133.{name}.dither_loss_delta"] = abs(float(L(c["noise"])) - float(L(0.0)))
        m.noise = 0.0; del m
    F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT = saved


def check_s134(dev, out):
    """§134 COPY-ANCHOR (Theory 26bc216e; the §116 lesson): the copy loss ALONE ([EOT, answer text] -> code -> [EOT, code] decoder CE)
    reaches W_down and W_up (and GPT-2), under CUDA bf16, in the soft (p_hard 0) and hard (p_hard 1) regimes; plus the chain loss."""
    import boxes.fsq120 as F120
    amp = torch.bfloat16 if str(dev).startswith("cuda") else None
    fam = F120.Family("s5"); m = F120.FSQ120(s5.load_gpt2(dev), fam).to(dev); m.train(); rng = np.random.default_rng(0)
    cans = torch.tensor(fam.answer_tokens(rng.integers(0, 120, 6)), device=dev)
    for ph in (0.0, 1.0):
        m.p_hard = ph
        def C():
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                e, qs, _, _ = m.copy_write(cans); return m.answer_ce(e, cans, qs)
        out[f"s134.copy_p{int(ph)}.main"] = term(m, C)
        for nm_, w_ in (("down", m.down.weight), ("up", m.up.weight)):
            out[f"s134.copy_p{int(ph)}.{nm_}_grad"] = float(w_.grad.norm()) if w_.grad is not None else 0.0
    s0, x, lab = F120.make_batch(fam, rng, 4, 4); X = torch.tensor(x, device=dev)
    def M():
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            return m.chain(m.start(4, dev, s0), X, lab)[0]
    out["s134.chain.main"] = term(m, M)


def check_s135(dev, out):
    """§135 (f1899abd / note 135-1): the aux InfoNCE ALONE gives a nonzero gradient into the carrier's ENCODER - W_down and the GPT-2
    trunk that writes z_b, through FSQ's straight-through (p_hard 0 and 1) - plus W_up / P_s / P_t, and NONE into the frozen target
    model. Under CUDA bf16. The target model here is pretrained GPT-2 (BASE's), layer 6, centred by its batch mean."""
    import boxes.fsq120 as F120
    from boxes.teacher135 import teacher_targets
    amp = torch.bfloat16 if str(dev).startswith("cuda") else None
    fam = F120.Family("s5"); m = F120.FSQ120(s5.load_gpt2(dev), fam).to(dev); m.enable_tsd(); m.train(); rng = np.random.default_rng(0)
    tg = s5.load_gpt2(dev).eval()
    for p_ in tg.parameters():
        p_.requires_grad_(False)
    s0, x, lab = F120.make_batch(fam, rng, 4, 4); X = torch.tensor(x, device=dev)
    T = teacher_targets(tg, 6, s0, x, lab, dev, amp); T = (T - T.mean((0, 1))).reshape(-1, 768)
    trunk = m.g.p["h_3_attn_c_attn_weight"]
    for ph in (0.0, 1.0):
        m.p_hard = ph
        def A():
            m._tsd_e = []
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                m.chain(m.start(4, dev, s0), X, lab)
            E = torch.stack(m._tsd_e, 1).reshape(-1, 768); m._tsd_e = []
            return F120.tsd_loss135(m, E, T, 0.1, lab)[0]
        out[f"s135.aux_p{int(ph)}.main"] = term(m, A)
        for nm_, w_ in (("down", m.down.weight), ("trunk_h3", trunk), ("up", m.up.weight), ("P_s", m.tsd_ps.weight), ("P_t", m.tsd_pt.weight)):
            out[f"s135.aux_p{int(ph)}.{nm_}_grad"] = float(w_.grad.norm()) if w_.grad is not None else 0.0
        out[f"s135.aux_p{int(ph)}.target_no_grad"] = 1.0 if all(p_.grad is None for p_ in tg.parameters()) else 0.0


def check_s138(dev, out):
    """§138 (fa1720ff / note 138-1) FULL-HINGE: d 768 CONT carrier, tanh in fp32; eps / directions measured on the probe batch as at 2k.
    The hinge term ALONE (rho-hat on the pre-quantisation z) reaches W_down, W_up and the GPT-2 trunk under CUDA bf16. The margin is set to 0
    for the check (term = rho-hat > 0, slope 1): at init rho-hat can sit below 0.3, where the hinge is correctly flat."""
    import math
    import boxes.fsq120 as F120
    saved = (F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT); F120.D_Q, F120.LEV = 768, 8; F120.HALF = (F120.LEV - 1) / 2; F120.SHIFT = math.atanh(F120.OFF / F120.HALF)
    amp = torch.bfloat16 if str(dev).startswith("cuda") else None; fam = F120.Family("s5"); m = F120.FSQ120(s5.load_gpt2(dev), fam).to(dev)
    m.no_round = True; m.tanh_fp32 = True; ref = F120.s138_hinge_ref(m, fam, dev, amp, 0.1); m.hinge_eps, m.hinge_dir, m.hinge_on = ref["eps"], ref["dir"], True
    m.train(); m.p_hard = 1.0; m.hinge_rows = 1.0; m.hinge_margin = 0.0; s0, x, lab = F120.make_batch(fam, np.random.default_rng(0), 8, 4)
    m.zero_grad(set_to_none=True); m._hinge = []
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        m.chain(m.start(8, dev, s0), torch.tensor(x, device=dev), lab)
    tot = torch.stack([h_ for h_, _ in m._hinge]).mean(); out["s138.hinge_rho_median"] = float(torch.cat([r_ for _, r_ in m._hinge]).median()); m._hinge = []
    tot.backward()
    for nm_, w_ in (("down", m.down.weight), ("up", m.up.weight), ("trunk", m.g.w("h.0.attn.c_attn.weight"))):
        out["s138.hinge_%s_grad" % nm_] = float(w_.grad.float().norm()) if w_.grad is not None else 0.0
    del m
    F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT = saved


def check_s142(dev, out):
    """§142 (a27cf1cf, note 142-1) MULT-READ on NOISE-64 (d 64, --no_round, training dither 0.25): each reader term CE_j (j = 1..3, the shared
    answer decoder on code_j vs S_j's 8 tokens) ALONE reaches W_down, W_up and the GPT-2 trunk under CUDA bf16; a REMOVED target has exactly 0
    gradient (with S_j removed, changing S_j's target tokens leaves every gradient bitwise identical), and with all three removed there is no
    reader term. Targets here are gold digits (gradient flow only; runs use the teacher cache)."""
    import math
    import boxes.fsq120 as F120
    import boxes.multcol as MC
    import boxes.multrow as R
    saved = (F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT); F120.D_Q, F120.LEV = 64, 8; F120.HALF = (F120.LEV - 1) / 2; F120.SHIFT = math.atanh(F120.OFF / F120.HALF)
    amp = torch.bfloat16 if str(dev).startswith("cuda") else None; fam = R.RowFam(); m = R.MultRow(s5.load_gpt2(dev), fam).to(dev)
    m.soft_beta = 0.0; m.no_round = True; m.noise = 0.25; m.train(); rng = np.random.default_rng(0); A, B = MC.gen_operands(rng, 8, 4), MC.gen_operands(rng, 8, 4)
    X = torch.tensor(np.tile(np.arange(4), (8, 1)), device=dev); lab = R.sj_values(A, B); rtok = torch.tensor(np.array(MC.DTOK)[R.digits8(lab[:, :3])], device=dev)
    def run(mask, tok, seed=7):
        m.zero_grad(set_to_none=True); R.set_ops(m, A, B, dev); torch.manual_seed(seed)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            fin, rd, n = m.chain_read(m.start(8, dev), X, lab, tok, torch.tensor(mask, device=dev))
        return fin, rd, n
    for j in (1, 2, 3):
        mk = np.zeros((8, 3), bool); mk[:, j - 1] = True; _, rd, n = run(mk, rtok); rd.backward()
        out[f"s142.reader{j}.down_grad"] = float(m.down.weight.grad.float().norm()); out[f"s142.reader{j}.up_grad"] = float(m.up.weight.grad.float().norm())
        out[f"s142.reader{j}.trunk_grad"] = float(m.g.w("h.0.attn.c_attn.weight").grad.float().norm()); out[f"s142.reader{j}.one_term"] = 1.0 if n == 1 else 0.0
    for j in (1, 2, 3):                                                     # removed S_j: its target tokens do not enter the graph
        mk = np.ones((8, 3), bool); mk[:, j - 1] = False; tok2 = rtok.clone(); tok2[:, j - 1] = torch.tensor(MC.DTOK[7], device=dev)
        fin, rd, _ = run(mk, rtok); (fin + rd).backward(); g1 = [p_.grad.detach().clone() for p_ in (m.down.weight, m.up.weight, m.g.w("h.0.attn.c_attn.weight"))]
        fin, rd, _ = run(mk, tok2); (fin + rd).backward(); g2 = [p_.grad.detach() for p_ in (m.down.weight, m.up.weight, m.g.w("h.0.attn.c_attn.weight"))]
        out[f"s142.removed{j}.zero_grad_exact"] = 1.0 if all(torch.equal(x_, y_) for x_, y_ in zip(g1, g2)) else 0.0
    _, rd, n = run(np.zeros((8, 3), bool), rtok); out["s142.all_removed_no_reader_term"] = 1.0 if (rd is None and n == 0) else 0.0
    del m
    F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT = saved


def check_s143(dev, out):
    """§143 FULL-GROW gates (dc546570): on the COPY+GROW-NOISE carrier (d0 8, switch 1 to 64 done, its fade-in complete), a forced switch 2
    (grow 704, new W_up columns 0, alpha 0) leaves every carried embedding and the answer decoder's logits bitwise identical (max |delta logits|
    = 0, eval mode, under CUDA bf16); at the first step after it (alpha = 1/500, training mode with dither) the step loss reaches the NEW W_up
    columns (nonzero grad); the new W_down rows get exactly 0 there (their outgoing weights are still 0), and the old 64 columns keep theirs."""
    import math
    import boxes.fsq120 as F120
    saved = (F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT); F120.D_Q, F120.LEV = 8, 8; F120.HALF = (F120.LEV - 1) / 2; F120.SHIFT = math.atanh(F120.OFF / F120.HALF)
    amp = torch.bfloat16 if str(dev).startswith("cuda") else None; fam = F120.Family("s5"); m = F120.FSQ120(s5.load_gpt2(dev), fam).to(dev)
    m.grow_noise = 0.25; m.grow(56); m.grow_alpha_0.fill_(1.0)
    with torch.no_grad():
        m.up_g[0].normal_(0, 0.02)                                          # switch 1's columns trained (non-zero)
    s0, x, lab = F120.make_batch(fam, np.random.default_rng(0), 8, 6); X = torch.tensor(x, device=dev)
    def fwd():
        m.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            _, e, _, embs = m.chain(m.start(8, dev, s0), X, record=True)
            xx = torch.cat([m.g.embed(torch.full((8, 1), F120.EOT, device=dev)), e[:, None].to(m.g.embed(torch.full((8, 1), F120.EOT, device=dev)).dtype)], 1)
            h, _ = m.g(xx, None); lg = m.g.logits(h).float()
        return torch.stack([v.float() for v in embs]), lg
    E0, L0 = fwd(); m.grow(704); E1, L1 = fwd()
    out["s143.preserve_width_768"] = 1.0 if m.width() == 768 else 0.0
    out["s143.preserve_embeddings_exact"] = 1.0 if torch.equal(E0, E1) else 0.0
    out["s143.preserve_logits_exact"] = 1.0 if float((L0 - L1).abs().max()) == 0.0 else 0.0
    m.grow_alpha_1.fill_(1 / 500); m.train(); m.p_hard = 1.0; m.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        loss, _, _, _ = m.chain(m.start(8, dev, s0), X, lab)
    loss.backward()
    out["s143.new_up_cols_grad"] = float(m.up_g[1].grad.float().norm()) if m.up_g[1].grad is not None else 0.0
    out["s143.old64_up_grad"] = float(m.up.weight.grad.float().norm()) + float(m.up_g[0].grad.float().norm())
    g2 = m.down_g[1].weight.grad; out["s143.new_down_rows_zero_at_first_step"] = 1.0 if (g2 is None or float(g2.abs().max()) == 0.0) else 0.0
    del m
    F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT = saved


def check_s145(dev, out):
    """§145 MULTI-SLOT (b1d7741c, note 145-1): K = 4 write positions, NOISE-64 (dither 0.25, training), the emission CE over a full 4x4 chain
    (16 pairs + flush) under CUDA bf16. SLOT4 (carry the first 4 = all): every slot's W_down_k / W_up_k / LN_k and every write embedding w_k get
    gradient. CARRY1 (--carry_from last: only w_4's slot carried and decoded): w_1..w_4 all get gradient (w_1..w_3 through attention into w_4),
    slot 4's maps get gradient, and the unused maps 1..3 get exactly 0 (flagged 1 when exactly 0)."""
    import math
    import boxes.fsq120 as F120
    import boxes.multcol as MC
    import boxes.multpair as MP
    saved = (F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT); F120.D_Q, F120.LEV = 64, 8; F120.HALF = (F120.LEV - 1) / 2; F120.SHIFT = math.atanh(F120.OFF / F120.HALF)
    amp = torch.bfloat16 if str(dev).startswith("cuda") else None; rng = np.random.default_rng(0); A, B = MC.gen_operands(rng, 8, 4), MC.gen_operands(rng, 8, 4)
    _, dig, _, pairs, col, emit = MP.accumulators(A, B); T = len(pairs)
    for arm, C, cf in (("slot4", 4, "first"), ("carry1", 1, "last")):
        m = MP.MultSlot(s5.load_gpt2(dev), MP.PairFam(), 4, C, cf).to(dev); m.no_round = True; m.noise = 0.25; m.train(); MP.set_ops(m, A, B, dev); m.zero_grad(set_to_none=True)
        X = torch.tensor(np.tile(np.arange(T), (8, 1)), device=dev); tm = torch.tensor(np.tile(emit, (8, 1)), device=dev); lab = np.where(dig >= 0, dig, 0)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            loss, _, _, _ = m.chain(m.start(8, dev), X, lab, tm)
        loss.backward()
        gn = lambda p_: float(p_.grad.float().norm()) if p_.grad is not None else 0.0  # noqa: E731
        for k in range(4):
            ln, dn, up = m.maps(k)
            out[f"s145.{arm}.w{k + 1}_grad"] = float(m.w.grad[k].float().norm()) if m.w.grad is not None else 0.0
            if k in m.cidx:
                out[f"s145.{arm}.down{k + 1}_grad"] = gn(dn.weight); out[f"s145.{arm}.up{k + 1}_grad"] = gn(up.weight); out[f"s145.{arm}.ln{k + 1}_grad"] = gn(ln.weight)
            else:
                out[f"s145.{arm}.unused_slot{k + 1}_maps_zero_grad"] = 1.0 if (gn(dn.weight) == 0.0 and gn(up.weight) == 0.0) else 0.0
        out[f"s145.{arm}.trunk_grad"] = gn(m.g.w("h.0.attn.c_attn.weight"))
        del m
    F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT = saved


def check_s144(dev, out):
    """§144 RUNNING-SUM (b1d7741c): fsq120 --family sum4 (random 4-digit start text, two-digit items, 4-token answer). The FINAL answer CE
    on an L 8 chain reaches W_down, W_up and the GPT-2 trunk under CUDA bf16 for both arms: F8 (FSQ-8, hard) and N64 (d 64, --no_round,
    dither 0.25)."""
    import math
    import boxes.fsq120 as F120
    saved = (F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT, F120.SUM_MOD); amp = torch.bfloat16 if str(dev).startswith("cuda") else None
    for arm, dq, nr in (("f8", 8, False), ("n64", 64, True)):
        F120.D_Q, F120.LEV = dq, 8; F120.HALF = (F120.LEV - 1) / 2; F120.SHIFT = math.atanh(F120.OFF / F120.HALF); F120.SUM_MOD = 10000
        fam = F120.Family("sum4"); m = F120.FSQ120(s5.load_gpt2(dev), fam).to(dev); m.no_round = nr; m.noise = 0.25 if nr else 0.0; m.p_hard = 1.0; m.train()
        s0, x, lab = F120.make_batch(fam, np.random.default_rng(0), 8, 8); m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            loss, _, _, _ = m.chain(m.start(8, dev, s0), torch.tensor(x, device=dev), lab)
        loss.backward()
        for nm_, w_ in (("down", m.down.weight), ("up", m.up.weight), ("trunk", m.g.w("h.0.attn.c_attn.weight"))):
            out[f"s144.{arm}.{nm_}_grad"] = float(w_.grad.float().norm()) if w_.grad is not None else 0.0
        del m
    F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT, F120.SUM_MOD = saved


def check_s147(dev, out):
    """§147 DELTA-STATE (b1d7741c / note 147-1): MultDelta on a full 4x4 chain (16 pairs + flush), emission CE under CUDA bf16.
    Step 0 (W_o = 0, fade 1/500): only W_o gets gradient (q / k / v / alpha / beta exactly 0: no path while W_o = 0, flagged). After one update
    (W_o small random, fade 1/500): q / k / v (qkv), W_o, the alpha / beta gate, LN_m and the GPT-2 trunk all get gradient; both head types
    carry it (the additive and delta halves of W_o's input columns)."""
    import boxes.multcol as MC
    import boxes.multpair as MP
    amp = torch.bfloat16 if str(dev).startswith("cuda") else None; rng = np.random.default_rng(0); A, B = MC.gen_operands(rng, 8, 4), MC.gen_operands(rng, 8, 4)
    _, dig, _, pairs, col, emit = MP.accumulators(A, B); T = len(pairs)
    m = MP.MultDelta(s5.load_gpt2(dev), MP.PairFam(), 6).to(dev); m.train(); MP.set_ops(m, A, B, dev)
    X = torch.tensor(np.tile(np.arange(T), (8, 1)), device=dev); tm = torch.tensor(np.tile(emit, (8, 1)), device=dev); lab = np.where(dig >= 0, dig, 0)
    gn = lambda p_: float(p_.grad.float().norm()) if p_.grad is not None else 0.0  # noqa: E731
    for phase in ("step0", "after1"):
        if phase == "after1":
            with torch.no_grad():
                m.mem.wo.weight.normal_(0, 0.02)
        m.mem.fade.fill_(1 / 500); m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            loss, _, _, _ = m.chain(m.start(8, dev), X, lab, tm)
        loss.backward(); w_ = m.mem.wo.weight.grad
        if phase == "step0":
            out["s147.step0.wo_grad"] = gn(m.mem.wo.weight)
            out["s147.step0.qkv_gate_zero_grad"] = 1.0 if (gn(m.mem.qkv.weight) == 0.0 and gn(m.mem.gate.weight) == 0.0) else 0.0
        else:
            for nm_, p_ in (("qkv", m.mem.qkv.weight), ("wo", m.mem.wo.weight), ("gate_alpha_beta", m.mem.gate.weight), ("ln_m", m.mem.ln.weight), ("trunk", m.g.w("h.0.attn.c_attn.weight"))):
                out[f"s147.after1.{nm_}_grad"] = gn(p_)
            out["s147.after1.wo_additive_cols_grad"] = float(w_[:, :384].float().norm()); out["s147.after1.wo_delta_cols_grad"] = float(w_[:, 384:].float().norm())
    del m


def check_s148(dev, out):
    """§148 LOOP-RETRO (b1d7741c): LoopGPT (prelude 0-3 / weight-tied loop 4-7 / coda 8-11, P zero-init, eta 0.25) under CUDA bf16; the deep-
    supervised loss (CE at r = 4 and at r = 2) reaches P, the prelude (h.0), the loop (h.4) and the coda (h.8)."""
    import boxes.loop148 as LP
    amp = torch.bfloat16 if str(dev).startswith("cuda") else None; rng = np.random.default_rng(0); A, B = rng.integers(0, 10, (8, 4)), rng.integers(0, 10, (8, 4))
    m = LP.LoopGPT(s5.load_gpt2(dev), 0.25); m.train(); P = LP.prompt_tokens(A, B); Y = np.array(LP.DTOK)[LP.answer_digits(A, B)]
    x = torch.tensor(np.c_[P, Y[:, :-1]], device=dev); y = torch.tensor(Y, device=dev); m.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        o, _, _ = m.run(x, 4, want={2, 4})
    loss = 0.5 * sum(torch.nn.functional.cross_entropy(o[r][:, P.shape[1] - 1:].float().reshape(-1, o[r].shape[-1]), y.reshape(-1)) for r in (2, 4)); loss.backward()
    gn = lambda p_: float(p_.grad.float().norm()) if p_.grad is not None else 0.0  # noqa: E731
    for nm_, p_ in (("P", m.P.weight), ("prelude_h0", m.g.w("h.0.attn.c_attn.weight")), ("loop_h4", m.g.w("h.4.attn.c_attn.weight")), ("coda_h8", m.g.w("h.8.attn.c_attn.weight"))):
        out[f"s148.{nm_}_grad"] = gn(p_)
    del m


def check_s141(dev, out):
    """§141 (9e45b92b / note 141-1): with R = 4 repeats per column (FSQ-8 and SOFT-16), the digit CE of column k (decoded after its last
    repeat) ALONE reaches every repeat's embedding of column k, the previous column's last-repeat embedding (the carried code) and
    W_down / W_up, under CUDA bf16, soft and hard."""
    import math
    import boxes.fsq120 as F120
    import boxes.multcol as MC
    saved = (F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT); amp = torch.bfloat16 if str(dev).startswith("cuda") else None; R = 4
    for cell, dq, sb in (("r4f8", 8, 0.0), ("r4s16", 16, 0.1)):
        F120.D_Q, F120.LEV = dq, 8; F120.HALF = (F120.LEV - 1) / 2; F120.SHIFT = math.atanh(F120.OFF / F120.HALF)
        fam = MC.MultFam(); m = MC.MultCol(s5.load_gpt2(dev), fam).to(dev); m.R = R; m.rep_index = True; m.soft_beta = sb; m.train()
        rng = np.random.default_rng(0); A, B = MC.gen_operands(rng, 8, 4), MC.gen_operands(rng, 8, 4); P, _, _ = MC.columns(A, B); m.ctx = torch.tensor(MC.ctx_tokens(A, B), device=dev)
        for ph in (0.0, 1.0):
            m.p_hard = ph
            for k in (2, 5):                                                # 1-based column
                m.zero_grad(set_to_none=True); prev = m.start(8, dev); keep = {}
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    for t in range(k * R):
                        h = m.segment(prev, torch.full((8,), t, device=dev)); qs, q, zb = m.quant(h); e = m.embed_code(qs)
                        if t >= (k - 1) * R - 1:                            # the previous column's last repeat + every repeat of column k
                            e.retain_grad(); keep[t] = e
                        prev = e[:, None]
                    loss = m.answer_ce(e, torch.tensor(fam.answer_tokens(P[:, k - 1]), device=dev), qs)
                loss.backward()
                for t, e_ in keep.items():
                    lab = "prev_col_last" if t == (k - 1) * R - 1 else "rep%d" % (t - (k - 1) * R + 1)
                    out[f"s141.{cell}.p{int(ph)}.col{k}.{lab}_grad"] = float(e_.grad.float().norm()) if t < k * R - 1 else 1.0   # the decoded repeat's own e: 1 (it is the CE input)
                out[f"s141.{cell}.p{int(ph)}.col{k}.down_grad"] = float(m.down.weight.grad.float().norm()); out[f"s141.{cell}.p{int(ph)}.col{k}.up_grad"] = float(m.up.weight.grad.float().norm())
        del m
    F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT = saved


def check_s140(dev, out):
    """§140 (ce1b3442 / note 140-1): the digit CE at a column's last pair ALONE reaches W_down / W_up and - through the carried code - the
    embedding carried out of earlier steps (the step before the column's first pair), under CUDA bf16, soft and hard."""
    import boxes.multcol as MC
    import boxes.multpair as MP
    amp = torch.bfloat16 if str(dev).startswith("cuda") else None; fam = MP.PairFam(); m = MP.MultPair(s5.load_gpt2(dev), fam).to(dev); m.train()
    rng = np.random.default_rng(0); A, B = MC.gen_operands(rng, 8, 4), MC.gen_operands(rng, 8, 4); _, dig, _, pairs, col, emit = MP.accumulators(A, B); MP.set_ops(m, A, B, dev)
    ends = [int(t) for t in np.where(emit)[0]]
    for ph in (0.0, 1.0):
        m.p_hard = ph
        for k in (2, 4, 7):                                                 # 1-based column whose digit CE is the only loss (its last pair)
            te = ends[k - 1]; t_carry = int(np.where(col == k - 1)[0][0]) - 1   # the hand-off before the column's first pair
            m.zero_grad(set_to_none=True); prev = m.start(8, dev); carried = None
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                for t in range(te + 1):
                    h = m.segment(prev, torch.full((8,), t, device=dev)); qs, q, zb = m.quant(h); e = m.embed_code(qs)
                    if t == t_carry:
                        e.retain_grad(); carried = e
                    prev = e[:, None]
                loss = m.answer_ce(e, torch.tensor(fam.answer_tokens(dig[:, te]), device=dev), qs)
            loss.backward()
            out[f"s140.p{int(ph)}.col{k}.carried_grad"] = float(carried.grad.float().norm())
            out[f"s140.p{int(ph)}.col{k}.down_grad"] = float(m.down.weight.grad.float().norm()); out[f"s140.p{int(ph)}.col{k}.up_grad"] = float(m.up.weight.grad.float().norm())
    del m


def check_s139(dev, out):
    """§139 (683318e2; notes 139-1 / 139-2, CE mode): the readout aux on e_j = W_up(code_j) ALONE (CE on the teacher's decoded S_j digits)
    reaches the readout, W_up, W_down and the GPT-2 trunk under CUDA bf16, soft and hard, and nothing reaches the frozen teacher. Schedule:
    arm b weight 1 at 3k, 0.5 at 3.5k, exactly 0 from 4k (aux + teacher skipped); arm c 1 throughout. Flags are 1 / 0."""
    import math
    import torch.nn as nn
    import boxes.fsq120 as F120
    import boxes.multcol as MC
    import boxes.multrow as R
    saved = (F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT); F120.D_Q, F120.LEV = 16, 8; F120.HALF = (F120.LEV - 1) / 2; F120.SHIFT = math.atanh(F120.OFF / F120.HALF)
    amp = torch.bfloat16 if str(dev).startswith("cuda") else None; fam = R.RowFam(); m = R.MultRow(s5.load_gpt2(dev), fam).to(dev); m.soft_beta = 0.1
    m.readout = nn.Linear(768, 80).to(dev); m.train(); T = R.Teacher(os.path.join(ROOT, "boxes", "runs_mult", "m98_tc_4x4_s1"), 8, None, dev)
    rng = np.random.default_rng(0); A, B = MC.gen_operands(rng, 8, 4), MC.gen_operands(rng, 8, 4)
    for ph in (0.0, 1.0):
        m.p_hard = ph; m.zero_grad(set_to_none=True); R.set_ops(m, A, B, dev); m.tsd_on = True; m._tsd_e = []
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            m.chain(m.start(8, dev), torch.tensor(np.tile(np.arange(4), (8, 1)), device=dev), R.sj_values(A, B))
        _, dec, ok = T.targets(A, B, 4, amp); E = torch.stack(m._tsd_e, 1).float(); lg = m.readout(E).view(8, 4, 8, 10).float()
        aux = (torch.nn.functional.cross_entropy(lg.reshape(-1, 10), dec.clamp_min(0).reshape(-1), reduction="none").view(8, 4, 8).mean(-1) * ok).sum() / ok.sum().clamp_min(1)
        aux.backward(); m.tsd_on = False; m._tsd_e = []
        out[f"s139.p{int(ph)}.readout_grad"] = float(m.readout.weight.grad.float().norm()); out[f"s139.p{int(ph)}.up_grad"] = float(m.up.weight.grad.float().norm())
        out[f"s139.p{int(ph)}.down_grad"] = float(m.down.weight.grad.float().norm()); out[f"s139.p{int(ph)}.trunk_grad"] = float(m.g.w("h.0.attn.c_attn.weight").grad.float().norm())
        out[f"s139.p{int(ph)}.teacher_no_grad"] = 1.0 if all(p_.grad is None for p_ in T.g.parameters()) else 0.0
        out[f"s139.p{int(ph)}.teacher_valid_rows"] = float(ok.float().mean())
    w = R.distill_weight
    out["s139.sched_b_1_at_3k"] = 1.0 if w(3000, True, 3000, 4000) == 1.0 else 0.0
    out["s139.sched_b_half_at_3500"] = 1.0 if abs(w(3500, True, 3000, 4000) - 0.5) < 1e-9 else 0.0
    out["s139.sched_b_zero_from_4k"] = 1.0 if all(w(s_, True, 3000, 4000) == 0.0 for s_ in (4000, 4001, 12000, 24000)) else 0.0
    out["s139.sched_c_one_throughout"] = 1.0 if all(w(s_, True, 3000, 0) == 1.0 for s_ in (1, 3000, 4000, 24000)) else 0.0
    out["s139.sched_a_zero"] = 1.0 if all(w(s_, False, 3000, 4000) == 0.0 for s_ in (1, 3000, 24000)) else 0.0
    del m, T
    F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT = saved


def check_s137(dev, out):
    """§137 (9c9b1899): the digit CE at a column k >= 2 ALONE reaches W_down / W_up and - through the carried code - the embedding
    carried out of column k-1 (dL_k / d e_{k-1} != 0), under CUDA bf16, soft (p_hard 0) and hard (p_hard 1)."""
    import boxes.multcol as M
    amp = torch.bfloat16 if str(dev).startswith("cuda") else None; fam = M.MultFam(); m = M.MultCol(s5.load_gpt2(dev), fam).to(dev); m.train()
    rng = np.random.default_rng(0); A, B = M.gen_operands(rng, 8, 4), M.gen_operands(rng, 8, 4); P, _, _ = M.columns(A, B); m.ctx = torch.tensor(M.ctx_tokens(A, B), device=dev)
    ans_all = torch.tensor(P, device=dev)
    for ph in (0.0, 1.0):
        m.p_hard = ph
        for k in (2, 4, 8):                                                 # 1-based column whose digit CE is the only loss
            m.zero_grad(set_to_none=True); torch.manual_seed(k)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                prev = m.start(8, dev); carried = None
                for t in range(k):
                    h = m.segment(prev, torch.full((8,), t + 1, device=dev)); qs, q, zb = m.quant(h); e = m.embed_code(qs)
                    if t == k - 2:
                        e.retain_grad(); carried = e
                    prev = e[:, None]
                ans = torch.tensor(fam.answer_tokens(P[:, k - 1]), device=dev); loss = m.answer_ce(e, ans, qs)
            loss.backward()
            out[f"s137.p{int(ph)}.col{k}.carried_grad"] = float(carried.grad.float().norm())
            out[f"s137.p{int(ph)}.col{k}.down_grad"] = float(m.down.weight.grad.float().norm()); out[f"s137.p{int(ph)}.col{k}.up_grad"] = float(m.up.weight.grad.float().norm())
    del ans_all, m


def check_s136(dev, out):
    """§136 GROW pre-launch asserts (Theory bccf643b) at a forced switch, every arm's widening op, CUDA bf16 autocast:
    (1) function preserved: max |delta logits| = 0 (decoder logits of the final carried code, eval / hard) across the switch;
    (2) not inert: >= 90 % of the new dims take >= 2 values on a fixed 64-chain batch (eval / hard);
    (3) gradflow: nonzero grad into the new W_up columns at the first step after the switch; the new W_down rows get none then
        (W_up columns exactly 0) and a nonzero grad at the next step (after one AdamW step with the warm-up LR);
    NESTED: chains of prefix width 4 send no grad into dims >= 4; full width does. Pass flags are 1 / 0 (all must be > 0);
    the raw numbers go to theory/results/s136_switch_asserts_<dev>.json."""
    import math
    import boxes.fsq120 as F120
    saved = (F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT); amp = torch.bfloat16 if str(dev).startswith("cuda") else None; raw = {}
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    def build(d0, base=None):
        """A forced switch happens on a trained trunk (loss <= 2.5 / a plateau), so the grow arms start from a trained S5 d-8 model.pt
        (plain s128_s5_s1 or COPY s134_copy_s5_s1); d0 4 (GROW-4 proxy): that trunk with the first 4 W_down rows / W_up columns."""
        F120.D_Q = d0; F120.LEV = 8; F120.HALF = (F120.LEV - 1) / 2; F120.SHIFT = math.atanh(F120.OFF / F120.HALF)
        torch.manual_seed(0); fam = F120.Family("s5"); m_ = F120.FSQ120(s5.load_gpt2(dev), fam).to(dev)
        if base:
            sd = torch.load(os.path.join(ROOT, "boxes", "runs_fsq", base, "model.pt"), map_location=dev)
            if d0 < 8:
                sd["down.weight"], sd["down.bias"], sd["up.weight"] = sd["down.weight"][:d0], sd["down.bias"][:d0], sd["up.weight"][:, :d0]
            m_.load_state_dict(sd); raw["base." + str(d0)] = base
        return fam, m_
    def logits(m, s0, X, lab):
        m.eval()
        with torch.no_grad(), ac():
            _, e, _, _ = m.chain(m.start(len(s0), dev, s0), X, lab); ans = torch.tensor(m.fam.answer_tokens(lab[:, -1]), device=dev); g = m.g
            eot = g.embed(torch.full((len(s0), 1), F120.EOT, device=dev)); h, _ = g(torch.cat([eot, e[:, None].to(eot.dtype), g.embed(ans[:, :-1])], 1), None)
            return g.logits(h[:, 1:]).float()
    def chain_loss(m, s0, X, lab):
        with ac():
            return m.chain(m.start(len(s0), dev, s0), X, lab)[0]
    arms = {"grow": [(8, 56)], "cgrow": [(8, 56)], "staged": [(8, 8), (16, 16)], "noise": [(8, 56)], "grow4": [(4, 60)]}
    bases = {"grow": "s128_s5_s1", "grow4": "s128_s5_s1", "cgrow": "s134_copy_s5_s1", "staged": "s134_copy_s5_s1", "noise": "s134_copy_s5_s1"}
    for name, ops in arms.items():
        fam, m = build(ops[0][0], bases[name]); m.grow_noise = 0.25 if name == "noise" else 0.0; rng = np.random.default_rng(0)
        s0, x, lab = F120.make_batch(fam, rng, 64, 4); X = torch.tensor(x, device=dev); opt = F120.make_opt(m, 1e-4); t_s = 2000
        for j, (d0, k) in enumerate(ops):
            before = logits(m, s0, X, lab); newp = m.grow(k); after = logits(m, s0, X, lab); md = float((after - before).abs().max())
            raw[f"{name}.{j}.max_abs_dlogits"] = md; out[f"s136.{name}.{j}.fn_preserved"] = 1.0 if md == 0.0 else 0.0
            m.eval()
            with torch.no_grad(), ac():
                _, _, codes, _ = m.chain(m.start(len(s0), dev, s0), X, lab, record=True)
            Q = torch.cat([c_.float() for c_ in codes])[:, d0:d0 + k]; nlev = [int(torch.unique(Q[:, i]).numel()) for i in range(k)]
            share = float(np.mean([n_ >= 2 for n_ in nlev])); raw[f"{name}.{j}.new_dims_ge2_levels_share"] = share; raw[f"{name}.{j}.levels_min"] = min(nlev)
            out[f"s136.{name}.{j}.not_inert"] = 1.0 if share >= 0.90 else 0.0
            gr = {"params": newp, "mult": 1.0, "warm_from": t_s}; opt.add_param_group(gr); dn, upw = m.down_g[-1], m.up_g[-1]; ia = m.n_grow - 1
            m.train(); m.p_hard = 1.0; s1, x1, l1 = F120.make_batch(fam, rng, 8, 4); X1 = torch.tensor(x1, device=dev)
            for st in (1, 2):                                               # the first two steps after the switch
                getattr(m, "grow_alpha_%d" % ia).fill_(st / 500.0)
                for g_ in opt.param_groups:
                    g_["lr"] = F120.lr_at(1e-4, g_, t_s + st, 24000)
                opt.zero_grad(set_to_none=True); torch.manual_seed(st); chain_loss(m, s1, X1, l1).backward()
                gu = float(upw.grad.float().norm()) if upw.grad is not None else 0.0; gd = float(dn.weight.grad.float().norm()) if dn.weight.grad is not None else 0.0
                raw[f"{name}.{j}.step{st}.new_up_grad"] = gu; raw[f"{name}.{j}.step{st}.new_down_grad"] = gd
                if st == 1:
                    out[f"s136.{name}.{j}.up_grad_step1"] = gu; out[f"s136.{name}.{j}.down_zero_step1"] = 1.0 if gd == 0.0 else 0.0
                else:
                    out[f"s136.{name}.{j}.down_grad_step2"] = gd
                opt.step()
            t_s += 1000
        if name == "noise":                                                 # grown dims continuous + dithered in training only
            m.train(); m.p_hard = 1.0
            with torch.no_grad(), ac():
                h = m.segment(m.start(8, dev, s0[:8]), X[:8, 0]); torch.manual_seed(1); qa = m.quant(h)[0]; torch.manual_seed(2); qb = m.quant(h)[0]
            out["s136.noise.dither_active"] = float((qa[:, 8:] - qb[:, 8:]).abs().max()); raw["noise.first8_identical"] = float((qa[:, :8] - qb[:, :8]).abs().max())
            out["s136.noise.first8_hard"] = 1.0 if raw["noise.first8_identical"] == 0.0 else 0.0
        del m, opt
    fam, m = build(64); m.train(); m.p_hard = 1.0; s0, x, lab = F120.make_batch(fam, np.random.default_rng(0), 8, 4); X = torch.tensor(x, device=dev)
    for w_ in (4, 64):                                                      # NESTED: the mask blocks / passes gradient beyond the prefix
        m.nest_mask = torch.tensor((np.arange(64) < w_).astype(np.float32), device=dev)[None].expand(8, 64); m.zero_grad(set_to_none=True)
        chain_loss(m, s0, X, lab).backward(); gb = float(m.down.weight.grad[4:].float().norm()); raw[f"nested.w{w_}.down_grad_dims_ge4"] = gb
        out[f"s136.nested.w{w_}." + ("blocked" if w_ == 4 else "passes")] = (1.0 if gb == 0.0 else 0.0) if w_ == 4 else gb
    m.nest_mask = None; del m
    F120.D_Q, F120.LEV, F120.HALF, F120.SHIFT = saved
    json.dump(raw, open(os.path.join(ROOT, "theory", "results", "s136_switch_asserts_%s.json" % str(dev).replace(":", "")), "w", newline="\n"), indent=1)
    print("s136 raw", raw, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--paths", nargs="*", default=["accel99", "mk108", "fsq120", "fsq120_std", "fsq120_rawh", "fsq120_anneal", "self110_copy", "self110_resume"])
    ap.add_argument("--device", default="cpu"); ap.add_argument("--out", default=os.path.join(ROOT, "theory", "results", "gradflow_check.json"))
    a = ap.parse_args(); out = {}
    for p in a.paths:
        globals()["check_" + p](a.device, out); print({k: v for k, v in out.items() if k.startswith(p.split("_")[0])}, flush=True)
    bad = {k: v for k, v in out.items() if not (v > 0)}
    out["_status"] = "PASS" if not bad else "FAIL"; out["_zero_terms"] = list(bad)
    json.dump(out, open(a.out, "w", newline="\n"), indent=1); print(out["_status"], bad, flush=True)
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
