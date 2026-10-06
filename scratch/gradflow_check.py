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
            return F120.tsd_loss135(m, E, T, 0.1)[0]
        out[f"s135.aux_p{int(ph)}.main"] = term(m, A)
        for nm_, w_ in (("down", m.down.weight), ("trunk_h3", trunk), ("up", m.up.weight), ("P_s", m.tsd_ps.weight), ("P_t", m.tsd_pt.weight)):
            out[f"s135.aux_p{int(ph)}.{nm_}_grad"] = float(w_.grad.norm()) if w_.grad is not None else 0.0
        out[f"s135.aux_p{int(ph)}.target_no_grad"] = 1.0 if all(p_.grad is None for p_ in tg.parameters()) else 0.0


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
