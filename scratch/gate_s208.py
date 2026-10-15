"""§208 gates (QONCE draft §12.4 / §16.4, AB draft §11.4) on A's final weights (gsm_lat_aug_s1), CPU fp32, FAST_ADD + CHUNK_DELTA 64 as the run, 8 Aug-train items
(rng 20801), K = 10. The model under test: ProsMem / FactorMem(nopass) with set_ab_qo(True, True, [EOT] + filler k16).
  g2a  every pass >= 2 write input row == x_f (mask all true), pass 1 == the question; the AB dec_loss rows assert ([EOT, ANS, a1.., EOS]).
  g2b  QO invariance: M_1 fixed, passes 2..K fed the pass_inputs of question-scrambled item records -> M_K bitwise equal; dec_loss on scrambled records equal.
  g2c  positive control: the same on the plain (no QO) path must differ.
  g3   gradflow (fade 1, gamma 1, W_o ~ N(0, 1e-3)): L_ans (lambda 0) -> nonzero grad on mem.qkv k/v rows, qkv q rows, gate, ln, ln_w, wo, s_add, s_del,
       trunk blocks 0 / 11; the pass-1 question embeddings get grad; with M_1.detach() they get none and the filler passes still give the write params grad;
       with S_K.detach(): zero grad on qkv k/v rows, gate, ln_w; at gamma 0: zero grad on ln_w; QO-2 at gamma 0: perturbing blocks 6-11 inside the filler
       passes leaves M_K bitwise unchanged (and changes it at gamma 1, the control).
  g5   K'0 constant: zero memory -> one distinct greedy answer over the items; the filler lower cache == a fresh gpt_lower bitwise.
  route  AB-only think == plain think bitwise (pass_inputs is the identity); FactorMem(nopass) + QO think == ProsMem + QO think bitwise; aux prompt NOPASS.
  ast  no decorator added on the §208 methods.
-> scratch/gate_s208/gate.json        python scratch/gate_s208.py"""
import ast
import json
import os
import sys

os.environ.setdefault("BOXES_NB", "5")
ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)
import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
import boxes.gsm8k_factor as GF  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.gsm8k_stepaux import Fwd, greedy_after  # noqa: E402
from boxes.prosqa168s2 import ANS, ENC, EOS, EOT, STEP_ID, COLON_ID, ProsMem  # noqa: E402

DEV = "cpu"; K = 10; A_PT = "boxes/runs_fsq/gsm_lat_aug_s1/model.pt"; DM.FAST_ADD = True; DM.CHUNK_DELTA = 64
META = json.load(open(os.path.join(G.AUG, "s200_test_meta.json"), encoding="utf-8")); XF = [EOT] + ENC.encode(META["filler"]["k16"])
ac = lambda: torch.autocast("cpu", enabled=False)  # noqa: E731


def build(kind="pros", ab=True, qo=True):
    g = s5.load_gpt2(DEV); m = (ProsMem(g) if kind == "pros" else GF.FactorMem(g, None, True, False, 1)).to(DEV)
    m.load_state_dict(torch.load(A_PT, map_location=DEV, weights_only=False)); m.eval()
    if ab:
        m.set_ab_qo(True, qo, XF)
    return g, m


def think(m, its, Kp=K, cache=True):
    x, msk = m.think_inputs(its, DEV); return m.think(m.mem.zero_state(len(its), DEV), x, msk, Kp, cache=cache)


def main():
    OUT = {}; R, _ = G.load_aug("train"); idx = np.random.default_rng(20801).permutation(len(R))
    its = [G.encode(R[i], ENC, EOS) for i in idx[:40] if len(R[i]["eqs"]) >= 2][:8]; scr = its[1:] + its[:1]
    scr_rec = [{**b, "a1": a["a1"], "ans": a["ans"]} for a, b in zip(its, scr)]          # question from another item, answer kept
    g, m = build()
    # g2a
    seen = []; w0 = m.write
    m.write = lambda S, x, mask, h_low=None: (seen.append((x, mask)), w0(S, x, mask, h_low))[1]
    with torch.no_grad():
        SK = think(m, its)
    del m.write
    xq, _ = m.think_inputs(its, DEV); xf = torch.tensor([XF] * len(its))
    ok_a = len(seen) == K and torch.equal(seen[0][0], xq) and all(torch.equal(x_.cpu(), xf) and bool(mk.all()) for x_, mk in seen[1:])
    try:
        with torch.no_grad():
            m.dec_loss(SK, its, DEV); dec_ok = True
    except AssertionError:
        dec_ok = False
    OUT["g2a"] = {"passes_seen": len(seen), "pass1_is_question": bool(torch.equal(seen[0][0], xq)), "passes_ge2_are_filler": ok_a, "dec_rows_assert_ok": dec_ok,
                  "filler_len": len(XF), "pass": ok_a and dec_ok}
    # g2b / g2c

    def cont(mod, base, recs):
        with torch.no_grad():
            S = think(mod, base, 1); x, msk = mod.think_inputs(recs, DEV); hl = mod.lower(x); pin = mod.pass_inputs(x, msk, hl)
            for t in range(2, K + 1):
                S = mod.write(S, *pin(t))
        return S
    with torch.no_grad():
        Sb = cont(m, its, scr); eq_b = bool(torch.equal(Sb, SK)); dl = (m.dec_loss(SK, its, DEV), m.dec_loss(Sb, scr_rec, DEV))
        Pa = greedy_after(Fwd(g, m.mem, SK), [[EOT, ANS]] * len(its), DEV, ac, 8); Pb = greedy_after(Fwd(g, m.mem, Sb), [[EOT, ANS]] * len(its), DEV, ac, 8)
    gp, mp = build(ab=False)
    with torch.no_grad():
        Sp = think(mp, its); Spc = cont(mp, its, scr)
    OUT["g2b"] = {"MK_bitwise": eq_b, "dec_loss_equal": bool(torch.equal(*dl)), "answers_equal": Pa == Pb, "pass": eq_b and bool(torch.equal(*dl)) and Pa == Pb}
    OUT["g2c"] = {"plain_MK_differs": not torch.equal(Sp, Spc), "max_abs": float((Sp - Spc).abs().max()), "pass": not torch.equal(Sp, Spc)}
    # route
    ga, ma = build(ab=True, qo=False)
    with torch.no_grad():
        Sab = think(ma, its)
    gf, mf = build("factor")
    with torch.no_grad():
        Sf = think(mf, its)
    OUT["route"] = {"ab_only_think_eq_plain": bool(torch.equal(Sab, Sp)), "factor_nopass_qo_eq_pros_qo": bool(torch.equal(Sf, SK)),
                    "factor_aux_prompt": mf.aux_prompt(3), "aux_prompt_nopass": mf.aux_prompt(3) == [EOT, STEP_ID, COLON_ID]}
    OUT["route"]["pass"] = OUT["route"]["ab_only_think_eq_plain"] and OUT["route"]["factor_nopass_qo_eq_pros_qo"] and OUT["route"]["aux_prompt_nopass"]
    # g5
    with torch.no_grad():
        S0 = m.mem.zero_state(len(its), DEV); P0 = greedy_after(Fwd(g, m.mem, S0), [[EOT, ANS]] * len(its), DEV, ac, 8)
        x, msk = m.think_inputs(its, DEV); hl = m.lower(x); hf = m.pass_inputs(x, msk, hl)(2)[2]; fresh = DM.gpt_lower(g, g.embed(xf), m.mem.layer)
    OUT["g5"] = {"k0_distinct": len(set(P0)), "k0_answer": P0[0], "filler_cache_bitwise": bool(torch.equal(hf, fresh))}
    OUT["g5"]["pass"] = OUT["g5"]["k0_distinct"] == 1 and OUT["g5"]["filler_cache_bitwise"]
    # g3
    g3 = {}

    def setup(mod, gam):
        with torch.no_grad():
            mod.mem.fade.fill_(1.0); mod.mem.gamma.fill_(gam); mod.mem.wo.weight.copy_(torch.randn(768, 768, generator=torch.Generator().manual_seed(5)) * 1e-3)
        mod.zero_grad(set_to_none=True)

    def gn(mod, name, rows=None):
        p_ = dict(mod.named_parameters())[name]; gr = p_.grad
        if gr is None:
            return 0.0
        return float((gr[rows] if rows is not None else gr).double().norm())
    D = 768; kv = slice(D, 3 * D); q_ = slice(0, D)
    names = {"qkv_kv": ("mem.qkv.weight", kv), "qkv_q": ("mem.qkv.weight", q_), "gate": ("mem.gate.weight", None), "ln": ("mem.ln.weight", None),
             "ln_w": ("mem.ln_w.weight", None), "wo": ("mem.wo.weight", None), "s_add": ("mem.s_add", None), "s_del": ("mem.s_del", None),
             "block0": ("g.p.h_0_attn_c_attn_weight", None), "block11": ("g.p.h_11_attn_c_attn_weight", None)}
    pn = {n for n, _ in m.named_parameters()}; names = {k_: v_ for k_, v_ in names.items() if v_[0] in pn}; g3["param_names_found"] = sorted(names)
    emb_caps = []; e0 = g.embed

    def emb(x):
        y = e0(x)
        if x.shape == xq.shape and torch.equal(x, xq):
            y.retain_grad(); emb_caps.append(y)
        return y
    g.embed = emb
    # full: L_ans alone
    setup(m, 1.0); S = think(m, its); L = m.dec_loss(S, its, DEV); L.backward()
    g3["full_gamma1"] = {k_: gn(m, *v_) for k_, v_ in names.items()}; g3["q_embed_grad"] = float(sum(float(e_.grad.norm()) for e_ in emb_caps if e_.grad is not None))
    # M_1 detached: no question grad; filler passes still train the write params
    setup(m, 1.0); emb_caps.clear()
    S1 = think(m, its, 1).detach(); x, msk = m.think_inputs(its, DEV); pin = m.pass_inputs(x, msk, None); S = S1
    for t in range(2, K + 1):
        S = m.write(S, *pin(t))
    m.dec_loss(S, its, DEV).backward()
    g3["m1_detached"] = {"q_embed_grad": float(sum(float(e_.grad.norm()) for e_ in emb_caps if e_.grad is not None)), "q_embed_grad_none": all(e_.grad is None for e_ in emb_caps),
                         **{k_: gn(m, *v_) for k_, v_ in names.items() if k_ in ("qkv_kv", "gate", "ln_w")}}
    # S_K detached
    setup(m, 1.0); S = think(m, its).detach(); m.dec_loss(S, its, DEV).backward(); g3["sk_detached"] = {k_: gn(m, *names[k_]) for k_ in ("qkv_kv", "gate", "ln_w") if k_ in names}
    # gamma 0: ln_w zero
    setup(m, 0.0); S = think(m, its); m.dec_loss(S, its, DEV).backward(); g3["gamma0"] = {k_: gn(m, *names[k_]) for k_ in ("ln_w", "qkv_kv") if k_ in names}
    g.embed = e0
    # QO-2 perturbation: blocks 6-11 perturbed inside the filler passes only
    w_orig = g.w

    def perturbed_think(gam):
        setup(m, gam); flag = {"on": False}; noise = {}

        def w(name):
            v_ = w_orig(name)
            if flag["on"] and name.startswith("h.") and int(name.split(".")[1]) >= 6:
                if name not in noise:
                    noise[name] = torch.randn(v_.shape, generator=torch.Generator().manual_seed(len(noise) + 7)) * 0.05
                return v_ + noise[name]
            return v_
        with torch.no_grad():
            Sa = think(m, its)
            g.w = w; x, msk = m.think_inputs(its, DEV); hl = m.lower(x); pin = m.pass_inputs(x, msk, hl); S = m.mem.zero_state(len(its), DEV)
            for t in range(1, K + 1):
                flag["on"] = t >= 2; S = m.write(S, *pin(t))
            g.w = w_orig
        return bool(torch.equal(Sa, S)), float((Sa - S).abs().max())
    e0g, d0g = perturbed_think(0.0); e1g, d1g = perturbed_think(1.0)
    g3["qo2_gamma0_unchanged"] = e0g; g3["qo2_gamma1_changes"] = not e1g; g3["qo2_max_abs"] = {"gamma0": d0g, "gamma1": d1g}
    full = g3["full_gamma1"]; must = [k_ for k_ in ("qkv_kv", "qkv_q", "gate", "ln", "ln_w", "wo", "s_add", "s_del", "block0", "block11") if k_ in full]
    g3["pass"] = (all(full[k_] > 0 for k_ in must) and g3["q_embed_grad"] > 0 and g3["m1_detached"]["q_embed_grad"] == 0
                  and g3["m1_detached"].get("qkv_kv", 1) > 0 and g3["m1_detached"].get("gate", 1) > 0
                  and all(v_ == 0 for v_ in g3["sk_detached"].values()) and g3["gamma0"].get("ln_w", 0) == 0 and e0g and not e1g)
    OUT["g3"] = g3
    # ast
    src = open("boxes/prosqa168s2.py", encoding="utf-8").read(); tree = ast.parse(src)
    decs = {f.name: [ast.unparse(d) for d in f.decorator_list] for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "ProsMem" for f in c.body
            if isinstance(f, ast.FunctionDef) and f.name in ("set_ab_qo", "pass_inputs", "think", "write", "dec_loss", "lower")}
    OUT["ast"] = {"decorators": decs, "pass": not any(decs.values())}
    OUT["pass_all"] = all(OUT[k_]["pass"] for k_ in ("g2a", "g2b", "g2c", "route", "g5", "g3", "ast"))
    os.makedirs("scratch/gate_s208", exist_ok=True); json.dump(OUT, open("scratch/gate_s208/gate.json", "w", newline="\n"), indent=1, default=str)
    print(json.dumps(OUT, indent=1, default=str)[:4000]); print("GATE_S208", OUT["pass_all"], flush=True)


if __name__ == "__main__":
    main()
