"""§223 P2 build gates (Note 223-8 R3 + Audit amendment 11a4b858; Theory 10-22 conditions a-d). Code under test: TREE/boxes/codi_ps.py
(scratch/_tmp/p2_tree, --m2_stage via soft_decoder) until committed (then --tree D:/Latent). Each part -> scratch/gate_s223p2/<part>.json ("pass").
  i    flags off == 45baec0d (HEAD) bitwise: DIRECT / M1-only / M2-only / PS (no --m2_stage), eff_bs 8, 12 steps, d_stages 1,2,3 (cg_bin 3, cg_cap 4:
       stage advances inside), log every step, on --device under scratch/s222_det_wrap.py (math SDP + deterministic): every log record but sec equal.
  ii   soft_decoder == the HF decoder call, fp64 CPU eval (dropout off), PS model (CP.build seed 1, LoRA B perturbed so LoRA is active), one d 3 batch:
       m 0 vs the HF call with no M2 mask, m 1 vs the registered M2 key mask -> max |logit diff| <= 1e-9 on real decoder positions.
  iii  question-content mass share (sum of attention on question + <bot> keys; mean over real decoder queries, heads, layers; per layer too) on the
       fixed batch, fp64 eval, at m 0 / .25 / .5 / .75 / 1: strictly decreasing, share(.5) >= .25 share(0), share(1) == 0.
  iv   gradflow (train mode off for determinism, fp64): the answer CE through soft_decoder at m .5 gives nonzero finite grads on every layer's LoRA
       (c_attn / c_proj / c_fc lora_B: lora_A is 0 at init because B starts at 0 - here B is perturbed, so both are checked) and on prj.
  v    layer 0 at m .5 vs m 0 (identical layer inputs): per decoder query and head, share(.5) == (1 - m) S0 / ((1 - m) S0 + 1 - S0) with S0 = the
       m 0 share, max |diff| <= 1e-10.
  vi   resume with --m2_stage (det wrapper): REF 40 steps vs KILL (exit 9 after 25, ckpt every 10 -> resume from 20) + RESUME; d_stages 1,2,3
       cg_bin 3 cg_cap 4 (stage 2 by ~step 8), --m2_ramp 10: records 21..40 equal (all keys but sec), m2_start equal, m2_m reaches 1.
    python scratch/gate_s223p2.py ii iii iv v [i vi --device cuda:2]"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys

TREE = "D:/Latent/scratch/_tmp/p2_tree"; HEAD = "D:/Latent"; OUT = "D:/Latent/scratch/gate_s223p2"; TMP = "D:/Latent/scratch/_tmp/s223p2_gate"
ap = argparse.ArgumentParser(); ap.add_argument("parts", nargs="+"); ap.add_argument("--tree", default=TREE); ap.add_argument("--device", default="cuda:2")
A = ap.parse_args(); TREE = A.tree
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "D:/Latent")
spec = importlib.util.spec_from_file_location("codi_ps_tree", os.path.join(TREE, "boxes", "codi_ps.py")); PS = importlib.util.module_from_spec(spec); spec.loader.exec_module(PS)
import boxes.codi_port as CP  # noqa: E402
import boxes.nested_words as NW  # noqa: E402

ENV = {**os.environ, "OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3"}; torch.set_num_threads(3)


def ns(**kw):
    d = dict(direct=False, recode=True, lam=1.0, bottleneck=True, codi_compat=False, m2_stage=True, k_cap=10)
    d.update(kw); return argparse.Namespace(**d)


def fixture():
    """the PS model in fp64 eval mode with LoRA B perturbed (seed 7) so the adapters are active, + one d 3 batch of 4 items (K 4)."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2"); PS.VALUE_TOKS.update(PS.value_tokens(enc))
    m, prj = CP.build("cpu", 1); D = PS.DHead(m)
    g = torch.Generator().manual_seed(7)
    for n_, p_ in m.named_parameters():
        if "lora_B" in n_:
            p_.data.copy_(torch.randn(p_.shape, generator=g) * 1e-2)
    m, prj, D = m.double().eval(), prj.double().eval(), D.double().eval()
    st = NW.Stream(22399, set()); its = [PS.tok_item(st.draw(3), enc) for _ in range(4)]; b = PS.collate(its, 4, "cpu")
    return m, prj, D, b


def real_dec(b):
    return (b["dl"] != -100)                                                # (B, Ld) real decoder tokens


def p_ii():
    m, prj, D, b = fixture(); o = {}
    with torch.no_grad():
        hf0 = PS.forward(m, prj, D, b, ns(bottleneck=False, m2_stage=False))[2]["logits"]; sd0 = PS.forward(m, prj, D, b, ns(), m2=0.0)[2]["logits"]
        hf1 = PS.forward(m, prj, D, b, ns(m2_stage=False))[2]["logits"]; sd1 = PS.forward(m, prj, D, b, ns(), m2=1.0)[2]["logits"]
    rd = real_dec(b)[..., None]
    o["max_abs_m0"] = float(((hf0 - sd0).abs() * rd).max()); o["max_abs_m1"] = float(((hf1 - sd1).abs() * rd).max()); o["dtype"] = str(hf0.dtype)
    o["m0_vs_m1_differ"] = float(((hf0 - hf1).abs() * rd).max())          # the mask matters (the comparison is informative)
    o["pass"] = o["max_abs_m0"] <= 1e-9 and o["max_abs_m1"] <= 1e-9 and o["m0_vs_m1_differ"] > 1e-3 and o["dtype"] == "torch.float64"; return o


def shares(m, prj, D, b, mv):
    cap = []
    with torch.no_grad():
        PS.forward(m, prj, D, b, ns(), m2=mv, capture=cap)
    Lq = b["em"].shape[1]; qf = b["qm"].to(torch.float64)                 # question + <bot> keys in the cached encoder part
    rd = real_dec(b).to(torch.float64)
    per = []
    for P in cap:                                                         # (B, H, Ld, Lp + Ld)
        s = (P[..., :Lq] * qf[:, None, None, :]).sum(-1)                   # (B, H, Ld)
        per.append(float((s.mean(1) * rd).sum() / rd.sum()))
    return per, cap


def p_iii():
    m, prj, D, b = fixture(); o = {"m": [0.0, 0.25, 0.5, 0.75, 1.0]}; o["per_layer"] = {}; o["mean"] = []
    for mv in o["m"]:
        per, _ = shares(m, prj, D, b, mv); o["per_layer"][str(mv)] = per; o["mean"].append(float(np.mean(per)))
    s = o["mean"]; o["strictly_decreasing"] = all(s[i] > s[i + 1] for i in range(4)); o["half_ratio"] = s[2] / s[0] if s[0] else None
    o["pass"] = o["strictly_decreasing"] and o["half_ratio"] is not None and o["half_ratio"] >= 0.25 and s[4] == 0.0 and s[0] > 0.0; return o


def p_iv():
    m, prj, D, b = fixture(); o = {}; m.zero_grad(set_to_none=True); prj.zero_grad(set_to_none=True)
    loss, parts, ex = PS.forward(m, prj, D, b, ns(recode=False, lam=0.0), m2=0.5); ex["ce"].backward()
    g = {n_: (float(p_.grad.norm()) if p_.grad is not None else None) for n_, p_ in m.named_parameters() if "lora_" in n_}
    gp = {n_: (float(p_.grad.norm()) if p_.grad is not None else None) for n_, p_ in prj.named_parameters()}
    ok = lambda v: v is not None and v > 0 and math.isfinite(v)  # noqa: E731
    o["lora_nonzero"] = sum(ok(v) for v in g.values()); o["lora_total"] = len(g); o["lora_zero_or_none"] = [k for k, v in g.items() if not ok(v)][:10]
    o["prj"] = gp; o["pass"] = o["lora_nonzero"] == o["lora_total"] and all(ok(v) for v in gp.values()) and o["lora_total"] > 0; return o


def p_v():
    m, prj, D, b = fixture(); mv = 0.5; Lq = b["em"].shape[1]; qf = b["qm"].to(torch.float64)
    _, c0 = shares(m, prj, D, b, 0.0); _, c5 = shares(m, prj, D, b, mv)
    S0 = (c0[0][..., :Lq] * qf[:, None, None, :]).sum(-1); S5 = (c5[0][..., :Lq] * qf[:, None, None, :]).sum(-1)
    pred = (1 - mv) * S0 / ((1 - mv) * S0 + 1 - S0); rd = real_dec(b)[:, None, :].expand_as(S0)
    o = {"max_abs": float((S5 - pred)[rd].abs().max()), "n": int(rd.sum()), "mean_S0": float(S0[rd].mean()), "mean_S5": float(S5[rd].mean())}
    o["pass"] = o["max_abs"] <= 1e-10 and o["mean_S0"] > 0.0; return o


def train(tree, out, extra):
    os.makedirs(out, exist_ok=True)
    for f_ in os.listdir(out):
        os.remove(os.path.join(out, f_))
    cmd = [sys.executable, "-B", "D:/Latent/scratch/s222_det_wrap.py", os.path.join(tree, "boxes", "codi_ps.py"), "--device", A.device, "--eff_bs", "8", "--micro", "8",
           "--d_stages", "1,2,3", "--cg_bin", "3", "--cg_cap", "4", "--log_every", "1", "--out", out] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, cwd="D:/Latent")
    return r.returncode, [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))] if os.path.exists(os.path.join(out, "log.jsonl")) else [], r.stderr[-1500:]


def p_i():
    o = {}; cells = {"DIRECT": ["--direct"], "M1": ["--recode", "--lam", "1"], "M2": ["--bottleneck"], "PS": ["--recode", "--lam", "1", "--bottleneck"]}
    for c, fl in cells.items():
        r = {}
        for tag, tree in (("head", HEAD), ("tree", TREE)):
            rc, lg, err = train(tree, "%s/i_%s_%s" % (TMP, c, tag), fl + ["--steps", "24000", "--stop_at", "12"]); r[tag] = (rc, lg, err)
        strip = lambda lg: [{k: v for k, v in x.items() if k != "sec"} for x in lg]  # noqa: E731
        o[c] = {"rc": [r["head"][0], r["tree"][0]], "n": [len(r["head"][1]), len(r["tree"][1])], "bitwise": strip(r["head"][1]) == strip(r["tree"][1]),
                "stage_max": max((x.get("stage", 0) for x in r["tree"][1]), default=None), "err": [r["head"][2][-300:] if r["head"][0] not in (0, 9) else "", r["tree"][2][-300:] if r["tree"][0] not in (0, 9) else ""]}
    o["pass"] = all(v["bitwise"] and v["n"][0] == 12 and v["rc"] == [9, 9] and v["stage_max"] >= 2 for k, v in o.items() if isinstance(v, dict)); return o


def p_vi():
    o = {}; base = ["--recode", "--lam", "1", "--bottleneck", "--m2_stage", "--m2_ramp", "10", "--steps", "24000", "--ckpt_every", "10"]
    rc0, ref, e0 = train(TREE, TMP + "/vi_ref", base + ["--stop_at", "40"])
    rc1, kil, e1 = train(TREE, TMP + "/vi_res", base + ["--stop_at", "25"])
    cmd_ok = os.path.exists(TMP + "/vi_res/ckpt_last.pt")
    r = subprocess.run([sys.executable, "-B", "D:/Latent/scratch/s222_det_wrap.py", os.path.join(TREE, "boxes", "codi_ps.py"), "--device", A.device, "--eff_bs", "8", "--micro", "8",
                        "--d_stages", "1,2,3", "--cg_bin", "3", "--cg_cap", "4", "--log_every", "1", "--out", TMP + "/vi_res", "--resume", "--stop_at", "40"] + base,
                       capture_output=True, text=True, env=ENV, cwd="D:/Latent")
    lg = [json.loads(l) for l in open(TMP + "/vi_res/log.jsonl")]; i2 = [i for i, x in enumerate(lg) if x["step"] == 21][-1]; rs = lg[i2:]
    strip = lambda L: [{k: v for k, v in x.items() if k != "sec"} for x in L]  # noqa: E731
    refm = [x for x in ref if x["step"] >= 21]
    o.update(rc=[rc0, rc1, r.returncode], ckpt=cmd_ok, n=[len(refm), len(rs)], bitwise=strip(refm) == strip(rs), m2_start=[ref[-1].get("m2_start"), rs[-1].get("m2_start") if rs else None],
             m2_m_last=ref[-1].get("m2_m"), first_diff=next(([a, b] for a, b in zip(strip(refm), strip(rs)) if a != b), None), err=r.stderr[-300:] if r.returncode not in (0, 9) else "")
    o["pass"] = o["bitwise"] and o["n"][0] == 20 and o["m2_start"][0] is not None and o["m2_start"][0] == o["m2_start"][1] and o["m2_m_last"] == 1.0 and o["m2_start"][0] <= 25
    return o


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    for part in A.parts:
        res = {"i": p_i, "ii": p_ii, "iii": p_iii, "iv": p_iv, "v": p_v, "vi": p_vi}[part](); res["tree"] = TREE
        json.dump(res, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", res["pass"], flush=True)


if __name__ == "__main__":
    main()
