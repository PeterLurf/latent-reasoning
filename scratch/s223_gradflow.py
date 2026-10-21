"""§223 build gates (PS2_v1 Addendum A item 4 + Addendum B item 3; Theory Q6 10-21), CPU, tiny batch (2 items, d 3, K 4; stream rng 777 - never a
registered stream). For each cell (PS / M1-only / M2-only / DIRECT / PLAIN):
  (1) per-term gradflow: each loss term backpropagated ALONE from fresh grads:
      - answer CE -> nonzero grad on LoRA and (latent cells) prj; zero on D;
      - recode CE (M1 cells) -> nonzero grad on D AND on prj AND on LoRA (through z_j: z_j is not detached);
  (2) the M2 direct-path probe: the decoder call's encoder-position past K / V and the latent rows' K / V are leaf tensors; answer CE's grad on the
      question-content + <bot> positions must be EXACTLY 0 with --bottleneck and nonzero without it; the latent rows' K / V grad must be nonzero
      (the only path left under M2); BOS's grad is reported (BOS stays visible);
  (3) ast: no decorator on codi_ps functions and no .detach() on z_j in the train path (grep);
  (4) plain-path smoke: the PLAIN cell under --codi_compat (no BOS, no latent / decoder masks, codi_port's CE dtype) with K = 6 gives the same
      student CE as boxes.codi_port.codi_loss's "ce" on the same batch and model (eval mode, no dropout) - bitwise; + two optimiser steps (step 1's warm-up lr is 0)
      without the new flags run and change the trainable parameters.
    python -B scratch/s223_gradflow.py -> scratch/gate_s223/gradflow.json"""
import argparse
import ast
import json
import os
import sys
import types

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.codi_port as CP  # noqa: E402
import boxes.codi_ps as PS  # noqa: E402
import boxes.nested_words as NW  # noqa: E402

CELLS = {"PS": dict(recode=True, lam=1.0, bottleneck=True), "M1-only": dict(recode=True, lam=1.0), "M2-only": dict(bottleneck=True),
         "DIRECT": dict(direct=True), "PLAIN": {}}


def args(**kw):
    base = dict(recode=False, lam=0.0, bottleneck=False, direct=False, codi_compat=False, seed=1)
    base.update(kw); return types.SimpleNamespace(**base)


def gnorm(params):
    return float(sum(float(p_.grad.abs().sum()) for p_ in params if p_.grad is not None))


AC = lambda: torch.autocast("cpu", dtype=torch.bfloat16)  # noqa: E731  (the trainer always runs under bf16 autocast)


def main():
    torch.set_num_threads(4); dev = "cpu"
    import tiktoken
    enc = tiktoken.get_encoding("gpt2"); PS.VALUE_TOKS.update(PS.value_tokens(enc))
    its = [PS.tok_item(NW.make(np.random.default_rng(777 + i), 3), enc) for i in range(2)]; out = {"batch": {"n": 2, "d": 3, "K": 4, "rng": "777 / 778"}, "cells": {}}
    for name, kw in CELLS.items():
        a = args(**kw); torch.manual_seed(1); m, prj, D, TP = PS.build_all(a, dev); m.eval(); prj.eval()
        lora = [p_ for n, p_ in TP if ".lora_" in n]; pj = [p_ for n, p_ in TP if n.startswith("prj.")]; dp = [p_ for n, p_ in TP if n.startswith("D.")]
        b = PS.collate(its, 0 if a.direct else 4, dev); r = {}
        for p_ in m.parameters():
            p_.grad = None
        with AC():
            _, pt, ex = PS.forward(m, prj, D, b, a)
        ex["ce"].backward()
        r["answer_ce"] = {"lora": gnorm(lora), "prj": gnorm(pj), "D": gnorm(dp)}
        for _, p_ in TP:
            p_.grad = None
        if a.recode:
            with AC():
                _, _, ex2 = PS.forward(m, prj, D, b, a); rce, _, _ = PS.recode_ce(m, D, ex2["zs"], b, dev)
            rce.backward()
            r["recode_ce"] = {"lora": gnorm(lora), "prj": gnorm(pj), "D": gnorm(dp), "value": float(rce)}
            for _, p_ in TP:
                p_.grad = None
        if not a.direct:
            for bn in (True, False):                                        # the probe with M2 on and off (the cell's own setting is one of them)
                a2 = args(**{**kw, "bottleneck": bn})
                with AC():
                    _, _, ex3 = PS.forward(m, prj, D, b, a2, probe=True)
                ex3["ce"].backward()
                qm = b["qm"].bool(); bos = (b["em"].bool() & ~qm)
                gq = sum(float((t.grad.abs().sum(1).sum(-1))[qm].sum()) for t in ex3["enc_k"] + ex3["enc_v"])
                gb = sum(float((t.grad.abs().sum(1).sum(-1))[bos].sum()) for t in ex3["enc_k"] + ex3["enc_v"])
                gl = sum(float(t.grad.abs().sum()) for t in ex3["lat_k"] + ex3["lat_v"])
                r["probe_M2_%s" % ("on" if bn else "off")] = {"question_bot_kv_grad": gq, "bos_kv_grad": gb, "latent_kv_grad": gl}
                for _, p_ in TP:
                    p_.grad = None
        ok = r["answer_ce"]["lora"] > 0 and r["answer_ce"]["D"] == 0 and (a.direct or r["answer_ce"]["prj"] > 0)
        if a.recode:
            ok = ok and all(r["recode_ce"][k] > 0 for k in ("lora", "prj", "D"))
        if not a.direct:
            ok = ok and r["probe_M2_on"]["question_bot_kv_grad"] == 0 and r["probe_M2_off"]["question_bot_kv_grad"] > 0 and r["probe_M2_on"]["latent_kv_grad"] > 0
        r["pass"] = bool(ok); out["cells"][name] = r; print(name, json.dumps(r), flush=True)
        del m, prj, D, TP
    src = open("boxes/codi_ps.py", encoding="utf-8").read(); tree = ast.parse(src)
    out["ast"] = {"decorated_functions": [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.decorator_list],
                  "no_grad_mentions": [l.strip()[:100] for l in src.split("\n") if "no_grad" in l],
                  "detach_lines": [l.strip()[:100] for l in src.split("\n") if ".detach()" in l]}
    # (4) plain path vs codi_port (eval mode, K 6, compat)
    a = args(codi_compat=True); torch.manual_seed(1); m, prj, D, TP = PS.build_all(a, dev); m.eval(); prj.eval()
    with AC():
        _, pt, _ = PS.forward(m, prj, D, PS.collate(its, 6, dev, compat=True), a)
    cits = [{"q": it["q"][1:], "ref": it["q"][1:] + it["dec"][1:], "nq": len(it["q"]) - 1, "dec": it["dec"]} for it in its]
    with torch.no_grad(), AC():
        _, cpp = CP.codi_loss(m, prj, CP.collate(cits, dev), n_latent=6, gamma=2.0)
    out["plain_vs_codi_port"] = {"codi_ps_ans_ce": pt["ans_ce"], "codi_port_ce": cpp["ce"], "bitwise_equal": pt["ans_ce"] == cpp["ce"]}
    a = args(); torch.manual_seed(1); m, prj, D, TP = PS.build_all(a, dev); m.train(); prj.train(); opt, sch, _ = PS.make_opt(types.SimpleNamespace(wd=0.1, lr=3e-3, warmup=0.03), TP, 100)
    before = {n: p_.detach().clone() for n, p_ in TP}
    for _ in range(2):                                                      # step 1's lr is 0 (linear warm-up from 0, as codi_port / HF): take two
        opt.zero_grad(set_to_none=True)
        with AC():
            l_, _, _ = PS.forward(m, prj, D, PS.collate(its, 4, dev), a)
        l_.backward(); opt.step(); sch.step()
    out["plain_one_step"] = {"loss": float(l_), "finite": bool(torch.isfinite(l_)), "params_changed": sum(int(not torch.equal(before[n], p_)) for n, p_ in TP), "n_params": len(TP)}
    out["PASS"] = bool(all(c["pass"] for c in out["cells"].values()) and not out["ast"]["decorated_functions"] and out["plain_vs_codi_port"]["bitwise_equal"]
                       and out["plain_one_step"]["finite"] and out["plain_one_step"]["params_changed"] > 0)
    os.makedirs("scratch/gate_s223", exist_ok=True); json.dump(out, open("scratch/gate_s223/gradflow.json", "w", newline="\n"), indent=1)
    print("GRADFLOW", "PASS" if out["PASS"] else "FAIL", json.dumps({k: out[k] for k in ("plain_vs_codi_port", "plain_one_step")}), flush=True)


if __name__ == "__main__":
    main()
