"""§202 WARM-START gates (THEORY 30401c78; docs/THEORY_WARMCOT_DRAFT_2026-10-14.md §5.3), before any launch:
  COPY-INIT      : W's construction (g <- C's model.pt strictly, then ProsMem(g)) has mem.qkv / mem.ln / mem.ln_w equal (torch.equal) to C's block 6
                   c_attn (transposed) / block 6 ln_1 / block 11 ln_1, and NOT equal to raw GPT-2's (the copy came from the loaded trunk).
  INIT-IDENTITY  : fp32, the first 200 Aug-valid items, W at step 0 (fade 0, g_add calibrated as gsm8k_stepaux does): for every K' in KPRIMES the
                   answers (greedy after [EOT] + pre + "###") equal C's plain trunk on the same prefix, item by item (strings), the final hidden of the
                   answer prefix is bitwise equal (max |d| reported), and acc_by_kprime is constant. direct_eot (= the plain answers' accuracy) is
                   printed beside theory/warmcot/probe_plan.json's C.direct_eot (its own item set).
  -> scratch/gate_s202/gate.json
    python scratch/gate_s202.py [device cpu] [n 200]"""
import json
import math
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
import boxes.gsm8k_stepaux as GS  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.deltas5 import fwd_read  # noqa: E402
from boxes.prosqa168s2 import ANS, ENC, EOS, EOT, ProsMem, pad  # noqa: E402

C_PT = "boxes/runs_fsq/gsm_txt_aug_s1/model.pt"


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; n = int(sys.argv[2]) if len(sys.argv) > 2 else 200; torch.set_num_threads(3)
    DM.FAST_ADD = True; DM.CHUNK_DELTA = 64                                 # §200 A's flags (W = A's flags exactly)
    od = "scratch/gate_s202"; os.makedirs(od, exist_ok=True); out = {"device": dev, "dtype": "float32 (no autocast)", "init_from": C_PT}
    raw = s5.load_gpt2(dev); m_raw = ProsMem(raw).to(dev)
    g = s5.load_gpt2(dev); sdC = torch.load(C_PT, map_location=dev, weights_only=False); g.load_state_dict(sdC, strict=True); m = ProsMem(g).to(dev)
    # ---- COPY-INIT
    pairs = {"qkv.weight": (m.mem.qkv.weight, g.w("h.6.attn.c_attn.weight").T, m_raw.mem.qkv.weight), "qkv.bias": (m.mem.qkv.bias, g.w("h.6.attn.c_attn.bias"), m_raw.mem.qkv.bias),
             "ln.weight": (m.mem.ln.weight, g.w("h.6.ln_1.weight"), m_raw.mem.ln.weight), "ln.bias": (m.mem.ln.bias, g.w("h.6.ln_1.bias"), m_raw.mem.ln.bias),
             "ln_w.weight": (m.mem.ln_w.weight, g.w("h.11.ln_1.weight"), m_raw.mem.ln_w.weight), "ln_w.bias": (m.mem.ln_w.bias, g.w("h.11.ln_1.bias"), m_raw.mem.ln_w.bias)}
    ci = {k: {"equal_C": bool(torch.equal(a.detach().float(), c.detach().float())), "differs_raw": not bool(torch.equal(a.detach().float(), r.detach().float()))}
          for k, (a, c, r) in pairs.items()}
    trunk_eq = all(torch.equal(v.detach().float(), sdC["p." + k.split(".", 1)[1]].float()) for k, v in g.state_dict().items() if k.startswith("p."))
    out["copy_init"] = {"tensors": ci, "trunk_equals_C_state_dict": bool(trunk_eq), "pass": bool(trunk_eq and all(v["equal_C"] and v["differs_raw"] for v in ci.values()))}
    print("COPY-INIT", json.dumps(out["copy_init"]), flush=True); del m_raw, raw
    # ---- INIT-IDENTITY (step 0: fade 0; g_add calibrated exactly as gsm8k_stepaux's note 147-2 block)
    Rtr, _ = G.load_aug("train"); Rva, _ = G.load_aug("valid", filter_bad=False); enc = lambda r: {**G.encode(r, ENC, EOS), "_q": r["q"]}  # noqa: E731
    its = [enc(Rtr[i]) for i in np.random.default_rng(16801).permutation(len(Rtr))[:32]]; x, msk = m.think_inputs(its, dev); m.mem.norms = []
    with torch.no_grad():
        m.think(m.mem.zero_state(len(its), dev), x, msk, 4)
    nr = m.mem.norms[1:]; m.mem.norms = None; r0 = float(torch.stack([d_ / a_.clamp_min(1e-12) for d_, a_ in nr]).median()) if nr else 1.0
    m.mem.s_add.data.fill_(math.log(max(r0, 1e-12))); m.eval(); assert float(m.mem.fade) == 0.0, "step 0 must have fade 0"
    D = [enc(r) for r in Rva[:n]]; bs = 64; ac = lambda: torch.autocast("cpu", enabled=False)  # noqa: E731
    seqs = [[EOT] + it["pre"] + [ANS] for it in D]
    P0 = GS.greedy_after(GS.Fwd(g), seqs, dev, ac, bs); ok0 = [GS.ans_ok(p_, it["ans"]) for p_, it in zip(P0, D)]
    res = {}; ident = True
    for kp in GS.KPRIMES:
        Ss = []
        for i in range(0, len(D), bs):
            ch = D[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            with torch.no_grad():
                Ss.append(m.think(m.mem.zero_state(len(ch), dev), x, msk, kp, cache=True))
        S = torch.cat(Ss); P = GS.greedy_after(GS.Fwd(g, m.mem, S), seqs, dev, ac, bs); dh = 0.0
        for i in range(0, len(D), bs):
            ids_, _ = pad(seqs[i:i + bs]); xx = torch.tensor(ids_, device=dev)
            with torch.no_grad():
                h1 = fwd_read(g, g.embed(xx), m.mem, S[i:i + bs]); h0 = g(g.embed(xx), None)[0]
            dh = max(dh, float((h1 - h0).abs().max()))
        same = sum(p_ == q_ for p_, q_ in zip(P, P0)); acc = float(np.mean([GS.ans_ok(p_, it["ans"]) for p_, it in zip(P, D)]))
        res[str(kp)] = {"acc": acc, "answers_equal_plain": int(same), "n": len(D), "max_abs_hidden_diff": dh}; ident &= (same == len(D))
        print("K'", kp, res[str(kp)], flush=True)
    accs = {v["acc"] for v in res.values()}
    pj = json.load(open("theory/warmcot/probe_plan.json"))
    out["init_identity"] = {"by_kprime": res, "direct_eot_plain": float(np.mean(ok0)), "probe_plan_C_direct_eot": pj.get("C", {}).get("direct_eot"), "g_add_init": r0,
                            "acc_constant_over_kprime": len(accs) == 1, "answers_equal_plain_all": bool(ident),
                            "hidden_bitwise_all": all(v["max_abs_hidden_diff"] == 0.0 for v in res.values()), "items": "Aug valid[:%d]" % n,
                            "pass": bool(ident and len(accs) == 1)}
    out["pass"] = bool(out["copy_init"]["pass"] and out["init_identity"]["pass"])
    json.dump(out, open(os.path.join(od, "gate.json"), "w", newline="\n"), indent=1); print("GATE_S202", "PASS" if out["pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
