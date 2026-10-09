"""THEORY §171 READ-SLOTS pre-launch gates (registration 2026-10-09 01:54, note 171-1), boxes/readslots171.py:
  (1) step-0 function (fade 0): the decoder's logits after K = 8 slot passes bitwise equal K = 0 (DIRECT by construction), and equal plain GPT-2 over the
      whole [EOT, input, " =", digits] sequence (max abs diff < 1e-3 and the same argmax, fp32: the masked SDPA kernel after a separate input KV vs one
      causal pass gives ~1e-4 round-off on logits of magnitude ~100; a wrong position / mask is O(1)); the same for the BLIND decoder against
      plain GPT-2 over [EOT, " =", digits]
  (2) gradflow (fade 0.5, W_o perturbed, gamma 0.5, K = 3), with the decoder fed a DETACHED copy of the input KV: every DeltaMem parameter and the slot
      embeddings get gradient; the memory after think pass 1 and 2 gets gradient; the input encoding (its KV and GPT-2's block-0 weights) gets gradient
      from the slot passes alone. Then K = 0 with the live KV: the decoder's gradient reaches the encoding.
  (3) the decoder attends to no slot position and the slot KV is discarded: after think + decode every input-KV tensor is unchanged and P0 long;
      perturbing the slot embeddings with the memory state held fixed leaves the decoder's logits bitwise unchanged; the decoder leaves S unchanged
  (4) the input encoding performs no memory read: its KV (fade 1, W_o random, after think passes) equals core.GPT2KV's own forward over the input bitwise
  (5) read_stats: per pass the attention mass over input + slots sums to 1, 0 <= entropy <= log P0, rho in [-1, 1] or None; compute_rs runs (both decoders)
    python scratch/gate171.py [device]   -> boxes/runs_fsq/s171_gates/gate171.json"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.readslots171 as R  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.fixedchunk import DT, EQ, product_digits  # noqa: E402
from boxes.fsq120 import EOT  # noqa: E402
from boxes.multcol import gen_operands  # noqa: E402


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; torch.set_num_threads(3); out = {"device": dev}; ok = True
    torch.manual_seed(0); rng = np.random.default_rng(0); g = s5.load_gpt2(dev); m = R.ReadSlots(g, 8).to(dev); mb = R.ReadSlots(g, 8, blind=True).to(dev)
    mb.mem = m.mem; mb.slots = m.slots                                       # one set of parameters, two decoders
    A, B = gen_operands(rng, 4, 4), gen_operands(rng, 4, 5); P = product_digits(A, B); ans = torch.tensor(DT[P], device=dev); P0 = A.shape[1] + B.shape[1] + 2
    # (1)
    m.mem.fade.fill_(0.0); m.eval()
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
        kv = m.encode(A, B, dev); S8 = m.think(m.mem.zero_state(4, dev), kv, 8); l8 = m.dec_logits(S8, kv, ans); l0 = m.dec_logits(m.mem.zero_state(4, dev), kv, ans)
        full = torch.cat([torch.tensor(R.inp_ids(A, B), device=dev), torch.full((4, 1), EQ, device=dev), ans[:, :-1]], 1)
        lp = g.logits(g(g.embed(full))[0])[:, -ans.shape[1]:].float()
        b8 = mb.dec_logits(S8, kv, ans); b0 = mb.dec_logits(m.mem.zero_state(4, dev), kv, ans)
        fb = torch.cat([torch.full((4, 1), EOT, device=dev), torch.full((4, 1), EQ, device=dev), ans[:, :-1]], 1); bp = g.logits(g(g.embed(fb))[0])[:, 1:].float()
    out["g1"] = {"k8_eq_k0": bool(torch.equal(l8, l0)), "max_abs_vs_plain": float((l8 - lp).abs().max()), "blind_k8_eq_k0": bool(torch.equal(b8, b0)),
                 "blind_max_abs_vs_plain": float((b8 - bp).abs().max()), "argmax_eq_plain": bool(torch.equal(l8.argmax(-1), lp.argmax(-1)))}
    out["g1_pass"] = out["g1"]["k8_eq_k0"] and out["g1"]["blind_k8_eq_k0"] and out["g1"]["max_abs_vs_plain"] < 1e-3 and out["g1"]["blind_max_abs_vs_plain"] < 1e-3 and out["g1"]["argmax_eq_plain"]; ok &= out["g1_pass"]
    # (2)
    m.mem.fade.fill_(0.5); m.mem.gamma.fill_(0.5); m.mem.s_add.data.fill_(-7.0); m.train(); m.zero_grad(set_to_none=True); kept = []; sp0 = R.ReadSlots.slot_pass

    def sp(self, S, kv_, cap=None):
        S2 = sp0(self, S, kv_, cap); S2.retain_grad(); kept.append(S2); return S2
    R.ReadSlots.slot_pass = sp
    try:
        kv = m.encode(A, B, dev)
        for k_, v_ in (kv[0], kv[11]):
            k_.retain_grad(); v_.retain_grad()
        kvd = tuple((k_.detach(), v_.detach()) for k_, v_ in kv); S = m.think(m.mem.zero_state(4, dev), kv, 3); lg = m.dec_logits(S, kvd, ans)
        Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), ans.reshape(-1)).backward()
    finally:
        R.ReadSlots.slot_pass = sp0
    gr = {n_: (float(p_.grad.float().norm()) if p_.grad is not None else 0.0) for n_, p_ in m.mem.named_parameters()}
    enc = {"kv0_k": float(kv[0][0].grad.norm()) if kv[0][0].grad is not None else 0.0, "kv11_v": float(kv[11][1].grad.norm()) if kv[11][1].grad is not None else 0.0,
           "h0_c_attn_w": float(g.w("h.0.attn.c_attn.weight").grad.norm()) if g.w("h.0.attn.c_attn.weight").grad is not None else 0.0}
    out["g2"] = {"mem_zero": [n_ for n_, v in gr.items() if v == 0.0], "slots": float(m.slots.grad.norm()) if m.slots.grad is not None else 0.0,
                 "S_grad": [float(k_.grad.norm()) if k_.grad is not None else 0.0 for k_ in kept], "enc_from_slots": enc}
    m.zero_grad(set_to_none=True); kv = m.encode(A, B, dev); kv[0][0].retain_grad(); lg = m.dec_logits(m.mem.zero_state(4, dev), kv, ans)
    Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), ans.reshape(-1)).backward(); out["g2"]["enc_from_decoder_k0"] = float(kv[0][0].grad.norm()); m.zero_grad(set_to_none=True)
    out["g2_pass"] = (not out["g2"]["mem_zero"] and out["g2"]["slots"] > 0 and min(out["g2"]["S_grad"][:2]) > 0 and min(enc.values()) > 0
                      and out["g2"]["enc_from_decoder_k0"] > 0); ok &= out["g2_pass"]
    # (3) + (4)
    m.eval(); m.mem.fade.fill_(1.0)
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.05); kv = m.encode(A, B, dev); kc = [(k_.clone(), v_.clone()) for k_, v_ in kv]
        S = m.think(m.mem.zero_state(4, dev), kv, 3); Sc = S.clone(); d1 = m.dec_logits(S, kv, ans); m.greedy_ans(S, kv, P.shape[1])
        same = all(torch.equal(a_, b_) and torch.equal(c_, d_) and a_.shape[2] == P0 for (a_, c_), (b_, d_) in zip(kv, kc))
        s0 = m.slots.data.clone(); m.slots.data.add_(torch.randn_like(s0)); d2 = m.dec_logits(S, kv, ans); S_alt = m.think(m.mem.zero_state(4, dev), kv, 3); m.slots.data.copy_(s0)
        out["g3"] = {"input_kv_unchanged_len_P0": bool(same), "dec_invariant_to_slots": bool(torch.equal(d1, d2)), "slots_change_S": bool(not torch.equal(S_alt, S)),
                     "S_unchanged_by_decoder": bool(torch.equal(S, Sc))}
        out["g3_pass"] = all(out["g3"].values()); ok &= out["g3_pass"]
        _, kp = g(g.embed(torch.tensor(R.inp_ids(A, B), device=dev)))
        out["g4_pass"] = bool(all(torch.equal(a_, b_) and torch.equal(c_, d_) for (a_, c_), (b_, d_) in zip(kv, kp))); ok &= out["g4_pass"]
    # (5)
    rA, rB = gen_operands(rng, 6, 4), gen_operands(rng, 6, 8); cap = {}
    with torch.no_grad():
        kv = m.encode(rA, rB, dev); m.think(m.mem.zero_state(6, dev), kv, 3, cap=cap)
    sums = [float(torch.stack(cap[t], 1).sum(-1).sub(1).abs().max()) for t in (1, 2, 3)]
    rmap, rent, rho, peak = R.read_stats(m, rA, rB, 3, dev, None, 4); P08 = 4 + 8 + 2
    msum = [abs(sum(v for v in rmap[str(t)].values()) - 1) for t in (1, 2, 3)]
    out["g5"] = {"prob_sum_err": sums, "map_sum_err": msum, "entropy": rent, "rho": rho, "peak": peak, "compute_4x4": R.compute_rs(m, 4, 4, 8)["flops"],
                 "compute_blind_4x4": R.compute_rs(mb, 4, 4, 8)["flops"]}
    out["g5_pass"] = (max(sums) < 1e-4 and max(msum) < 1e-4 and all(0 <= v <= math.log(P08) + 1e-6 for v in rent.values()) and (rho is None or -1 <= rho <= 1)
                      and len(peak) == 3); ok &= out["g5_pass"]
    out["all_pass"] = bool(ok); dd = os.path.join(ROOT, "boxes", "runs_fsq", "s171_gates"); os.makedirs(dd, exist_ok=True)
    json.dump(out, open(os.path.join(dd, "gate171.json"), "w", newline="\n"), indent=1); print("GATE171", json.dumps(out), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
