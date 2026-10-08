"""THEORY §165 FIXED-CHUNK pre-launch gates (60e55135), boxes/fixedchunk.py:
  (1) step-0 function: fade 0 -> a chunk pass's logits under DeltaMem-R bitwise equal to plain GPT-2 (branch skipped)
  (2) gradflow (fade 0.5, W_o perturbed, gamma 0.5, K = 3): every DeltaMem parameter (incl. LN_w) gets gradient from the answer CE; the memory
      after the input chunks (>= 2 think passes before the decoder) and after think pass 1 get nonzero gradient
  (3) the decoder writes nothing: the state is bitwise unchanged by dec_logits / greedy, and two decodes of the same state agree
  (4) chunking: for k in {1, 3, 4, 5} and 4 x n (n = 2..16) the chunks are exactly k tokens, boundaries at multiples of k, their concatenation =
      the problem text + " ." padding (no task boundary); the problem text = a digits, " *", b digits, " =" LSB first
  (5) labels: product_digits = a x b's digits LSB first (4 x 2 .. 4 x 16)
    python scratch/gate165.py [device]   -> boxes/runs_fsq/s165_gates/gate165.json (exit 1 on a failed gate)"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.fixedchunk as FC  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.multcol import gen_operands  # noqa: E402


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; torch.set_num_threads(3); out = {"device": dev}; ok = True
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    torch.manual_seed(0); rng = np.random.default_rng(0); g = s5.load_gpt2(dev); m = FC.FixedChunk(g).to(dev); A, B = gen_operands(rng, 4, 4), gen_operands(rng, 4, 3)
    # (1)
    m.mem.fade.fill_(0.0); ch = torch.tensor(FC.chunks(FC.problem_tokens(A, B), 4)[0], device=dev)
    with torch.no_grad(), FC.amp_ctx(amp):
        x = g.embed(m.ids(ch)); h1, _ = DM.gpt_forward_mem(g, x, m.mem, m.mem.zero_state(4, dev)); h0, _ = g(x, None)
    out["g1_max_abs_dlogits_vs_plain"] = float((g.logits(h1).float() - g.logits(h0).float()).abs().max()); out["g1_pass"] = out["g1_max_abs_dlogits_vs_plain"] == 0.0; ok &= out["g1_pass"]
    # (2)
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5); m.mem.gamma.fill_(0.5); m.mem.s_add.data.fill_(-7.0); m.train(); m.zero_grad(set_to_none=True)
    with FC.amp_ctx(amp):
        S_in = m.encode(A, B, 4, dev); S_in.retain_grad(); S1 = m.write(S_in, m.ids(torch.full((4, 1), FC.PAD, device=dev))); S1.retain_grad(); S = m.think(S1, 2)
        P = FC.product_digits(A, B); ans = torch.tensor(FC.DT[P], device=dev); lg = m.dec_logits(S, ans); loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), ans.reshape(-1))
    loss.backward()
    gr = {n_: (float(p_.grad.float().norm()) if p_.grad is not None else 0.0) for n_, p_ in m.mem.named_parameters()}
    out["g2_zero"] = [n_ for n_, v in gr.items() if v == 0.0]; out["g2_S_grad"] = [float(S_in.grad.float().norm()), float(S1.grad.float().norm())]
    out["g2_pass"] = not out["g2_zero"] and min(out["g2_S_grad"]) > 0; ok &= out["g2_pass"]
    # (3)
    m.eval()
    with torch.no_grad(), FC.amp_ctx(amp):
        S = m.think(m.encode(A, B, 4, dev), 2); Sc = S.clone(); m.dec_logits(S, ans); d1 = m.greedy(S, 7); d2 = m.greedy(S, 7)
    out["g3_state_unchanged"] = bool(torch.equal(S, Sc)); out["g3_decodes_agree"] = bool((d1 == d2).all()); out["g3_pass"] = out["g3_state_unchanged"] and out["g3_decodes_agree"]; ok &= out["g3_pass"]
    # (4) + (5)
    bad4, bad5 = [], []
    for nb in range(2, 17):
        A_, B_ = gen_operands(rng, 16, 4), gen_operands(rng, 16, nb); tok = FC.problem_tokens(A_, B_); L = tok.shape[1]
        if not (L == nb + 6 and (tok[:, 4] == FC.STAR).all() and (tok[:, -1] == FC.EQ).all() and (tok[:, :4] == FC.DT[A_]).all() and (tok[:, 5:5 + nb] == FC.DT[B_]).all()):
            bad4.append(("text", nb))
        for k in (1, 3, 4, 5):
            cs = FC.chunks(tok, k); cat = np.concatenate(cs, 1)
            if not (all(c.shape[1] == k for c in cs) and len(cs) == -(-L // k) and (cat[:, :L] == tok).all() and (cat[:, L:] == FC.PAD).all()):
                bad4.append((k, nb))
        a_ = [int("".join(str(d) for d in r[::-1])) for r in A_]; b_ = [int("".join(str(d) for d in r[::-1])) for r in B_]; P_ = FC.product_digits(A_, B_)
        if not all(int("".join(str(d) for d in P_[r][::-1])) == a_[r] * b_[r] for r in range(16)) or P_.shape[1] != 4 + nb:
            bad5.append(nb)
    out["g4_bad"] = bad4; out["g4_pass"] = not bad4; out["g5_bad"] = bad5; out["g5_pass"] = not bad5; ok &= out["g4_pass"] and out["g5_pass"]
    out["all_pass"] = bool(ok); dd = os.path.join(ROOT, "boxes", "runs_fsq", "s165_gates"); os.makedirs(dd, exist_ok=True)
    json.dump(out, open(os.path.join(dd, "gate165.json"), "w", newline="\n"), indent=1)
    print("GATE165", json.dumps(out), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
