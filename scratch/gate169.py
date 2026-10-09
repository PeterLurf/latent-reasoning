"""THEORY §169 INJECT pre-launch gates (5fffefd1), boxes/inject169.py:
  (1) step-0 function: fade 0 -> a think pass's logits under DeltaMem-R bitwise equal to plain GPT-2 (branch skipped)
  (2) gradflow (fade 0.5, W_o perturbed, gamma 0.5, K = 3): every DeltaMem parameter (incl. LN_w) gets gradient from the answer CE; the memory after
      think pass 1 (>= 2 passes before the decoder) and after pass 2 get nonzero gradient
  (3) the decoder writes nothing: the state is bitwise unchanged by dec_logits / greedy_ans, and two decodes agree
  (4) identical problem text per pass: every think pass's input ids (captured at the write) are the same [EOT, a * b, " ."] for all t; the decoder's
      prefix is [EOT, a * b, " ="]; K = 0 runs no think pass (the decoder reads the zero memory)
    python scratch/gate169.py [device]   -> boxes/runs_fsq/s169_gates/gate169.json"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.inject169 as IJ  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.multcol import gen_operands  # noqa: E402


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; torch.set_num_threads(3); out = {"device": dev}; ok = True
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    torch.manual_seed(0); rng = np.random.default_rng(0); g = s5.load_gpt2(dev); m = IJ.Inject(g).to(dev); A, B = gen_operands(rng, 4, 4), gen_operands(rng, 4, 3)
    # (1)
    m.mem.fade.fill_(0.0)
    with torch.no_grad(), IJ.amp_ctx(amp):
        x = g.embed(m.ids(torch.tensor(IJ.think_text(A, B), device=dev))); h1, _ = DM.gpt_forward_mem(g, x, m.mem, m.mem.zero_state(4, dev)); h0, _ = g(x, None)
    out["g1_max_abs_dlogits_vs_plain"] = float((g.logits(h1).float() - g.logits(h0).float()).abs().max()); out["g1_pass"] = out["g1_max_abs_dlogits_vs_plain"] == 0.0; ok &= out["g1_pass"]
    # (2)
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5); m.mem.gamma.fill_(0.5); m.mem.s_add.data.fill_(-7.0); m.train(); m.zero_grad(set_to_none=True); w0 = IJ.Inject.write; kept = []

    def wr(self, S, ids, cap=None):
        S2 = w0(self, S, ids) if cap is None else w0(self, S, ids, cap); S2.retain_grad(); kept.append(S2); return S2
    IJ.Inject.write = wr
    try:
        with IJ.amp_ctx(amp):
            S = m.think(m.mem.zero_state(4, dev), A, B, 3); P = IJ.product_digits(A, B); ans = torch.tensor(IJ.DT[P], device=dev)
            lg = m.dec_logits(S, A, B, ans); loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), ans.reshape(-1))
        loss.backward()
    finally:
        IJ.Inject.write = w0
    gr = {n_: (float(p_.grad.float().norm()) if p_.grad is not None else 0.0) for n_, p_ in m.mem.named_parameters()}
    out["g2_zero"] = [n_ for n_, v in gr.items() if v == 0.0]; out["g2_S_grad"] = [float(k_.grad.float().norm()) if k_.grad is not None else 0.0 for k_ in kept]
    out["g2_pass"] = not out["g2_zero"] and len(kept) == 3 and min(out["g2_S_grad"][:2]) > 0; ok &= out["g2_pass"]
    # (3)
    m.eval()
    with torch.no_grad(), IJ.amp_ctx(amp):
        S = m.think(m.mem.zero_state(4, dev), A, B, 2); Sc = S.clone(); m.dec_logits(S, A, B, ans); d1 = m.greedy_ans(S, A, B, 7); d2 = m.greedy_ans(S, A, B, 7)
    out["g3_state_unchanged"] = bool(torch.equal(S, Sc)); out["g3_decodes_agree"] = bool((d1 == d2).all()); out["g3_pass"] = out["g3_state_unchanged"] and out["g3_decodes_agree"]; ok &= out["g3_pass"]
    # (4)
    seen = []
    def wr2(self, S, ids, cap=None):
        seen.append(ids.detach().cpu().numpy().copy()); return w0(self, S, ids)
    IJ.Inject.write = wr2
    try:
        with torch.no_grad(), IJ.amp_ctx(amp):
            m.think(m.mem.zero_state(4, dev), A, B, 5); n_k0 = len(seen); IJ.predict(m, A, B, 0, dev, amp, 4); n_after_k0 = len(seen)
    finally:
        IJ.Inject.write = w0
    exp = np.concatenate([np.full((4, 1), IJ.EOT), IJ.think_text(A, B)], 1); pre = IJ.dec_prefix(A, B)
    out["g4"] = {"passes": n_k0, "all_identical": all(np.array_equal(s_, exp) for s_ in seen[:n_k0]), "think_ends_pad": bool((exp[:, -1] == IJ.PAD).all()),
                 "dec_prefix_ends_eq": bool((pre[:, -1] == IJ.EQ).all()), "k0_writes": n_after_k0 - n_k0}
    out["g4_pass"] = n_k0 == 5 and out["g4"]["all_identical"] and out["g4"]["think_ends_pad"] and out["g4"]["dec_prefix_ends_eq"] and out["g4"]["k0_writes"] == 0; ok &= out["g4_pass"]
    out["all_pass"] = bool(ok); dd = os.path.join(ROOT, "boxes", "runs_fsq", "s169_gates"); os.makedirs(dd, exist_ok=True)
    json.dump(out, open(os.path.join(dd, "gate169.json"), "w", newline="\n"), indent=1); print("GATE169", json.dumps(out), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
