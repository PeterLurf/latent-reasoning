"""THEORY §152 DELTA-READ pre-launch gates (378fe8e6 / note 152-1), on a local CUDA card (bf16 where the run uses it):
  (1) step-0 function preservation: with fade 0, a segment through multrow.fwd_mem has max |delta logits| = 0 against plain GPT-2 (the same op
      sequence with no branch), and fwd_mem equals boxes.deltamem.gpt_forward_mem bitwise (logits and state)
  (2) gradient flow under CUDA bf16 (fade 0.5, W_o perturbed, g_add calibrated as in training): nonzero grad into every DeltaMem parameter from
      the reader CE alone and from the final CE alone (alpha exempt, note 147-3: reported, not gated)
  (3) the reader writes nothing back: states into every segment j + 1 and the final logits bitwise identical with the reader passes on / off
    python scratch/gate152.py cuda:1      -> boxes/runs_fsq/s152_gates/gate152.json (exit 1 on a failed gate)"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.multrow as MR  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.multcol import DTOK, gen_operands  # noqa: E402


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"; out = {"device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"}
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None; out["amp"] = str(amp)
    torch.manual_seed(0); rng = np.random.default_rng(0); B = 8; A, Bd = gen_operands(rng, B, 4), gen_operands(rng, B, 4)
    m = MR.MultRowDelta(s5.load_gpt2(dev), MR.RowFam(), 6).to(dev); m.train(); MR.set_ops(m, A, Bd, dev); g = m.g; ok = True
    # (1) step 0: fade 0
    m.mem.fade.fill_(0.0)
    with torch.no_grad(), MR.amp_ctx(amp):
        x = g.embed(m.seg_ids(0, B, dev)); S0 = m.start(B, dev)
        h1, S1 = MR.fwd_mem(g, x, m.mem, S0); h2, S2 = DM.gpt_forward_mem(g, x, m.mem, S0)
        lay = m.mem.layer; m.mem.layer = 99; h0, _ = MR.fwd_mem(g, x, m.mem, S0); m.mem.layer = lay                # plain GPT-2: no branch
        hc, _ = g(x, None)                                                                                         # core.GPT.forward (info)
        L1, L0, Lc = g.logits(h1).float(), g.logits(h0).float(), g.logits(hc).float()
    out["g1_max_abs_dlogits_vs_plain"] = float((L1 - L0).abs().max()); out["g1_fwd_mem_eq_gpt_forward_mem"] = bool(torch.equal(h1, h2) and torch.equal(S1, S2))
    out["g1_info_max_abs_dlogits_vs_core_forward"] = float((L1 - Lc).abs().max())
    out["g1_pass"] = out["g1_max_abs_dlogits_vs_plain"] == 0.0 and out["g1_fwd_mem_eq_gpt_forward_mem"]; ok &= out["g1_pass"]
    # (2) gradient flow: fade 0.5, W_o perturbed, g_add calibrated (note 147-2)
    _, med0 = MR.read_ratio_drow(m, A, Bd, dev, amp, n=B); m.mem.s_add.data.fill_(math.log(med0)); out["g2_g_add_init"] = float(med0)
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5); MR.set_ops(m, A, Bd, dev); X = torch.tensor(np.tile(np.arange(4), (B, 1)), device=dev); lab = MR.sj_values(A, Bd)
    rtok = torch.tensor(np.array(DTOK)[MR.digits8(lab[:, :3])], device=dev); rmask = torch.ones(B, 3, dtype=torch.bool, device=dev)
    for term in ("reader", "final"):
        m.zero_grad(set_to_none=True)
        with MR.amp_ctx(amp):
            fin, rd, nrd = m.chain_read(m.start(B, dev), X, lab, rtok, rmask)
        (rd if term == "reader" else fin).backward()
        gr = {n_: (float(p_.grad.float().norm()) if p_.grad is not None else 0.0) for n_, p_ in m.mem.named_parameters()}
        gr["trunk.h.0.attn.c_attn.weight"] = float(g.w("h.0.attn.c_attn.weight").grad.float().norm()) if g.w("h.0.attn.c_attn.weight").grad is not None else 0.0
        gw = m.mem.gate.weight.grad; gr["gate.alpha_rows"] = float(gw[:DM.H_DEL].float().norm()) if gw is not None else 0.0; gr["gate.beta_rows"] = float(gw[DM.H_DEL:].float().norm()) if gw is not None else 0.0
        out["g2_%s_grad" % term] = gr; out["g2_%s_n_handoffs" % term] = nrd
        need = [n_ for n_, _ in m.mem.named_parameters()]
        out["g2_%s_zero" % term] = [n_ for n_ in need if gr[n_] == 0.0]
        out["g2_%s_pass" % term] = not out["g2_%s_zero" % term]; ok &= out["g2_%s_pass" % term]
    # (3) the reader writes nothing back
    m.eval()
    with torch.no_grad(), MR.amp_ctx(amp):
        def run(with_reader):
            MR.set_ops(m, A, Bd, dev); S = m.start(B, dev); st = []; same = True
            for t in range(4):
                S = m.write(S, t); st.append(S.clone())
                if with_reader:
                    Sc = S.clone(); m.read_logits(S, rtok[:, min(t, 2)]); m.read_greedy(S); same &= bool(torch.equal(S, Sc))
            return st, m.read_logits(S, None), same
        st_on, lg_on, unchanged = run(True); st_off, lg_off, _ = run(False)
    out["g3_states_bitwise_equal"] = all(bool(torch.equal(a_, b_)) for a_, b_ in zip(st_on, st_off)); out["g3_final_logits_bitwise_equal"] = bool(torch.equal(lg_on, lg_off))
    out["g3_state_object_unchanged_by_reader"] = unchanged
    out["g3_pass"] = out["g3_states_bitwise_equal"] and out["g3_final_logits_bitwise_equal"] and unchanged; ok &= out["g3_pass"]
    out["all_pass"] = bool(ok); d = os.path.join(ROOT, "boxes", "runs_fsq", "s152_gates"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "gate152.json"), "w", newline="\n"), indent=1); print("GATE152", json.dumps(out), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
