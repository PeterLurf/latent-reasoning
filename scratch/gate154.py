"""THEORY §154 SHIFT-MEM pre-launch gates (ed7857cc), MultDelta --add_reset column on a full 4x4 digit-pair chain:
  (1) step-0 function preservation: fade 0 -> every emission's logits bitwise equal to plain GPT-2 (the branch skipped) along the chain
  (2) gradient flow (fade 0.5, W_o perturbed, g_add calibrated as in training): nonzero grad into every DeltaMem parameter from the emission CE
  (3) the reset acts where registered: entering every new-column segment and the flush, the additive state is exactly 0 and the delta state is not
    python scratch/gate154.py [device]   -> boxes/runs_fsq/s154_gates/gate154_shift.json (exit 1 on a failed gate)"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.multcol as MC  # noqa: E402
import boxes.multpair as MP  # noqa: E402
from boxes import s5  # noqa: E402


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"; out = {"device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"}
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None; out["amp"] = str(amp)
    torch.manual_seed(0); rng = np.random.default_rng(0); B = 8; A, Bd = MC.gen_operands(rng, B, 4), MC.gen_operands(rng, B, 4)
    _, dig, _, pairs, col, emit = MP.accumulators(A, Bd); T = len(pairs)
    m = MP.MultDelta(s5.load_gpt2(dev), MP.PairFam(), 6).to(dev); m.add_reset = "column"; m.train(); MP.set_ops(m, A, Bd, dev); ok = True
    # (1) fade 0: chain logits vs the same chain with the branch skipped
    m.mem.fade.fill_(0.0); dmax = 0.0
    with torch.no_grad(), MP.amp_ctx(amp):
        S = m.start(B, dev)
        for t in range(T):
            lg, S = m.step(S, t); lay = m.mem.layer; m.mem.layer = 99; lg0, _ = m.step(m.start(B, dev), t); m.mem.layer = lay
            dmax = max(dmax, float((lg - lg0).abs().max()))
    out["g1_max_abs_dlogits_vs_plain"] = dmax; out["g1_pass"] = dmax == 0.0; ok &= out["g1_pass"]
    # (3) reset placement
    starts = [t for t in range(1, T) if m.newcol[t]]; seen = []
    with torch.no_grad(), MP.amp_ctx(amp):
        S = m.start(B, dev)
        for t in range(T):
            if t in starts:
                Sr = torch.cat([torch.zeros_like(S[:, :DM.H_ADD]), S[:, DM.H_ADD:]], 1)
                seen.append({"t": t, "add_norm_before": float(S[:, :DM.H_ADD].float().norm()), "delta_norm_before": float(S[:, DM.H_ADD:].float().norm())})
                _, S1 = m.step(S, t); _, S2 = m.step(Sr, t); seen[-1]["step_equals_explicit_reset"] = bool(torch.equal(S1, S2)); S = S1
            else:
                _, S = m.step(S, t)
    out["g3_reset_steps"] = starts; out["g3_detail"] = seen
    out["g3_pass"] = len(starts) == len(set(col.tolist())) - 1 and all(x["step_equals_explicit_reset"] for x in seen); ok &= out["g3_pass"]
    # (2) gradient flow
    _, med0 = MP.read_ratio_by_handoff(m, A, Bd, dev, amp, n=B); m.mem.s_add.data.fill_(math.log(med0)); out["g2_g_add_init"] = float(med0)
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5); m.zero_grad(set_to_none=True); MP.set_ops(m, A, Bd, dev)
    X = torch.tensor(np.tile(np.arange(T), (B, 1)), device=dev); tm = torch.tensor(np.tile(emit, (B, 1)), device=dev); lab = np.where(dig >= 0, dig, 0)
    with MP.amp_ctx(amp):
        loss, _, _, _ = m.chain(m.start(B, dev), X, lab, tm)
    loss.backward()
    gr = {n_: (float(p_.grad.float().norm()) if p_.grad is not None else 0.0) for n_, p_ in m.mem.named_parameters()}
    w_ = m.mem.wo.weight.grad; gr["wo.additive_cols"] = float(w_[:, :384].float().norm()); gr["wo.delta_cols"] = float(w_[:, 384:].float().norm())
    out["g2_grad"] = gr; out["g2_zero"] = [n_ for n_ in gr if gr[n_] == 0.0]; out["g2_pass"] = not out["g2_zero"]; ok &= out["g2_pass"]
    out["all_pass"] = bool(ok); d = os.path.join(ROOT, "boxes", "runs_fsq", "s154_gates"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "gate154_shift.json"), "w", newline="\n"), indent=1); print("GATE154", json.dumps(out), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
