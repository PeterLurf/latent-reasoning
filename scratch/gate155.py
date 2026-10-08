"""THEORY §155 DELTA-R pre-launch gates (e9ee8a6c), MultDelta --mem_write top|block9 [--add_reset column] on a full 4x4 digit-pair chain:
  (1) step-0 function preservation: fade 0 -> every step's logits bitwise equal to plain GPT-2 (the branch skipped), at gamma 0 and 1
  (2) gradient flow (CUDA bf16 where available; gamma 0.5, fade 0.5, W_o perturbed, g_add calibrated): the chain's emission CE gives nonzero grad
      into every DeltaMem parameter incl. LN_w
  (3) BPTT through >= 2 hand-offs: the CE at emission e with LN_w live only in writes at steps <= e - 2 (later steps use a frozen copy) still
      gives nonzero grad into LN_w; and dL/dS_{e-2} differs between gamma 0.5 and 0 (the write-after-read path carries gradient)
  (4) rho_jvp computable (s155_recurrence)
    python scratch/gate155.py [device] [top|block9] [shift]   -> <out dir or boxes/runs_fsq/s155_gates>/gate155_<write>[_shift].json"""
import copy
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.multcol as MC  # noqa: E402
import boxes.multpair as MP  # noqa: E402
from boxes import s5  # noqa: E402


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"; wr = sys.argv[2] if len(sys.argv) > 2 else "top"; shift = len(sys.argv) > 3 and sys.argv[3] == "shift"
    odir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(ROOT, "boxes", "runs_fsq", "s155_gates")
    out = {"device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "mem_write": wr, "add_reset": "column" if shift else None}
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None; out["amp"] = str(amp)
    torch.manual_seed(0); rng = np.random.default_rng(0); B = 8; A, Bd = MC.gen_operands(rng, B, 4), MC.gen_operands(rng, B, 4)
    _, dig, _, pairs, col, emit = MP.accumulators(A, Bd); T = len(pairs)
    m = MP.MultDelta(s5.load_gpt2(dev), MP.PairFam(), 6, wr).to(dev); m.add_reset = "column" if shift else None; m.train(); MP.set_ops(m, A, Bd, dev); ok = True
    # (1)
    dmax = 0.0
    for gm in (0.0, 1.0):
        m.mem.fade.fill_(0.0); m.mem.gamma.fill_(gm)
        with torch.no_grad(), MP.amp_ctx(amp):
            S = m.start(B, dev)
            for t in range(T):
                lg, S = m.step(S, t); wm, lay = m.mem.write_mode, m.mem.layer; m.mem.write_mode, m.mem.layer = None, 99
                lg0, _ = m.step(m.start(B, dev), t); m.mem.write_mode, m.mem.layer = wm, lay; dmax = max(dmax, float((lg - lg0).abs().max()))
    out["g1_max_abs_dlogits_vs_plain"] = dmax; out["g1_pass"] = dmax == 0.0; ok &= out["g1_pass"]
    # (2)
    _, med0 = MP.read_ratio_by_handoff(m, A, Bd, dev, amp, n=B); m.mem.s_add.data.fill_(math.log(med0)); out["g2_g_add_init"] = float(med0)
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5); m.mem.gamma.fill_(0.5); m.zero_grad(set_to_none=True); MP.set_ops(m, A, Bd, dev)
    X = torch.tensor(np.tile(np.arange(T), (B, 1)), device=dev); tm = torch.tensor(np.tile(emit, (B, 1)), device=dev); lab = np.where(dig >= 0, dig, 0)
    with MP.amp_ctx(amp):
        loss, _, _, _ = m.chain(m.start(B, dev), X, lab, tm)
    loss.backward()
    gr = {n_: (float(p_.grad.float().norm()) if p_.grad is not None else 0.0) for n_, p_ in m.mem.named_parameters()}
    out["g2_grad"] = gr; out["g2_zero"] = [n_ for n_ in gr if gr[n_] == 0.0]; out["g2_pass"] = not out["g2_zero"]; ok &= out["g2_pass"]
    # (3) emission e = the last pair of column 3 (t = 9 in 4x4); LN_w live only at steps <= e - 2
    e = int(np.nonzero(emit)[0][3]); frozen = copy.deepcopy(m.mem.ln_w)
    for p_ in frozen.parameters():
        p_.requires_grad_(False)
    live = m.mem.ln_w; m.zero_grad(set_to_none=True); MP.set_ops(m, A, Bd, dev); ans = torch.tensor(np.array(MP.DTOK)[lab[:, e]], device=dev)
    with MP.amp_ctx(amp):
        S = m.start(B, dev)
        for t in range(e + 1):
            m.mem.ln_w = live if t <= e - 2 else frozen
            lg, S = m.step(S, t)
        Le = Fn.cross_entropy(lg, ans)
    Le.backward(); m.mem.ln_w = live
    out["g3_lnw_grad_writes_le_e_minus_2"] = float(live.weight.grad.float().norm()) if live.weight.grad is not None else 0.0
    gS = {}
    for gm in (0.5, 0.0):
        m.mem.gamma.fill_(gm); MP.set_ops(m, A, Bd, dev)
        with torch.no_grad(), MP.amp_ctx(amp):
            S = m.start(B, dev)
            for t in range(e - 1):
                _, S = m.step(S, t)
        S0 = S.detach().float().requires_grad_(True)
        with MP.amp_ctx(amp):
            S = S0
            for t in range(e - 1, e + 1):
                lg, S = m.step(S, t)
            Le = Fn.cross_entropy(lg, ans)
        gS[gm] = torch.autograd.grad(Le, S0)[0].detach()
    out["g3_dL_dS_norm_gamma_0.5"] = float(gS[0.5].norm()); out["g3_dL_dS_norm_gamma_0"] = float(gS[0.0].norm()); out["g3_dL_dS_diff_norm"] = float((gS[0.5] - gS[0.0]).norm())
    out["g3_pass"] = out["g3_lnw_grad_writes_le_e_minus_2"] > 0 and out["g3_dL_dS_diff_norm"] > 0; ok &= out["g3_pass"]
    # (4)
    m.mem.gamma.fill_(0.5); rr = MP.s155_recurrence(m, A, Bd, dev, amp, n=B, rows=200); out["g4_recurrence"] = rr; out["g4_pass"] = rr.get("rho_jvp") is not None; ok &= out["g4_pass"]
    out["all_pass"] = bool(ok); os.makedirs(odir, exist_ok=True); fn = os.path.join(odir, "gate155_%s%s.json" % (wr, "_shift" if shift else ""))
    json.dump(out, open(fn, "w", newline="\n"), indent=1); print("GATE155", json.dumps({k: v for k, v in out.items() if k != "g2_grad"}), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
