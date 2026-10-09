"""Gate for deltamem.FAST_ADD (Audit 2026-10-09): in R mode the reads see only the pass-start state, so the additive heads' writes within a pass commute
and are summed in one einsum; the gated-delta heads keep the token-sequential update. Checks, FAST_ADD on vs off (fp32):
  (1) write_seq on real GPT-2 activations (h6 / h_top of random-token inputs), T = 16 and 64, from a non-zero start state: max abs diff on S < 1e-5 OR
      relative to max |S| < 1e-6 (|S| reaches ~1.8e3 here, where the fp32 ulp is ~1.2e-4, so an absolute 1e-5 is below resolution), with no mask and with
      a right-padded mask; a masked row equals writing its unpadded prefix alone (relative < 1e-6); the delta heads' block bitwise equal
  (2) train mode: gradients of every DeltaMem parameter through 2 passes + a read agree (max abs diff relative to the grad's max < 1e-4)
  (3) end to end: inject169's Inject model (fade 1, W_o / gamma perturbed), K = 4 think passes on 4x5 items -> decoder logits: max abs diff < 1e-4 and
      identical argmax; §170 InjectN (right-padded, masked) the same
  (4) timing (descriptive): one write_seq at T = 16, B = 64, forward + backward, both modes
    python scratch/gate_fastadd.py [device]   -> boxes/runs_fsq/fastadd_gates/gate_fastadd.json"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
from boxes import s5  # noqa: E402


def both(fn):
    DM.FAST_ADD = False; a = fn(); DM.FAST_ADD = True
    try:
        b = fn()
    finally:
        DM.FAST_ADD = False
    return a, b


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; torch.set_num_threads(3); torch.manual_seed(0); out = {"device": dev}; ok = True
    g = s5.load_gpt2(dev); mem = DM.DeltaMem(g, 6, "top").to(dev)
    with torch.no_grad():
        mem.wo.weight.normal_(0, 0.02); mem.gamma.fill_(0.5); mem.fade.fill_(1.0); mem.gate.weight.normal_(0, 0.01)
    # (1)
    r1 = {}
    for T in (16, 64):
        x = g.embed(torch.randint(0, 50000, (4, T), device=dev)); S0 = torch.zeros(4, 12, 64, 64, device=dev)
        with torch.no_grad():
            for _ in range(2):
                cap = {}; DM.gpt_forward_mem_r(g, x, mem, S0, cap=cap); S0 = DM.write_seq(mem, cap["h6"], cap["hw"], S0)
            cap = {}; DM.gpt_forward_mem_r(g, x, mem, S0, cap=cap); h6, hw = cap["h6"], cap["hw"]
            msk = torch.ones(4, T, dtype=torch.bool, device=dev); msk[1, T // 2:] = False; msk[3, T - 3:] = False
            for nm, m_ in (("nomask", None), ("mask", msk)):
                a, b = both(lambda: DM.write_seq(mem, h6, hw, S0, m_))
                d = float((a - b).abs().max()); r1["T%d_%s" % (T, nm)] = {"max_abs": d, "rel": d / float(a.abs().max()), "delta_block_equal": bool(torch.equal(a[:, 6:], b[:, 6:])),
                                                                         "max_abs_S": float(a.abs().max())}
            Sm = DM.write_seq(mem, h6[1:2, :T // 2], hw[1:2, :T // 2], S0[1:2])   # the masked row equals writing its unpadded prefix alone
            DM.FAST_ADD = True
            try:
                Sf = DM.write_seq(mem, h6, hw, S0, msk)
            finally:
                DM.FAST_ADD = False
            r1["T%d_mask_vs_prefix_rel" % T] = float((Sf[1] - Sm[0]).abs().max() / Sm[0].abs().max())
    out["g1"] = r1; out["g1_pass"] = all(v["max_abs"] < 1e-5 or v["rel"] < 1e-6 for k, v in r1.items() if isinstance(v, dict)) and all(
        v["delta_block_equal"] for v in r1.values() if isinstance(v, dict)) and all(v < 1e-6 for k, v in r1.items() if not isinstance(v, dict)); ok &= out["g1_pass"]
    # (2)
    x = g.embed(torch.randint(0, 50000, (4, 16), device=dev)).detach()

    def grads():
        mem.zero_grad(set_to_none=True); S = torch.zeros(4, 12, 64, 64, device=dev)
        for _ in range(2):
            _, S = DM.gpt_forward_mem_r(g, x, mem, S)
        h, _ = DM.gpt_forward_mem_r(g, x, mem, S); h.pow(2).mean().backward()
        return {n_: p_.grad.detach().clone() for n_, p_ in mem.named_parameters() if p_.grad is not None}
    ga, gb = both(grads); g.zero_grad(set_to_none=True); mem.zero_grad(set_to_none=True)
    out["g2"] = {n_: float((ga[n_] - gb[n_]).abs().max() / ga[n_].abs().max().clamp_min(1e-30)) for n_ in ga}; out["g2_missing"] = sorted(set(ga) ^ set(gb))
    out["g2_pass"] = not out["g2_missing"] and max(out["g2"].values()) < 1e-4; ok &= out["g2_pass"]
    # (3)
    from boxes.inject169 import Inject
    from boxes.multcol import gen_operands
    from boxes.fixedchunk import DT, product_digits
    m = Inject(g, 6, "top").to(dev); m.mem = mem; rng = np.random.default_rng(1); A, B = gen_operands(rng, 8, 4), gen_operands(rng, 8, 5)
    ans = torch.tensor(DT[product_digits(A, B)], device=dev)
    with torch.no_grad():
        la, lb = both(lambda: m.dec_logits(m.think(mem.zero_state(8, dev), A, B, 4), A, B, ans))
    out["g3_inject"] = {"max_abs": float((la - lb).abs().max()), "argmax_equal": bool(torch.equal(la.argmax(-1), lb.argmax(-1)))}
    import boxes.nested170 as N
    mn = N.InjectN(g, 6, "top").to(dev); mn.mem = mem; its = N.train_items(np.random.default_rng(2), 8, 8, set())
    with torch.no_grad():
        xn, mk = mn.think_inputs(its, dev); na, nb = both(lambda: mn.dec_logits(mn.think(mem.zero_state(8, dev), xn, mk, 4), its, dev))
    out["g3_nested"] = {"max_abs": float((na - nb).abs().max()), "argmax_equal": bool(torch.equal(na.argmax(-1), nb.argmax(-1))), "padded": bool((~mk).any())}
    out["g3_pass"] = all(v["max_abs"] < 1e-4 and v["argmax_equal"] for v in (out["g3_inject"], out["g3_nested"])); ok &= out["g3_pass"]
    # (4)
    h6 = torch.randn(64, 16, 768, device=dev, requires_grad=True); hw = torch.randn(64, 16, 768, device=dev); S0 = torch.zeros(64, 12, 64, 64, device=dev); tm = {}
    for fa in (False, True):
        DM.FAST_ADD = fa; ts = []
        for _ in range(4):
            if dev.startswith("cuda"):
                torch.cuda.synchronize(dev)
            t0 = time.time(); DM.write_seq(mem, h6, hw, S0).sum().backward()
            if dev.startswith("cuda"):
                torch.cuda.synchronize(dev)
            ts.append(time.time() - t0)
        tm["fast" if fa else "loop"] = float(np.median(ts[1:]))
    DM.FAST_ADD = False; out["g4_sec_fwd_bwd_T16_B64"] = tm
    out["all_pass"] = bool(ok); dd = os.path.join(ROOT, "boxes", "runs_fsq", "fastadd_gates"); os.makedirs(dd, exist_ok=True)
    json.dump(out, open(os.path.join(dd, "gate_fastadd_%s.json" % ("gpu" if dev.startswith("cuda") else "cpu")), "w", newline="\n"), indent=1)
    print("GATE_FASTADD", json.dumps(out), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
