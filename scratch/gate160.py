"""THEORY §160 R-ROWS pre-launch gates (785c3838): DeltaMem-R's write source (--mem_write top|block9) on boxes.multrow's MultRowDelta.
  (0) §152's path unchanged: a plain MultRowDelta (no --mem_write) built and run in the patched module and in a reference copy (MULTROW_REF,
      default boxes._multrow_head_tmp = git show HEAD:boxes/multrow.py) -> chain_read losses and every gradient bitwise equal
  (1) step-0 function preservation: fade 0 -> a writer segment's logits under DeltaMem-R bitwise equal to plain GPT-2 (branch skipped)
  (2) gamma 0: DeltaMem-R's states after segments 1..4 and the reader's final logits equal §152's (the same weights; writes from LN_6(h_6) only)
  (3) gradient flow (fade 0.5, zero-init maps perturbed, gamma 0.5): every DeltaMem parameter (incl. LN_w) from the final CE alone; the gradient
      on S_1 (3 hand-offs before the final read) and S_2 nonzero; and S_1's gradient differs between gamma 0.5 and gamma 0 (the read -> write path)
  (4) the reader writes nothing back: states into each segment and the final logits bitwise identical with the reader passes on / off
    python scratch/gate160.py [device] [write_mode]   -> boxes/runs_fsq/s160_gates/gate160_<mode>.json (exit 1 on a failed gate)"""
import importlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

MR = importlib.import_module(os.environ.get("MULTROW_MOD", "boxes.multrow"))
import boxes.deltamem as DM  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.multcol import DTOK, gen_operands  # noqa: E402


def batch(MR_, m, A, Bd, dev):
    MR_.set_ops(m, A, Bd, dev); B = len(A); X = torch.tensor(np.tile(np.arange(4), (B, 1)), device=dev); lab = MR_.sj_values(A, Bd)
    rtok = torch.tensor(np.array(DTOK)[MR_.digits8(lab[:, :3])], device=dev); rmask = torch.ones(B, 3, dtype=torch.bool, device=dev); return X, lab, rtok, rmask


def perturb(m):
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5)


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; mode = sys.argv[2] if len(sys.argv) > 2 else "top"
    out = {"device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "write_mode": mode, "module": MR.__name__}; ok = True
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None; out["amp"] = str(amp)
    rng = np.random.default_rng(0); B = 8; A, Bd = gen_operands(rng, B, 4), gen_operands(rng, B, 4); g = s5.load_gpt2(dev)
    # (0) §152's path in the patched module vs the reference copy
    REF = importlib.import_module(os.environ.get("MULTROW_REF", "boxes._multrow_head_tmp")); res = []
    torch.use_deterministic_algorithms(True, warn_only=True)               # the wte embedding backward is nondeterministic otherwise (REF vs REF differs)
    for M_ in (MR, REF):
        torch.manual_seed(0); m = M_.MultRowDelta(g, M_.RowFam(), 6).to(dev); m.train(); perturb(m); X, lab, rtok, rmask = batch(M_, m, A, Bd, dev); m.zero_grad(set_to_none=True)
        with M_.amp_ctx(amp):
            fin, rd, _ = m.chain_read(m.start(B, dev), X, lab, rtok, rmask)
        (fin + rd).backward(); res.append((float(fin.detach()), float(rd.detach()), {n_: p_.grad.clone() for n_, p_ in m.named_parameters() if p_.grad is not None})); m.zero_grad(set_to_none=True)
    torch.use_deterministic_algorithms(False); out["g0_losses"] = [res[0][:2], res[1][:2]]; out["g0_grad_names_equal"] = sorted(res[0][2]) == sorted(res[1][2])
    out["g0_pass"] = res[0][:2] == res[1][:2] and out["g0_grad_names_equal"] and all(torch.equal(res[0][2][k], res[1][2][k]) for k in res[0][2]); ok &= out["g0_pass"]
    # (1) step 0 under R
    torch.manual_seed(0); m = MR.MultRowDelta(g, MR.RowFam(), 6).to(dev); m.mem = DM.DeltaMem(g, 6, mode).to(dev); m.train(); MR.set_ops(m, A, Bd, dev)
    m.mem.fade.fill_(0.0)
    with torch.no_grad(), MR.amp_ctx(amp):
        x = g.embed(m.seg_ids(0, B, dev)); S0 = m.start(B, dev); h1, _ = MR.fwd_mem(g, x, m.mem, S0)
        wm = m.mem.write_mode; m.mem.write_mode = None; lay = m.mem.layer; m.mem.layer = 99; h0, _ = MR.fwd_mem(g, x, m.mem, S0); m.mem.layer = lay; m.mem.write_mode = wm
    out["g1_max_abs_dlogits_vs_plain"] = float((g.logits(h1).float() - g.logits(h0).float()).abs().max()); out["g1_pass"] = out["g1_max_abs_dlogits_vs_plain"] == 0.0; ok &= out["g1_pass"]
    # (2) gamma 0: R's states / reader logits vs §152's with the same weights
    _, med0 = MR.read_ratio_drow(m, A, Bd, dev, amp, n=B); m.mem.s_add.data.fill_(math.log(med0)); out["g_add_init"] = float(med0); perturb(m)
    torch.manual_seed(1); mp = MR.MultRowDelta(g, MR.RowFam(), 6).to(dev); mp.load_state_dict(m.state_dict(), strict=False); mp.mem.fade.fill_(0.5); m.eval(); mp.eval()
    m.mem.gamma.fill_(0.0); st = {}
    with torch.no_grad(), MR.amp_ctx(amp):
        for nm, mm in (("r", m), ("p", mp)):
            MR.set_ops(mm, A, Bd, dev); S = mm.start(B, dev); st[nm] = []
            for t in range(4):
                S = mm.write(S, t); st[nm].append(S.clone())
            st[nm].append(mm.read_logits(S, None).float())
    out["g2_max_abs_dstate"] = [float((a_.float() - b_.float()).abs().max()) for a_, b_ in zip(st["r"][:4], st["p"][:4])]
    out["g2_max_abs_dlogits"] = float((st["r"][4] - st["p"][4]).abs().max()); out["g2_pass"] = max(out["g2_max_abs_dstate"]) == 0.0 and out["g2_max_abs_dlogits"] == 0.0; ok &= out["g2_pass"]
    # (3) gradient flow at gamma 0.5 (and S_1's gradient at gamma 0 for the read -> write path)
    m.train(); X, lab, rtok, rmask = batch(MR, m, A, Bd, dev); w0 = MR.MultRowDelta.write; gS = {}
    for gam in (0.5, 0.0):
        m.mem.gamma.fill_(gam); m.zero_grad(set_to_none=True); kept = []

        def wr(self, S, t, cap=None):
            S2 = w0(self, S, t, cap); S2.retain_grad(); kept.append(S2); return S2
        MR.MultRowDelta.write = wr
        try:
            with MR.amp_ctx(amp):
                fin, _, _ = m.chain_read(m.start(B, dev), X, lab, None, None)
            fin.backward()
        finally:
            MR.MultRowDelta.write = w0
        gS[gam] = [k_.grad.float().clone() if k_.grad is not None else None for k_ in kept]
        if gam == 0.5:
            gr = {n_: (float(p_.grad.float().norm()) if p_.grad is not None else 0.0) for n_, p_ in m.mem.named_parameters()}
            out["g3_final_grad"] = gr; out["g3_zero"] = [n_ for n_, v in gr.items() if v == 0.0]
            out["g3_S_grad_norm"] = [float(x_.norm()) if x_ is not None else 0.0 for x_ in gS[gam]]
    out["g3_S1_grad_diff_gamma"] = float((gS[0.5][0] - gS[0.0][0]).norm()) if gS[0.5][0] is not None and gS[0.0][0] is not None else 0.0
    out["g3_pass"] = not out["g3_zero"] and min(out["g3_S_grad_norm"][:2]) > 0 and out["g3_S1_grad_diff_gamma"] > 0; ok &= out["g3_pass"]
    # (4) the reader writes nothing back (gamma 0.5)
    m.eval(); m.mem.gamma.fill_(0.5)
    with torch.no_grad(), MR.amp_ctx(amp):
        def run(with_reader):
            MR.set_ops(m, A, Bd, dev); S = m.start(B, dev); st_ = []; same = True
            for t in range(4):
                S = m.write(S, t); st_.append(S.clone())
                if with_reader:
                    Sc = S.clone(); m.read_logits(S, rtok[:, min(t, 2)]); m.read_greedy(S); same &= bool(torch.equal(S, Sc))
            return st_, m.read_logits(S, None), same
        st_on, lg_on, unchanged = run(True); st_off, lg_off, _ = run(False)
    out["g4_states_bitwise_equal"] = all(bool(torch.equal(a_, b_)) for a_, b_ in zip(st_on, st_off)); out["g4_final_logits_bitwise_equal"] = bool(torch.equal(lg_on, lg_off))
    out["g4_pass"] = out["g4_states_bitwise_equal"] and out["g4_final_logits_bitwise_equal"] and unchanged; ok &= out["g4_pass"]
    out["all_pass"] = bool(ok); d = os.path.join(ROOT, "boxes", "runs_fsq", "s160_gates"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "gate160_%s.json" % mode), "w", newline="\n"), indent=1)
    print("GATE160", json.dumps({k: v for k, v in out.items() if not k.endswith("_grad")}), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
