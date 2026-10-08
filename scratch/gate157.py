"""THEORY §157 READ-PANEL pre-launch gates (c13b78da + note 157-1), per arm (periodic | multiread | ccq):
  (1) step-0 function preservation: fade 0 (and MULTIREAD's read-token fade 0) -> a segment's logits bitwise equal to plain GPT-2 (branch skipped)
  (2) gradient flow into the new read parameters and every DeltaMem parameter (zero-init maps perturbed, fades 0.5, g_add calibrated): from the
      reader CE alone and from the final CE alone
  (3) the reader writes nothing back: states into each segment and the final logits bitwise identical with the reader passes on / off
    python scratch/gate157.py [device] [arm]   -> boxes/runs_fsq/s157_gates/gate157_<arm>.json (exit 1 on a failed gate)
MULTROW_MOD=boxes._multrow157_tmp runs it against a staged copy."""
import importlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

MR = importlib.import_module(os.environ.get("MULTROW_MOD", "boxes.multrow"))
from boxes import s5  # noqa: E402
from boxes.multcol import DTOK, gen_operands  # noqa: E402


def build(dev, arm):
    m = MR.MultRowDelta(s5.load_gpt2(dev), MR.RowFam(), 6).to(dev)
    if arm == "periodic":
        m.mem = MR.PeriodicMem(m.g, 6).to(dev)
    elif arm == "multiread":
        m.mem.mr = MR.MultiRead(dev)
    elif arm == "ccq":
        m.mem = MR.CCQMem(m.g, 6).to(dev)
    return m


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"; arm = sys.argv[2] if len(sys.argv) > 2 else "periodic"
    out = {"device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "arm": arm, "module": MR.__name__}; ok = True
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None; out["amp"] = str(amp)
    torch.manual_seed(0); rng = np.random.default_rng(0); B = 8; A, Bd = gen_operands(rng, B, 4), gen_operands(rng, B, 4)
    m = build(dev, arm); m.train(); MR.set_ops(m, A, Bd, dev); g = m.g
    # (1)
    m.mem.fade.fill_(0.0)
    if getattr(m.mem, "mr", None) is not None:
        m.mem.mr.fade.fill_(0.0)
    with torch.no_grad(), MR.amp_ctx(amp):
        x = g.embed(m.seg_ids(0, B, dev)); S0 = m.start(B, dev); h1, _ = MR.fwd_mem(g, x, m.mem, S0)
        lay = m.mem.layer; m.mem.layer = 99; h0, _ = MR.fwd_mem(g, x, m.mem, S0); m.mem.layer = lay
    out["g1_max_abs_dlogits_vs_plain"] = float((g.logits(h1).float() - g.logits(h0).float()).abs().max()); out["g1_pass"] = out["g1_max_abs_dlogits_vs_plain"] == 0.0; ok &= out["g1_pass"]
    # (2)
    _, med0 = MR.read_ratio_drow(m, A, Bd, dev, amp, n=B); m.mem.s_add.data.fill_(math.log(med0)); out["g2_g_add_init"] = float(med0)
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
        if arm == "periodic":
            m.mem.pm.weight.normal_(0, 0.02)
        if arm == "multiread":
            m.mem.mr.Z.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5)
    if getattr(m.mem, "mr", None) is not None:
        m.mem.mr.fade.fill_(0.5)
    MR.set_ops(m, A, Bd, dev); X = torch.tensor(np.tile(np.arange(4), (B, 1)), device=dev); lab = MR.sj_values(A, Bd)
    rtok = torch.tensor(np.array(DTOK)[MR.digits8(lab[:, :3])], device=dev); rmask = torch.ones(B, 3, dtype=torch.bool, device=dev)
    for term in ("reader", "final"):
        m.zero_grad(set_to_none=True)
        with MR.amp_ctx(amp):
            fin, rd, nrd = m.chain_read(m.start(B, dev), X, lab, rtok, rmask)
        (rd if term == "reader" else fin).backward()
        gr = {n_: (float(p_.grad.float().norm()) if p_.grad is not None else 0.0) for n_, p_ in m.mem.named_parameters()}
        out["g2_%s_grad" % term] = gr; out["g2_%s_zero" % term] = [n_ for n_, v in gr.items() if v == 0.0]
        out["g2_%s_pass" % term] = not out["g2_%s_zero" % term]; ok &= out["g2_%s_pass" % term]
    # (3)
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
    out["g3_pass"] = out["g3_states_bitwise_equal"] and out["g3_final_logits_bitwise_equal"] and unchanged; ok &= out["g3_pass"]
    out["state_shape"] = list(st_on[0].shape); out["all_pass"] = bool(ok); d = os.path.join(ROOT, "boxes", "runs_fsq", "s157_gates"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "gate157_%s.json" % arm), "w", newline="\n"), indent=1)
    print("GATE157", json.dumps({k: v for k, v in out.items() if not k.endswith("_grad")}), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
