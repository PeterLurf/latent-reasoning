"""THEORY §162 DELTA-S5 pre-launch gates (9f7d4793), boxes/deltas5.py:
  (1) step-0 function: fade 0 -> a writer segment's logits under DeltaMem-R bitwise equal to plain GPT-2 (branch skipped)
  (2) gradflow (fade 0.5, W_o perturbed, gamma 0.5): every DeltaMem parameter (incl. LN_w) gets gradient from the final CE alone; the memory
      after the start segment and after item 1 (>= 2 hand-offs before the decoder) gets nonzero gradient
  (3) labels: make_batch's answers = the COMP chain (random start); bench S5=512.csv's targets = the COMP chain from the identity
  (4) the probes + edit on a synthetic memory that encodes g exactly (state one-hot in the first 120 coordinates, zeros elsewhere; writes apply COMP; decode =
      argmax): state_probe 1.0 at every p, the edit propagates (propagation_share 1.0), the random edit and the null do not
  (5) DIRECT: the teacher-forced loss path and greedy decode run; the prefix layout is [start text (6), 6 per item, " ="]
    python scratch/gate162.py [device]   -> boxes/runs_fsq/s162_gates/gate162.json (exit 1 on a failed gate)"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltas5 as D5  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.fsq120 import Family, make_batch  # noqa: E402


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; torch.set_num_threads(3); out = {"device": dev}; ok = True
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None; out["amp"] = str(amp)
    torch.manual_seed(0); rng = np.random.default_rng(0); g = s5.load_gpt2(dev); m = D5.DeltaS5(g).to(dev); B = 4
    s0, X, Y = make_batch(Family("s5"), rng, B, 3)
    # (1)
    m.mem.fade.fill_(0.0)
    with torch.no_grad(), D5.amp_ctx(amp):
        x = g.embed(m.ids_item(X[:, 0], dev)); S0 = m.start(B, dev); h1, _ = DM.gpt_forward_mem(g, x, m.mem, S0); h0, _ = g(x, None)
    out["g1_max_abs_dlogits_vs_plain"] = float((g.logits(h1).float() - g.logits(h0).float()).abs().max()); out["g1_pass"] = out["g1_max_abs_dlogits_vs_plain"] == 0.0; ok &= out["g1_pass"]
    # (2)
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5); m.mem.gamma.fill_(0.5); m.mem.s_add.data.fill_(np.log(D5.read_ratio(m, dev, amp, n=8, L=3))); m.train(); m.zero_grad(set_to_none=True)
    w0 = D5.DeltaS5.write; kept = []

    def wr(self, S, ids, cap=None):
        S2 = w0(self, S, ids, cap); S2.retain_grad(); kept.append(S2); return S2
    D5.DeltaS5.write = wr
    try:
        with D5.amp_ctx(amp):
            l_ = m.loss(m.run(s0, X, dev), Y[:, -1], dev)
        l_.backward()
    finally:
        D5.DeltaS5.write = w0
    gr = {n_: (float(p_.grad.float().norm()) if p_.grad is not None else 0.0) for n_, p_ in m.mem.named_parameters()}
    out["g2_grad"] = gr; out["g2_zero"] = [n_ for n_, v in gr.items() if v == 0.0]; out["g2_S_grad_norm"] = [float(k_.grad.float().norm()) if k_.grad is not None else 0.0 for k_ in kept]
    out["g2_pass"] = not out["g2_zero"] and min(out["g2_S_grad_norm"][:2]) > 0; ok &= out["g2_pass"]
    # (3)
    s0b, Xb, Yb = make_batch(Family("s5"), rng, 256, 16); Xt, Yt = s5.load_csv(os.path.join(ROOT, "bench", "data", "s5", "S5=512.csv"), limit=200)
    out["g3_batch_labels_comp"] = bool((Yb == D5.comp_chain(s0b, Xb)).all()); out["g3_test_labels_comp"] = bool((Yt == D5.comp_chain(np.zeros(len(Xt), np.int64), Xt)).all())
    out["g3_pass"] = out["g3_batch_labels_comp"] and out["g3_test_labels_comp"]; ok &= out["g3_pass"]
    # (4) synthetic memory: S_flat[:120] = one-hot(g), zeros elsewhere; write = COMP, decode = argmax
    d = 12 * DM.DH * DM.DH; nrng = np.random.default_rng(1)

    def enc(gs):
        f = np.zeros((len(gs), d), np.float32); f[np.arange(len(gs)), gs] = 1.0; return f   # exact (standardisation would blow i.i.d. noise up to unit scale)

    def run_fn(s0_, X_, keep):
        st = D5.comp_chain(s0_, X_); return {p: enc(st[:, p - 1]) for p in keep}

    def cont_fn(Sf, Xr):
        return D5.comp_chain(Sf[:, :120].argmax(1), Xr)[:, -1]
    pr = D5.probes(run_fn, [2, 4], dev, n_fit=2000, n_score=300)   # 2000: every one of the 120 states is seen in the fit
    se = D5.state_edit(run_fn, cont_fn, 8, dev, n=300, n_fit=2000)
    out["g4_probes"] = {"state_probe": pr["state_probe"], "op_probe_2": pr["op_probe"]["2"]}; out["g4_state_edit"] = se
    out["g4_pass"] = all(v == 1.0 for v in pr["state_probe"].values()) and se["propagation_share"] == 1.0 and se["n"] >= 250 and se["random_edit_share"] < 0.1 and se["null_value"] < 0.1
    ok &= out["g4_pass"]
    # (5) DIRECT
    md = D5.Direct(g).to(dev); pre = md.prefix(s0[:2], X[:2], dev)
    with D5.amp_ctx(amp):
        ld = md.loss(s0[:2], X[:2], Y[:2, -1], dev)
    ld.backward(); gd = md.greedy(s0[:2], X[:2], dev)
    out["g5_prefix_len"] = int(pre.shape[1]); out["g5_loss"] = float(ld); out["g5_greedy"] = gd.tolist()
    out["g5_pass"] = pre.shape[1] == 6 + 6 * X.shape[1] + 1 and np.isfinite(out["g5_loss"]) and len(gd) == 2; ok &= out["g5_pass"]
    out["all_pass"] = bool(ok); dd = os.path.join(ROOT, "boxes", "runs_fsq", "s162_gates"); os.makedirs(dd, exist_ok=True)
    json.dump(out, open(os.path.join(dd, "gate162.json"), "w", newline="\n"), indent=1, default=float)
    print("GATE162", json.dumps({k: v for k, v in out.items() if not k.endswith("_grad")}, default=float)[:1500], flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
