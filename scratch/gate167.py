"""THEORY §167 DAS-EDIT pre-launch gates (c08fbbd0 + note 167-1), boxes/das167.py, on real kept models (small n, CPU or a local card):
  (1) MULT: the counterfactual generator = donor_delta's expected digits (multpair.later_gt on the donor's accumulator, as donor_delta), and the k = D
      patch (whole-state swap) through das167's continuation gives row-level IIA = the whole-state SERIAL share computed independently with
      multpair.rollout_delta(override=(t, donor states)) on the same pairs; also the per-row decodes agree exactly
  (2) S5: the k = D IIA = deltas5.rollout(override at p) final-state agreement with g_don o later items on the same pairs
  (3) a fitted k = 64 basis trains (loss falls) and R is orthonormal (|R^T R - I| < 1e-4)
    python scratch/gate167.py <mult run> <s5 run> [device]   -> boxes/runs_fsq/s167_gates/gate167.json"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.das167 as DS  # noqa: E402


def main():
    mrun, srun = sys.argv[1], sys.argv[2]; dev = sys.argv[3] if len(sys.argv) > 3 else "cpu"; torch.set_num_threads(3); out = {"device": dev}; ok = True
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None; n = 48
    # (1) MULT
    F = DS.MultFam(mrun, dev, amp); B1, B2 = F.sets["B"]; B1, B2 = B1[:400], B2[:400]; r1 = {}
    for t in F.points():
        SB = F.states(B1, B2, t); ir, idn, gt, sl, Pr = F.make_pairs(B1, B2, t, np.random.default_rng(1), n)
        iia_k = DS.evaluate(F, (B1, B2), SB, SB, ir, idn, gt, t, DS.FixedRot())
        kept = SB.reshape((len(B1),) + DS.SH); got, _, _ = F.MP.rollout_delta(F.m, B1[ir], B2[ir], dev, amp, 64, override=(t, kept[idn]))
        acc = F.MP.accumulators(B1, B2)[0]; gt2, sl2 = F.MP.later_gt(B1[ir], B2[ir], acc[idn, t - 1], t, F.pairs, F.col, F.emit, F.MP.columns(B1[ir], B2[ir])[0])
        serial = float((got[:, sl] == gt[:, sl]).all(1).mean())
        r1[str(t)] = {"iia_kD": iia_k, "serial_rollout_override": serial, "gt_equal_later_gt": bool((gt == gt2).all() and (sl == sl2).all()), "n": int(len(ir))}
    out["g1"] = r1; out["g1_pass"] = all(v["gt_equal_later_gt"] and abs(v["iia_kD"] - v["serial_rollout_override"]) < 1e-9 for v in r1.values()); ok &= out["g1_pass"]
    # (3) on MULT t = 8: a k = 64 fit for 40 steps
    t = 8; A1, A2 = F.sets["A"]; A1, A2 = A1[:600], A2[:600]; SA = F.states(A1, A2, t); tr = F.make_pairs(A1[:500], A2[:500], t, np.random.default_rng(2), 400)[:3]
    va_ir, va_idn, va_gt, _, _ = F.make_pairs(A1[500:], A2[500:], t, np.random.default_rng(3), 64); va = (va_ir + 500, va_idn + 500, va_gt)
    torch.manual_seed(0); rot = DS.Rot(64, dev); opt = torch.optim.Adam(rot.parameters(), lr=1e-3); ls = []
    for st in range(40):
        b = np.random.default_rng(st).integers(0, len(tr[0]), 32); Sr = torch.tensor(SA[tr[0][b]], device=dev); Sd = torch.tensor(SA[tr[1][b]], device=dev)
        o_, c_ = F.cont_logits(A1[tr[0][b]], A2[tr[0][b]], rot.patch(Sr, Sd), t); loss = F.loss(o_, c_, tr[2][b]); opt.zero_grad(); loss.backward(); opt.step(); ls.append(float(loss))
    R = rot.R().detach(); orth = float((R.T @ R - torch.eye(64, device=dev)).abs().max())
    out["g3"] = {"loss_first5": float(np.mean(ls[:5])), "loss_last5": float(np.mean(ls[-5:])), "orth_err": orth}; out["g3_pass"] = out["g3"]["loss_last5"] < out["g3"]["loss_first5"] and orth < 1e-4; ok &= out["g3_pass"]
    del F
    # (2) S5
    G = DS.S5Fam(srun, dev, amp); s0, X = G.sets["B"]; s0, X = s0[:400], X[:400]; p = G.points()[0]; SB = G.states(s0, X, p)
    ir, idn, gt, sl, own = G.make_pairs(s0, X, p, np.random.default_rng(4), n); iia_k = DS.evaluate(G, (s0, X), SB, SB, ir, idn, gt, p, DS.FixedRot())
    Sd = SB[idn].reshape((len(ir),) + DS.SH); fin, _ = G.D5.rollout(G.m, s0[ir], X[ir], dev, amp, 64, every=False, override=(p, lambda S, r_: torch.tensor(Sd[r_], device=dev, dtype=S.dtype)))
    serial = float((fin == gt[:, 0]).mean()); out["g2"] = {"iia_kD": iia_k, "serial_rollout_override": serial, "n": int(len(ir)), "p": p}
    out["g2_pass"] = abs(iia_k - serial) < 1e-9; ok &= out["g2_pass"]
    out["all_pass"] = bool(ok); dd = os.path.join(ROOT, "boxes", "runs_fsq", "s167_gates"); os.makedirs(dd, exist_ok=True)
    json.dump(out, open(os.path.join(dd, "gate167.json"), "w", newline="\n"), indent=1); print("GATE167", json.dumps(out), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
