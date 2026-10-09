"""§177 (f) note 177-5 gate: the masked single-batch kmatch form equals the bucketed (per-K groups) form, CPU fp32.
One mixed batch: n = 1..8 x K in {n, n + 1, n + 2} (24 items; K_max = 10), InjectN with fade 0.5, W_o perturbed, gamma 0.5. Compared:
the loss and each item's final memory (relative to max |S|): rel < 1e-6; every parameter's gradient (norm-relative) within 3x max(1e-6, its noise floor) - the
SAME bucketed math regrouped one item per group (cancellation-sensitive trunk grads move ~1e-4 from grouping alone); the same with deltamem.FAST_ADD on.   python scratch/gate177_kmask.py -> boxes/runs_fsq/s177_gates/gate177_kmask.json"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes import s5  # noqa: E402


def main():
    torch.set_num_threads(3); torch.manual_seed(0); g = s5.load_gpt2("cpu"); m = N.InjectN(g, 6, "top")
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5); m.mem.gamma.fill_(0.5); m.train()
    rng = np.random.default_rng(17950); items, ks = [], []
    for n in range(1, 9):
        for dk in (0, 1, 2):
            r, t = N.generate(rng, n); items.append({"syms": N.render(r), "t": t, "root": r, "n": n}); ks.append(n + dk)
    ac = lambda: torch.autocast("cuda", enabled=False)  # noqa: E731
    out = {"n_items": len(items), "k_max": max(ks)}; ok = True
    for fa in (False, True):
        DM.FAST_ADD = fa; res = {}
        for nm in ("masked", "bucketed"):
            m.zero_grad(set_to_none=True); loss, S = N.kmatch_loss(m, items, ks, "cpu", ac, masked=nm == "masked"); loss.backward()
            res[nm] = (float(loss), S.detach().clone(), {k: (p.grad.detach().clone() if p.grad is not None else None) for k, p in m.named_parameters()})
        (la, Sa, ga), (lb, Sb, gb) = res["masked"], res["bucketed"]
        # the noise floor: the SAME bucketed math regrouped (one item per group) - some trunk grads (e.g. h.4 ln_2) are cancellation-sensitive to the
        # accumulation grouping at ~1e-4 (norm-relative), so a flat 1e-6 is below fp32 resolution for them
        tgt_ = torch.tensor(N.NUM_IDS[[it["t"] for it in items]]); m.zero_grad(set_to_none=True); tot_ = 0
        for i in range(len(items)):
            x_, mk_ = m.think_inputs([items[i]], "cpu"); S_ = m.think(m.mem.zero_state(1, "cpu"), x_, mk_, ks[i])
            tot_ = tot_ + torch.nn.functional.cross_entropy(m.dec_logits(S_, [items[i]], "cpu"), tgt_[i:i + 1]) / len(items)
        tot_.backward(); gs = {k: (p.grad.detach().clone() if p.grad is not None else None) for k, p in m.named_parameters()}
        rel = lambda x, y: float((x - y).norm() / y.norm().clamp_min(1e-30))  # noqa: E731
        gr, fl = {}, {}
        for k in ga:
            if ga[k] is None and gb[k] is None:
                continue
            if (ga[k] is None) != (gb[k] is None):
                gr[k] = float("inf"); fl[k] = 0.0; continue
            gr[k] = rel(ga[k], gb[k]); fl[k] = rel(gs[k], gb[k])
        s_rel = float((Sa - Sb).abs().max() / Sb.abs().max()); over = {k: (gr[k], fl[k]) for k in gr if gr[k] > 3 * max(1e-6, fl[k])}
        worst = max(gr, key=gr.get) if gr else None
        r_ = {"loss_masked": la, "loss_bucketed": lb, "loss_rel": abs(la - lb) / abs(lb), "state_rel": s_rel, "state_bitwise": bool(torch.equal(Sa, Sb)),
              "grad_relnorm_max": gr.get(worst), "grad_worst_param": worst, "noise_floor_worst_param": fl.get(worst), "noise_floor_max": max(fl.values()) if fl else None,
              "params_over_3x_floor": over, "n_params_with_grad": len(gr)}
        r_["pass"] = bool(r_["loss_rel"] < 1e-6 and s_rel < 1e-6 and not over); out["fast_add" if fa else "loop"] = r_; ok &= r_["pass"]
    DM.FAST_ADD = False; out["all_pass"] = bool(ok); gd = os.path.join(ROOT, "boxes", "runs_fsq", "s177_gates"); os.makedirs(gd, exist_ok=True)
    json.dump(out, open(os.path.join(gd, "gate177_kmask.json"), "w", newline="\n"), indent=1); print("GATE177_KMASK", json.dumps(out), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
