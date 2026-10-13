"""note 192-1 (Theory 77757876) PLATEAU read, eval-only, descriptive: 256 fixed test items with n in {6, 7, 8} (rng 19201 pick, the same for every
model), passes t = 1..8 from M_0 = 0. Per pass t:
  repeat(t)   = share of items whose aux HANDLE prediction at t equals the one at t - 1 (t >= 2; the deepaux call: teacher-forced m.aux_logits at pass t,
                argmax at the handle index of step t; items with t <= n only, i.e. both t - 1 and t have a target step)
  collapse_t (192-1b, PRIMARY) = mean over items of the mean pairwise cosine of that item's reads r_t, t = 3..n (t != t')
  collapse_pass(t) (192-1a) = mean over items of cos(r_t, r_{t-1}) for each item (t >= 2)
  collapse_items(t) = mean pairwise cosine over the items of the additive read r_t at the main decoder's read position (" =": ReadTap + read_vec's additive
                part, post-normalisation, as sizeprobe)
    python scratch/plateau192.py <run dir> [device]   -> <run>/eval_s192_plateau.json"""
import json
import os
import sys
from contextlib import nullcontext

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.halt173 as H  # noqa: E402
from boxes.nested170 import test_set, aux_of, ids, TID, EOT as NEOT  # noqa: E402


@torch.no_grad()
def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"; kind, m, a = H.load_model(run, dev)
    amp = None if (not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ctx = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else nullcontext
    TE = test_set(a.get("per_n", 1000)); pool = [it for it in TE if it["n"] in (6, 7, 8)]; its = [pool[i] for i in np.random.default_rng(19201).permutation(len(pool))[:256]]
    ha = getattr(m.mem, "h_add", H.DM.H_ADD); T = 8; bs = 128
    hpred = {t: {} for t in range(1, T + 1)}; htrue = {t: {} for t in range(1, T + 1)}; reads = {t: [] for t in range(1, T + 1)}
    for i in range(0, len(its), bs):
        ch = its[i:i + bs]; x, msk = m.think_inputs(ch, dev)
        seqs = [[NEOT] + ids(it["syms"]) + [TID[" ="]] for it in ch]; rpos = torch.tensor([len(q) - 1 for q in seqs], device=dev)
        with ctx():
            S = m.mem.zero_state(len(ch), dev)
            for t in range(1, T + 1):
                S = m.write(S, x, msk)
                with H.ReadTap() as tap:
                    tap.arm(); m.dec_logits(S, ch, dev)
                assert tap.x is not None, "PLATEAU: the read tap did not fire"
                xr = tap.x[torch.arange(len(ch), device=dev), rpos]; reads[t].append(H.read_vec(m.mem, xr, S)[:, :ha * H.DM.DH].float().cpu())
                idx = [j for j, it in enumerate(ch) if len(aux_of(it)) >= t]
                if idx:
                    lg_a, y_a = m.aux_logits(S[idx], [ch[j] for j in idx], t, dev); pr = lg_a.argmax(-1).cpu().numpy(); yy = y_a.cpu().numpy()
                    for r_, j in enumerate(idx):
                        hi_ = aux_of(ch[j])[t - 1][1]; hpred[t][i + j] = int(pr[r_, hi_]); htrue[t][i + j] = int(yy[r_, hi_])
    by_t = {}
    for t in range(1, T + 1):
        R = torch.cat(reads[t]); Rn = R / R.norm(dim=1, keepdim=True).clamp_min(1e-12); C = Rn @ Rn.T; n_ = len(R)
        col = float((C.sum() - C.diagonal().sum()) / (n_ * (n_ - 1)))
        both = [k for k in hpred[t] if k in hpred.get(t - 1, {})] if t >= 2 else []
        rep = float(np.mean([hpred[t][k] == hpred[t - 1][k] for k in both])) if both else None
        rep_true = float(np.mean([htrue[t][k] == htrue[t - 1][k] for k in both])) if both else None   # the data's base rate (correct handles repeating)
        hacc = float(np.mean([hpred[t][k] == htrue[t][k] for k in hpred[t]])) if hpred[t] else None
        cp_ = None
        if t >= 2:                                                          # amendment 192-1a: cos(r_t, r_{t-1}) per item, mean over items
            Rp = torch.cat(reads[t - 1]); Rpn = Rp / Rp.norm(dim=1, keepdim=True).clamp_min(1e-12); cp_ = float((Rn * Rpn).sum(1).mean())
        by_t[str(t)] = {"repeat": rep, "repeat_truth": rep_true, "handle_acc": hacc, "collapse_items": col, "collapse_pass": cp_, "n_repeat_items": len(both), "n_read_items": n_}
    Rall = torch.stack([torch.cat(reads[t]) for t in range(1, T + 1)], 1); Rall = Rall / Rall.norm(dim=2, keepdim=True).clamp_min(1e-12)   # (items, T, D)
    ct_ = []                                                                # amendment 192-1b (primary): per item, mean pairwise cos of r_t over t = 3..n, t != t'
    for k_, it in enumerate(its):
        V = Rall[k_, 2:it["n"]]; G = V @ V.T; m_ = len(V)
        ct_.append(float((G.sum() - G.diagonal().sum()) / (m_ * (m_ - 1))))
    collapse_t = float(np.mean(ct_))
    out = {"collapse_t": collapse_t, "collapse_t_by_n": {str(n): float(np.mean([c for c, it in zip(ct_, its) if it["n"] == n])) for n in (6, 7, 8)}, "by_t": by_t, "n_items": len(its), "n_by_n": {str(n): sum(1 for it in its if it["n"] == n) for n in (6, 7, 8)}, "run": run, "kind": kind,
           "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "eval_dtype": str(amp), "nopass": bool(getattr(m, "nopass", False)),
           "def": "items: test n in {6,7,8}, rng 19201 permutation[:256]; passes from M_0 = 0. repeat(t): handle-token argmax (teacher-forced aux_logits at "
                  "pass t, the step-t handle index) == the pass t-1 prediction, over items with t <= n; collapse_t: per item mean pairwise cos of r_3..r_n, averaged; collapse_pass(t): mean cos(r_t, r_t-1) per item; collapse_items(t): mean off-diagonal cosine of the "
                  "additive read (read_vec[:, :h_add*DH], post-normalisation) at the main decoder's ' =' position"}
    json.dump(out, open(os.path.join(run, "eval_s192_plateau.json"), "w", newline="\n"), indent=1)
    print("PLATEAU", run, "collapse_t", round(collapse_t, 4), {t: (v["repeat"], round(v["collapse_items"], 4), v["collapse_pass"] and round(v["collapse_pass"], 4)) for t, v in by_t.items()}, flush=True)


if __name__ == "__main__":
    main()
