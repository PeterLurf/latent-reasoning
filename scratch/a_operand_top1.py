"""§208 QO-CAPACITY-BOUND FIDELITY difficulty control (Theory, note 208-2): A's aux-slot operand top-1 (full vocabulary, the first token of each
question-literal operand, kind 'q'; theory/stepfactor/diag_A.py's bookkeeping) decoded from M_t with the pass-t aux prompt, on the recomputable Aug-valid
items (qonce_read's LIT-RECALL set), split by the item's n_lit = len(used_q): <= 2 vs >= 4. Kind 'r' and all-operand values beside. fp32.
-> scratch/gate_s208/A_operand_top1.json  (merged into <§208 run>/eval_qonce.json as "A_operand_top1_by_nlit")   python scratch/a_operand_top1.py [--device cuda:1]"""
import argparse
import json
import os
import sys

sys.path.insert(0, "D:/Latent/theory/stepfactor"); sys.path.insert(0, "D:/Latent")
import diag_A as DA  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.gsm8k_data as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:1"); a = ap.parse_args(); DA.DEV = a.device
    m = DA.build(); R, _ = G.load_aug("valid", filter_bad=False); D = []
    for r in R:
        tp, _ = G.parse_recomputable(r)
        if tp is not None:
            y = DA.enc(r); y["nlit"] = len(tp["used_q"]); D.append(y)
    rows = []
    with torch.no_grad():
        for i in range(0, len(D), DA.BS):
            ch = D[i:i + DA.BS]; Ss = DA.states(m, ch)
            for t in range(1, DA.TMAX + 1):
                ix = [j for j, it in enumerate(ch) if t <= it["m"]]
                if not ix:
                    continue
                its = [ch[j] for j in ix]; lg, y = m.aux_logits(Ss[t][ix], its, t, a.device); top = lg.argmax(-1)
                for r_, it in enumerate(its):
                    for (j, kind) in it["fo"][t - 1]:
                        rows.append((it["nlit"], kind, int(top[r_, j]) == int(y[r_, j])))
    out = {}
    for kind in ("q", "r", "all"):
        sel = [r_ for r_ in rows if kind == "all" or r_[1] == kind]; le = [h for nl, _, h in sel if nl <= 2]; ge = [h for nl, _, h in sel if nl >= 4]
        out[kind] = {"le2": float(np.mean(le)) if le else None, "ge4": float(np.mean(ge)) if ge else None, "n_le2": len(le), "n_ge4": len(ge), "all": float(np.mean([h for *_, h in sel]))}
    res = {"A_operand_top1_by_nlit": {"le2": out["q"]["le2"], "ge4": out["q"]["ge4"]}, "detail": out, "model": DA.RUN + "/model.pt", "device": a.device, "dtype": "fp32",
           "items": "recomputable Aug-valid items (%d); operand = the first token of each question-literal operand (kind q), M_t with the pass-t aux prompt, t <= m" % len(D)}
    os.makedirs("D:/Latent/scratch/gate_s208", exist_ok=True); json.dump(res, open("D:/Latent/scratch/gate_s208/A_operand_top1.json", "w", newline="\n"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
