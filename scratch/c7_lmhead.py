"""C7 readability check (Theory 2026-10-07, LOCAL_BACKLOG; Audit's steward idea; descriptive, $0): is the carried code human-readable
through GPT-2's own output head? For one kept exact fsq120 run: 1,024 eval-mode chains of length 8 (fsq120.make_batch, seed 14100), every
hand-off's embedding e = W_up(code) (fsq120._probe_chain, hard codes) projected through GPT-2's final LN and tied LM head
(layer_norm(e; ln_f) @ wte^T); top-1 / top-5 agreement with the state's text token (S5: the first token of the state text; FF / CYC / MOD /
RDFA: the answer token), pooled and by chain position. Also the chance rate of each target under a frequency-matched guess.
    python scratch/c7_lmhead.py <run dir> [device]      -> <run dir>/eval_c7_lmhead.json"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
try:                                                                        # local hard memory cap
    from boxes import memcap  # noqa: E402
except ImportError:
    memcap = None
import boxes.fsq120 as F  # noqa: E402
from boxes import s5  # noqa: E402


def main():
    d = sys.argv[1].rstrip("/"); dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"; a = json.load(open(os.path.join(d, "args.json")))
    F.D_Q, F.LEV = int(a.get("d_q", 8)), int(a.get("levels", 8)); F.HALF = (F.LEV - 1) / 2; F.SHIFT = math.atanh(F.OFF / F.HALF)
    memcap and memcap.apply(dev)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    fam = F.Family(a["family"]); m = F.FSQ120(s5.load_gpt2(dev), fam).to(dev)
    m.no_round = bool(a.get("no_round")); m.soft_beta = float(a.get("soft_beta") or 0.0); m.carrier_ln = bool(a.get("carrier_ln"))
    assert not (a.get("std") or a.get("std2") or a.get("anneal") or a.get("raw_h") or a.get("skip_dec") or a.get("grow")), "plain FSQ carriers only"
    m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev)); m.eval(); m.p_hard = 1.0
    s0, x, lab = F.make_batch(fam, np.random.default_rng(14100), 1024, 8)
    tgt = (F.STTOK[lab.reshape(-1)][:, 0] if fam.name == "s5" else fam.answer_tokens(lab.reshape(-1))[:, 0]).reshape(lab.shape)
    hs = F._probe_chain(m, fam, dev, amp, s0 if fam.name == "s5" else None, x); g = m.g
    with torch.no_grad():
        E = torch.stack([e.float() for _, _, e in hs], 1)                    # (N, L, 768)
        lg = Fn.layer_norm(E, (768,), g.w("ln_f.weight").float(), g.w("ln_f.bias").float()) @ g.w("wte.weight").float().T
        top5 = lg.topk(5, -1).indices.cpu().numpy()
    T = torch.tensor(tgt).numpy(); hit1 = top5[..., 0] == T; hit5 = (top5 == T[..., None]).any(-1)
    vals, cnt = np.unique(T, return_counts=True)
    res = {"run": os.path.basename(d), "family": a["family"], "d_q": F.D_Q, "soft_beta": m.soft_beta, "target": "first state token" if fam.name == "s5" else "answer token",
           "n_handoffs": int(T.size), "top1": float(hit1.mean()), "top5": float(hit5.mean()), "top1_by_pos": [round(float(v), 4) for v in hit1.mean(0)],
           "top5_by_pos": [round(float(v), 4) for v in hit5.mean(0)], "chance_majority_target": float(cnt.max() / cnt.sum()), "n_distinct_targets": int(len(vals)),
           "most_common_top1_tokens": [int(t) for t in np.bincount(top5[..., 0].reshape(-1)).argsort()[::-1][:5]], "eval_dtype": str(amp),
           "device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"}
    json.dump(res, open(os.path.join(d, "eval_c7_lmhead.json"), "w", newline="\n"), indent=1); print("C7", json.dumps({k: res[k] for k in ("run", "top1", "top5", "chance_majority_target")}), flush=True)


if __name__ == "__main__":
    main()
