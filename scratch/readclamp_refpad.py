"""Audit's note on 187-8 / 8a (2026-10-12): the READ-CLAMP refs (halt173.readclamp_fields) pooled every token incl. right-padding. For the same 500 n <= 8
train items (rng 18708) and the same calls, recompute the per-(path, k) p99 / mean of each additive head's post-gain read norm WITHOUT pad tokens,
the pad share of the samples, and the relative difference vs the with-pad refs -> <run>/eval_s187_readclamp_refpad.json. Eval-only, descriptive.
    python scratch/readclamp_refpad.py <run> [device]"""
import json
import os
import sys
from contextlib import nullcontext

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
from boxes.nested170 import test_set, train_items, aux_of, ids, TID, EOT as NEOT  # noqa: E402

CUR = {"path": None, "k": 0, "valid": None}; STORE = {}


def rc_collect(oa):
    nrm = oa.norm(dim=-1).detach(); v = CUR["valid"].to(nrm.device)          # (B, T) True = a real token
    d = STORE.setdefault((CUR["path"], CUR["k"]), {"all": [], "real": []})
    d["all"].append(nrm.reshape(-1, nrm.shape[-1]).cpu()); d["real"].append(nrm[v[:, :nrm.shape[1]]].cpu())
    return oa


@torch.no_grad()
def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"; kind, m, a = H.load_model(run, dev)
    amp = None if (not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ctx = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else nullcontext
    TE = test_set(a.get("per_n", 1000)); EX = {tuple(it["syms"]) for it in TE if it["n"] >= int(a.get("excl_min_n", 1))}
    tr = train_items(np.random.default_rng(18708), 500, 8, EX); DM.READ_CLAMP = {"mode": "collect"}; DM.read_clamp = rc_collect; bs = 125
    try:
        for i in range(0, len(tr), bs):
            ch = tr[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            dec_len = [len([NEOT] + ids(it["syms"]) + [TID[" ="]]) for it in ch]; Td = max(dec_len)
            with ctx():
                S = m.mem.zero_state(len(ch), dev)
                for k in range(0, 13):
                    CUR.update(path="dec", k=k, valid=torch.arange(Td)[None] < torch.tensor(dec_len)[:, None]); m.dec_logits(S, ch, dev)
                    idx = [j for j, it in enumerate(ch) if 1 <= k <= len(aux_of(it))]
                    if idx:
                        al = [4 + len(ids(aux_of(ch[j])[k - 1][0])) for j in idx]        # [EOT, Step, k, :] + target tokens (+ EOT) - 1
                        CUR.update(path="aux", k=k, valid=torch.arange(max(al))[None] < torch.tensor(al)[:, None])
                        if getattr(m, "nopass", False):
                            CUR["valid"] = torch.arange(max(al) - 1)[None] < (torch.tensor(al) - 1)[:, None]
                        m.aux_logits(S[idx], [ch[j] for j in idx], k, dev)
                    CUR.update(path="think", k=k, valid=msk.cpu()); S = m.write(S, x, msk)
    finally:
        DM.READ_CLAMP = None
    out = {}
    for (p_, k_), d in sorted(STORE.items()):
        A = torch.cat(d["all"]).float(); R = torch.cat(d["real"]).float()
        pa, pr = torch.quantile(A, 0.99, dim=0), torch.quantile(R, 0.99, dim=0); ma, mr = A.mean(0), R.mean(0)
        out["%s_k%d" % (p_, k_)] = {"pad_share": float(1 - len(R) / len(A)), "p99_with_pad": pa.tolist(), "p99_real": pr.tolist(), "mean_with_pad": ma.tolist(), "mean_real": mr.tolist(),
                                    "p99_max_rel_diff": float(((pa - pr).abs() / pr.clamp_min(1e-12)).max()), "mean_max_rel_diff": float(((ma - mr).abs() / mr.clamp_min(1e-12)).max())}
    summ = {"max_pad_share": max(v["pad_share"] for v in out.values()), "max_p99_rel_diff": max(v["p99_max_rel_diff"] for v in out.values()),
            "max_mean_rel_diff": max(v["mean_max_rel_diff"] for v in out.values())}
    json.dump({"by_path_k": out, "summary": summ, "run": run, "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
               "def": "same 500 n<=8 train items (rng 18708) and calls as halt173.readclamp_fields' refs; 'real' excludes right-padding tokens (think: the think mask; dec: [EOT] + expr + ' ='; aux: the prompt + teacher-forced target)"},
              open(os.path.join(run, "eval_s187_readclamp_refpad.json"), "w", newline="\n"), indent=1)
    print("REFPAD", run, summ, flush=True)


if __name__ == "__main__":
    main()
