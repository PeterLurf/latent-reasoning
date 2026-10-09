"""THEORY note 172-1 (2fa81289): PCCoT multi-latent / whole-block swaps (eval only), on whynlp/pccot-gpt2 at T = 3 (scratch/s172_probe.py's loader and
student path). Pairs: §149's same-template funnel exactly as §172 (valid chain, recipient correct, a later step references step t, unambiguous reference,
same template, donor step-t value differs, substituted answer differs; <= 20 candidates per recipient, <= 300 pairs; rng 14901) at t = 1 and t = 2.
Swap: the donor's latents replace the recipient's in the block fed to the final latent run (z_{T-1}), for the top-m latents by §172's step-t ridge R^2
(boxes/runs_fsq/s172_pccot/eval.json alignment.r2_grid[t], descending), m in {1, 4, 8}, and all 24 (the whole block).
Per (t, m): n and the shares predicting the recipient's own answer (recipient), the recipient's chain with step t := the donor's value (hybrid), and
the donor's own final answer (donor_full). Null: the whole block from a random DIFFERENT-template donor (rng 17211), over t = 1's recipients: donor_full.
    python scratch/s172_swap.py [device]   -> boxes/runs_fsq/s172_pccot/eval_s172_swap.json"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch"))
from boxes import core as bc  # noqa: E402
import s149_probe as P149  # noqa: E402
import s172_probe as S  # noqa: E402

MS = (1, 4, 8, 24)


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"; t0 = time.time(); torch.set_num_threads(3)
    try:
        from boxes import memcap
        memcap.apply(dev)
    except ImportError:
        pass
    out_d = os.path.join(ROOT, "boxes", "runs_fsq", "s172_pccot"); ev = json.load(open(os.path.join(out_d, "eval.json"))); grid = ev["alignment"]["r2_grid"]
    lines = open(os.path.join(ROOT, "bench", "data", "gsm8k_aug", "test.txt"), encoding="utf-8").read().splitlines(); assert len(lines) == 1319
    N = len(lines); qtext = [l.split("||")[0] for l in lines]; gold_s = [l.split("####")[-1].strip() for l in lines]; gold = np.array([S.num(x) for x in gold_s])
    chains = [P149.parse_chain(l.split("||")[1].split("####")[0]) if "||" in l else None for l in lines]; qn = [P149.qnumbers(q) for q in qtext]
    valid = np.array([c is not None and abs(c[-1][1] - gold[i]) < 1e-4 for i, c in enumerate(chains)])
    Q = [bc.ENC.encode(q) for q in qtext]; M = S.PCCoT(S.load_model(dev), dev); ALL = list(range(N)); bs = 64
    s_, zz = S.answers(M, Q, ALL, bs, T=S.T_REL, keep_z=True); ok = np.array([s_[i].strip() == gold_s[i] for i in ALL]); Zf = np.stack([zz[i][S.T_REL - 1] for i in ALL])
    res = {"model": "whynlp/pccot-gpt2", "T": S.T_REL, "acc_release_rerun": float(ok.mean()), "acc_release_s172": ev["acc_release"], "ms": list(MS), "swap": {},
           "order": {}, "device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"}
    print("release", res["acc_release_rerun"], "(s172 %.4f)" % ev["acc_release"], flush=True)
    rng = np.random.default_rng(14901); rec1 = None
    for t in (1, 2):
        order = [int(j) for j in np.argsort(-np.array(grid[str(t)]), kind="stable")]; res["order"][str(t)] = order
        f0 = [i for i in ALL if valid[i]]; f1 = [i for i in f0 if ok[i]]; f2 = [i for i in f1 if len(chains[i]) > t]
        f3 = [i for i in f2 if P149.referenced_downstream(chains[i], t)]; rec = [i for i in f3 if not P149.ambiguous_ref(chains[i], t, qn[i])]; pairs = []
        for i in rec:
            tp = P149.template(chains[i]); pool = [k for k in ALL if k != i and valid[k] and len(chains[k]) >= t and P149.template(chains[k]) == tp]; rng.shuffle(pool)
            for k in pool[:20]:
                vd = chains[k][t - 1][1]
                if abs(vd - chains[i][t - 1][1]) < 1e-9:
                    continue
                e = P149.substitute(chains[i], t, vd)
                if e is None or abs(e - gold[i]) < 1e-9:
                    continue
                pairs.append((i, k, e)); break
        pairs = pairs[:300]; cell = {}
        if t == 1:
            rec1 = [p[0] for p in pairs]
        if pairs:
            ri = [p[0] for p in pairs]; di = np.array([p[1] for p in pairs]); hyb = np.array([p[2] for p in pairs])
            for m_ in MS:
                js = list(range(24)) if m_ == 24 else order[:m_]
                sw, _ = S.answers(M, Q, ri, bs, override={j: Zf[di, j] for j in js}); pv = np.array([S.num(sw[i]) for i in ri])
                cell[str(m_)] = {"n": len(ri), "latents": js, "recipient": float((np.abs(pv - gold[ri]) < 1e-4).mean()), "hybrid": float((np.abs(pv - hyb) < 1e-4).mean()),
                                 "donor_full": float((np.abs(pv - gold[di]) < 1e-4).mean())}
                print("t", t, "m", m_, cell[str(m_)]["recipient"], cell[str(m_)]["hybrid"], cell[str(m_)]["donor_full"], flush=True)
        res["swap"][str(t)] = cell
    r2 = np.random.default_rng(17211); nr, nd = [], []
    for i in (rec1 or []):
        tp = P149.template(chains[i]); pool = [k for k in ALL if k != i and valid[k] and P149.template(chains[k]) != tp and abs(gold[k] - gold[i]) > 1e-9]
        if pool:
            nr.append(i); nd.append(int(pool[int(r2.integers(0, len(pool)))]))
    if nr:
        sw, _ = S.answers(M, Q, nr, bs, override={j: Zf[np.array(nd), j] for j in range(24)}); pv = np.array([S.num(sw[i]) for i in nr])
        res["null_whole_block"] = {"n": len(nr), "donor_full": float((np.abs(pv - gold[np.array(nd)]) < 1e-4).mean()), "recipient": float((np.abs(pv - gold[nr]) < 1e-4).mean()),
                                   "def": "the whole block from a random different-template donor with a different answer, over t = 1's recipients (rng 17211)"}
    res["sec"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(out_d, "eval_s172_swap.json"), "w", newline="\n"), indent=1, default=float); print("DONE", res["sec"], flush=True)


if __name__ == "__main__":
    main()
