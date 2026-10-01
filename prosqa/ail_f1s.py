"""THEORY §61 (F1-S): the frozen §54b AI-soft path (prosqa/ail_gpt2.py at 02a03ac: AIL model, soft forward, tau 1 -> 0.3 over
2/3, binarisation penalty ramped to 0.1 over the last 1/3, lab_std 0.144, bias -4, bs 4 x accum 2, lr 5e-5 / 1e-3) trained
on prosqa/p8data/f1s_train.json (make_f1s.py): micro-batches bucketed by question depth d with P = d + U{0,1,2}; curriculum
d = 1 only for the first 1/6 of steps, widening linearly to d = 1..6 by 1/3. Optimizer steps = 1486, i.e. the same number of
labels (steps x 8 = 11,888) as §54b AI-soft (5,946 problems x 2 epochs = 11,892).
Test: p8_na_test deep problems (d3..14, first 100 per depth) + held-out shallow questions (f1s_test_shallow.json), fixed
P = 16 with hardened labels; also soft-test accuracy, fixed point, accuracy at P = d, edit IIA, cue control, label prec/recall.
    python prosqa/ail_f1s.py --out prosqa/runs_ail/f1s_s1 --seed 1 [--device cuda:0]
    python prosqa/ail_f1s.py --eval prosqa/runs_ail/f1s_s1
"""
import argparse
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ail_gpt2 import AIL, answer, collate, cue_variant, hops  # noqa: E402

STEPS, BS, ACCUM, LR, LR_HEAD, TAU_END, LAM, EPS = 1486, 4, 2, 5e-5, 1e-3, 0.3, 0.1, 1e-3


def train(a):
    dev = a.device
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    sd = torch.load(a.init, map_location="cpu", weights_only=True)
    m = AIL(sd, "mono", 0.144, -4.0).to(dev); m.mode = "soft"
    data = json.load(open(os.path.join(HERE, "p8data", "f1s_train.json")))
    by_d = {}
    for i, x in enumerate(data):
        by_d.setdefault(len(x["steps"]), []).append(i)
    head = list(m.lab.parameters()) + list(m.head.parameters())
    opt = torch.optim.AdamW([{"params": m.lm.parameters(), "lr": LR}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    sq = lambda x: x * (1 - 2 * EPS) + EPS
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n")
    t0 = time.time(); hist = []; seen_d = {}
    N = a.steps
    for step in range(N):
        fr = step / N
        m.tau = 1 + (TAU_END - 1) * min(1, fr / (2 / 3))
        lam = LAM * max(0.0, (fr - 2 / 3) / (1 / 3))
        dmax = 1 if fr < 1 / 6 else min(6, 1 + int((fr - 1 / 6) / (1 / 6) * 5 + 1e-9))
        for gp, lr in zip(opt.param_groups, (LR, LR_HEAD)):
            gp["lr"] = lr * min(1, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / N))
        opt.zero_grad()
        for _ in range(ACCUM):
            pool = [i for d in range(1, dmax + 1) for i in by_d.get(d, [])]
            d = len(data[pool[int(rng.integers(len(pool)))]]["steps"])           # bucket: depth of a uniformly drawn question
            idx = rng.choice(by_d[d], BS, replace=False)
            xs = [data[i] for i in idx]; seen_d[d] = seen_d.get(d, 0) + BS
            P = d + int(rng.integers(0, 3))
            b = collate(xs, dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                lab, logit, traj = m.run(b, P, ckpt=True)
            r = torch.arange(len(xs), device=dev)
            loss = F.binary_cross_entropy(sq(lab[r, b["tgt"]]), torch.ones(len(xs), device=dev)) + \
                   F.binary_cross_entropy(sq(lab[r, b["neg"]]), torch.zeros(len(xs), device=dev))
            if lam > 0:
                msk = torch.zeros_like(lab)
                for j, x in enumerate(xs):
                    msk[j, : len(x["idx_to_symbol"])] = 1
                loss = loss + lam * sum(((t_ * (1 - t_)) * msk).sum() for t_ in traj[1:]) / (msk.sum() * P)
            (loss / ACCUM).backward()
            hist.append((loss.item(), float(answer(lab.detach(), logit.detach(), b).float().mean())))
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (step + 1) % 50 == 0 or step + 1 == N:
            h = np.array(hist); hist = []
            rec = {"step": step + 1, "of": N, "loss": float(h[:, 0].mean()), "train_acc": float(h[:, 1].mean()),
                   "tau": m.tau, "lam": lam, "dmax": dmax, "sec": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    torch.save({"model": m.state_dict(), "args": vars(a), "tau_end": m.tau, "labels_seen_by_depth": seen_d},
               os.path.join(a.out, "model.pt"))
    print("TRAIN_DONE", a.out, "labels by depth", dict(sorted(seen_d.items())), f"{time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def evaluate(a):
    dev = a.device
    ck = torch.load(os.path.join(a.eval, "model.pt"), map_location="cpu", weights_only=False)
    sd = torch.load(ck["args"]["init"], map_location="cpu", weights_only=True)
    m = AIL(sd, "mono"); m.load_state_dict(ck["model"]); m.to(dev).eval()
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    rng = np.random.default_rng(0)
    te = json.load(open(os.path.join(HERE, "p8data", "p8_na_test.json")))
    seen, sel = {}, []
    for x in te:
        d = len(x["steps"])
        if seen.get(d, 0) < 100:
            seen[d] = seen.get(d, 0) + 1; sel.append(x)
    sel += json.load(open(os.path.join(HERE, "p8data", "f1s_test_shallow.json")))
    if a.max_eval:                                                            # smoke only
        sel = sel[:a.max_eval // 2] + sel[-a.max_eval // 2:]
    rows = []
    for i in range(0, len(sel), 4):
        xs = sel[i:i + 4]; b = collate(xs, dev)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            m.mode, m.tau = "hard", None
            lab, logit, traj = m.run(b, 16)
            ed = torch.zeros_like(b["start"]); has = []
            for j, x in enumerate(xs):                                        # edit: farthest ancestor (<= 3 hops) of the negative ON
                radj = {}
                for u, v in x["edges"]:
                    radj.setdefault(v, []).append(u)
                dd = {x["neg_target"]: 0}; q = deque([x["neg_target"]])
                while q:
                    u = q.popleft()
                    for v in radj.get(u, []):
                        if v not in dd and dd[u] < 3:
                            dd[v] = dd[u] + 1; q.append(v)
                c = [v for v, k in dd.items() if k == max(dd.values()) and k > 0]
                if c:
                    ed[j, int(rng.choice(c))] = 1; has.append(True)
                else:
                    has.append(False)
            lab_e, _, _ = m.run(b, 16, edit=ed)
            xc = [cue_variant(x, rng) for x in xs]
            bc = collate([c if c is not None else x for c, x in zip(xc, xs)], dev)
            lab_c, logit_c, _ = m.run(bc, 16); acc_c = answer(lab_c, logit_c, bc)
            m.mode, m.tau = "soft", ck.get("tau_end", TAU_END)
            lab_s, logit_s, _ = m.run(b, 16); acc_s = answer(lab_s, logit_s, b)
            m.mode, m.tau = "hard", None
        acc = answer(lab, logit, b)
        for j, x in enumerate(xs):
            d = len(x["steps"])
            T = [traj[p][j] > 0.5 for p in range(len(traj))]
            ch = [bool((T[p] != T[p - 1]).any()) for p in range(1, len(T))]
            row = {"depth": d, "kind": x.get("f1s_kind", "deep"), "acc": bool(acc[j]), "acc_soft": bool(acc_s[j]),
                   "fixed_point": next((p for p in range(len(ch)) if not ch[p]), len(ch))}
            lt, ln = bool(traj[d][j, x["target"]] > 0.5), bool(traj[d][j, x["neg_target"]] > 0.5)
            row["acc_at_P_eq_d"] = (lt and not ln) if lt != ln else None
            h = hops(x); nV = len(x["idx_to_symbol"]); fin = traj[16][j, :nV].cpu().numpy() > 0.5
            reach = np.array([v in h for v in range(nV)])
            row["label_prec"] = float(fin[reach].sum() / max(1, fin.sum())); row["label_recall"] = float(fin[reach].mean())
            if xc[j] is not None:
                row["acc_cue"] = bool(acc_c[j])
            if has[j]:
                row["edit_neg_on"] = bool(lab_e[j, x["neg_target"]] > 0.5)
                row["edit_tgt_same"] = bool((lab_e[j, x["target"]] > 0.5) == (lab[j, x["target"]] > 0.5))
            rows.append(row)
    json.dump({"model": a.eval, "P": 16, "n": len(rows), "rows": rows}, open(os.path.join(a.eval, "eval.json"), "w", newline="\n"))
    print("EVAL_DONE", a.eval, len(rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--eval"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init", default=os.path.join(HERE, "ckpt", "gpt2_pretrained.bin"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=STEPS, help="smoke tests only; the registered run uses 1486")
    ap.add_argument("--max_eval", type=int, default=0, help="smoke tests only")
    a = ap.parse_args()
    evaluate(a) if a.eval else train(a)


if __name__ == "__main__":
    main()
