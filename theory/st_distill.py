"""THEORY §65: restart-distillation on S3 permutation tracking (the Pilot L analogue of §32 on the §64 task).

Student = a gated §64 S3 model; teacher = a frozen copy of it. Each step:
  normal batch : N ~ U{1..8}, K = 8, cross-entropy on the true final arrangement (the original training task)
  deep batch   : one N ~ U{9..16} per batch, the student runs ONE loop with K' = N; targets from --teacher:
    rd   : the frozen copy's written-state restarts every 8 operations (§64's restart chain; label-free)
    gt   : the true final arrangement (ceiling)
    self : the frozen copy's own single loop at K' = N (self-training control)
Then §64's evaluation (st_model.evaluate -> eval_st.json) and the share of single-loop answers that are valid
arrangements at N = 12, 16, 20 (valid.json).
    python theory/st_distill.py --init theory/runs/st/s3_s1 --teacher rd --out theory/runs/st/s3rd_s1 [--device cuda:0]
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_model import KTRAIN, NMAX_TRAIN, STReasoner, evaluate, perms, sample, tt  # noqa: E402


@torch.no_grad()
def restart_chain(m, init, ops, pos, N, dev):
    n = m.n; Gt = torch.as_tensor(perms(n), device=dev); cur = init.clone(); done = 0
    while done < N:
        k = min(NMAX_TRAIN, N - done)
        lp = F.log_softmax(m.run(cur, ops[:, done:done + k], pos[:, done:done + k],
                                 torch.full((len(cur),), k, device=dev, dtype=torch.long), KTRAIN), -1)
        cur = Gt[lp[:, torch.arange(n, device=dev)[None], Gt].sum(-1).argmax(1)]
        done += k
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--teacher", required=True, choices=["rd", "gt", "self"])
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_eval", type=int, default=1024)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    ck = torch.load(os.path.join(a.init, "model.pt"), map_location=dev, weights_only=False)
    group = ck["args"]["group"]
    m = STReasoner(group).to(dev); m.load_state_dict(ck["model"])
    tm = copy.deepcopy(m).eval()
    for p_ in tm.parameters():
        p_.requires_grad_(False)
    n = m.n
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time()
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        init, ops, pos, nops, final, _ = sample(rng, group, a.bs, 1, NMAX_TRAIN)
        I, O, P, Nn, Fi = tt(dev, init, ops, pos, nops, final)
        loss_n = F.cross_entropy(m.run(I, O, P, Nn, KTRAIN).reshape(-1, n), Fi.reshape(-1))
        N = int(rng.integers(9, 17))
        init, ops, pos, nops, final, _ = sample(rng, group, a.bs, N, N, Nmax=N)
        I, O, P, Nn, Fi = tt(dev, init, ops, pos, nops, final)
        with torch.no_grad():
            if a.teacher == "gt":
                y = Fi
            elif a.teacher == "self":
                y = tm.run(I, O, P, Nn, N).argmax(-1)
            else:
                y = restart_chain(tm, I, O, P, N, dev)
        loss_d = F.cross_entropy(m.run(I, O, P, Nn, N).reshape(-1, n), y.reshape(-1))
        loss = loss_n + loss_d
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == a.steps:
            log.write(json.dumps({"step": step, "loss_normal": loss_n.item(), "loss_deep": loss_d.item(), "N": N,
                                  "teacher_acc": float((y == Fi).all(1).float().mean()), "sec": round(time.time() - t0)}) + "\n")
            log.flush()
    args = dict(ck["args"]); args.update(distill=vars(a))
    torch.save({"model": m.state_dict(), "args": args}, os.path.join(a.out, "model.pt"))
    evaluate(a.out, dev, a.n_eval)
    m.eval(); val = {}
    with torch.no_grad():
        for N in (12, 16, 20):
            init, ops, pos, nops, final, _ = sample(np.random.default_rng(5000 + N), group, 512, N, N, Nmax=24)
            p = m.run(*tt(dev, init, ops, pos, nops), N).argmax(-1).cpu().numpy()
            val[N] = float(np.mean([len(set(r)) == n for r in p]))
    json.dump(val, open(os.path.join(a.out, "valid.json"), "w", newline="\n"), indent=1)
    print(a.out, "valid share", val, flush=True)


if __name__ == "__main__":
    main()
