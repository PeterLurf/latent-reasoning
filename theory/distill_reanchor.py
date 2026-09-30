"""THEORY §32 / PROPOSALS §9 F5a: distil re-anchored inference into one uninterrupted latent loop (Pilot L).

Fine-tunes a trained P30 model (fresh AdamW) on two batches per step:
  * normal : ordinary training graphs at full depth, K = 6, delayed + upfront BCE (keeps the in-distribution skill)
  * deep   : long-chain graphs (latent.length_gen generator: a 12-hop chain A -> B plus look-alike decoys D, C), queries
             on every A/B/C/D node, answered by the student in ONE episode of K' = 12 latent steps, BCE against targets
             from --teacher:
               t8   : a frozen copy of the initial model run with re-anchored inference (§26 re2top3: episode 1 from
                      START, episode 2 from each of the 3 most-attended edges' targets at step K; max-combined logits > 0)
               self : the same frozen copy run as one episode at K' = 12 (plain self-training; isolates re-anchoring)
               gt   : the true labels (A, B reachable; C, D not) — upper bound
Output <out>/model.pt (config as the P30 run), arch.json, log.jsonl (with an inherited full-depth marker, so
theory/eval_arch.py scores it like any run).
    python theory/distill_reanchor.py --init_from runs/p30_s126/model.pt --teacher t8 --head 3 --out theory/runs/pl_t8_s126
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.data import make_batch  # noqa: E402
from latent.graphs import NPB, sample_graphs, set_shapes  # noqa: E402
from latent.length_gen import long_chain_graphs  # noqa: E402
from latent.model import LatentReasoner  # noqa: E402
from rollout import rollout  # noqa: E402


def one_episode(m, K, Kp, src, dst, start, q):
    h, cache, _ = m.encode(m.embed_prefix(src, dst, start))
    for k in range(1, Kp + 1):
        h, cache, _ = m.latent_step(h, k, cache)
    return m.query(cache, q)


@torch.no_grad()
def teacher_targets(kind, tm, K, head, g, br, q, dev):
    T = lambda a: torch.as_tensor(a, device=dev)
    n = len(g); bi = np.arange(n)
    if kind == "gt":
        y = np.concatenate([np.ones((n, 2 * NPB)), np.zeros((n, 2 * NPB))], 1)
        return T(y).float()
    src, dst = T(g.src), T(g.dst)
    if kind == "self":
        return (one_episode(tm, K, 12, src, dst, T(g.start), q) > 0).float()
    o1 = rollout(tm, K, src, dst, T(g.start), K, qnodes=q)
    best = o1["qlogit"]
    a = o1["attn1"][K - 1][:, head, :o1["P"] - 1]
    order = torch.argsort(a, 1, descending=True).cpu().numpy()
    for r in range(3):
        anc = g.dst[bi, order[:, r]]
        best = torch.maximum(best, rollout(tm, K, src, dst, T(anc), K, qnodes=q)["qlogit"])
    return (best > 0).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", required=True)
    ap.add_argument("--teacher", required=True, choices=["t8", "self", "gt"])
    ap.add_argument("--head", type=int, required=True, help="sweep head of the initial model (exp12_reanchor.json)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.init_from, map_location=dev, weights_only=False)
    c = dict(ck["config"])
    set_shapes(c.get("shapes", "default"), c.get("p_fork", 0.5))
    m = LatentReasoner(c["d"], c["layers"], c["heads"], k_max=max(c["K"], 8), mode=c["mode"], emb=c.get("emb", "slots"),
                       d_id=c.get("d_id", 64), init=c.get("init", "torch"), feedback=c.get("feedback", "post_ln"),
                       step_emb=not c.get("no_step_emb", False)).to(dev)
    m.load_state_dict(ck["model"])
    tm = copy.deepcopy(m).eval()
    for p_ in tm.parameters():
        p_.requires_grad_(False)
    K = c["K"]
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = lambda s: a.lr * min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / a.steps))
    amp = torch.bfloat16 if dev != "cpu" and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    c.update(distill={"teacher": a.teacher, "init_from": a.init_from, "steps": a.steps, "lr": a.lr, "bs": a.bs, "seed": a.seed})
    json.dump({"arch": "coconut", "distill": c["distill"]}, open(os.path.join(a.out, "arch.json"), "w"))
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    log.write(json.dumps({"step": 0, "curriculum_advance_to": None, "note": "inherited: initial model was at full depth"}) + "\n")
    t0 = time.time()
    T = lambda x: torch.as_tensor(x, device=dev)
    m.train()
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = sched(step)
        g = sample_graphs(rng, a.bs, split="train", max_bdepth=None)
        b = make_batch(rng, g, 16, dev)
        gl, br = long_chain_graphs(rng, a.bs, [2, 4, 6])
        bi = np.arange(a.bs)
        q = T(np.concatenate([gl.nodes[bi, br[k]] for k in ("A", "B", "C", "D")], 1)).long()
        y = teacher_targets(a.teacher, tm, K, a.head, gl, br, q, dev)
        with torch.autocast("cuda", dtype=amp or torch.float32, enabled=amp is not None):
            ld, lu, _, _ = m.run_both(b["src"], b["dst"], b["start"], K, b["q"], b["tgt"])
            ldeep = one_episode(m, K, 12, T(gl.src), T(gl.dst), T(gl.start), q)
        loss_n = F.binary_cross_entropy_with_logits(ld.float(), b["qy"]) + F.binary_cross_entropy_with_logits(lu.float(), b["ty"])
        loss_d = F.binary_cross_entropy_with_logits(ldeep.float(), y)
        loss = loss_n + loss_d
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == a.steps:
            agree = float(((ldeep > 0).float() == y).float().mean())
            log.write(json.dumps({"step": step, "loss_normal": loss_n.item(), "loss_deep": loss_d.item(),
                                  "deep_agree_teacher": agree, "sec": time.time() - t0}) + "\n"); log.flush()
    c["distilled_from"] = a.init_from
    torch.save({"model": m.state_dict(), "config": c, "step": a.steps}, os.path.join(a.out, "model.pt"))
    print("done", a.out, "sec %.0f" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
