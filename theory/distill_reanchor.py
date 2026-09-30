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
               t8f  : t8 with an agreement filter (loss only on queries where top-1 and top-3 re-anchoring agree; §38)
Output <out>/model.pt (config as the P30 run), arch.json, log.jsonl (with an inherited full-depth marker, so
theory/eval_arch.py scores it like any run).
Round 2 of the ladder (THEORY §38, Pilot M): --deep_R 4 --deep_widths 2,3 --student_Kp 24 --teacher_ep 12 starts from a
round-1 student, draws 24-hop chains (theory/deep_chain.py) and re-anchors with 12-step episodes. The defaults
(--deep_R 2, --student_Kp 12, --teacher_ep = the model's K) are round 1 exactly.
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
from deep_chain import deep_chain_graphs  # noqa: E402


def one_episode(m, K, Kp, src, dst, start, q):
    h, cache, _ = m.encode(m.embed_prefix(src, dst, start))
    for k in range(1, Kp + 1):
        h, cache, _ = m.latent_step(h, k, cache)
    return m.query(cache, q)


@torch.no_grad()
def teacher_targets(kind, tm, K, head, g, br, q, dev, Kp=12, ep=None):
    T = lambda a: torch.as_tensor(a, device=dev)
    n = len(g); bi = np.arange(n)
    ep = ep or K
    if kind == "gt":
        h = q.shape[1] // 2
        y = np.concatenate([np.ones((n, h)), np.zeros((n, h))], 1)
        return T(y).float()
    src, dst = T(g.src), T(g.dst)
    if kind == "self":
        return (one_episode(tm, K, Kp, src, dst, T(g.start), q) > 0).float()
    o1 = rollout(tm, K, src, dst, T(g.start), ep, qnodes=q)
    best = o1["qlogit"]
    a = o1["attn1"][ep - 1][:, head, :o1["P"] - 1]
    order = torch.argsort(a, 1, descending=True).cpu().numpy()
    for r in range(3):
        anc = g.dst[bi, order[:, r]]
        best = torch.maximum(best, rollout(tm, K, src, dst, T(anc), ep, qnodes=q)["qlogit"])
        if r == 0:
            top1 = best.clone()
    if kind == "t8f":                    # agreement filter: train only where the top-1 and top-3 re-anchorings agree
        return (best > 0).float(), ((best > 0) == (top1 > 0)).float()
    return (best > 0).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", required=True)
    ap.add_argument("--teacher", required=True, choices=["t8", "t8f", "self", "gt"])
    ap.add_argument("--head", type=int, required=True, help="sweep head of the initial model (exp12_reanchor.json)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--deep_R", type=int, default=2, help="chain length 6R; 2 = latent.length_gen (round 1)")
    ap.add_argument("--deep_widths", default="2,4,6")
    ap.add_argument("--deep_sup_max", type=int, default=0, help="THEORY §41: supervise deep-chain queries only up to this "
                    "distance (reachable node and its look-alike at the same index); 0 = all")
    ap.add_argument("--student_Kp", type=int, default=12)
    ap.add_argument("--teacher_ep", type=int, default=0, help="re-anchoring episode length (0 = the model's K)")
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
    if a.deep_R != 2 or a.student_Kp != 12 or a.teacher_ep or a.deep_sup_max:
        c["distill"].update(deep_R=a.deep_R, deep_widths=a.deep_widths, student_Kp=a.student_Kp, teacher_ep=a.teacher_ep,
                            deep_sup_max=a.deep_sup_max)
    widths = [int(w) for w in a.deep_widths.split(",")]
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
        bi = np.arange(a.bs)
        if a.deep_R == 2:
            gl, br = long_chain_graphs(rng, a.bs, widths)
            q = T(np.concatenate([gl.nodes[bi, br[k]] for k in ("A", "B", "C", "D")], 1)).long()
        else:
            gl, br = deep_chain_graphs(rng, a.bs, widths, a.deep_R)
            q = T(np.concatenate([br["reach"], br["decoy"]], 1)).long()
        y = teacher_targets(a.teacher, tm, K, a.head, gl, br, q, dev, Kp=a.student_Kp, ep=a.teacher_ep or None)
        w = None
        if isinstance(y, tuple):
            y, w = y
        if a.deep_sup_max:
            if a.deep_R == 2:       # columns A0..A5, B0..B5 (distance 1..12), C0..C5 (look-alike 7..12), D0..D5 (1..6)
                dist = np.r_[np.arange(1, 13), np.arange(7, 13), np.arange(1, 7)]
            else:                   # reach 1..6R, decoy 1..6R
                dist = np.r_[np.arange(1, 6 * a.deep_R + 1), np.arange(1, 6 * a.deep_R + 1)]
            wm = T((dist <= a.deep_sup_max).astype(np.float32))[None].expand(a.bs, -1)
            w = wm if w is None else w * wm
        with torch.autocast("cuda", dtype=amp or torch.float32, enabled=amp is not None):
            ld, lu, _, _ = m.run_both(b["src"], b["dst"], b["start"], K, b["q"], b["tgt"])
            ldeep = one_episode(m, K, a.student_Kp, T(gl.src), T(gl.dst), T(gl.start), q)
        loss_n = F.binary_cross_entropy_with_logits(ld.float(), b["qy"]) + F.binary_cross_entropy_with_logits(lu.float(), b["ty"])
        loss_d = F.binary_cross_entropy_with_logits(ldeep.float(), y) if w is None else             (F.binary_cross_entropy_with_logits(ldeep.float(), y, reduction="none") * w).sum() / w.sum().clamp_min(1)
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
