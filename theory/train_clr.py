"""THEORY §45 (checkpointed latent reasoner, first pilot): teach a trained model to restart one latent episode from a
*set* of written node-id checkpoints (START tokens), without step labels.

Fine-tunes a trained P30 model (fresh AdamW) on two batches per step, as theory/distill_reanchor.py:
  * normal : ordinary training graphs at full depth, K = 6, delayed + upfront BCE (keeps the in-distribution skill)
  * deep   : long-chain graphs (12-hop chain + look-alike decoys), all A/B/C/D chain nodes queried. The student's answer is
             two segments: episode 1 (K steps from START) and episode 2 (K steps) restarted from a written checkpoint --
             the W nodes the student's own sweep head attends most at step K of episode 1 (chosen without gradient),
             written as W START tokens -- combined by elementwise max. BCE against the targets of a frozen copy of the
             initial model run with §26's re-anchoring (re2top3: separate episodes per anchor; no labels).
Arms: --W 3 (set-valued checkpoint) and --W 1 (single anchor, same teacher and data).
Output <out>/model.pt, arch.json, log.jsonl (inherited full-depth marker), like distill_reanchor.
    python theory/train_clr.py --init_from runs/p30_s111/model.pt --head 4 --W 3 --seed 111 --out theory/runs/pc3_s111
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
from distill_reanchor import one_episode, teacher_targets  # noqa: E402
from latent.data import make_batch  # noqa: E402
from latent.graphs import sample_graphs, set_shapes  # noqa: E402
from latent.length_gen import long_chain_graphs  # noqa: E402
from latent.model import T_START, LatentReasoner  # noqa: E402
from rollout import rollout  # noqa: E402


def set_episode(m, K, src, dst, starts, q):
    """One episode (with gradient) whose prefix ends in START tokens for starts[:, 0..W-1]; the last seeds the loop."""
    x = m.embed_prefix(src, dst, starts[:, -1])
    te = m.type_emb.weight
    extra = [(m.node_emb(starts[:, w]) + te[T_START])[:, None] for w in range(starts.shape[1] - 1)]
    x = torch.cat([x[:, :-1]] + extra + [x[:, -1:]], 1)
    h, cache, _ = m.encode(x)
    for k in range(1, K + 1):
        h, cache, _ = m.latent_step(h, k, cache)
    return m.query(cache, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--W", type=int, required=True, help="checkpoint size (written START tokens)")
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
    c.update(clr={"W": a.W, "init_from": a.init_from, "steps": a.steps, "lr": a.lr, "bs": a.bs, "seed": a.seed, "head": a.head})
    json.dump({"arch": "coconut", "clr": c["clr"]}, open(os.path.join(a.out, "arch.json"), "w", newline="\n"))
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n")
    log.write(json.dumps({"step": 0, "curriculum_advance_to": None, "note": "inherited: initial model was at full depth"}) + "\n")
    t0 = time.time()
    T = lambda x: torch.as_tensor(x, device=dev)
    for step in range(1, a.steps + 1):
        m.train()
        for gp in opt.param_groups:
            gp["lr"] = sched(step)
        g = sample_graphs(rng, a.bs, split="train", max_bdepth=None)
        b = make_batch(rng, g, 16, dev)
        gl, br = long_chain_graphs(rng, a.bs, [2, 4, 6])
        bi = np.arange(a.bs)
        q = T(np.concatenate([gl.nodes[bi, br[k]] for k in ("A", "B", "C", "D")], 1)).long()
        y = teacher_targets("t8", tm, K, a.head, gl, br, q, dev)
        src, dst, st = T(gl.src), T(gl.dst), T(gl.start)
        with torch.no_grad():                                            # the student writes its own checkpoint
            m.eval()
            o = rollout(m, K, src, dst, st, K)
            att = o["attn1"][K - 1][:, a.head, :o["P"] - 1]
            order = torch.argsort(att, 1, descending=True)[:, :a.W].cpu().numpy()
            anc = T(np.stack([gl.dst[bi, order[:, w]] for w in range(a.W)][::-1], 1)).long()   # top-1 anchor last
            m.train()
        with torch.autocast("cuda", dtype=amp or torch.float32, enabled=amp is not None):
            ld, lu, _, _ = m.run_both(b["src"], b["dst"], b["start"], K, b["q"], b["tgt"])
            l1 = one_episode(m, K, K, src, dst, st, q)
            l2 = set_episode(m, K, src, dst, anc, q)
        ldeep = torch.maximum(l1.float(), l2.float())
        loss_n = F.binary_cross_entropy_with_logits(ld.float(), b["qy"]) + F.binary_cross_entropy_with_logits(lu.float(), b["ty"])
        loss_d = F.binary_cross_entropy_with_logits(ldeep, y)
        loss = loss_n + loss_d
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == a.steps:
            agree = float(((ldeep > 0).float() == y).float().mean())
            log.write(json.dumps({"step": step, "loss_normal": loss_n.item(), "loss_deep": loss_d.item(),
                                  "deep_agree_teacher": agree, "sec": time.time() - t0}) + "\n"); log.flush()
    c["clr_from"] = a.init_from
    torch.save({"model": m.state_dict(), "config": c, "step": a.steps}, os.path.join(a.out, "model.pt"))
    print("done", a.out, "sec %.0f" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
