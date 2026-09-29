"""Theory stream trainer: latent/train.py (copied 2026-09-29, recipe unchanged) plus architecture and schedule options.

  --arch coconut|markov|markov_l2   model variant (theory/arch.py); written to <out>/arch.json
  --budget_match                    latent budget per step = min(K, curriculum depth + 1) (no dwell needed; eval uses K)
  --overrun M                       latent budget drawn from K..K+M each step (the last thought is sometimes fed on)
  --device cpu                      eager step instead of the CUDA-graph step (smoke tests)
Example:
    python theory/train_arch.py --arch markov --K 6 --seed 0 --device cuda:0 --out theory/runs/mk_s0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latent.data import make_batch  # noqa: E402
from latent.graphs import E, NPB, WMAX, sample_graphs, set_shapes  # noqa: E402
from latent.graphstep import GraphedStep  # noqa: E402
from arch import build  # noqa: E402


class EagerStep:
    """CPU fallback with the GraphedStep loss (smoke tests only)."""
    def __init__(self, model, opt, upfront_weight, clip=1.0):
        self.model, self.opt, self.clip = model, opt, clip
        self.w_children = torch.tensor(0.0); self.w_aux = torch.tensor(0.0); self.w_up = torch.tensor(float(upfront_weight))

    def step(self, b, K):
        self.opt.zero_grad()
        ld, lu, lp, lc = self.model.run_both(b["src"], b["dst"], b["start"], K, b["q"], b["tgt"], b["pq"], b["cq"])
        self.loss_d = F.binary_cross_entropy_with_logits(ld, b["qy"]); self.loss_u = F.binary_cross_entropy_with_logits(lu, b["ty"])
        self.loss_p = F.cross_entropy(lp.flatten(0, 1), b["py"].flatten())
        self.loss_c = F.binary_cross_entropy_with_logits(lc, b["cy"]) * lc.shape[-1] / 4
        self.loss = self.loss_d + self.w_up * self.loss_u + self.w_aux * self.loss_p + self.w_aux * self.w_children * self.loss_c
        self.loss.backward()
        self.gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.opt.step()


class AnytimeMixin:
    """Anytime (minimum-progress) supervision, --anytime C: after every latent step k the delayed/upfront answers are
    supervised for reachable nodes within distance k + C and for unreachable nodes; reachable nodes farther than k + C
    are masked (no loss), so reaching them early is never penalised. The final step k = K is the ordinary loss."""
    def _anytime_loss(self, ld, lu, qd, qy, tgtd, ty, C):
        K = ld.shape[0]
        ks = torch.arange(1, K + 1, device=ld.device, dtype=qd.dtype)
        md = ((qy[None] == 0) | (qd[None] <= ks[:, None, None] + C)).float()
        md[-1] = 1.0
        mu = ((ty[None] == 0) | (tgtd[None] <= ks[:, None] + C)).float()
        mu[-1] = 1.0
        bd = F.binary_cross_entropy_with_logits(ld.float(), qy[None].expand_as(ld), reduction="none")
        bu = F.binary_cross_entropy_with_logits(lu.float(), ty[None].expand_as(lu), reduction="none")
        return (bd * md).sum() / md.sum(), (bu * mu).sum() / mu.sum()


class AnytimeGraphedStep(AnytimeMixin, GraphedStep):
    def __init__(self, *a, anytime_C=1, **kw):
        self.C = anytime_C
        super().__init__(*a, **kw)

    def _fwd_bwd(self, K):
        from arch import run_both_anytime
        i = self.inp
        if "qd" not in i:            # created during warm-up (before capture); step() then copies batch values in
            i["qd"] = torch.zeros_like(i["q"]); i["tgtd"] = torch.zeros_like(i["tgt"])
        with torch.autocast("cuda", dtype=self.amp or torch.float32, enabled=self.amp is not None, cache_enabled=False):
            ld, lu, lp, lc = run_both_anytime(self.model, i["src"], i["dst"], i["start"], K, i["q"], i["tgt"], i["pq"], i["cq"])
        self.loss_d, self.loss_u = self._anytime_loss(ld, lu, i["qd"], i["qy"], i["tgtd"], i["ty"], self.C)
        self.loss_p = F.cross_entropy(lp.float().flatten(0, 1), i["py"].flatten())
        self.loss_c = F.binary_cross_entropy_with_logits(lc.float(), i["cy"]) * lc.shape[-1] / 4
        self.loss = (self.loss_d + self.w_up * self.loss_u + self.w_aux * self.loss_p
                     + self.w_aux * self.w_children * self.loss_c)
        self.loss.backward()
        self.gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.opt.step()


class AnytimeEagerStep(AnytimeMixin):
    def __init__(self, model, opt, upfront_weight, anytime_C=1, clip=1.0):
        self.model, self.opt, self.clip, self.C = model, opt, clip, anytime_C
        self.w_children = torch.tensor(0.0); self.w_aux = torch.tensor(0.0); self.w_up = torch.tensor(float(upfront_weight))

    def step(self, b, K):
        from arch import run_both_anytime
        self.opt.zero_grad()
        ld, lu, lp, lc = run_both_anytime(self.model, b["src"], b["dst"], b["start"], K, b["q"], b["tgt"], b["pq"], b["cq"])
        self.loss_d, self.loss_u = self._anytime_loss(ld, lu, b["qd"], b["qy"], b["tgtd"], b["ty"], self.C)
        self.loss_p = F.cross_entropy(lp.flatten(0, 1), b["py"].flatten())
        self.loss_c = F.binary_cross_entropy_with_logits(lc, b["cy"]) * lc.shape[-1] / 4
        self.loss = self.loss_d + self.w_up * self.loss_u + self.w_aux * self.loss_p + self.w_aux * self.w_children * self.loss_c
        self.loss.backward()
        self.gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.opt.step()


def evaluate(model, K, n=2048, split="discovery", seed=12345, device="cuda", Q=16, widths=None,
             max_bdepth=None):
    """Accuracy by format / width / depth on a fixed held-out-family sample."""
    rng = np.random.default_rng(seed)
    widths = widths or range(1, WMAX + 1)
    g = sample_graphs(rng, n, widths=widths, split=split, max_bdepth=max_bdepth)
    model.eval()
    res = {"delayed": [], "upfront": []}
    with torch.no_grad():
        for i in range(0, n, 128):          # small eval chunks: lower peak memory next to the captured graph
            sub = g.subset(np.arange(i, min(n, i + 128)))
            b = make_batch(rng, sub, Q, device)
            o = model.run(b["src"], b["dst"], b["start"], K, qnodes=b["q"])
            pd = (o["logit"] > 0).float()
            res["delayed"].append(np.stack([np.repeat(sub.width[:, None], Q, 1).ravel(),
                                            b["qd"].cpu().numpy().ravel(),
                                            (pd == b["qy"]).cpu().numpy().ravel(),
                                            b["qy"].cpu().numpy().ravel()], 1))
            o = model.run(b["src"], b["dst"], b["start"], K, target=b["tgt"])
            pu = (o["logit"][:, 0] > 0).float()
            res["upfront"].append(np.stack([sub.width, b["tgtd"].cpu().numpy(),
                                            (pu == b["ty"]).cpu().numpy(), b["ty"].cpu().numpy()], 1))
    model.train()
    out = {}
    for f, rows in res.items():
        a = np.concatenate(rows)
        pos, neg = a[a[:, 3] == 1], a[a[:, 3] == 0]
        out[f] = {"acc": float(a[:, 2].mean()),
                  "bal_acc": float(0.5 * (pos[:, 2].mean() + neg[:, 2].mean())),
                  "by_width": {int(w): float(a[a[:, 0] == w, 2].mean()) for w in np.unique(a[:, 0])},
                  "by_depth": {int(d): float(a[a[:, 1] == d, 2].mean()) for d in np.unique(a[:, 1])}}
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="coconut", choices=["coconut", "pause"])
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--Q", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--upfront_weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--curriculum", default="", help='e.g. "1:1500,2:1500,3:1500,4:1500"')
    p.add_argument("--emb", default="slots", choices=["slots", "untied"])
    p.add_argument("--d_id", type=int, default=64)
    p.add_argument("--init", default="torch", choices=["gpt2", "torch"])
    p.add_argument("--aux_steps", type=int, default=0, help="steps with PARENT-lookup aux loss at weight 1")
    p.add_argument("--aux_anneal", type=int, default=0, help="steps to anneal aux weight 1 -> 0 afterwards")
    p.add_argument("--children_weight", type=float, default=0.0, help="relative weight of CHILDREN lookup aux")
    p.add_argument("--gate", type=float, default=0.9)
    p.add_argument("--depth_cap", type=int, default=None,
                   help="PROTOCOL6 addendum M: never train deeper than this within-branch depth (the gated curriculum "
                        "still advances to 'full', which then means the cap)")
    p.add_argument("--feedback", default="post_ln", choices=["post_ln", "pre_ln"])
    p.add_argument("--save_every", type=int, default=0, help="also keep ckpts/step_N.pt every N steps and at curriculum advances")
    p.add_argument("--no_step_emb", action="store_true", help="ablation: latent steps get no step embedding")
    p.add_argument("--K_random", default="", help='"lo:hi": latent budget drawn uniformly from lo..hi per batch '
                                                  "(evaluation and the curriculum gate use K = hi)")
    p.add_argument("--shapes", default="default", choices=["default", "fork"],
                   help="branch shapes: default random trees, or fork = chains and two-armed forks (frontier size != W)")
    p.add_argument("--p_fork", type=float, default=0.5, help="--shapes fork: probability a branch is a fork")
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp32"],
                   help="auto: bf16 autocast on sm>=80, fp32 otherwise; fp32 reproduces RTX 2070 runs on newer cards")
    p.add_argument("--resume", action="store_true", help="continue from <out>/state.pt (full training state) if present")
    p.add_argument("--arch", default="coconut", choices=["coconut", "markov", "markov_l2", "restore", "markov_restore", "markov_sigread", "markov_restore_sigread"])
    p.add_argument("--budget_match", action="store_true")
    p.add_argument("--overrun", type=int, default=0)
    p.add_argument("--anytime", type=int, default=None, help="anytime supervision slack C (queries after step k must "
                   "know reachable nodes within distance k + C); off by default")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    Ks = [a.K]
    if a.K_random:
        lo, hi = map(int, a.K_random.split(":"))
        Ks, a.K = list(range(lo, hi + 1)), hi

    set_shapes(a.shapes, a.p_fork)
    os.makedirs(a.out, exist_ok=True)
    state_path = os.path.join(a.out, "state.pt")
    resume = a.resume and os.path.exists(state_path)
    if not resume:
        json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=2)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    # TF32 does not exist on the RTX 2070, so a forced-fp32 run disables it to match those runs
    torch.backends.cuda.matmul.allow_tf32 = a.amp != "fp32"
    os.makedirs(a.out, exist_ok=True)
    dev = a.device

    json.dump({"arch": a.arch, "budget_match": a.budget_match, "overrun": a.overrun, "anytime": a.anytime},
              open(os.path.join(a.out, "arch.json"), "w"))
    model = build(a.arch, d=a.d, n_layers=a.layers, n_heads=a.heads, k_max=max(a.K + a.overrun, 8), mode=a.mode,
                  emb=a.emb, d_id=a.d_id, init=a.init, feedback=a.feedback, step_emb=not a.no_step_emb).to(dev)
    lr_t = torch.tensor(a.lr, device=dev)
    cpu = dev == "cpu"
    opt = torch.optim.AdamW(model.parameters(), lr=lr_t, weight_decay=a.wd, betas=(0.9, 0.98),
                            capturable=not cpu)
    lr_at = lambda s: a.lr * min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    # bf16 autocast where supported; plain fp32 on sm<80 (RTX 2070) or when --amp fp32 is forced
    if a.amp == "auto":
        amp = None if dev == "cpu" else torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else None
    else:
        amp = torch.bfloat16 if a.amp == "bf16" else None
    if a.budget_match:
        Ks = sorted(set(range(2, a.K + 1)))
    if a.overrun:
        Ks = sorted(set(Ks) | set(range(a.K, a.K + a.overrun + 1)))
    if a.anytime is not None:
        gstep = (AnytimeEagerStep(model, opt, a.upfront_weight, anytime_C=a.anytime) if cpu else
                 AnytimeGraphedStep(model, opt, Ks, a.bs, E, a.Q, a.upfront_weight, dev, amp, anytime_C=a.anytime))
    else:
        gstep = EagerStep(model, opt, a.upfront_weight) if cpu else GraphedStep(model, opt, Ks, a.bs, E, a.Q, a.upfront_weight, dev, amp)
    gstep.w_children.fill_(a.children_weight)
    # depth curriculum: "1:1500,2:1500" -> max within-branch depth 1 for 1500 steps, then 2, ...
    # or "gated": start at max depth 1, advance when held-out delayed bal-acc at the
    # current stage >= gate (checked every 500 steps), full depth after stage NPB-2.
    stages = []
    gated = a.curriculum == "gated"
    cur = {"d": 1, "hist": []}
    for tok in filter(None, "" if gated else a.curriculum.split(",")):
        dmax, n = tok.split(":")
        stages.append((int(dmax), int(n)))
    bounds = np.cumsum([n for _, n in stages]) if stages else []

    def max_depth_at(s):
        d = _max_depth_at(s)
        if a.depth_cap is None:
            return d
        return a.depth_cap if d is None else min(d, a.depth_cap)

    def _max_depth_at(s):
        if gated:
            return cur["d"]
        for (dm, _), bnd in zip(stages, bounds):
            if s <= bnd:
                return dm
        return None

    def aux_at(s):
        """Auxiliary PARENT-lookup weight: 1 until aux_steps, then linear anneal to 0."""
        if s <= a.aux_steps:
            return 1.0
        return max(0.0, 1.0 - (s - a.aux_steps) / max(1, a.aux_anneal))

    def save_atomic(obj, path):
        """Write to a temporary file and rename, so a crash mid-write never leaves a truncated file."""
        torch.save(obj, path + ".tmp")
        os.replace(path + ".tmp", path)

    def save_state(step, sec):
        save_atomic({"model": model.state_dict(), "step": step, "sec": sec, "cur": cur,
                     "opt": [{k: v.detach().clone() for k, v in opt.state[p_].items()} for p_ in model.parameters()],
                     "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                     "cuda_rng": None if cpu else torch.cuda.get_rng_state(dev)}, state_path)

    start, sec0 = 1, 0.0
    log_path = os.path.join(a.out, "log.jsonl")
    if resume:
        st = torch.load(state_path, map_location=dev, weights_only=False)
        with torch.no_grad():                         # copy in place: the captured graph keeps its tensors
            model.load_state_dict(st["model"])
            for p_, sd in zip(model.parameters(), st["opt"]):
                for k, v in opt.state[p_].items():
                    v.copy_(sd[k])
        rng.bit_generator.state = st["np_rng"]
        torch.set_rng_state(st["torch_rng"].cpu())
        if not cpu:
            torch.cuda.set_rng_state(st["cuda_rng"].cpu(), dev)
        cur.clear(); cur.update(st["cur"])
        start, sec0 = st["step"] + 1, st["sec"]
        lines = open(log_path).read().splitlines() if os.path.exists(log_path) else []
        keep = [l for l in lines if json.loads(l).get("step", 0) <= st["step"]]
        with open(os.path.join(a.out, "log_discarded.jsonl"), "a") as f:
            f.write("".join(l + "\n" for l in lines if l not in keep))
        with open(log_path, "w") as f:
            f.write("".join(l + "\n" for l in keep) + json.dumps({"step": st["step"], "resumed_from": st["step"]}) + "\n")
        print(f"[{a.out}] resumed from step {st['step']}", flush=True)
    log = open(log_path, "a")
    t0, peak = time.time() - sec0, 0

    def keep_ckpt(step):
        os.makedirs(os.path.join(a.out, "ckpts"), exist_ok=True)
        save_atomic({"model": model.state_dict(), "config": vars(a), "step": step},
                    os.path.join(a.out, "ckpts", f"step_{step:05d}.pt"))

    for step in range(start, a.steps + 1):
        g = sample_graphs(rng, a.bs, split="train", max_bdepth=max_depth_at(step))
        b = make_batch(rng, g, a.Q, dev)
        lr_t.fill_(lr_at(step))
        gstep.w_aux.fill_(aux_at(step))
        # K drawn from its own (seed, step) stream: the data stream is untouched and resume stays exact
        K_step = int(np.random.default_rng([a.seed, step]).integers(Ks[0], Ks[-1] + 1)) if a.K_random else a.K
        if a.budget_match:
            dcur = max_depth_at(step)
            K_step = a.K if dcur is None else min(a.K, dcur + 1)
        if a.overrun:
            K_step = K_step + int(np.random.default_rng([a.seed, step, 7]).integers(0, a.overrun + 1))
        gstep.step(b, K_step)
        loss, loss_d, gn = gstep.loss, gstep.loss_d, gstep.gn
        if step % 200 == 0:
            peak = 0 if cpu else max(peak, torch.cuda.max_memory_allocated(dev) / 2**20)
            rec = {"step": step, "loss": loss.item(), "loss_delayed": loss_d.item(),
                   "loss_upfront": gstep.loss_u.item(), "loss_parent": gstep.loss_p.item(),
                   "loss_children": gstep.loss_c.item(),
                   "w_aux": aux_at(step), "max_bdepth": max_depth_at(step), "K": K_step,
                   "gn": gn.item(), "lr": lr_at(step), "sec": time.time() - t0, "peak_mb": peak}
            log.write(json.dumps(rec) + "\n"); log.flush()
        if gated and cur["d"] is not None and step % 500 == 0:
            K_gate = min(a.K, cur["d"] + 1) if a.budget_match else a.K     # budget_match: gate at the stage's budget
            ev = evaluate(model, K_gate, n=512, device=dev, max_bdepth=cur["d"], seed=step)
            if ev["delayed"]["bal_acc"] >= a.gate:
                cur["hist"].append((step, cur["d"]))
                cur["d"] = None if cur["d"] >= NPB - 2 else cur["d"] + 1
                log.write(json.dumps({"step": step, "curriculum_advance_to": cur["d"],
                                      "acc": ev["delayed"]["bal_acc"]}) + "\n"); log.flush()
                print(f"[{a.out}] step {step}: curriculum -> max_bdepth {cur['d']}", flush=True)
                if a.save_every:
                    keep_ckpt(step)
        if a.save_every and step % a.save_every == 0:
            keep_ckpt(step)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(model, a.K, n=1024, device=dev)
            rec = {"step": step, "eval": ev, "sec": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[{a.out}] step {step} loss {loss.item():.4f} delayed {ev['delayed']['bal_acc']:.3f} "
                  f"upfront {ev['upfront']['bal_acc']:.3f} depth {ev['delayed']['by_depth']} "
                  f"{time.time()-t0:.0f}s", flush=True)
            save_atomic({"model": model.state_dict(), "config": vars(a), "step": step},
                        os.path.join(a.out, "model.pt"))
            save_state(step, time.time() - t0)


if __name__ == "__main__":
    main()
