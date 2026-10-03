"""THEORY §89 (fade to latent; the user's idea; note 89-1): text CoT whose thought inputs fade continuously into a
function of the previous hidden state, on §66's boxes task (4 boxes, S4 swaps, both question types; boxes/core.py).
Sequence (TC's format): prompt[:-3] + " cot:" | thought positions | " a:" + answer. The thought positions are every
CoT token of TC's format (" >" + 4 letters after every op: 5N positions), processed one at a time with the KV cache.
Input at thought position j (j = 1..5N), from the previous position's final hidden state h and logits:
    x_j = (1 - alpha) * E[y] + alpha * g(h),   y = argmax of the logits at the previous position (detached; the model's
    own token), or the gold CoT token j (fade_gold only).
alpha rises linearly 0 -> 1 over the first 60 % of updates, then 1 (pure latent). Loss: (1 - alpha) * mean CE of the
logits at the previous position against gold CoT token j (the thought loss) + CE on the answer tokens (teacher-forced
after " a:", weight 1).
Arms (--arm):
  fade_std      g(h) = mu_E + sigma_E * (h - mu_h) / sigma_h (per coordinate); mu_h, sigma_h an EMA (momentum 0.99) over
                thought-position hidden states from alpha = 0, frozen once alpha = 1 and used unchanged at test; mu_E,
                sigma_E the per-coordinate mean / std of GPT-2's token embeddings (fixed at init).
  fade_std_dec  fade_std + a constant-weight auxiliary linear head (768 -> the 5 CoT tokens " >", " A".." D") on h at the
                previous position decoding gold CoT token j; side loss only, never fed back (note 89-1); its decode
                accuracy at alpha = 1 is reported.
  fade_raw      g(h) = h.
  fade_soft     g(h) = softmax(logits(h)) @ E (soft token, T = 1 fixed; note 89-1); the mean distance of the thought input
                from the mean embedding is logged.
  fade_gold     fade_std with the gold CoT token in E[.] (leak control).
  coco_stage    Coconut's stage schedule: stage s in 0..8 (equal shares of the updates) makes the first s ops' 5 positions
                continuous (input = h, raw) with no thought loss on them; the remaining CoT positions take the model's own
                token (E[y]) with the thought CE; the final stage is all-latent with 5N positions.
TC / NC are boxes/train.py's tc / nc arms (same updates).
Training: GPT-2 small from base, §66's budget (8,000 updates, bs 32, AdamW lr 1e-4, wd 0.01, 100-step warmup + cosine),
one shape (N ~ U{1..8}, question type all / where) per update; bf16 autocast on compute capability >= 8, fp32 otherwise.
Eval (pure latent, alpha = 1; eval.json + rows.json): the §66 bench items (bench/tasks.py boxes_load, seed 6600: 50 per
N and question type) for N 1-16; accuracy at N 5-8 (primary), 1-8, 9-16, per question type; C1 removal: thought inputs
replaced by zeros, and by the mean thought input (over 512 training-distribution problems at alpha = 1); nearest-token
cosine of the thought inputs (max cosine to any token embedding, mean); fade_std_dec: the aux decode accuracy.
    python -m boxes.fade89 --arm fade_std --seed 1 --out boxes/runs/f89_fade_std_s1 [--device cuda:0]
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes.core import DIG_IDS, LET, LET_IDS, NB, greedy, make_problem, prompt, tc_text, tok, truth  # noqa: E402
from boxes.train import load_model  # noqa: E402

ARMS = ("fade_std", "fade_std_dec", "fade_raw", "fade_soft", "fade_gold", "coco_stage")
COT_IDS = tok(" >") + LET_IDS                                          # the 5 CoT token ids (aux decoder classes)


class Fade(nn.Module):
    def __init__(self, m, arm):
        super().__init__()
        self.m = m; self.arm = arm
        E = m.w("wte.weight").detach()
        self.register_buffer("muE", E.mean(0).clone()); self.register_buffer("sdE", E.std(0).clone())
        self.register_buffer("muH", torch.zeros(768)); self.register_buffer("sdH", torch.ones(768))
        self.register_buffer("frozen", torch.tensor(0)); self.register_buffer("seen", torch.tensor(0))
        self.aux = nn.Linear(768, len(COT_IDS)) if arm == "fade_std_dec" else None
        self.cot = torch.tensor(COT_IDS)
        self.inv = None                         # §94 INV: letter -> box side head (set by --inv_dec; never fed back)

    def g(self, h, update_stats):
        if self.arm in ("fade_raw", "coco_stage"):
            return h
        if self.arm == "fade_soft":
            return F.softmax(self.lg(h).float(), -1).to(self._W.dtype) @ self._W
        if update_stats and not int(self.frozen):                          # collected; the EMA is applied after the update
            self._acc.append(h.detach().float())
        return (self.muE + self.sdE * (h.float() - self._mu) / self._sd).to(h.dtype)

    def lg(self, h):
        return h.to(self._W.dtype) @ self._W.T

    @torch.no_grad()
    def update_stats(self):
        """EMA (momentum 0.99) over this update's thought-position hidden states; from alpha = 0, frozen at alpha = 1."""
        if self._acc and not int(self.frozen):
            H = torch.cat(self._acc); hm, hs = H.mean(0), H.std(0) + 1e-5
            if int(self.seen) == 0:
                self.muH.copy_(hm); self.sdH.copy_(hs)
            else:
                self.muH.mul_(0.99).add_(0.01 * hm); self.sdH.mul_(0.99).add_(0.01 * hs)
            self.seen += 1
        self._acc = []

    def forward(self, batch, dev, alpha, stage=None, train=True, ablate=None, mean_in=None, collect=None):
        """Returns (loss terms dict) when train, else (answer texts, aux decode correct count, n aux)."""
        m = self.m; B = len(batch); N = len(batch[0]["ops"]); qt = batch[0]["qtype"]
        self._mu, self._sd, self._acc = self.muH.clone(), self.sdH.clone(), []   # stats snapshot for this pass
        W = m.w("wte.weight")                                              # cast once per pass (memory: autocast re-casts the
        self._W = W.to(torch.get_autocast_gpu_dtype()) if torch.is_autocast_enabled() else W   # 50k x 768 matrix per call)
        pre = torch.tensor([tok(prompt(p)[:-3] + " cot:") for p in batch], device=dev)
        cot = torch.tensor([tok(tc_text(p)[len(" cot:"):-len(" a:")]) for p in batch], device=dev)   # (B, 5N) gold CoT
        assert cot.shape[1] == 5 * N
        h, past = m(m.embed(pre)); hl = h[:, -1]
        th_ce, aux_ce, aux_ok, soft_dist, inv_ce = [], [], 0, [], []
        if self.inv is not None and train:                                 # §94: letter -> box of the state after each op
            from boxes.core import apply as _apply
            inv_t = torch.tensor([[[_apply(p["start"], p["ops"][:i + 1]).index(c) for c in LET] for i in range(N)]
                                  for p in batch], device=dev)                 # (B, N, 4)
        for j in range(5 * N):
            lg = self.lg(hl)
            latent_pos = self.arm == "coco_stage" and j // 5 < (stage if stage is not None else N)
            if train and not latent_pos and alpha < 1:
                th_ce.append(F.cross_entropy(lg.float(), cot[:, j], reduction="none"))
            if self.aux is not None:
                al = self.aux(hl.float()); tgt = (cot[:, j][:, None] == self.cot.to(dev)[None]).float().argmax(-1)
                aux_ce.append(F.cross_entropy(al, tgt)); aux_ok += int((al.argmax(-1) == tgt).sum())
            if self.inv is not None and train:
                inv_ce.append(F.cross_entropy(self.inv(hl.float()).view(-1, 4), inv_t[:, j // 5].reshape(-1)))
            if self.arm == "coco_stage":
                x = hl if latent_pos else m.embed(lg.argmax(-1).detach())
            else:
                y = cot[:, j] if self.arm == "fade_gold" else lg.argmax(-1).detach()
                gh = self.g(hl, update_stats=train)
                x = gh if alpha >= 1 else (1 - alpha) * m.embed(y) + alpha * gh
                if self.arm == "fade_soft":
                    soft_dist.append((gh.float() - self.muE).norm(dim=-1).mean().item())
            if ablate == "zero":
                x = torch.zeros_like(x)
            elif ablate == "mean":
                x = mean_in.to(x.dtype)[None].expand_as(x)
            if collect is not None:
                collect.append(x.detach().float())
            h, past = m(x[:, None, :], past); hl = h[:, -1]
        mid = torch.tensor([tok(" a:")] * B, device=dev); h, past = m(m.embed(mid), past); hl = h[:, -1]
        if not train:
            nans = NB if qt == "all" else 1
            ans = greedy(m, hl, past, nans, None, LET_IDS if qt == "all" else DIG_IDS)
            from boxes.core import ENC
            return [" " + " ".join(ENC.decode([int(t)]).strip() for t in row) for row in ans], aux_ok, 5 * N * B
        tgt = torch.tensor([tok(truth(p)) for p in batch], device=dev)
        hs = [hl[:, None]]
        if tgt.shape[1] > 1:
            h3, _ = m(m.embed(tgt[:, :-1]), past); hs.append(h3)
        lga = self.lg(torch.cat(hs, 1))
        out = {"answer": F.cross_entropy(lga.reshape(-1, lga.shape[-1]).float(), tgt.reshape(-1))}
        if th_ce:
            out["thought"] = torch.stack(th_ce).mean()
        if aux_ce:
            out["aux"] = torch.stack(aux_ce).mean()
        if inv_ce:
            out["inv"] = torch.stack(inv_ce).mean()
        if soft_dist:
            out["soft_dist"] = float(np.mean(soft_dist))
        return out


def alpha_at(step, steps, frac=0.6):
    return min(1.0, step / (frac * steps))


def stage_at(step, steps, n_stages=9):
    return min(n_stages - 1, (step - 1) * n_stages // steps)


def train(a):
    dev = a.device; torch.manual_seed(a.seed); rng = random.Random(a.seed)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    f = Fade(load_model(a.init, dev), a.arm).to(dev)
    if a.inv_dec:
        f.inv = nn.Linear(768, 16).to(dev)                                 # §94 STD-DEC-INV: 4 letters x 4 boxes
    opt = torch.optim.AdamW([p for p in f.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = {}
    for step in range(1, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        al = 1.0 if a.arm == "coco_stage" else alpha_at(step, a.steps)
        st = stage_at(step, a.steps) if a.arm == "coco_stage" else None
        if al >= 1 and a.arm != "coco_stage":
            f.frozen.fill_(1)
        n = rng.randint(1, 8); qt = rng.choice(["all", "where"]); batch = [make_problem(rng, n, qt) for _ in range(a.bs)]
        opt.zero_grad(); mb = a.bs // a.grad_acc                            # memory: micro-batches, same effective batch
        for i in range(a.grad_acc):
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                out = f(batch[i * mb:(i + 1) * mb], dev, al, stage=st, train=True)
                loss = out["answer"] + ((1 - al) * out["thought"] if "thought" in out and a.arm != "coco_stage" else 0) \
                    + (out["thought"] if "thought" in out and a.arm == "coco_stage" else 0) + (out["aux"] if "aux" in out else 0) + (out["inv"] if "inv" in out else 0)
            (loss / a.grad_acc).backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0); opt.step()
        f.update_stats()
        for k, v in out.items():
            hist.setdefault(k, []).append(v.item() if torch.is_tensor(v) else v)
        if step % 100 == 0 or step == a.steps:
            rec = {"step": step, "alpha": round(al, 4), "stage": st, **{k: float(np.mean(v)) for k, v in hist.items()},
                   "sec": round(time.time() - t0)}
            hist = {}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    torch.save(f.state_dict(), os.path.join(a.out, "model.pt"))
    evaluate(f, a, dev, amp, round(time.time() - t0))


@torch.no_grad()
def evaluate(f, a, dev, amp, train_sec):
    f.eval(); from bench.tasks import boxes_load
    items = [it for it in boxes_load()[0] if it["depth"] <= 16]
    if a.smoke:                                                           # plumbing only
        items = [it for it in items if it["depth"] in (1, 6, 9) and int(it["id"].split(":")[-1]) < 2]
    groups = {}
    for it in items:
        groups.setdefault((it["depth"], it["raw"]["qtype"]), []).append(it)
    coll = []; rng = random.Random(8900)                                  # mean thought input (pure latent)
    for _ in range(1 if a.smoke else 16):
        n = rng.randint(1, 8); qt = rng.choice(["all", "where"]); b = [make_problem(rng, n, qt) for _ in range(32)]
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            f(b, dev, 1.0, stage=None, train=False, collect=coll)
    mean_in = torch.cat(coll).mean(0)
    E = f.m.w("wte.weight").float(); En = F.normalize(E, dim=-1)
    sample = torch.cat(coll)[:4096]; near = (F.normalize(sample, dim=-1) @ En.T).max(-1).values
    res = {"arm": a.arm, "seed": a.seed, "train_sec": train_sec, "eval_dtype": str(amp),
           "nearest_token_cos_mean": float(near.mean()), "thought_input_norm_mean": float(sample.norm(dim=-1).mean()),
           "frozen_stats": bool(int(f.frozen)), "modes": {}}
    rows = []
    for mode in ("none", "zero", "mean"):
        ok = {}; aux_ok = aux_n = 0
        for (n, qt), its in sorted(groups.items()):
            for s in range(0, len(its), 64):
                b = its[s:s + 64]
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    txt, ao, an = f([it["raw"] for it in b], dev, 1.0, stage=None, train=False,
                                    ablate=None if mode == "none" else mode, mean_in=mean_in)
                aux_ok += ao; aux_n += an
                for it, t in zip(b, txt):
                    c = t.strip() == it["answer"].strip(); ok.setdefault((n, qt), []).append(c)
                    if mode == "none":
                        rows.append({"id": it["id"], "N": n, "qtype": qt, "pred": t, "correct": c})
        band = lambda lo, hi, q=None: float(np.mean([c for (n, qt), cs in ok.items() for c in cs if lo <= n <= hi and (q is None or qt == q)]))  # noqa: E731
        r = {"acc_N5_8": band(5, 8), "acc_N1_8": band(1, 8), "acc_N9_16": band(9, 16),
             "acc_N5_8_all": band(5, 8, "all"), "acc_N5_8_where": band(5, 8, "where"),
             "by_N": {str(n): float(np.mean([c for (nn, qt), cs in ok.items() for c in cs if nn == n])) for n in range(1, 17)}}
        if f.aux is not None and mode == "none":
            r["aux_decode_acc_alpha1"] = aux_ok / max(1, aux_n)
        res["modes"][mode] = r
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    json.dump(rows, open(os.path.join(a.out, "rows.json"), "w", newline="\n"))
    print("EVAL", json.dumps({m_: r["acc_N5_8"] for m_, r in res["modes"].items()}), flush=True); print("DONE", a.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad_acc", type=int, default=2, help="micro-batches per update (bs 32 x 1 OOMs 8 GB at N 8)")
    ap.add_argument("--inv_dec", action="store_true", help="§94: letter -> box side head at thought positions (weight 1)")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--init", default=os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"))
    ap.add_argument("--smoke", action="store_true", help="plumbing only: tiny eval")
    ap.add_argument("--eval_only", action="store_true", help="evaluate <out>/model.pt only (a training run's eval was stopped)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.eval_only:
        eval_only(a)
    else:
        train(a)


def eval_only(a):
    """Re-run evaluate() on a finished training run's model.pt (same args; train_sec from the last log line)."""
    dev = a.device; amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    f = Fade(load_model(a.init, dev), a.arm).to(dev)
    if a.inv_dec:
        f.inv = nn.Linear(768, 16).to(dev)
    f.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev))
    last = [json.loads(l) for l in open(os.path.join(a.out, "log.jsonl"))][-1]
    assert last["step"] == a.steps, "model.pt is not from a finished run"
    evaluate(f, a, dev, amp, last["sec"])


if __name__ == "__main__":
    main()
