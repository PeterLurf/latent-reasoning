"""THEORY §99 (note 99-1): speeding up the latent hand-off on mult 4x4 (§98's GEN-M recipe and code path, boxes/mult98.py
imported unchanged). Arms (--arm99) as flags on GEN-M:
  ctrl            plain GEN-M (eligible from 0)
  kcurr           training chunk length 1 (steps 1-2k), 2 (2k-6k), then 4; a short chunk starts from the true previous step
                  text with the history cut (as GEN-M's later chunks); test k = 4 (eligible from 6k)
  std             slot input g(h) = mu_E + sd_E * (h - mu_h) / max(sd_h, 0.05 * median(sd_h)), per coordinate; mu_h, sd_h an
                  EMA (0.99) over slot-input hiddens of each update (first update initialises), frozen at the stop and
                  used unchanged at test; mu_E, sd_E = GPT-2's initial wte statistics (fade89's g) (eligible from 0)
  textfirst       steps 1-3k train TC-M's text-CoT objective (mult98 seq arm "tc", micro 2), then GEN-M from those weights;
                  the text steps count (eligible from 3k)
  textfirst_mask  textfirst with §98's masked fork (--fork mask) (eligible from 3k)
  combo           text CoT 1-3k, then k = 1 (3k-5k), k = 2 (5k-9k), then k = 4, STD throughout the latent phase (from 9k)
  tfslot          each training update, at each slot position j < c (not the chunk's last slot) a coin with probability
                  rho(step) (one coin per position per batch) replaces slot j's latent with the re-embedded GOLD step-j
                  text tokens, in context, no cut; that slot's fork CE is skipped. rho = 1 - step / 12k, 0 from 12k (from 12k)
Every 1k steps, on the repo's 4x4 valid split (1,000 items; never the bench test): per-slot step-text decode and answer
accuracy in the k = 4 test rollout (curve entries {step, per_slot, val_acc, gold_share, k_train, phase}). A run stops at the
first ELIGIBLE log (step >= eligible_from) with every slot >= 0.95 and answer >= 0.95 (steps_to_threshold), else at 24k
(censored: steps_to_threshold null). LR schedule: §98's (100-step warmup + cosine over the 24k cap), unchanged by the stop.
At the stop: bench test answer accuracy, per-slot decode, zero / mean removal of slot inputs 1..k-1 (answer accuracy, as
§98), wall-clock, and training FLOPs (Meter forward FLOPs per trained problem of each step's configuration x 3 x bs, summed;
tfslot updates are counted at the latent configuration).
    python -m boxes.accel99 --arm99 kcurr --seed 1 --out boxes/runs_mult/a99_kcurr_4x4_s1
"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes.mult98 import (DIGIT_IDS, TOK, MeterR, MultGen, gen_mean_inputs, gen_rollout, load, seq_batch,  # noqa: E402
                          seq_forward)

ARMS = {"ctrl": 0, "kcurr": 6000, "std": 0, "textfirst": 3000, "textfirst_mask": 3000, "combo": 9000, "tfslot": 12000}


def config(arm, step):
    """-> (phase 'text' | 'latent', training chunk length, gold share rho) at this update."""
    if arm in ("textfirst", "textfirst_mask", "combo") and step <= 3000:
        return "text", 4, 0.0
    if arm == "kcurr":
        return "latent", 1 if step <= 2000 else (2 if step <= 6000 else 4), 0.0
    if arm == "combo":
        return "latent", 1 if step <= 5000 else (2 if step <= 9000 else 4), 0.0
    if arm == "tfslot":
        return "latent", 4, max(0.0, 1.0 - step / 12000)
    return "latent", 4, 0.0


class MultGen99(MultGen):
    def __init__(self, g, fork, std):
        super().__init__(g, fork)
        self.std = std
        E = g.w("wte.weight").detach()
        self.register_buffer("muE", E.mean(0).clone()); self.register_buffer("sdE", E.std(0).clone())
        self.register_buffer("muH", torch.zeros(768)); self.register_buffer("sdH", torch.ones(768)); self.register_buffer("seen", torch.tensor(0))
        self._raw = []; self.gold = None

    def g_std(self, h):
        sd = torch.maximum(self.sdH, 0.05 * self.sdH.median())
        return (self.muE + self.sdE * (h.float() - self.muH) / sd).to(h.dtype)

    @torch.no_grad()
    def update_stats(self):
        if self._raw:
            H = torch.cat(self._raw); hm, hs = H.mean(0), H.std(0) + 1e-5
            if int(self.seen) == 0:
                self.muH.copy_(hm); self.sdH.copy_(hs)
            else:
                self.muH.mul_(0.99).add_(0.01 * hm); self.sdH.mul_(0.99).add_(0.01 * hs)
            self.seen += 1
        self._raw = []

    def chunk(self, prefix, start, ops, step_tgt=None, ans_tgt=None, step_len=None, ans_len=None, decode="all",
              ablate=None, mean_in=None, meter=None, meter_x=None, snap=False, collect=False):
        """mult98.MultGen.chunk with (a) STD slot inputs and (b) tfslot gold text at the slot positions in self.gold
        (a set of j; training only; the gold tokens are step_tgt[j])."""
        B, c = ops.shape; g = self.g
        parts = [g.embed(prefix)] + ([g.embed(start)] if start is not None else []) + [g.embed(self.digtok[ops[:, 0]])[:, None]]
        x = torch.cat(parts, 1); past = None; losses, decs, ins = [], [], []; ans = None
        for j in range(c):
            ctx = 0 if past is None else past[0][0].shape[2]
            h, past = g(x, past)
            if meter:
                meter.step(x.shape[1], ctx, 0)
            if self.gold and j in self.gold and step_tgt is not None and j < c - 1:   # tfslot: gold step text in place of slot j
                gt = step_tgt[j]; ctx = past[0][0].shape[2]; _, past = g(g.embed(gt), past)
                ins.append(torch.zeros_like(h[:, -1]))
            else:
                s_in = h[:, -1]
                if self.std:
                    if collect:
                        self._raw.append(s_in.detach().float())
                    s_in = self.g_std(s_in)
                if ablate in ("zero", "mean") and j < c - 1:
                    s_in = torch.zeros_like(s_in) if ablate == "zero" else mean_in[j][None].expand_as(s_in).to(s_in.dtype)
                ins.append(s_in)
                ctx = past[0][0].shape[2]; hs, past = g(s_in[:, None], past)
                if meter:
                    meter.step(1, ctx, 0)
                hslot = hs[:, 0]
                if step_tgt is not None:
                    losses.append(self.fork_tf(hslot, past, step_tgt[j]))
                if step_len is not None and (decode == "all" or j == c - 1):
                    decs.append(self.fork_greedy(hslot, past, step_len[j], meter if (snap and j == c - 1) else meter_x))
            if j + 1 < c:
                x = g.embed(self.digtok[ops[:, j + 1]])[:, None]
        if ans_tgt is not None:
            losses.append(self.fork_tf(hslot, past, ans_tgt))
        if ans_len is not None:
            ans = self.fork_greedy(hslot, past, ans_len, meter)
        loss = torch.stack(losses).mean() if losses else None
        return loss, decs, ans, torch.stack(ins, 1)


def latent_batch(D, rng, bs, dev, L):
    """A training chunk of length L at an aligned start c0 in {0, L, 2L, ..} < n (one draw per batch); true start text."""
    n = D["n"]; starts = list(range(0, n, L)); c0 = starts[int(rng.integers(len(starts)))]; c1 = min(c0 + L, n)
    idx = rng.integers(0, len(D["A"]), bs); T = lambda a: torch.tensor(a, device=dev)  # noqa: E731
    prefix = T(np.concatenate([DIGIT_IDS[D["A"][idx]], np.full((bs, 1), TOK["*"])], 1))
    start = T(D["steps"][c0 - 1][idx]) if c0 > 0 else None
    return prefix, start, T(D["B"][idx, c0:c1]), [T(D["steps"][j][idx]) for j in range(c0, c1)], (T(D["ans"][idx]) if c1 == n else None)


@torch.no_grad()
def flops_per_problem(m, g, D, dev, phase, L):
    """Meter forward FLOPs of one trained problem in this configuration (averaged over the aligned chunk starts)."""
    if phase == "text":
        mt = MeterR(g); q, text = seq_batch(D, np.random.default_rng(0), 1, dev, "tc", 0)
        seq_forward(g, q, 0, text, meter=mt); return mt.flops + 2 * 768 * 50257 * text.shape[1]
    n = D["n"]; out = []
    for c0 in range(0, n, L):
        c1 = min(c0 + L, n); mt = MeterR(g); idx = np.array([0]); T = lambda a: torch.tensor(a, device=dev)  # noqa: E731
        pre = T(np.concatenate([DIGIT_IDS[D["A"][idx]], np.full((1, 1), TOK["*"])], 1)); stt = T(D["steps"][c0 - 1][idx]) if c0 > 0 else None
        gold, m.gold = m.gold, None; m.chunk(pre, stt, T(D["B"][idx, c0:c1]), meter=mt); m.gold = gold
        fork_tokens = sum(D["steps"][j].shape[1] for j in range(c0, c1)) + (D["ans"].shape[1] if c1 == n else 0)
        ctx_end = pre.shape[1] + (0 if stt is None else stt.shape[1]) + 2 * (c1 - c0)
        for _ in range(fork_tokens - ((c1 - c0) + (1 if c1 == n else 0))):
            mt.step(1, ctx_end, 0)
        out.append(mt.flops + 2 * 768 * 50257 * fork_tokens)
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm99", required=True, choices=list(ARMS)); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log_every", type=int, default=1000); ap.add_argument("--eval_bs", type=int, default=250)
    ap.add_argument("--val_n", type=int, default=None, help="smoke only"); ap.add_argument("--test_n", type=int, default=None, help="smoke only")
    ap.add_argument("--train_limit", type=int, default=0, help="smoke only"); ap.add_argument("--sched_scale", type=float, default=1.0, help="smoke only: scales the schedule step boundaries")
    ap.add_argument("--train_dir", default=os.path.join(ROOT, "bench", "data", "mult_train"))
    ap.add_argument("--final_eval_bs", type=int, default=100, help="stop-eval batch (OOM: halved and retried)")
    ap.add_argument("--eval_only", action="store_true", help="re-run the stop eval of a finished run (model.pt, log.jsonl)")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); t_start = time.time()
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    Dtr = load(os.path.join(a.train_dir, "4x4_train.txt"), limit=a.train_limit or None)
    Dva = load(os.path.join(a.train_dir, "4x4_valid.txt"), limit=a.val_n)
    Dte = load(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt"), limit=a.test_n)
    sc = a.sched_scale; arm = a.arm99; elig = int(ARMS[arm] * sc)
    cfg = lambda s: config(arm, s / sc)  # noqa: E731
    if a.eval_only:                                                         # re-run the stop eval of a finished run from its model.pt
        g = s5.load_gpt2(dev); m = MultGen99(g, "mask" if arm == "textfirst_mask" else "genm", std=arm in ("std", "combo")).to(dev)
        m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev))
        recs = [json.loads(x) for x in open(os.path.join(a.out, "log.jsonl"))]; curve = [
            {"step": r["step"], "per_slot": r["per_slot"], "val_acc": r["val_acc"], "gold_share": r["gold_share"], "k_train": r["k_train"],
             "phase": r["phase"], "eligible": r["eligible"]} for r in recs if "per_slot" in r]
        hit = next((c["step"] for c in curve if c["eligible"] and min(c["per_slot"]) >= 0.95 and c["val_acc"] >= 0.95), None)
        stop_step = hit or a.steps; assert recs[-1]["step"] == stop_step, "log does not end at the stop step"
        flops = 0.0; fc = {}
        m.eval()
        for st in range(1, stop_step + 1):
            ph, L, _ = cfg(st)
            if (ph, L) not in fc:
                fc[(ph, L)] = flops_per_problem(m, g, Dtr, dev, ph, L)
            flops += 3 * fc[(ph, L)] * a.bs
        finish(a, m, Dtr, Dte, dev, amp, arm, elig, hit, stop_step, curve, flops, None, recs[-1]["sec"]); return
    os.makedirs(a.out, exist_ok=True); json.dump({**vars(a), "eligible_from": ARMS[a.arm99]}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    g = s5.load_gpt2(dev)
    m = MultGen99(g, "mask" if arm == "textfirst_mask" else "genm", std=arm in ("std", "combo")).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []; curve = []; flops = 0.0; fcache = {}
    stop_step, hit = a.steps, None
    for step in range(1, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        phase, L, rho = cfg(step); opt.zero_grad(); tot = 0.0
        if (phase, L) not in fcache:
            m.eval(); fcache[(phase, L)] = flops_per_problem(m, g, Dtr, dev, phase, L); m.train()
        flops += 3 * fcache[(phase, L)] * a.bs
        if phase == "text":
            q, text = seq_batch(Dtr, rng, a.bs, dev, "tc", 0); mb = a.bs // 2
            for i in range(2):
                sl = slice(i * mb, (i + 1) * mb)
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    hs, _ = seq_forward(g, q[sl], 0, text[sl]); lg = g.logits(hs).float()
                    loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), text[sl].reshape(-1))
                (loss / 2).backward(); tot += loss.item() / 2
        else:
            prefix, start, ops, st, ans = latent_batch(Dtr, rng, a.bs, dev, L)
            m.gold = {j for j in range(ops.shape[1] - 1) if rng.random() < rho} if rho > 0 else None
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                loss, _, _, _ = m.chunk(prefix, start, ops, step_tgt=st, ans_tgt=ans, collect=True)
            loss.backward(); tot = loss.item(); m.gold = None
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(tot)
        if m.std and phase == "latent":
            m.update_stats()
        if step % 100 == 0 or step % a.log_every == 0:
            rec = {"step": step, "loss": float(np.mean(hist)), "sec": round(time.time() - t0), "phase": phase, "k_train": L, "gold_share": rho}; hist = []
            if step % a.log_every == 0:
                m.eval(); per, ok, _ = gen_rollout(m, Dva, dev, amp, a.eval_bs); m.train()
                e = {"step": step, "per_slot": per.mean(0).tolist(), "val_acc": float(ok.mean()), "gold_share": rho, "k_train": L,
                     "phase": phase, "eligible": step >= elig}
                curve.append(e); rec.update({k: e[k] for k in ("per_slot", "val_acc", "eligible")})
                if e["eligible"] and min(e["per_slot"]) >= 0.95 and e["val_acc"] >= 0.95:
                    hit = step
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
            if hit:
                stop_step = step; break
    wall = round(time.time() - t_start)
    torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    del opt; m.zero_grad(set_to_none=True)
    finish(a, m, Dtr, Dte, dev, amp, arm, elig, hit, stop_step, curve, flops, wall, round(time.time() - t0))


USED_BS = []


def rollout_safe(m, D, dev, amp, bs, **kw):
    """gen_rollout, halving the batch on CUDA OOM (three runs on one card can finish and evaluate together)."""
    while True:
        try:
            out = gen_rollout(m, D, dev, amp, bs, **kw); USED_BS.append(bs); return out
        except torch.OutOfMemoryError:
            if bs <= 10:
                raise
            torch.cuda.empty_cache(); bs //= 2; print("eval OOM: retry at bs", bs, flush=True)


def finish(a, m, Dtr, Dte, dev, amp, arm, elig, hit, stop_step, curve, flops, wall, train_sec):
    m.eval(); bs = a.final_eval_bs
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    per, ok, _ = rollout_safe(m, Dte, dev, amp, bs); mu = gen_mean_inputs(m, Dtr, dev, amp)
    res = {"arm": arm, "task": "4x4", "seed": a.seed, "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "eval_dtype": str(amp), "eligible_from": elig, "steps_to_threshold": hit, "stopped_step": stop_step, "curve": curve,
           "test_acc": float(ok.mean()), "per_slot_decode_acc": per.mean(0).tolist(), "removal": {},
           "training": {"flops": flops, "wall_sec": wall, "train_sec": train_sec,
                        "flops_convention": "sum over updates of Meter forward FLOPs per trained problem (config of that update) x 3 x bs; tfslot updates at the latent config"},
           "std_stats": {"seen": int(m.seen), "sd_floor": float(0.05 * m.sdH.median())} if m.std else None,
           "code": {f: hashlib.md5(open(os.path.join(ROOT, "boxes", f), "rb").read()).hexdigest() for f in ("accel99.py", "mult98.py")}}
    for ab in ("zero", "mean"):
        _, o_, _ = rollout_safe(m, Dte, dev, amp, bs, ablate=ab, mean_in=mu); res["removal"][ab] = {"answer_acc": float(o_.mean())}
    res["eval_bs"] = {"requested": a.final_eval_bs, "used": sorted(set(USED_BS))}
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({k: res[k] for k in ("steps_to_threshold", "stopped_step", "test_acc")}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
