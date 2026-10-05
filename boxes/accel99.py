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
  anchorread      §121: markov + the reader (at every carried slot j < c-1, a fresh segment [EOT, carried latent] decodes
                  step j's text, CE weight 1) + the anchor (at every latent-fed segment j, (1 - cos) + |log-norm gap| to the
                  stop-grad text path [a, gold step j-1 text, op j]); eval adds the random donor (loop114.donor), reader_acc,
                  anchor_cos_by_pos (eligible from 6k)
  read            §121: markov + the reader only (eligible from 6k)
  --textmix f     §122 (with anchorread / read): a share f of every latent-phase batch's rows are k = 1 chunks from the TRUE
                  previous step text at a uniform 1-based position j in {2, 3, 4} (CE on step j's text, + the answer at
                  j = 4), row-weighted with the main chunk's loss; every log adds the text path's step-j decode (T1 = j 3)
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

ARMS = {"ctrl": 0, "kcurr": 6000, "std": 0, "textfirst": 3000, "textfirst_mask": 3000, "combo": 9000, "tfslot": 12000, "markov": 6000, "statepath": 0, "lat2": 6000, "lat4": 6000, "pause4": 6000, "anchorread": 6000, "read": 6000}
EOT = 50256
MARKOV = ("markov", "lat2", "lat4", "pause4", "anchorread", "read")


def config(arm, step):
    """-> (phase 'text' | 'latent', training chunk length, gold share rho) at this update."""
    if arm in ("textfirst", "textfirst_mask", "combo") and step <= 3000:
        return "text", 4, 0.0
    if arm in ("kcurr",) + MARKOV:                                       # markov (note 99-3), §107, §121: K-CURR's schedule
        return "latent", 1 if step <= 2000 else (2 if step <= 6000 else 4), 0.0
    if arm == "combo":
        return "latent", 1 if step <= 5000 else (2 if step <= 9000 else 4), 0.0
    if arm == "tfslot":
        return "latent", 4, max(0.0, 1.0 - step / 12000)
    return "latent", 4, 0.0


class MultGen99(MultGen):
    def __init__(self, g, fork, std):
        super().__init__(g, fork)
        self.std = std; self.markov = False; self.statepath = False; self.ablate_mode = None
        self.m_lat = 1; self.n_pause = 0; self.rec_lat = False; self._latrec = []
        E = g.w("wte.weight").detach()
        self.register_buffer("muE", E.mean(0).clone()); self.register_buffer("sdE", E.std(0).clone())
        self.register_buffer("muH", torch.zeros(768)); self.register_buffer("sdH", torch.ones(768)); self.register_buffer("seen", torch.tensor(0))
        self._raw = []; self.gold = None
        self.reader = False; self.anchor = False; self.override = None; self.parts = (None, None, None)   # §121 (default off)

    def enable_mlat(self, m_lat, n_pause):
        """§107: m chained latents per op (lat2, lat4) or n_pause shared learned pause embeddings before 1 latent (pause4)."""
        self.m_lat = m_lat; self.n_pause = n_pause
        if n_pause:
            self.pause = torch.nn.Parameter(self.g.w("wte.weight").detach().mean(0).clone())

    def enable_statepath(self):
        """§102: h(4)_slot_j += gamma * W h(9)_slot_{j-1} within a chunk; W = identity, gamma = 0 at init."""
        self.statepath = True
        self.Wsp = torch.nn.Linear(768, 768, bias=False); torch.nn.init.eye_(self.Wsp.weight); self.gamma = torch.nn.Parameter(torch.zeros(()))

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
        if self.n_pause:                                                    # §107 pause4: pauses after op 1 as well
            parts = parts + [self.pause.to(parts[0].dtype)[None, None].expand(B, self.n_pause, -1)]
        x = torch.cat(parts, 1); past = None; losses, decs, ins = [], [], []; ans = None; h9_prev = None; pos_prev = None
        rlosses, alosses = [], []
        for j in range(c):
            if self.markov:                                                 # §99 note 99-3: every slot step is a fresh-cache segment
                past = None
            ctx = 0 if past is None else past[0][0].shape[2]
            h, past = g(x, past)
            if meter:
                meter.step(x.shape[1], ctx, 0)
            if self.m_lat > 1 or self.n_pause > 0:                          # §107: m chained latents per op (or pauses + 1 latent)
                cur = h[:, -1]; lats = []
                for i in range(self.m_lat):
                    if j < c - 1 and ablate in ("zero", "mean"):
                        cur = torch.zeros_like(cur) if ablate == "zero" else mean_in[j][i][None].expand_as(cur).to(cur.dtype)
                    if j < c - 1 and self.ablate_mode == f"latent{i}":
                        cur = torch.zeros_like(cur)
                    lats.append(cur); ctx = past[0][0].shape[2]; hs, past = g(cur[:, None], past)
                    if meter:
                        meter.step(1, ctx, 0)
                    cur = hs[:, 0]                                          # latent i+1's input = latent i's last hidden
                hslot = cur; ins.append(torch.stack(lats, 1))               # (B, m, D): carried to the next segment
                if self.rec_lat:
                    self._latrec.append(torch.stack(lats, 1).float())
                final = j == c - 1 and (ans_tgt is not None or ans_len is not None)
                if step_tgt is not None:
                    tgt = torch.cat([step_tgt[j], ans_tgt], 1) if (final and ans_tgt is not None) else step_tgt[j]
                    losses.append(self.fork_tf(hslot, past, tgt))
                if final and ans_len is not None:
                    d = self.fork_greedy(hslot, past, step_len[j] + ans_len, meter)
                    decs.append(d[:, :step_len[j]]); ans = d[:, step_len[j]:]
                elif step_len is not None and (decode == "all" or j == c - 1):
                    decs.append(self.fork_greedy(hslot, past, step_len[j], meter if (snap and j == c - 1) else meter_x))
                if j + 1 < c:                                               # [a, op j's m latents, op j+1, pauses]
                    dt = g.embed(prefix).dtype
                    x = torch.cat([g.embed(prefix), ins[-1].to(dt), g.embed(self.digtok[ops[:, j + 1]])[:, None]]
                                  + ([self.pause.to(dt)[None, None].expand(B, self.n_pause, -1)] if self.n_pause else []), 1)
                continue
            if self.gold and j in self.gold and step_tgt is not None and j < c - 1:   # tfslot: gold step text in place of slot j
                gt = step_tgt[j]; ctx = past[0][0].shape[2]; _, past = g(g.embed(gt), past)
                ins.append(torch.zeros_like(h[:, -1]))
            else:
                s_in = h[:, -1]
                if self.anchor and self.markov and j > 0 and step_tgt is not None:   # §121 anchor: latent-fed segment j vs the
                    with torch.no_grad():                                            # stop-grad text path [a, gold step j-1 text, op j]
                        hb, _ = g(torch.cat([g.embed(prefix), g.embed(step_tgt[j - 1]), g.embed(self.digtok[ops[:, j]])[:, None]], 1), None)
                        b = hb[:, -1].float()
                    a_ = s_in.float(); cos = F.cosine_similarity(a_, b, dim=-1)
                    alosses.append(((1 - cos) + (a_.norm(dim=-1).log() - b.norm(dim=-1).log()).abs()).mean())
                if self.std:
                    if collect:
                        self._raw.append(s_in.detach().float())
                    s_in = self.g_std(s_in)
                if ablate in ("zero", "mean") and j < c - 1:
                    s_in = torch.zeros_like(s_in) if ablate == "zero" else mean_in[j][None].expand_as(s_in).to(s_in.dtype)
                if self.override is not None and self.override[0] == j:     # §121 donor patch (j, (B, D)); default off
                    s_in = self.override[1].to(s_in.dtype)
                if self.reader and step_tgt is not None and j < c - 1:       # §121 reader: [EOT, carried latent] -> step j text (CE, weight 1)
                    st_ = step_tgt[j]; eot = g.embed(torch.full((B, 1), EOT, device=ops.device))
                    hr, _ = g(torch.cat([eot, s_in[:, None].to(eot.dtype), g.embed(st_[:, :-1])], 1), None); lr_ = g.logits(hr[:, 1:]).float()
                    rlosses.append(F.cross_entropy(lr_.reshape(-1, lr_.shape[-1]), st_.reshape(-1), reduction="none").view(st_.shape).sum(-1).mean())
                ins.append(s_in)
                ctx = past[0][0].shape[2]
                if self.statepath:                                          # §102 state path (off under removal / wpath ablation)
                    off = ablate in ("zero", "mean") or self.ablate_mode == "wpath" or h9_prev is None
                    inj = None if off else (4, self.gamma * self.Wsp(h9_prev.to(self.Wsp.weight.dtype)))
                    kvm = [pos_prev] if (self.ablate_mode == "attn_prev" and pos_prev is not None) else None
                    pos_prev = ctx; hs, past, h9_prev = g(s_in[:, None], past, inject=inj, capture=9, kv_mask=kvm)
                else:
                    hs, past = g(s_in[:, None], past)
                if meter:
                    meter.step(1, ctx, 0)
                hslot = hs[:, 0]
                final = j == c - 1 and (ans_tgt is not None or ans_len is not None)
                # note 98-3: one final fork, step text then " #### <product>" (as mult98)
                if step_tgt is not None:
                    tgt = torch.cat([step_tgt[j], ans_tgt], 1) if (final and ans_tgt is not None) else step_tgt[j]
                    losses.append(self.fork_tf(hslot, past, tgt))
                if final and ans_len is not None:
                    d = self.fork_greedy(hslot, past, step_len[j] + ans_len, meter)
                    decs.append(d[:, :step_len[j]]); ans = d[:, step_len[j]:]
                elif step_len is not None and (decode == "all" or j == c - 1):
                    decs.append(self.fork_greedy(hslot, past, step_len[j], meter if (snap and j == c - 1) else meter_x))
            if j + 1 < c:
                if self.markov:                                             # [operand prefix, slot j's latent, op j+1], positions from 0
                    x = torch.cat([g.embed(prefix), ins[-1][:, None].to(g.embed(prefix).dtype), g.embed(self.digtok[ops[:, j + 1]])[:, None]], 1)
                else:
                    x = g.embed(self.digtok[ops[:, j + 1]])[:, None]
        loss = torch.stack(losses).mean() if losses else None
        rl = torch.stack(rlosses).mean() if rlosses else None; al = torch.stack(alosses).mean() if alosses else None
        self.parts = (None if loss is None else float(loss.detach()), None if rl is None else float(rl.detach()), None if al is None else float(al.detach()))
        if loss is not None:
            loss = loss + (rl if rl is not None else 0) + (al if al is not None else 0)
        return loss, decs, ans, torch.stack(ins, 1)

    @torch.no_grad()
    def read_greedy(self, s_in, L):
        """§121 reader decode: a fresh segment [EOT, carried latent], L greedy tokens."""
        g = self.g; eot = g.embed(torch.full((s_in.shape[0], 1), EOT, device=s_in.device))
        h, past = g(torch.cat([eot, s_in[:, None].to(eot.dtype)], 1), None); t = g.logits(h[:, -1]).argmax(-1); out = [t]
        for _ in range(L - 1):
            h, past = g(g.embed(t)[:, None], past); t = g.logits(h[:, 0]).argmax(-1); out.append(t)
        return torch.stack(out, 1)


def latent_batch(D, rng, bs, dev, L):
    """A training chunk of length L at an aligned start c0 in {0, L, 2L, ..} < n (one draw per batch); true start text."""
    n = D["n"]; starts = list(range(0, n, L)); c0 = starts[int(rng.integers(len(starts)))]; c1 = min(c0 + L, n)
    idx = rng.integers(0, len(D["A"]), bs); T = lambda a: torch.tensor(a, device=dev)  # noqa: E731
    prefix = T(np.concatenate([DIGIT_IDS[D["A"][idx]], np.full((bs, 1), TOK["*"])], 1))
    start = T(D["steps"][c0 - 1][idx]) if c0 > 0 else None
    return prefix, start, T(D["B"][idx, c0:c1]), [T(D["steps"][j][idx]) for j in range(c0, c1)], (T(D["ans"][idx]) if c1 == n else None)


def textmix_batch(D, rng, bs, dev, j):
    """§122: bs rows of the k = 1 chunk at 0-based op j >= 1 from the TRUE step j-1 text (answer target when j is last)."""
    n = D["n"]; idx = rng.integers(0, len(D["A"]), bs); T = lambda a: torch.tensor(a, device=dev)  # noqa: E731
    prefix = T(np.concatenate([DIGIT_IDS[D["A"][idx]], np.full((bs, 1), TOK["*"])], 1))
    return prefix, T(D["steps"][j - 1][idx]), T(D["B"][idx, j:j + 1]), [T(D["steps"][j][idx])], (T(D["ans"][idx]) if j == n - 1 else None)


@torch.no_grad()
def text_path_acc(m, D, dev, amp, bs):
    """§122: exact decode of step j's text (1-based j = 2, 3, 4) from the TRUE step j-1 text, k = 1 (T1 = j 3)."""
    N, n = D["B"].shape; T = lambda a: torch.tensor(a, device=dev); out = {}  # noqa: E731
    for j in range(1, n):
        ok = []
        for s in range(0, N, bs):
            e = min(N, s + bs); pre = T(np.concatenate([DIGIT_IDS[D["A"][s:e]], np.full((e - s, 1), TOK["*"])], 1))
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                _, decs, _, _ = m.chunk(pre, T(D["steps"][j - 1][s:e]), T(D["B"][s:e, j:j + 1]), step_len=[D["steps"][j].shape[1]])
            ok.append((decs[0].cpu().numpy() == D["steps"][j][s:e]).all(1))
        out[str(j + 1)] = float(np.concatenate(ok).mean())
    return out


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
        for _ in range(fork_tokens - (c1 - c0)):                           # one fork per slot (note 98-3)
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
    ap.add_argument("--donor_n", type=int, default=2000, help="§121 random-donor recipients")
    ap.add_argument("--textmix", type=float, default=0.0, help="§122 TEXT-MIX share of rows (0.25)")
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
        g = s5.load_gpt2(dev); m = MultGen99(g, "mask" if arm == "textfirst_mask" else "genm", std=arm in ("std", "combo")).to(dev); m.markov = arm in MARKOV; m.reader = arm in ("anchorread", "read"); m.anchor = arm == "anchorread"
        if arm in ("lat2", "lat4", "pause4"):
            m.enable_mlat({"lat2": 2, "lat4": 4, "pause4": 1}[arm], 3 if arm == "pause4" else 0); m.to(dev)
        if arm == "statepath":
            m.enable_statepath(); m.to(dev)
        m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev))
        recs = [json.loads(x) for x in open(os.path.join(a.out, "log.jsonl"))]; curve = [
            {"step": r["step"], "per_slot": r["per_slot"], "val_acc": r["val_acc"], "gold_share": r["gold_share"], "k_train": r["k_train"],
             "phase": r["phase"], "eligible": r["eligible"], "gate": r.get("gate")} for r in recs if "per_slot" in r]
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
    m = MultGen99(g, "mask" if arm == "textfirst_mask" else "genm", std=arm in ("std", "combo")).to(dev); m.markov = arm in MARKOV; m.reader = arm in ("anchorread", "read"); m.anchor = arm == "anchorread"
    if arm in ("lat2", "lat4", "pause4"):
        m.enable_mlat({"lat2": 2, "lat4": 4, "pause4": 1}[arm], 3 if arm == "pause4" else 0); m.to(dev)
    if arm == "statepath":
        m.enable_statepath(); m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []; aux = []; curve = []; flops = 0.0; fcache = {}
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
            n_tm = int(round(a.bs * a.textmix)); n_main = a.bs - n_tm
            prefix, start, ops, st, ans = latent_batch(Dtr, rng, n_main, dev, L)
            m.gold = {j for j in range(ops.shape[1] - 1) if rng.random() < rho} if rho > 0 else None
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                loss, _, _, _ = m.chunk(prefix, start, ops, step_tgt=st, ans_tgt=ans, collect=True)
                if n_tm:                                                    # §122 TEXT-MIX rows: true step j-1 text -> step j (k = 1)
                    parts_main = m.parts; loss = loss * (n_main / a.bs); pos = rng.integers(1, Dtr["n"], n_tm)   # 0-based j in {1, 2, 3}
                    for j in np.unique(pos):
                        tb = textmix_batch(Dtr, rng, int((pos == j).sum()), dev, int(j))
                        lt, _, _, _ = m.chunk(*tb[:3], step_tgt=tb[3], ans_tgt=tb[4]); loss = loss + lt * (int((pos == j).sum()) / a.bs)
                    m.parts = parts_main
            loss.backward(); tot = loss.item(); m.gold = None
            if m.reader:
                aux.append(m.parts)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(tot)
        if m.std and phase == "latent":
            m.update_stats()
        if step % 100 == 0 or step % a.log_every == 0:
            rec = {"step": step, "loss": float(np.mean(hist)), "sec": round(time.time() - t0), "phase": phase, "k_train": L, "gold_share": rho}; hist = []
            for i_, nm in ((0, "main_loss"), (1, "reader_loss"), (2, "anchor_loss")):
                v_ = [p_[i_] for p_ in aux if p_[i_] is not None]
                if v_:
                    rec[nm] = float(np.mean(v_))
            aux = []
            if dev.startswith("cuda"):
                rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
            if step % a.log_every == 0:
                m.eval(); per, ok, _ = gen_rollout(m, Dva, dev, amp, a.eval_bs); m.train()
                e = {"step": step, "per_slot": per.mean(0).tolist(), "val_acc": float(ok.mean()), "gold_share": rho, "k_train": L, "gate": float(m.gamma) if m.statepath else None,
                     "phase": phase, "eligible": step >= elig}
                if a.textmix:                                               # §122: the anchor's target is competent (T1 = text_acc["3"])
                    m.eval(); e["text_acc"] = text_path_acc(m, Dva, dev, amp, a.eval_bs); e["t1"] = e["text_acc"]["3"]; m.train(); rec.update({"text_acc": e["text_acc"], "t1": e["t1"]})
                curve.append(e); rec.update({k: e[k] for k in ("per_slot", "val_acc", "eligible", "gate")})
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


@torch.no_grad()
def read_anchor(m, D, dev, amp, bs):
    """§121 test: per carried slot j (0-based 0..2) the reader's exact decode of step j's text, and per latent-fed segment
    j (1..3) the cosine / |log-norm gap| between its carried latent and the text path [a, TRUE step j-1 text, op j]."""
    N, n = D["B"].shape; T = lambda x: torch.tensor(x, device=dev); g = m.g  # noqa: E731
    rd = [[] for _ in range(n - 1)]; cs = [[] for _ in range(n)]; gp = [[] for _ in range(n)]
    for s in range(0, N, bs):
        e = min(N, s + bs); pre = T(np.concatenate([DIGIT_IDS[D["A"][s:e]], np.full((e - s, 1), TOK["*"])], 1)); ops = T(D["B"][s:e])
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            _, _, _, ins = m.chunk(pre, None, ops)
            for j in range(n - 1):
                tj = D["steps"][j][s:e]; rd[j].append((m.read_greedy(ins[:, j], tj.shape[1]).cpu().numpy() == tj).all(1))
            for j in range(1, n):
                hb, _ = g(torch.cat([g.embed(pre), g.embed(T(D["steps"][j - 1][s:e])), g.embed(m.digtok[ops[:, j]])[:, None]], 1), None)
                a_, b = ins[:, j].float(), hb[:, -1].float()
                cs[j].append(F.cosine_similarity(a_, b, dim=-1).cpu().numpy()); gp[j].append((a_.norm(dim=-1).log() - b.norm(dim=-1).log()).abs().cpu().numpy())
    return {"reader_acc": [float(np.concatenate(r).mean()) for r in rd],
            "anchor_cos_by_pos": {str(j): float(np.concatenate(cs[j]).mean()) for j in range(1, n)},
            "anchor_lognorm_gap_by_pos": {str(j): float(np.concatenate(gp[j]).mean()) for j in range(1, n)}}


def finish(a, m, Dtr, Dte, dev, amp, arm, elig, hit, stop_step, curve, flops, wall, train_sec):
    m.eval(); bs = a.final_eval_bs
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    meter = MeterR(m.g); per, ok, _ = rollout_safe(m, Dte, dev, amp, bs, meter=meter); mu = gen_mean_inputs(m, Dtr, dev, amp)
    res = {"arm": arm + ("_textmix" if getattr(a, "textmix", 0) else ""), "textmix": getattr(a, "textmix", 0), "task": "4x4", "seed": a.seed, "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "eval_dtype": str(amp), "eligible_from": elig, "steps_to_threshold": hit, "stopped_step": stop_step, "curve": curve,
           "test_acc": float(ok.mean()), "per_slot_decode_acc": per.mean(0).tolist(), "removal": {}, "compute": meter.as_dict(),
           "training": {"flops": flops, "wall_sec": wall, "train_sec": train_sec,
                        "flops_convention": "sum over updates of Meter forward FLOPs per trained problem (config of that update) x 3 x bs; tfslot updates at the latent config"},
           "std_stats": {"seen": int(m.seen), "sd_floor": float(0.05 * m.sdH.median())} if m.std else None,
           "code": {f: hashlib.md5(open(os.path.join(ROOT, "boxes", f), "rb").read()).hexdigest() for f in ("accel99.py", "mult98.py", "loop114.py")}}
    if m.m_lat > 1:                                                         # §107 lat-m: per-latent removal and latent cosine
        res["per_latent_removal"] = {}
        for i in range(m.m_lat):
            m.ablate_mode = f"latent{i}"; _, o_, _ = rollout_safe(m, Dte, dev, amp, bs); res["per_latent_removal"][str(i + 1)] = {"zero_answer_acc": float(o_.mean())}
        m.ablate_mode = None; m.rec_lat = True; m._latrec = []
        rollout_safe(m, Dte, dev, amp, bs); m.rec_lat = False
        L = torch.cat(m._latrec); Ln = torch.nn.functional.normalize(L, dim=-1); C = Ln @ Ln.transpose(1, 2)   # (N, m, m)
        iu = torch.triu_indices(m.m_lat, m.m_lat, 1); res["latent_cosine"] = float(C[:, iu[0], iu[1]].mean()); m._latrec = []
    if m.statepath:                                                         # §102: separate test ablations
        res["ablation"] = {}; res["gate_curve"] = [[c["step"], c.get("gate")] for c in curve]
        for mode in ("wpath", "attn_prev"):
            m.ablate_mode = mode; _, o_, _ = rollout_safe(m, Dte, dev, amp, bs); res["ablation"][mode] = {"answer_acc": float(o_.mean())}
        m.ablate_mode = None
    for ab in ("zero", "mean"):
        _, o_, _ = rollout_safe(m, Dte, dev, amp, bs, ablate=ab, mean_in=mu); res["removal"][ab] = {"answer_acc": float(o_.mean())}
    if m.reader:                                                            # §121: random-donor test (§104 protocol, loop114's), reader, anchor cos
        from boxes.loop114 import donor
        res["donor"] = donor(m, dev, amp, n=a.donor_n, bs=bs); res.update(read_anchor(m, Dte, dev, amp, bs))
        if getattr(a, "textmix", 0):
            res["text_acc"] = text_path_acc(m, Dte, dev, amp, bs); res["t1"] = res["text_acc"]["3"]
        res["training"]["flops_convention"] += "; §121 reader/anchor auxiliary passes not counted"
    res["eval_bs"] = {"requested": a.final_eval_bs, "used": sorted(set(USED_BS))}
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({k: res[k] for k in ("steps_to_threshold", "stopped_step", "test_acc")}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
