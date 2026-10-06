"""THEORY §120 (notes 120-1): no snaps, no hand-designed auto-snaps - a pretrained GPT-2 carries an FSQ-quantised state
between Markov segments. Same code and hyperparameters for both families (S5, flip-flop).
Segment per input item: [EOT, carried state embedding, item tokens] -> last hidden h (fresh cache, positions from 0).
FSQ (Mentzer et al.): z = W_down(LN(h)) (d = 8); z_b = tanh(z + shift) * half_l - offset with L = 8 levels (half_l = 3.5,
offset 0.5, shift = atanh(offset / half_l): the half-integer offset for even L), q = round(z_b), q_st = z_b + sg(q - z_b).
Carried embedding = W_up(q_st) (8 -> 768, learned bias). Rounding: soft (q = z_b) for steps 1-2k, hard with probability
rising 0 -> 1 over 2k-4k (one coin per hand-off), hard after; test always hard.
Start: S5 = the start state's text at position 0 (the only text input; training starts uniform over the 120 states as
§110b; test = identity); flip-flop = EOT + the empty state W_up(0) + bias.
Answer decoder: a fresh segment [EOT, carried embedding] -> answer tokens through the tied head (S5: the 6-token state
text; flip-flop: the bit token). Training: lengths uniform 1..L_max, L_max 2 / 4 / 8 / 16 / 32 from steps 0 / 2k / 5k /
9k / 14k; full backprop through the chain; 24k steps, bs 64, AdamW lr 1e-4, warmup 100 + cosine.
Arms: final (answer at the chain's end only), trace (+ the decoder at every native label: S5 every op's state, flip-flop
every read's bit; weight 1).
Flip-flop (Liu 2306.00946): items write 0/1, ignore 0/1, read; p_ignore 0.8, p_write = p_read = 0.1; first a write, last a
read; answer = the last written bit.
eval.json: long_acc (S5: cumulative state accuracy at 512 on DeltaProduct's 2,000 x 512, identity start; flip-flop: mean
accuracy over every read of 2,000 x 512 chains at p_ignore 0.8, long_acc_seq = all reads right), acc_by_length
{16..512}, descriptive (S5 2,048 on 500; flip-flop p_ignore 0.98 / 0.1), code_perplexity / n_codes / codes_per_state on
the test chains, cert {holds, n_reachable, n_moore, capped} (BFS over the code machine from the start codes with every
item, cap --cert_cap; every code decoded with the answer decoder; holds iff all decode validly and decode(next(c, x)) =
delta(decode(c), x) for every reachable pair incl. the start; n_moore = Moore-minimised size), donor {pos2, pos4, pos6}
(single-code patch on length-8 chains), margin_flip_rate (hand-offs whose rounded code changes under U(-0.5, 0.5) noise on
z_b), coverage_drawn (note 120-1: share of (state, item) transitions labelled by the length-1 / length-2 training
problems drawn: (s, x1), and (delta(s, x1), x2)).
    python -m boxes.fsq120 --family s5 --arm final --seed 1 --out boxes/runs_fsq/s120_s5_final_s1
THEORY §123 (same harness): --no_round carries z_b continuously (W_up(z_b), never rounded, test too); --dither s adds
U(-s, s) per coordinate to z_b at every TRAINING hand-off after 2k (test clean); --fsq_dither s keeps §120's rounding and
adds U(-s, s) to z_b before rounding at every training hand-off after 4k. Eval adds (all runs): rho / basin (contraction
diag ported: n chains of length T0 + K = 4 + 16 from training-distribution starts; at T0 z_b += eps * std_d * u, std_d =
per-coordinate std of clean carried z_b, u Gaussian per chain; rho = geometric mean over k = 1..8 of the median distance
ratio ||z'_k - z_k|| / ||z'_{k-1} - z_{k-1}|| at eps 0.1; basin = the largest eps in {0.03, 0.1, 0.3, 1.0} with decode
agreement >= 0.99 at k = 8), grid_share (share of test-time z_b coordinates within 0.1 of an integer grid point; z_b
already carries the -0.5 offset). Continuous runs skip the code statistics, CERT and margin_flip.
THEORY §127 (one change per arm on §120 FINAL; none draws from §120's random streams: new buffers start at zero, SKIP's maps
use their own generator, probes / checks run in eval mode with their own numpy RNG):
  --var_res   (A) VAR: r = z_b - m[x, j] (x = the item the segment read, j = its chain position; m = stop-gradient EMA 0.99 of
              z_b per (item, position), a pair's first sighting initialises its row); over all hand-offs of each forward
              w(step) * [mean_d relu(1 - sqrt(Var(r_d) + 1e-4)) + 0.04 * (1/8) * sum_{d != d'} Cov(r)^2]; w = --var_w to 2k,
              linear to 0 at 6k.
  --skip_dec  (B) SKIP: S_l (8 -> 768, no bias) adds S_l(q_st) to the residual stream entering GPT-2 blocks 6..11 and S_f(q_st)
              to the final hidden before ln_f, at every answer-decoder position after EOT (carried slot + answer positions;
              training CE, greedy decoding, donor, CERT, contraction). Init N(0, 0.02^2) from torch.Generator(12700 + seed).
  --std2      (C) STD2: every carried embedding entering GPT-2 becomes mu_E + sd_E * (e - mean(e)) / max(sd(e), 0.5 sd_E);
              training: mean / sd over the batch rows of the current hand-off (detached); test: EMA 0.99 of the per-forward
              average of those statistics (updated after every optimiser step, frozen at the end; batch stats until the first
              update).
  --lr_new F  (D) LR10: LN / W_down / W_up in their own AdamW group at F x lr (same warmup + cosine, wd 0.01, global clip).
  --probe127  the note 127-1 logging without an arm (the control); any arm implies it. log.jsonl gets {"step": s, "mcheck":
              {...}} at s = 1, 500, 2000 (after that step's update; fixed probe batch, eval mode with training semantics (soft
              codes unless p_hard = 1, STD2 on batch statistics; no draws); the two gradient checks on its first 256 chains;
              *_eval keys repeat row norm / resid std in plain eval mode): skip_grad_q = ||dL_ans/dq_st||
              through the skip maps only (q_st a leaf), var_grad_down = ||d VAR / d W_down|| (w = 1, EMA not updated),
              resid_std = mean over dims of the pooled within-(item, position) std of z_b, resid_std_ema (VAR: vs the EMA rows
              seen), row_norm / wte_row_norm (carried embeddings), lr_ratio) and {"step": 0 / 2000, "probe": {r2_item,
              r2_state, abl_dloss}} (1,024 chains of length 2, own RNG; per-group means of z_b fit on chains 1-512, R^2 on
              513-1024, unseen groups -> the fit half's global mean; abl_dloss = answer CE at the last hand-off with the code
              (slot + skip) replaced by the batch-mean code, minus the intact CE). No "loss" key on these lines (123-3).
  --stop_plateau  stop at 2k if 123-3's flag fires (loss(2k) >= 0.9 loss(1k) and >= 0.25 loss(100)): log kept, no eval.json.
"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes import core as bc  # noqa: E402
from boxes.gen97 import NST, STTOK, decode_to_el  # noqa: E402
from boxes.latent86 import OPTOK  # noqa: E402

EOT, D_Q, LEV = 50256, 8, 8
HALF = (LEV - 1) / 2; OFF = 0.5; SHIFT = math.atanh(OFF / HALF)
FF_ITEMS = [bc.tok(" write 0"), bc.tok(" write 1"), bc.tok(" ignore 0"), bc.tok(" ignore 1"), bc.tok(" read.")]
assert all(len(t) == 2 for t in FF_ITEMS), FF_ITEMS
FF_BIT = [bc.tok(" 0")[0], bc.tok(" 1")[0]]
W0, W1, I0, I1, RD = range(5)


class Family:
    """S5: states 0..119 (element ids), items 0..119 (ops), delta = COMP. Flip-flop: states 0 (none) / 1 (last bit 0) /
    2 (last bit 1), items W0 W1 I0 I1 RD; delta: writes set the bit, others keep it; the answer is the bit (state - 1)."""
    def __init__(self, name):
        self.name = name
        if name == "s5":
            self.n_items, self.n_states = 120, 120; self.item_tok = np.array([OPTOK[e] for e in range(120)])
        else:
            self.n_items, self.n_states = 5, 3; self.item_tok = np.array(FF_ITEMS)

    def delta(self, s, x):
        if self.name == "s5":
            return s5.COMP[s, x]
        s = np.array(s); x = np.array(x)
        return np.where(x == W0, 1, np.where(x == W1, 2, s))

    def answer_tokens(self, s):
        """Answer token rows for states s (S5: 6-token state text; ff: the bit token)."""
        return STTOK[s] if self.name == "s5" else np.array(FF_BIT)[np.clip(np.array(s) - 1, 0, 1)][:, None]

    def decode(self, toks):
        """Decoded answer token rows -> states (-1 = invalid)."""
        if self.name == "s5":
            return decode_to_el(toks)[0]
        t = toks[:, 0]; return np.where(t == FF_BIT[0], 1, np.where(t == FF_BIT[1], 2, -1))


def ff_sequences(rng, n, L, p_ignore=0.8):
    p_w = p_r = (1 - p_ignore) / 2
    x = rng.choice([W0, W1, I0, I1, RD], size=(n, L), p=[p_w / 2, p_w / 2, p_ignore / 2, p_ignore / 2, p_r])
    x[:, 0] = rng.integers(0, 2, n)                                         # first a write
    if L > 1:
        x[:, -1] = RD                                                       # last a read
    return x


class FSQ120(nn.Module):
    def __init__(self, g, fam):
        super().__init__()
        self.g = g; self.fam = fam; self.ln = nn.LayerNorm(768); self.down = nn.Linear(768, D_Q); self.up = nn.Linear(D_Q, 768)
        self.register_buffer("itok", torch.tensor(fam.item_tok))
        self.p_hard = 1.0                                                   # training schedule sets it; test = 1
        self.no_round = False; self.noise = 0.0                             # §123: continuous carry; training dither (set per step)
        self.std = False; self._acc = []                                    # §125 (enable_std registers its buffers)
        self.raw_h = False                                                  # §126 RAW-H: no bottleneck, carried = std(h) (needs enable_std)
        self.anneal = False; self.lam = 1.0; self._h_last = None; self._dist = []; self.distill_on = False   # §126 ANNEAL (test lam = 1)
        self.skip = False; self.std2 = False; self.var_on = False; self._var = []; self.last_qs = None   # §127 (enable_* register)
        self.det_train = False                                              # §127 checks: training semantics without RNG draws

    def enable_skip(self, seed):
        """§127 B: 7 maps 8 -> 768 (blocks 6..11, then the final hidden), no bias, N(0, 0.02^2) from their own generator."""
        gen = torch.Generator().manual_seed(12700 + seed); dev = self.up.weight.device
        self.skip_w = nn.Parameter((torch.randn(7, D_Q, 768, generator=gen) * 0.02).to(dev)); self.skip = True

    def enable_std2(self):
        """§127 C: wte statistics, test-time EMA buffers (zeros; seen counts updates)."""
        E = self.g.w("wte.weight").detach().float()
        self.register_buffer("e2mu", E.mean(0).clone()); self.register_buffer("e2sd", E.std(0).clone())
        self.register_buffer("s2mu", torch.zeros(768, device=E.device)); self.register_buffer("s2sd", torch.ones(768, device=E.device))
        self.register_buffer("s2seen", torch.tensor(0, device=E.device)); self.register_buffer("s2frozen", torch.tensor(0, device=E.device))
        self._s2acc = []; self.std2 = True

    def std2_map(self, e, acc=True):
        """acc=False (the flip-flop start, not a hand-off): EMA statistics once they exist, never accumulated."""
        ef = e.float(); flat = ef.detach().reshape(-1, ef.shape[-1])
        if int(self.s2seen) == 0 or ((self.training or self.det_train) and acc):   # training / checks / before any update: this batch
            mu, sd = flat.mean(0), flat.std(0, unbiased=False)
            if self.training and acc and not int(self.s2frozen):
                self._s2acc.append((mu, sd))
        else:
            mu, sd = self.s2mu, self.s2sd
        return (self.e2mu + self.e2sd * (ef - mu) / torch.maximum(sd, 0.5 * self.e2sd)).to(e.dtype)

    @torch.no_grad()
    def update_std2(self):
        if self.std2 and self._s2acc and not int(self.s2frozen):
            mu = torch.stack([a for a, _ in self._s2acc]).mean(0); sd = torch.stack([b for _, b in self._s2acc]).mean(0)
            if int(self.s2seen) == 0:
                self.s2mu.copy_(mu); self.s2sd.copy_(sd)
            else:
                self.s2mu.mul_(0.99).add_(0.01 * mu); self.s2sd.mul_(0.99).add_(0.01 * sd)
            self.s2seen += 1
        self._s2acc = []

    def enable_var(self, n_items, jmax=32):
        """§127 A: per-(item, position) EMA rows of z_b (zeros; seen marks a pair's first sighting)."""
        dev = self.up.weight.device; self.jmax = jmax
        self.register_buffer("vm", torch.zeros(n_items * jmax, D_Q, device=dev)); self.register_buffer("vseen", torch.zeros(n_items * jmax, dtype=torch.bool, device=dev))

    def var_loss(self, update=True):
        """VICReg variance + covariance on the item-and-position residual of this forward's hand-offs; update=False (checks)
        leaves the EMA untouched."""
        Z = torch.cat([z for z, _, _ in self._var]).float(); key = torch.cat([x * self.jmax + j for _, x, j in self._var]); self._var = []
        with torch.no_grad():
            n = self.vm.shape[0]; cnt = torch.bincount(key, minlength=n).float()
            gm = torch.zeros(n, D_Q, device=Z.device).index_add_(0, key, Z.detach()) / cnt.clamp_min(1)[:, None]
            present = cnt > 0; new = present & ~self.vseen
            vm = self.vm.clone(); vm[new] = gm[new]
            if update:
                upd = present & self.vseen
                self.vm[new] = gm[new]; self.vm[upd] = 0.99 * self.vm[upd] + 0.01 * gm[upd]; self.vseen |= present
        r = Z - vm[key]
        v = torch.relu(1 - torch.sqrt(r.var(0) + 1e-4)).mean()
        rc = r - r.mean(0); C = rc.T @ rc / max(1, r.shape[0] - 1); off = C - torch.diag(torch.diag(C))
        return v + 0.04 * (off ** 2).sum() / D_Q

    def gpt_skip(self, x, past, qs, mask):
        """GPT2KV.forward with §127 B's skip: mask (T,) marks the positions after EOT; S_l(q_st) enters blocks 6..11, S_f(q_st)
        the final hidden before ln_f."""
        g = self.g; B, T, D = x.shape; P0 = 0 if past is None else past[0][0].shape[2]; hd = D // g.H
        add = torch.einsum("bq,kqd->kbd", qs.to(self.skip_w.dtype), self.skip_w)[:, :, None, :] * mask.view(1, 1, T, 1).to(self.skip_w.dtype)
        h = x + g.w("wpe.weight")[P0:P0 + T][None]; new = []
        for i in range(g.L):
            pre = f"h.{i}."
            if i >= 6:
                h = h + add[i - 6]
            a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
            q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
            q, k, v = (t.view(B, T, g.H, hd).transpose(1, 2) for t in (q, k, v))
            if past is not None:
                k = torch.cat([past[i][0], k], 2); v = torch.cat([past[i][1], v], 2)
            new.append((k, v))
            if P0 == 0:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(T, P0 + T, dtype=torch.bool, device=x.device).tril(P0))
            h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
            a = F.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
            h = h + F.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh") @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")
        h = h + add[6]
        return F.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias")), new

    def enable_std(self):
        """§125 STD: every carried embedding e = W_up(q) entering GPT-2 becomes mu_E + sd_E * (e - mu_e) / sd_e (wte stats; EMA
        0.99 of the carried embeddings of each update from step 1, sd floored at 0.05 median as fade89 / accel99; frozen at the
        end). Draws no random numbers."""
        E = self.g.w("wte.weight").detach().float()
        self.register_buffer("muE", E.mean(0).clone()); self.register_buffer("sdE", E.std(0).clone())
        self.register_buffer("muH", torch.zeros(768, device=E.device)); self.register_buffer("sdH", torch.ones(768, device=E.device))
        self.register_buffer("seen", torch.tensor(0, device=E.device)); self.register_buffer("frozen", torch.tensor(0, device=E.device)); self.std = True

    def std_map(self, e):
        if self.training and not int(self.frozen):
            self._acc.append(e.detach().float().reshape(-1, e.shape[-1]))
        sd = torch.maximum(self.sdH, 0.05 * self.sdH.median())
        return (self.muE + self.sdE * (e.float() - self.muH) / sd).to(e.dtype)

    @torch.no_grad()
    def update_stats(self):
        """§125: EMA (momentum 0.99) over this update's carried embeddings (first update initialises), after opt.step()."""
        if self.std and self._acc and not int(self.frozen):
            H = torch.cat(self._acc); hm, hs = H.mean(0), H.std(0) + 1e-5
            if int(self.seen) == 0:
                self.muH.copy_(hm); self.sdH.copy_(hs)
            else:
                self.muH.mul_(0.99).add_(0.01 * hm); self.sdH.mul_(0.99).add_(0.01 * hs)
            self.seen += 1
        self._acc = []

    def quant(self, h, noise=None):
        if self.raw_h:                                                      # §126: the carrier IS the segment's last hidden (768-d)
            hh = h.float() + (noise if noise is not None else 0)
            return hh, hh, hh
        if self.anneal:                                                     # §126 ANNEAL: keep h for the mixed carrier (embed_code)
            self._h_last = h
        z = self.down(self.ln(h.float())); zb = torch.tanh(z + SHIFT) * HALF - OFF
        if self.training and self.noise > 0:                                # §123 dither (continuous) / pre-rounding dither (FSQ)
            zb = zb + (torch.rand_like(zb) * 2 - 1) * self.noise
        if noise is not None:
            zb = zb + noise
        if self.no_round:                                                   # §123: carried continuously, never rounded
            return zb, zb, zb
        q = torch.round(zb)
        if self.det_train:                                                  # §127 checks (eval mode): soft unless fully hard, no draw
            hard = self.p_hard >= 1.0
        else:
            hard = self.training is False or (torch.rand(()) < self.p_hard).item()
        qs = zb + (q - zb).detach() if hard else zb
        return qs, q, zb

    def embed_code(self, qs):
        if self.std2:                                                       # §127 C (the only change in its arm)
            return self.std2_map(self.up(qs))
        if self.anneal:                                                     # §126 ANNEAL: (1 - lam) std(h) + lam W_up(z_b) (+ QBN distillation)
            ec = self.up(qs); h = self._h_last; self._h_last = None
            if self.lam >= 1.0 or h is None or h.shape[0] != ec.shape[0]:
                return ec
            sh = self.std_map(h.float())
            if self.training and self.distill_on:                          # ||W_up(z_b) - sg(std h)||^2 / ||sg(std h)||^2 (weight 1, RAMP ONLY:
                # before 2k the distill gradient would reach GPT-2 through z_b(h), so ANNEAL would not equal RAW-H there)
                t = sh.detach().float(); self._dist.append((((ec.float() - t) ** 2).sum(-1) / (t ** 2).sum(-1).clamp_min(1e-6)).mean())
            return (1 - self.lam) * sh.float() + self.lam * ec.float()     # fp32 while lam < 1, as RAW-H (note 126-5; ANNEAL s2 ran bf16-cast)
        e = qs if self.raw_h else self.up(qs)                              # §126: std(h) directly; else W_up(code)
        return self.std_map(e) if self.std else e                          # §125: the same map everywhere a code enters GPT-2

    def segment(self, prev, items):
        """prev (B, P, D) carried embedding(s) or start text embeddings; items (B,) -> last hidden (B, D)."""
        g = self.g; B = items.shape[0]; eot = g.embed(torch.full((B, 1), EOT, device=items.device))
        x = torch.cat([eot, prev.to(eot.dtype), g.embed(self.itok[items])], 1); h, _ = g(x, None)
        return h[:, -1]

    def start(self, B, dev, states=None):
        self._h_last = None
        if self.fam.name == "s5":
            return self.g.embed(torch.tensor(STTOK[states], device=dev))
        z0 = torch.zeros(B, D_Q, device=dev)
        if self.std2:                                                       # §127 C: the start is not a hand-off (see std2_map)
            return self.std2_map(self.up(z0), acc=False)[:, None]
        return self.embed_code(z0)[:, None]

    def answer_ce(self, e, ans, qs=None):
        """Answer decoder: [EOT, carried embedding] + teacher-forced answer tokens -> CE (summed over tokens, mean rows).
        §127 SKIP: qs (the carried q_st) feeds the skip maps at every position after EOT."""
        g = self.g; B = e.shape[0]; eot = g.embed(torch.full((B, 1), EOT, device=e.device))
        x = torch.cat([eot, e[:, None].to(eot.dtype)] + ([g.embed(ans[:, :-1])] if ans.shape[1] > 1 else []), 1)
        if self.skip:
            assert qs is not None, "§127 SKIP: the decoder needs q_st"
            h, _ = self.gpt_skip(x, None, qs, (torch.arange(x.shape[1], device=x.device) > 0))
        else:
            h, _ = g(x, None)
        lg = g.logits(h[:, 1:]).float()
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ans.reshape(-1), reduction="none").view(ans.shape).sum(-1).mean()

    @torch.no_grad()
    def answer_greedy(self, e, n_tok, qs=None):
        g = self.g; B = e.shape[0]; eot = g.embed(torch.full((B, 1), EOT, device=e.device))
        if self.skip:                                                       # §127 SKIP: the carried slot and every generated position
            assert qs is not None, "§127 SKIP: the decoder needs q_st"
            fwd = lambda x_, p_, m_: self.gpt_skip(x_, p_, qs, m_)  # noqa: E731
        else:
            fwd = lambda x_, p_, m_: g(x_, p_)  # noqa: E731
        dv = e.device
        h, past = fwd(torch.cat([eot, e[:, None].to(eot.dtype)], 1), None, torch.tensor([False, True], device=dv)); t = g.logits(h[:, -1]).argmax(-1); out = [t]
        for _ in range(n_tok - 1):
            h, past = fwd(g.embed(t)[:, None], past, torch.tensor([True], device=dv)); t = g.logits(h[:, 0]).argmax(-1); out.append(t)
        return torch.stack(out, 1)

    def chain(self, start_emb, items, labels=None, trace_mask=None, record=False, override=None):
        """Run a chain: start_emb (B, P, D); items (B, L). labels (B, L) true states after each item (training CE at the
        final position, and at trace_mask positions). Returns (loss, final carried embedding, per-position codes/embeds)."""
        B, L = items.shape; prev = start_emb; losses = []; codes = []; embs = []
        for t in range(L):
            h = self.segment(prev, items[:, t]); qs, q, zb = self.quant(h)
            if self.var_on:                                                 # §127 A: (z_b, item, position) of every hand-off
                self._var.append((zb, items[:, t], torch.full_like(items[:, t], t)))
            if override is not None and override[0] == t:
                q = override[1]; qs = q.to(qs.dtype)
            e = self.embed_code(qs)
            if record:
                codes.append(q.detach()); embs.append(e.detach())
            if labels is not None and (t == L - 1 or (trace_mask is not None and trace_mask[:, t].any())):
                rows = torch.arange(B, device=items.device) if t == L - 1 else torch.nonzero(trace_mask[:, t]).squeeze(1)
                ans = torch.tensor(self.fam.answer_tokens(labels[rows.cpu().numpy(), t]), device=items.device)
                losses.append(self.answer_ce(e[rows], ans, qs[rows]))
            prev = e[:, None]
        self.last_qs = qs
        return (torch.stack(losses).mean() if losses else None), e, codes, embs


def lmax_at(step):
    for lim, L in ((2000, 2), (5000, 4), (9000, 8), (14000, 16)):
        if step <= lim:
            return L
    return 32


def make_batch(fam, rng, bs, L):
    if fam.name == "s5":
        s0 = rng.integers(0, 120, bs); x = rng.integers(0, 120, (bs, L))
    else:
        s0 = np.zeros(bs, np.int64); x = ff_sequences(rng, bs, L)
    lab = np.zeros((bs, L), np.int64); s = s0.copy()
    for t in range(L):
        s = fam.delta(s, x[:, t]); lab[:, t] = s
    return s0, x, lab


@torch.no_grad()
def run_test(m, fam, s0, X, dev, amp, bs, every=True, grid=None):
    """Hard chains; decode at every position (every=True) or at reads only (flip-flop). Returns decoded states (N, L) and
    the codes (N, L, 8) int8."""
    N, L = X.shape; pred = np.full((N, L), -1); ntok = NST if fam.name == "s5" else 1
    codes = np.zeros((N, L, 1 if m.raw_h else D_Q), np.float32 if (m.no_round or m.raw_h) else np.int8)   # §126: 768-d carriers not stored
    for i in range(0, N, bs):
        sl = slice(i, i + bs); B = len(X[sl]); prev = m.start(B, dev, s0[sl] if fam.name == "s5" else None)
        for t in range(L):
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                h = m.segment(prev, torch.tensor(X[sl, t], device=dev)); qs, q, zb = m.quant(h); e = m.embed_code(qs)
                if every or (X[sl, t] == RD).any():
                    pred[sl, t] = fam.decode(m.answer_greedy(e, ntok, qs).cpu().numpy())
            if not m.raw_h:
                codes[sl, t] = q.float().cpu().numpy()
            prev = e[:, None]
            if grid is not None and not m.raw_h:                            # §123 grid_share
                zf = zb.float(); grid[0] += int(((zf - torch.round(zf)).abs() <= 0.1).sum()); grid[1] += zf.numel()
    return pred, codes


@torch.no_grad()
def certificate(m, fam, dev, amp, cap, bs=2048, time_limit=None, first_items=None):
    """BFS over the code machine; decode each code; check the transition law on every reachable (code, item) pair.
    time_limit (s, Audit 2026-10-05): stop the BFS when exceeded -> timed_out (inconclusive, like the cap)."""
    ntok = NST if fam.name == "s5" else 1; idx = {}; codes = []; dec = []; trans = {}; t_start = time.time(); timed_out = False
    def add(qrows):
        out = []
        for r in qrows:
            k = r.tobytes()
            if k not in idx:
                idx[k] = len(codes); codes.append(r)
            out.append(idx[k])
        return out
    def decode_codes(ids):
        res = []
        for i in range(0, len(ids), bs):
            q = torch.tensor(np.stack([codes[j] for j in ids[i:i + bs]]).astype(np.float32), device=dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                res.append(fam.decode(m.answer_greedy(m.embed_code(q), ntok, q).cpu().numpy()))
        return np.concatenate(res) if res else np.zeros(0, int)
    # start transitions
    starts = np.arange(120) if fam.name == "s5" else np.array([0]); ok = True; frontier = []; viol = 0
    S, X = np.meshgrid(starts, np.arange(fam.n_items) if first_items is None else np.array(first_items), indexing="ij"); S, X = S.ravel(), X.ravel()   # note 120-7: FF task start = a write
    for i in range(0, len(S), bs):
        B = len(S[i:i + bs])
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            h = m.segment(m.start(B, dev, S[i:i + bs] if fam.name == "s5" else None), torch.tensor(X[i:i + bs], device=dev)); _, q, _ = m.quant(h)
        ids = add(q.float().cpu().numpy().astype(np.int8)); frontier += ids
        for s_, x_, c in zip(S[i:i + bs], X[i:i + bs], ids):
            trans[("start", int(s_), int(x_))] = c
    frontier = sorted(set(frontier)); seen = set(frontier); capped = False
    layers = [{"depth": 1, "frontier": len(frontier), "cum": len(codes), "sec": round(time.time() - t_start)}]   # Theory: growth curve
    while frontier:
        nxt = []
        pairs = [(c, x) for c in frontier for x in range(fam.n_items)]
        for i in range(0, len(pairs), bs):
            if time_limit and time.time() - t_start > time_limit:
                timed_out = True; break
            pc = pairs[i:i + bs]; q = torch.tensor(np.stack([codes[c] for c, _ in pc]).astype(np.float32), device=dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                h = m.segment(m.embed_code(q)[:, None], torch.tensor([x for _, x in pc], device=dev)); _, qn, _ = m.quant(h)
            ids = add(qn.float().cpu().numpy().astype(np.int8))
            for (c, x), n in zip(pc, ids):
                trans[(c, x)] = n
                if n not in seen:
                    seen.add(n); nxt.append(n)
        if len(codes) > cap:
            capped = True; break
        layers.append({"depth": layers[-1]["depth"] + 1, "frontier": len(nxt), "cum": len(codes), "sec": round(time.time() - t_start),
                       "complete": not timed_out})
        if timed_out:
            break
        frontier = nxt
    if timed_out or capped:                                                 # inconclusive: no decode / Moore pass on a partial machine (the cap too:
        return {"holds": None, "status": "inconclusive_time" if timed_out else "inconclusive_cap", "n_reachable_partial": len(codes),   # a 10^6-code
                "capped": capped, "timed_out": timed_out, "time_limit": time_limit, "sec": round(time.time() - t_start),   # Python Moore ran for hours)
                "layers": layers, "n_start_pairs": int(len(S))}
    dec = decode_codes(list(range(len(codes))))
    valid = bool((dec >= 0).all())
    for k, n in trans.items():
        if k[0] == "start":
            want = fam.delta(np.array([k[1]]), np.array([k[2]]))[0]
        else:
            if dec[k[0]] < 0:
                viol += 1; continue
            want = fam.delta(np.array([dec[k[0]]]), np.array([k[1]]))[0]
        viol += int(dec[n] != want)
    holds = valid and viol == 0 and not capped
    # Moore minimisation of the code machine (outputs = decodes)
    n = len(codes); part = {c: int(dec[c]) for c in range(n)}; t_moore = time.time(); moore_timed_out = False
    while True:
        if time_limit and time.time() - t_moore > time_limit:              # Moore bounded like the BFS (n_moore then None)
            moore_timed_out = True; break
        sig = {c: (part[c],) + tuple(part.get(trans.get((c, x), -1), -2) for x in range(fam.n_items)) for c in range(n)}
        relabel = {}; newp = {c: relabel.setdefault(sig[c], len(relabel)) for c in range(n)}
        if len(set(newp.values())) == len(set(part.values())):
            break
        part = newp
    return {"holds": bool(holds), "n_reachable": n, "n_moore": None if moore_timed_out else len(set(part.values())), "capped": capped,
            "violations": viol, "all_decode_valid": valid, "status": "done", "timed_out": False, "moore_timed_out": moore_timed_out,
            "sec": round(time.time() - t_start), "layers": layers, "n_start_pairs": int(len(S))}


@torch.no_grad()
def cert_task_reads(m, fam, dev, amp, cap, bs=2048, time_limit=None):
    """Note 120-9 (descriptive, flip-flop only): product-machine BFS over (code c, true state q), from (start, q0 = no
    write); first item w0 / w1, then all 5 items: c' = next(c, x), q' = delta(q, x). Checked on READ transitions only:
    decode(c') must equal the bit q holds (states 1 / 2 = bit 0 / 1, as fam.decode returns). holds iff every reachable read
    transition is right and nothing was capped / timed out."""
    assert fam.name == "ff"; t0 = time.time(); idx = {}; codes = []; nxt_tab = {}; dec = {}; ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    def add(rows):
        out = []
        for r in rows:
            k = r.tobytes()
            if k not in idx:
                idx[k] = len(codes); codes.append(r)
            out.append(idx[k])
        return out
    def step(cs, xs):                                                       # next codes for (code, item) pairs, batched
        out = []
        for i in range(0, len(cs), bs):
            q = torch.tensor(np.stack([codes[c] for c in cs[i:i + bs]]).astype(np.float32), device=dev)
            with ac():
                h = m.segment(m.embed_code(q)[:, None], torch.tensor(xs[i:i + bs], device=dev)); _, qn, _ = m.quant(h)
            out += add(qn.float().cpu().numpy().astype(np.int8))
        return out
    def decode(cs):
        todo = [c for c in cs if c not in dec]
        for i in range(0, len(todo), bs):
            q = torch.tensor(np.stack([codes[c] for c in todo[i:i + bs]]).astype(np.float32), device=dev)
            with ac():
                d = fam.decode(m.answer_greedy(m.embed_code(q), 1, q).cpu().numpy())
            dec.update(zip(todo[i:i + bs], d.tolist()))
    with ac():                                                              # first item: a write from the empty start
        h = m.segment(m.start(2, dev, None), torch.tensor([W0, W1], device=dev)); _, q, _ = m.quant(h)
    c1 = add(q.float().cpu().numpy().astype(np.int8))
    frontier = [(c1[0], int(fam.delta(np.array([0]), np.array([W0]))[0])), (c1[1], int(fam.delta(np.array([0]), np.array([W1]))[0]))]
    seen = set(frontier); viol = 0; n_read = 0; capped = timed_out = False
    while frontier:
        need = sorted({c for c, _ in frontier if c not in nxt_tab})
        for x in range(fam.n_items):
            if time_limit and time.time() - t0 > time_limit:
                timed_out = True; break
            if need:
                for c, n in zip(need, step(need, [x] * len(need))):
                    nxt_tab[(c, x)] = n
        if timed_out:
            break
        for c in need:
            nxt_tab[c] = True
        reads = [nxt_tab[(c, RD)] for c, _ in frontier]; decode(reads); new = []
        for c, qs in frontier:
            for x in range(fam.n_items):
                c2 = nxt_tab[(c, x)]; q2 = int(fam.delta(np.array([qs]), np.array([x]))[0])
                if x == RD:
                    n_read += 1; viol += int(dec[c2] != qs)
                if (c2, q2) not in seen:
                    seen.add((c2, q2)); new.append((c2, q2))
        if len(seen) > cap:
            capped = True; break
        frontier = new
    holds = None if (capped or timed_out) else (viol == 0)
    return {"holds": holds, "status": "inconclusive_time" if timed_out else ("inconclusive_cap" if capped else "done"), "n_pairs": len(seen),
            "n_codes": len(codes), "n_read_transitions": n_read, "violations": viol, "capped": capped, "timed_out": timed_out,
            "sec": round(time.time() - t0)}


@torch.no_grad()
def carried_stats(m, fam, dev, amp, n=256, L=4, seed=12500):
    """§125: on a fixed batch (own RNG; eval mode, so no torch draws), the carried embeddings that enter GPT-2: mean row norm and
    the per-coordinate std across inputs (mean over coordinates, and its norm), plus GPT-2 wte's row norm for reference."""
    was = m.training; m.eval(); s0, x, _ = make_batch(fam, np.random.default_rng(seed), n, L)
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        _, _, _, embs = m.chain(m.start(n, dev, s0 if fam.name == "s5" else None), torch.tensor(x, device=dev), record=True)
    E = torch.cat([e.float() for e in embs]); sd = E.std(0); W = m.g.w("wte.weight").float()
    out = {"row_norm": float(E.norm(dim=-1).mean()), "per_coord_std_across_inputs": float(sd.mean()), "std_vector_norm": float(sd.norm()),
           "wte_row_norm": float(W.norm(dim=-1).mean()), "n": int(E.shape[0])}
    m.train(was); return out


def make_opt(m, lr, lr_new=1.0):
    """AdamW wd 0.01; §127 D (lr_new != 1): LN / W_down / W_up in their own group at lr_new x lr (the loop scales by "mult")."""
    if lr_new == 1.0:
        return torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    newp = list(m.ln.parameters()) + list(m.down.parameters()) + list(m.up.parameters()); ids = {id(p_) for p_ in newp}
    return torch.optim.AdamW([{"params": [p_ for p_ in m.parameters() if id(p_) not in ids], "mult": 1.0},
                              {"params": newp, "mult": lr_new}], lr=lr, weight_decay=0.01)


def lr_at(base, gr, step, steps):
    return base * gr.get("mult", 1.0) * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))


def probe_batch127(fam):
    return make_batch(fam, np.random.default_rng(12702), 1024, 2)


def _probe_chain(m, fam, dev, amp, s0, x, grad=False, train_sem=False):
    """Run the probe chains in eval mode (no torch draws); returns per-hand-off (zb, qs, e). train_sem: what training feeds
    GPT-2 at this step, deterministically (soft codes unless p_hard = 1; STD2 on this hand-off's batch statistics)."""
    m.det_train = train_sem; prev = m.start(len(x), dev, s0 if fam.name == "s5" else None); X = torch.tensor(x, device=dev); out = []
    try:
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            for t in range(X.shape[1]):
                h = m.segment(prev, X[:, t]); qs, q, zb = m.quant(h); e = m.embed_code(qs); out.append((zb, qs, e)); prev = e[:, None]
    finally:
        m.det_train = False
    return out


def _r2(g_fit, g_score, Zf, Zs):
    """Per-group means fit on (Zf, g_fit); R^2 of (Zs, g_score) (unseen groups -> the fit half's global mean)."""
    mu = Zf.mean(0); means = {}
    for gg in np.unique(g_fit):
        means[int(gg)] = Zf[g_fit == gg].mean(0)
    pred = np.stack([means.get(int(gg), mu) for gg in g_score])
    return float(1 - ((Zs - pred) ** 2).sum() / max(1e-12, ((Zs - mu) ** 2).sum()))


def probe127(m, fam, dev, amp):
    """§127 probe (every arm and the control): R^2 of z_b on the item read / on the state after it; z-ablation dloss."""
    was = m.training; m.eval(); s0, x, lab = probe_batch127(fam); n = len(x)
    with torch.no_grad():
        hs = _probe_chain(m, fam, dev, amp, s0, x)
        Z = torch.stack([z.float() for z, _, _ in hs], 1).cpu().numpy()       # (n, 2, 8)
        half = n // 2; it = x; st = lab
        Zf, Zs = Z[:half].reshape(-1, Z.shape[-1]), Z[half:].reshape(-1, Z.shape[-1])
        r2i = _r2(it[:half].reshape(-1), it[half:].reshape(-1), Zf, Zs); r2s = _r2(st[:half].reshape(-1), st[half:].reshape(-1), Zf, Zs)
        _, qs, e = hs[-1]; ans = torch.tensor(fam.answer_tokens(lab[:, -1]), device=dev)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            l0 = float(m.answer_ce(e, ans, qs)); qm = qs.float().mean(0, keepdim=True).expand_as(qs).to(qs.dtype)
            l1 = float(m.answer_ce(m.embed_code(qm), ans, qm))
    m.train(was)
    return {"r2_item": r2i, "r2_state": r2s, "abl_dloss": l1 - l0, "ce_intact": l0, "ce_ablated": l1, "n": n}


def mcheck127(m, fam, dev, amp, opt, ng=256):
    """Note 127-1 delivery checks on the fixed probe batch (eval mode; grads by autograd.grad, so .grad is untouched). The two
    gradient checks use its first ng chains (activation memory), the other keys all 1,024. Every check uses training semantics
    (_probe_chain train_sem): what the intervention feeds GPT-2 at this step; *_eval keys repeat row norm / resid std in eval
    mode (hard codes, STD2's EMA)."""
    was = m.training; m.eval(); s0, x, lab = probe_batch127(fam); out = {"grad_rows": ng, "semantics": "train (deterministic)"}
    key = torch.cat([torch.tensor(x[:, t] * 32 + t, device=dev) for t in range(x.shape[1])])
    for sfx, ts in (("", True), ("_eval", False)):
        with torch.no_grad():
            hs = _probe_chain(m, fam, dev, amp, s0, x, train_sem=ts)
            E = torch.cat([e.float() for _, _, e in hs]); out["row_norm" + sfx] = float(E.norm(dim=-1).mean())
            Z = torch.cat([z.float() for z, _, _ in hs])
            cnt = torch.bincount(key).float(); gm = torch.zeros(len(cnt), Z.shape[1], device=dev).index_add_(0, key, Z) / cnt.clamp_min(1)[:, None]
            R = Z - gm[key]; G = int((cnt > 0).sum())
            out["resid_std" + sfx] = float(torch.sqrt((R ** 2).sum(0) / max(1, len(Z) - G)).mean())
            if hasattr(m, "vm") and ts:
                seen = m.vseen[key]
                out["resid_std_ema"] = float((Z[seen] - m.vm[key[seen]]).std(0).mean()) if seen.sum() > 1 else None
    out["wte_row_norm"] = float(m.g.w("wte.weight").float().norm(dim=-1).mean())
    s0, x, lab = s0[:ng], x[:ng], lab[:ng]
    if m.skip:
        hs_ = _probe_chain(m, fam, dev, amp, s0, x, train_sem=True); _, qs, e = hs_[-1]; ql = qs.detach().float().requires_grad_(True)
        ans = torch.tensor(fam.answer_tokens(lab[:, -1]), device=dev)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            L_ = m.answer_ce(e.detach(), ans, ql)
        out["skip_grad_q"] = float(torch.autograd.grad(L_, ql)[0].norm())
    if hasattr(m, "vm"):
        on = m.var_on; m.var_on = True; m._var = []
        hs_ = _probe_chain(m, fam, dev, amp, s0, x, grad=True, train_sem=True)
        m._var = [(z, torch.tensor(x[:, t], device=dev), torch.full((len(x),), t, device=dev)) for t, (z, _, _) in enumerate(hs_)]
        vl = m.var_loss(update=False); m.var_on = on
        out["var_grad_down"] = float(torch.autograd.grad(vl, m.down.weight, retain_graph=True)[0].norm()); out["var_term"] = float(vl.detach())
    if opt is not None:
        lrs = [gr["lr"] for gr in opt.param_groups]; out["lr_ratio"] = lrs[1] / lrs[0] if len(lrs) > 1 and lrs[0] > 0 else 1.0; out["lrs"] = lrs
    m.train(was)
    return out


CONTRACT_EPS = (0.03, 0.1, 0.3, 1.0)


@torch.no_grad()
def contraction(m, fam, dev, amp, n=500, T0=4, K=16, seed=12301):
    """§123 rho / basin (theory/contraction_diag.py ported): perturb carried z_b at T0 by eps * std_d * u, run K hand-offs."""
    rng = np.random.default_rng(seed); s0, x, _ = make_batch(fam, rng, n, T0 + K); X = torch.tensor(x, device=dev)
    ntok = NST if fam.name == "s5" else 1; ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    def step(prev, t, add=None):
        with ac():
            h = m.segment(prev, X[:, t]); qs, q, zb = m.quant(h, noise=add); e = m.embed_code(qs)
        e.qs_ = qs                                                          # §127 SKIP: the decoder's q_st rides on e
        return zb.float(), e
    def dec(e):
        with ac():
            return fam.decode(m.answer_greedy(e, ntok, e.qs_).cpu().numpy())
    prev = m.start(n, dev, s0 if fam.name == "s5" else None); zs = []
    for t in range(T0 - 1):
        z, e = step(prev, t); zs.append(z); prev = e[:, None]
    pre_T0 = prev; zT0, eT0 = step(prev, T0 - 1); zs.append(zT0); clean = [zT0]; cdec = []; c = eT0[:, None]
    for k in range(K):
        z, e = step(c, T0 + k); zs.append(z); clean.append(z); cdec.append(dec(e)); c = e[:, None]
    std_d = torch.cat(zs).std(0)                                            # per-coordinate std of clean carried z_b
    U = torch.tensor(rng.standard_normal((n, zT0.shape[-1])), dtype=torch.float32, device=dev); R = {}   # 8 (FSQ) or 768 (§126 RAW-H)
    for eps in CONTRACT_EPS:
        delta = eps * std_d * U; zp, ep = step(pre_T0, T0 - 1, add=delta); dist = [(zp - zT0).norm(dim=-1)]; agree = []; c = ep[:, None]
        distn = [((zp - zT0) / std_d.clamp_min(1e-6)).norm(dim=-1)]                        # note 123-2: std-normalised coordinates
        for k in range(K):
            z, e = step(c, T0 + k); dist.append((z - clean[k + 1]).norm(dim=-1)); agree.append(float((dec(e) == cdec[k]).mean())); c = e[:, None]
            distn.append(((z - clean[k + 1]) / std_d.clamp_min(1e-6)).norm(dim=-1))
        D = torch.stack(dist).cpu().numpy(); ratio = D[1:] / np.maximum(D[:-1], 1e-6); rho_k = np.median(ratio, axis=1)
        Dn = torch.stack(distn).cpu().numpy(); rho_kn = np.median(Dn[1:] / np.maximum(Dn[:-1], 1e-6), axis=1)
        R[str(eps)] = {"rho_k": rho_k.round(4).tolist(), "rho_geo_1_8": float(np.exp(np.log(np.maximum(rho_k[:8], 1e-6)).mean())),
                       "agree_k": [round(v, 4) for v in agree], "dist_med_k": np.median(D, axis=1).round(4).tolist(),
                       "rho_k_norm": rho_kn.round(4).tolist(), "rho_norm_geo_1_8": float(np.exp(np.log(np.maximum(rho_kn[:8], 1e-6)).mean()))}
        # note 123-1 (FSQ arms): chains rounded back exactly (D = 0 at every k = 1..8), and rho over chains with D_{k-1} > 0 only
        R[str(eps)]["exact_recovery_share"] = float((D[1:9] == 0).all(0).mean())
        rk_nz = [float(np.median(ratio[k][D[k] > 0])) for k in range(8) if (D[k] > 0).any()]
        R[str(eps)]["rho_k_nonzero"] = [round(v, 4) for v in rk_nz]
        R[str(eps)]["rho_nonzero"] = float(np.exp(np.log(np.maximum(rk_nz, 1e-6)).mean())) if rk_nz else None
    basin = max([e_ for e_ in CONTRACT_EPS if R[str(e_)]["agree_k"][7] >= 0.99], default=0.0)
    return {"rho": R["0.1"]["rho_geo_1_8"], "basin": basin, "exact_recovery_share": R["0.1"]["exact_recovery_share"],
            "rho_nonzero": R["0.1"]["rho_nonzero"], "rho_norm": R["0.1"]["rho_norm_geo_1_8"], "detail": {"T0": T0, "K": K, "n": n, "std_d": std_d.cpu().numpy().round(4).tolist(), "eps": R}}


GROWTH_POS = (16, 64, 128, 256, 512, 1024, 2048)


def code_growth(codes, lab=None):
    """Theory (Audit): distinct codes visited by position p over all chains (and the mean distinct codes per true state)."""
    out = {}
    for p_ in GROWTH_POS:
        if p_ > codes.shape[1]:
            continue
        flat = codes[:, :p_].reshape(-1, codes.shape[-1]); r = {"distinct": int(len(np.unique(flat, axis=0)))}
        if lab is not None:
            per = {}
            for c, s_ in zip(map(bytes, flat), lab[:, :p_].reshape(-1)):
                per.setdefault(int(s_), set()).add(c)
            r["per_state_mean"] = float(np.mean([len(v) for v in per.values()]))
        out[str(p_)] = r
    return out


@torch.no_grad()
def donor(m, fam, dev, amp, n=2000, seed=12001, bs=500):
    rng = np.random.default_rng(seed); res = {}; ntok = NST if fam.name == "s5" else 1
    sA, xA, labA = make_batch(fam, rng, n, 8); sB, xB, labB = make_batch(fam, rng, n, 8)   # S5: random given starts (as training)
    for pos in (2, 4, 6):
        got = []
        for i in range(0, n, bs):
            sl = slice(i, i + bs); B = len(xA[sl])
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                _, _, cB, _ = m.chain(m.start(B, dev, sB[sl] if fam.name == "s5" else None), torch.tensor(xB[sl], device=dev), record=True)
                _, eA, _, _ = m.chain(m.start(B, dev, sA[sl] if fam.name == "s5" else None), torch.tensor(xA[sl], device=dev),
                                      override=(pos - 1, cB[pos - 1].float()))
                got.append(fam.decode(m.answer_greedy(eA, ntok, m.last_qs).cpu().numpy()))
        got = np.concatenate(got); d = labB[:, pos - 1].copy()
        for t in range(pos, 8):
            d = fam.delta(d, xA[:, t])
        res[f"pos{pos}"] = {"donor_consistent": float((got == d).mean()), "recipient_consistent": float((got == labA[:, -1]).mean()),
                            "chance_overlap": float((d == labA[:, -1]).mean())}
    return res


@torch.no_grad()
def margin_flip(m, fam, s0, X, dev, amp, bs=250, seed=12002):
    g_ = torch.Generator(device="cpu").manual_seed(seed); flips = tot = 0
    for i in range(0, len(X), bs):
        sl = slice(i, i + bs); B = len(X[sl]); prev = m.start(B, dev, s0[sl] if fam.name == "s5" else None)
        for t in range(X.shape[1]):
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                h = m.segment(prev, torch.tensor(X[sl, t], device=dev)); qs, q, zb = m.quant(h)
                nz = (torch.rand(zb.shape, generator=g_) - 0.5).to(zb.device); qn = torch.round(zb + nz)
            flips += int((qn != q).any(-1).sum()); tot += B
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                prev = m.embed_code(qs)[:, None]                            # bf16 qs: embed under autocast (as run_test)
    return flips / max(1, tot)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", required=True, choices=["s5", "ff"]); ap.add_argument("--arm", required=True, choices=["final", "trace"])
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=2000)
    ap.add_argument("--test_len", type=int, default=512); ap.add_argument("--ext_n", type=int, default=500); ap.add_argument("--cert_cap", type=int, default=1_000_000)
    ap.add_argument("--donor_n", type=int, default=2000); ap.add_argument("--margin_n", type=int, default=500); ap.add_argument("--sched_scale", type=float, default=1.0, help="smoke only")
    ap.add_argument("--log_every", type=int, default=100); ap.add_argument("--data_dir", default=os.path.join(ROOT, "bench", "data", "s5"))
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    ap.add_argument("--no_round", action="store_true", help="§123: carry z_b continuously (never rounded)")
    ap.add_argument("--contract_n", type=int, default=500, help="§123 rho / basin chains (< 500: smoke only)")
    ap.add_argument("--micro", type=int, default=1, help="micro-batches (gradient accumulation) for training chains of length >= 32 (OOM guard)")
    ap.add_argument("--dither", type=float, default=0.0, help="§123: U(-s, s) on carried z_b at training hand-offs after 2k (with --no_round)")
    ap.add_argument("--fsq_dither", type=float, default=0.0, help="§123: U(-s, s) on z_b before rounding at training hand-offs after 4k")
    ap.add_argument("--no_cert", action="store_true", help="skip CERT in the main eval (run post-hoc with --cert_only; Audit 2026-10-05)")
    ap.add_argument("--growth_only", action="store_true", help="distinct codes vs position on ext_n 2,048-step rollouts -> <out>/eval_growth.json")
    ap.add_argument("--var_res", action="store_true", help="§127 A VAR: item-and-position residual variance on z_b")
    ap.add_argument("--var_w", type=float, default=1.0, help="§127 A: VAR weight up to 2k (note 127-1 rerun: 10)")
    ap.add_argument("--skip_dec", action="store_true", help="§127 B SKIP: S_l(q_st) into decoder blocks 6-11 and before ln_f")
    ap.add_argument("--std2", action="store_true", help="§127 C STD2: standardised carried embedding with an absolute floor")
    ap.add_argument("--lr_new", type=float, default=1.0, help="§127 D LR10: LN / W_down / W_up at this multiple of lr")
    ap.add_argument("--probe127", action="store_true", help="§127 / note 127-1 logging (implied by any §127 arm)")
    ap.add_argument("--mcheck_only", action="store_true", help="§127 (note 127-2): end-of-training mcheck from <out>/model.pt (p_hard 1) -> eval_mcheck_end.json")
    ap.add_argument("--stop_plateau", action="store_true", help="§127: stop at 2k when 123-3's flag fires (log kept, no eval.json)")
    ap.add_argument("--anneal", action="store_true", help="§126 Stage 2 ANNEAL: (1-lam) std(h) + lam W_up(z_b), lam 2k->6k, QBN distill, FSQ 6k->8k")
    ap.add_argument("--raw_h", action="store_true", help="§126 RAW-H: no bottleneck; carried vector = §125-standardised last hidden h")
    ap.add_argument("--std", action="store_true", help="§125 STD: standardised carried embedding (task-agnostic)")
    ap.add_argument("--fp32", action="store_true", help="note 123-5: --contraction_only in fp32 (autocast off) -> eval_contraction_fp32.json")
    ap.add_argument("--contraction_only", action="store_true", help="note 123-2: rho / rho_norm / basin alone from <out>/model.pt -> <out>/eval_contraction.json")
    ap.add_argument("--cert_only", action="store_true", help="CERT alone from <out>/model.pt -> <out>/eval_cert.json")
    ap.add_argument("--cert_time", type=int, default=2700, help="CERT wall-clock limit (s); exceeded -> inconclusive_time")
    ap.add_argument("--eval_only", action="store_true", help="re-run the eval from <out>/model.pt (training batch draws replayed for coverage_drawn)")
    a = ap.parse_args(); dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); sc = a.sched_scale
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    fam = Family(a.family); os.makedirs(a.out, exist_ok=True)
    assert not (a.dither and not a.no_round) and not (a.fsq_dither and a.no_round), "§123: --dither needs --no_round; --fsq_dither is FSQ"
    def mk():
        m_ = FSQ120(s5.load_gpt2(dev), fam).to(dev); m_.no_round = a.no_round; m_.raw_h = a.raw_h
        if a.std or a.raw_h or a.anneal:
            m_.enable_std()
        if a.anneal:                                                        # std buffers serve std(h) only; embed_code mixes (no std on W_up)
            m_.std = False; m_.anneal = True
        if a.skip_dec:                                                      # §127 (no global RNG draws)
            m_.enable_skip(a.seed)
        if a.std2:
            m_.enable_std2()
        if a.var_res:
            m_.enable_var(fam.n_items)
        return m_
    p127 = a.probe127 or a.var_res or a.skip_dec or a.std2 or a.lr_new != 1.0
    if a.growth_only:                                                       # Theory / Audit: distinct codes vs length on 2,048-step rollouts
        m = mk(); m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.eval()
        if fam.name == "s5":
            Xx, Yx = s5.load_csv(os.path.join(a.data_dir, "S5=2048.csv"), limit=a.ext_n); s5.check_labels(Xx, Yx)
            px, cx = run_test(m, fam, np.zeros(len(Xx), np.int64), Xx, dev, amp, a.eval_bs); acc = float((px == Yx).all(1).mean())
        else:
            Xx = ff_sequences(np.random.default_rng(12004), a.ext_n, 2048); Yx = np.zeros_like(Xx); s_ = np.zeros(len(Xx), np.int64)
            for t in range(Xx.shape[1]):
                s_ = fam.delta(s_, Xx[:, t]); Yx[:, t] = s_
            px, cx = run_test(m, fam, None, Xx, dev, amp, a.eval_bs, every=False); r_ = Xx == RD; acc = float(((px == Yx) & r_).sum() / r_.sum())
        cg = code_growth(cx, Yx)
        g_ = {"n": len(Xx), "len": int(Xx.shape[1]), "acc": acc, "code_growth": cg, "eval_dtype": str(amp),
              "codes_per_state": {k: v.get("per_state_mean") for k, v in cg.items()}, "codes_per_state_2048": (cg.get("2048") or {}).get("per_state_mean")}
        json.dump(g_, open(os.path.join(a.out, "eval_growth.json"), "w", newline="\n"), indent=1); print("GROWTH", json.dumps(g_), flush=True); return
    if a.mcheck_only:                                                       # note 127-2: STD2 train / test row norm at the end, post-hoc
        m = mk(); m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.p_hard = 1.0
        mc = mcheck127(m, fam, dev, amp, None); mc["eval_dtype"] = str(amp)
        json.dump(mc, open(os.path.join(a.out, "eval_mcheck_end.json"), "w", newline="\n"), indent=1); print("MCHECK", json.dumps(mc), flush=True); return
    if a.contraction_only:                                                  # note 123-2: post-hoc contraction (rho, rho_norm, ...) from model.pt
        m = mk(); m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.eval()
        amp_c = None if a.fp32 else amp                                     # note 123-5: bf16 makes 8-d chains bit-identical in 3-4 hand-offs
        ct = contraction(m, fam, dev, amp_c, n=a.contract_n); ct.update({"eval_dtype": str(amp_c)})
        json.dump(ct, open(os.path.join(a.out, "eval_contraction_fp32.json" if a.fp32 else "eval_contraction.json"), "w", newline="\n"), indent=1)
        print("CONTRACTION", json.dumps({k: ct[k] for k in ("rho", "rho_norm", "rho_nonzero", "exact_recovery_share", "basin")}), flush=True); return
    if a.cert_only:                                                         # post-hoc CERT (scoring as registered; only its scheduling moved)
        m = mk(); m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.eval()
        c = certificate(m, fam, dev, amp, a.cert_cap, time_limit=a.cert_time); c.update({"eval_dtype": str(amp), "cert_cap": a.cert_cap})
        if fam.name == "ff":                                                # note 120-7 (amendment after data): task-reachable CERT, first item w0 / w1
            c["cert_task"] = certificate(m, fam, dev, amp, a.cert_cap, time_limit=a.cert_time, first_items=[W0, W1]); c["cert_task"]["first_items"] = ["w0", "w1"]
            c["cert_task_reads"] = cert_task_reads(m, fam, dev, amp, a.cert_cap, time_limit=a.cert_time)   # note 120-9 (descriptive)
        json.dump(c, open(os.path.join(a.out, "eval_cert.json"), "w", newline="\n"), indent=1); print("CERT", json.dumps(c), flush=True); return
    json.dump({**vars(a), **({} if a.eval_only else {"train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"})},
              open(os.path.join(a.out, "args_eval.json" if a.eval_only else "args.json"), "w", newline="\n"), indent=1)   # notes 123-8 / 125-3
    m = mk(); drawn = np.zeros((fam.n_states, fam.n_items), bool)
    if a.eval_only:                                                         # replay the training draws (rng only) for coverage_drawn
        for step in range(1, a.steps + 1):
            L = int(rng.integers(1, lmax_at(step / sc) + 1)); s0, x, lab = make_batch(fam, rng, a.bs, L)
            if L <= 2:
                drawn[s0, x[:, 0]] = True
                if L == 2:
                    drawn[lab[:, 0], x[:, 1]] = True
        m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev))
        _last = [json.loads(r) for r in open(os.path.join(a.out, "log.jsonl"))][-1]; train_sec = _last["sec"]; last_loss = _last.get("loss")
    opt = None if a.eval_only else make_opt(m, a.lr, a.lr_new)
    log = None if a.eval_only else open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []
    carried = {} if a.eval_only else {"init": carried_stats(m, fam, dev, amp)}   # §125 (every arm; own RNG, eval mode: no draws)
    if not a.eval_only:
        last_loss = None
    reest = False; dist_hist = []; var_hist = []; L127 = {}
    if p127 and not a.eval_only:                                            # note 127-1: probe at step 0
        log.write(json.dumps({"step": 0, "probe": probe127(m, fam, dev, amp)}) + "\n"); log.flush()
    m.train()
    for step in range(1, 0 if a.eval_only else a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        m.p_hard = 0.0 if step <= 2000 * sc else min(1.0, (step - 2000 * sc) / (2000 * sc)); m.train()
        if a.anneal:                                                        # §126: lam 0 -> 1 over 2k-6k; FSQ soft -> hard over 6k-8k
            m.lam = 0.0 if step <= 2000 * sc else min(1.0, (step - 2000 * sc) / (4000 * sc))
            m.distill_on = 2000 * sc < step <= 6000 * sc
            m.p_hard = 0.0 if step <= 6000 * sc else min(1.0, (step - 6000 * sc) / (2000 * sc))
            if step == int(2000 * sc) + 1:
                m.frozen.fill_(1)                                           # std(h) statistics frozen at 2k
            if step == int(6000 * sc) + 1:                                  # ... re-estimated once at the FSQ switch, then frozen
                m.frozen.fill_(0); m.seen.fill_(0); reest = True
        m.noise = (a.dither if step > 2000 * sc else 0.0) if a.no_round else (a.fsq_dither if step > 4000 * sc else 0.0)
        var_w = (a.var_w * (1.0 if step <= 2000 * sc else max(0.0, 1 - (step - 2000 * sc) / (4000 * sc)))) if a.var_res else 0.0
        m.var_on = var_w > 0; m._var = []
        L = int(rng.integers(1, lmax_at(step / sc) + 1)); s0, x, lab = make_batch(fam, rng, a.bs, L)
        if L <= 2:                                                          # note 120-1: transitions labelled by short problems
            drawn[s0, x[:, 0]] = True
            if L == 2:
                drawn[lab[:, 0], x[:, 1]] = True
        tm = None
        if a.arm == "trace":
            tm = torch.ones(a.bs, L, dtype=torch.bool, device=dev) if fam.name == "s5" else torch.tensor(x == RD, device=dev)
        nmb = a.micro if L >= 32 else 1; opt.zero_grad(); tot = 0.0           # gradient accumulation at L >= 32 (OOM, 2026-10-05)
        for mi in range(nmb):
            sl = slice(mi * a.bs // nmb, (mi + 1) * a.bs // nmb)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                loss, _, _, _ = m.chain(m.start(len(x[sl]), dev, s0[sl] if fam.name == "s5" else None), torch.tensor(x[sl], device=dev), lab[sl],
                                        None if tm is None else tm[sl])
            if m.var_on and m._var:                                         # §127 A: VAR over this forward's hand-offs
                vl = m.var_loss(); loss = loss + var_w * vl; var_hist.append(float(vl.detach()))
            if m.anneal and m._dist:                                        # §126 QBN distillation during the ramp (weight 1)
                dl = torch.stack(m._dist).mean(); m._dist = []; loss = loss + dl; dist_hist.append(float(dl.detach()))
            m._dist = []
            (loss / nmb).backward(); tot += loss.item() / nmb
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(tot)
        if m.anneal:                                                        # std(h) EMA (std_map collects while not frozen)
            m.std = True; m.update_stats(); m.std = False
            if reest:
                m.frozen.fill_(1); reest = False
        else:
            m.update_stats()
        m.update_std2() if m.std2 else None
        if p127 and step in (1, int(500 * sc), int(2000 * sc), a.steps):     # note 127-1 delivery checks (after this update); + end (127-2)
            log.write(json.dumps({"step": step, "mcheck": mcheck127(m, fam, dev, amp, opt)}) + "\n"); log.flush(); m.train()
        if step == int(2000 * sc):                                         # §125: carried-embedding statistics at 2k (every arm)
            carried["2k"] = carried_stats(m, fam, dev, amp); m.train()
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)), "L_max": lmax_at(step / sc), "p_hard": m.p_hard, "noise": m.noise, "sec": round(time.time() - t0)}; hist = []; last_loss = rec["loss"]
            if m.anneal:
                rec.update({"lam": m.lam, "distill": float(np.mean(dist_hist)) if dist_hist else None}); dist_hist = []
            if a.var_res:
                rec.update({"var_w": var_w, "var": float(np.mean(var_hist)) if var_hist else None}); var_hist = []
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True); L127[step] = rec["loss"]
        if p127 and step == int(2000 * sc):                                 # note 127-1: probe at 2k (before any stop)
            log.write(json.dumps({"step": step, "probe": probe127(m, fam, dev, amp)}) + "\n"); log.flush(); m.train()
        if a.stop_plateau and step == int(2000 * sc) and L127:              # 123-3's flag (log steps nearest 100 / 1k / 2k)
            gl = lambda t_: L127[min(L127, key=lambda s_: abs(s_ - t_ * sc))]  # noqa: E731
            if gl(2000) >= 0.9 * gl(1000) and gl(2000) >= 0.25 * gl(100):
                log.write(json.dumps({"step": step, "stopped": "plateau (123-3 flag at 2k)", "loss_100": gl(100), "loss_1k": gl(1000), "loss_2k": gl(2000),
                                      "carried_norm": carried}) + "\n"); log.close(); print("STOPPED plateau at 2k", flush=True); return
    if not a.eval_only:
        train_sec = round(time.time() - t0)
        if m.std:
            m.frozen.fill_(1)                                               # §125: statistics frozen at the end, used unchanged at test
        if m.std2:
            m.s2frozen.fill_(1)                                             # §127 C: EMA frozen at the end
        torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    del opt; m.eval(); m.noise = 0.0; m.lam = 1.0; t1 = time.time()            # §126: test at lam = 1, hard
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    res = {"family": a.family, "arm": a.arm, "seed": a.seed, "train_sec": train_sec, "eval_only": a.eval_only, "std": a.std, "raw_h": a.raw_h, "anneal": a.anneal,
           "s127": {"var_res": a.var_res, "var_w": a.var_w if a.var_res else None, "skip_dec": a.skip_dec, "std2": a.std2, "lr_new": a.lr_new, "probe127": p127},
           "carried_norm": carried or None, "final_loss": last_loss, "no_round": a.no_round, "dither": a.dither, "fsq_dither": a.fsq_dither, "eval_dtype": str(amp), "coverage_drawn": float(drawn.mean()),
           "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "train_gpu": json.load(open(os.path.join(a.out, "args.json"))).get("train_gpu")}
    if fam.name == "s5":
        Xte, Yte = s5.load_csv(os.path.join(a.data_dir, "S5=512.csv"), limit=a.test_n); s5.check_labels(Xte, Yte); Xte, Yte = Xte[:, :a.test_len], Yte[:, :a.test_len]
        grid = [0, 0]; pred, codes = run_test(m, fam, np.zeros(len(Xte), np.int64), Xte, dev, amp, a.eval_bs, grid=grid); C = pred == Yte
        res["long_acc"] = float(C.all(1).mean()); res["acc_by_length"] = {str(t): float(C[:, :t].all(1).mean()) for t in (16, 32, 64, 128, 256, 512) if t <= C.shape[1]}
        lab = Yte
        if a.ext_n:
            el = 2048 if a.test_len >= 512 else a.test_len
            Xx, Yx = s5.load_csv(os.path.join(a.data_dir, "S5=2048.csv"), limit=a.ext_n); s5.check_labels(Xx, Yx); Xx, Yx = Xx[:, :el], Yx[:, :el]
            px, cx = run_test(m, fam, np.zeros(len(Xx), np.int64), Xx, dev, amp, a.eval_bs); res["ext_2048"] = {"n": len(Xx), "len": Xx.shape[1], "cumulative": float((px == Yx).all(1).mean())}
            if not (a.no_round or a.raw_h):
                res["code_growth_ext"] = code_growth(cx, Yx)
        s0m, Xm = np.zeros(a.margin_n, np.int64), Xte[:a.margin_n, :128]
    else:
        trng = np.random.default_rng(12000); Xte = ff_sequences(trng, a.test_n, a.test_len)
        lab = np.zeros_like(Xte); s = np.zeros(len(Xte), np.int64)
        for t in range(Xte.shape[1]):
            s = fam.delta(s, Xte[:, t]); lab[:, t] = s
        grid = [0, 0]; pred, codes = run_test(m, fam, None, Xte, dev, amp, a.eval_bs, every=False, grid=grid); rd = Xte == RD; ok = (pred == lab) & rd
        res["long_acc"] = float(ok.sum() / rd.sum()); res["long_acc_seq"] = float(((pred == lab) | ~rd).all(1).mean())
        res["acc_by_length"] = {str(t): float(ok[:, :t].sum() / max(1, rd[:, :t].sum())) for t in (16, 32, 64, 128, 256, 512) if t <= Xte.shape[1]}
        for p_ig in (0.98, 0.1):
            Xd = ff_sequences(np.random.default_rng(12003), 500, a.test_len, p_ig); ld = np.zeros_like(Xd); s = np.zeros(len(Xd), np.int64)
            for t in range(Xd.shape[1]):
                s = fam.delta(s, Xd[:, t]); ld[:, t] = s
            pd_, _ = run_test(m, fam, None, Xd, dev, amp, a.eval_bs, every=False); r_ = Xd == RD
            res[f"long_acc_pignore_{p_ig}"] = float(((pd_ == ld) & r_).sum() / r_.sum())
        s0m, Xm = None, Xte[:a.margin_n, :128]
    res["grid_share"] = None if a.raw_h else grid[0] / max(1, grid[1])   # §123 implicit quantisation
    ct = contraction(m, fam, dev, amp, n=a.contract_n); res["rho"] = ct["rho"]; res["basin"] = ct["basin"]; res["contraction"] = ct["detail"]
    res["exact_recovery_share"] = ct["exact_recovery_share"]; res["rho_nonzero"] = ct["rho_nonzero"]; res["rho_norm"] = ct["rho_norm"]   # note 123-2
    if a.no_round or a.raw_h:                                               # continuous: no codes / CERT / margin (§123 / §126)
        res["n_codes"] = res["code_perplexity"] = res["codes_per_state"] = res["margin_flip_rate"] = None
        res["cert"] = {"status": "n/a (continuous)", "holds": None}; res["donor"] = donor(m, fam, dev, amp, n=a.donor_n)
    else:
        flat = codes.reshape(-1, D_Q); keys, cnt = np.unique(flat, axis=0, return_counts=True); p = cnt / cnt.sum()
        res["n_codes"] = int(len(keys)); res["code_perplexity"] = float(np.exp(-(p * np.log(p)).sum()))
        st = lab.reshape(-1); per = {}
        for c, s_ in zip(map(bytes, flat), st):
            per.setdefault(int(s_), set()).add(c)
        res["codes_per_state"] = float(np.mean([len(v) for v in per.values()]))
        res["cert"] = {"status": "deferred", "holds": None} if a.no_cert else certificate(m, fam, dev, amp, a.cert_cap, time_limit=a.cert_time); res["donor"] = donor(m, fam, dev, amp, n=a.donor_n)
        res["margin_flip_rate"] = margin_flip(m, fam, s0m, Xm, dev, amp)
    res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({k: res[k] for k in ("long_acc", "n_codes", "cert")}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
