"""§223 PORT-STATE trainer (docs/research_portstate_2026-10-21/PS2_v1.md at 54f63381 §2-§3, Addenda A / B; Theory's answers Q1-Q7 10-21). NEW module:
boxes/codi_port.py is imported (build / LoRA / prj / trainable / token ids) and never edited; CurGate from boxes/vbank.py; items from
boxes/nested_words.py (NESTED-WORDS v1.2). Cells by flags (single-change diffs, Addendum A item 3):
    PS       --recode --lam 1 --bottleneck        M1-only  --recode --lam 1        M2-only  --bottleneck        DIRECT  --direct (no latents)
  * sequence: encoder = BOS (EOS 50256, every cell; Q5) + question tokens + <bot>, left padded, key mask; K latents (K = d + U{0..2}, capped at
    --k_cap 10; one d and one K per optimiser step = batches bucketed by d): z_1 = prj(h_last at <bot>), z_{j+1} = prj(h_last at latent j), fed as
    input embeddings with the encoder in past KV and a key mask (the question is visible to every latent step); decoder = <eot> + "The answer is: N" +
    EOS. DIRECT: no latents (the decoder follows the encoder directly). --codi_compat (smokes only): no BOS, no key mask on latent / decoder calls
    (codi_port's fix_attn_mask False; the encoder call keeps its key mask, as codi_port) and codi_port's CE dtype, so the plain path can be compared
    with codi_port's student CE.
  * (M1) --recode: D = a copy of GPT-2 blocks 0-1 + wpe + a copy of ln_f, fp32 trainable, with the frozen tied lm_head; D reads z_j ONLY (no question
    KV): input [z_j, wte(R_j tokens)[:-1]], causal, teacher-forced CE on tok(R_j) + EOS, per-token mean; mean over the K latents; z_j's target is
    R_j for j <= d and the HOLD target R_d for j > d (Q4; hold indices logged). Value tokens (numbers) CE logged separately. Not detached (gradflow).
  * (M2) --bottleneck: the decoder call's key mask zeroes every question-content AND <bot> position (BOS stays visible: it attends only to itself,
    so its KV is item-independent); the answer reads the latent rows + its own prompt tokens only.
  * loss (Q1): CE(answer) = the decoder CE over all decoder tokens (codi_port's ce; Q2) [+ lam x mean_j CE(D(z_j) -> R_j) with --recode]; NO
    distillation, NO teacher pass. CurGate (Q7) advances on the NUMBER-token CE only (thr 0.3, 1k bins, need 2, cap 2,500 per stage, d_max 1 -> 8);
    every advance (threshold or cap) and the entry step (first 100-step mean number-token CE < 0.5) are logged.
  * recipe (codi_port): LoRA r128 alpha32 dropout 0.1 on frozen bf16 GPT-2 + prj (+ D); AdamW lr 3e-3 wd 0.1 (no decay on biases / LayerNorm), cosine
    with 3% warm-up over --steps (24k), clip 2.0, bf16 autocast, effective batch 64 = micro x accumulation. Train stream nested_words.Stream(22330,
    excl = the test set + the pair candidates' signatures), frames 0-5.
  * per-term grad norms (Audit 10-21, logged only): on each log step, from the first micro-batch, |d answer CE / d shared| and |d lam recode CE /
    d shared| over LoRA + prj, and their cosine (gn_ans_shared / gn_rec_shared / cos_ans_rec), via torch.autograd.grad (no .grad change).
  * outputs: args.json (--args_only stops there; --args_ref -> args_diff.json + an ARGS-DIFF line), log.jsonl every --log_every (step / loss / sec
    adjacent for cost_guard), stages.json (every advance), a full-state ckpt_last.pt + ckpt_fingerprint.json every --ckpt_every (LoRA / prj / D,
    optimiser, scheduler, CurGate, the item stream and the step-draw generator states, torch / cuda RNG, elapsed), trainable.pt + train_done.json
    at the end. --time_steps N -> timing.json; --mem_check -> mem_check.json (the deepest shape: d 8, K 10, the longest R). No eval here (a separate
    eval module).
  * (Note 223-8 R3, Audit amendment 11a4b858; Theory 10-22) --m2_stage (needs --bottleneck; default off): the M2 mask is staged in MASS space.
    m = 0 while CurGate stage < 2; from the first step at stage >= 2 (m2_start, checkpointed) m = (step - m2_start) / --m2_ramp (2,000), capped
    at 1; the training decoder call is soft_decoder (a self-contained GPT-2 forward from the cache through the model's own modules) with
    + log(max(1 - m, 1e-6)) on the question-content + <bot> keys for m < 1 and the registered key mask for m >= 1. It is used for EVERY m (the
    cell never switches implementation mid-run); forward() without m2 (flags off, ps_eval) is the HF call, unchanged. Implementation difference
    of the PS cell only: soft_decoder's dropout RNG draws differ from the HF call's order (same distribution). m2_m / m2_start are logged.
  * (§226 ORDERED-WORDS stage 0, registered 09368b14; Theory 10-23) --task ow (default nested): items from boxes/ow_words.py (forward / topo layout
    --order_mode, mod 11; --dag on | off: off = a strict previous-state chain), its own test set / pair candidates (excluded from the stream) and
    stream seed (--stream_seed default: 22330 nested, ow_words.SEED_TRAIN 22930 ow). OW items carry wrap_final, so each step also logs the counts
    num_ok_wrap / n_wrap / num_ok_nowrap / n_nowrap (the number token correct, split by whether the final step wrapped mod 11); a nested run logs
    nothing new. args.json gains ow_words_sha256 for --task ow only.
  * (§226 P0d, Note 226-1 + Audit) --full_ft (default off): no LoRA; every GPT-2 parameter trainable in fp32 (master weights) under the same bf16
    autocast, AdamW / wd / clip / warm-up / cosine and no-decay rule; the base is codi_port.build's without its bf16 cast and adapters (build_full:
    the same pretrained load and seed + 2090 embedding resize; prj after the same seed + 2091 reseed). The lr is the cell's --lr (P0d: 1e-4).
  * (§227 TEXT-SERIAL, registered f69fd658) --text_cot (needs --direct; default off): the decoder target is the step sentences R_step then
    " The answer is <v_d>." + EOS (after <eot>, the question in the encoder prompt as DIRECT); teacher-forced CE on every target token. CurGate reads
    step_value_ce (the teacher-forced CE of each step sentence's VALUE token; args cg_signal), never num_ce (the final answer copies the last step).
    entry_step on step_value_ce. Greedy free decoding every --decode_every (1,000) steps and at stop on a sha-pinned held-out set (200 items per d,
    train frames 0-5, seed 22950, excluded from the stream by rendered match) for d 1..min(d_max + 1, 8): max_new 24 d + 16, stop at EOS; the first
    " The answer is N." is parsed; unparseable / non-terminating / context overflow = wrong -> decode_evals.jsonl (acc, per-step value acc, wrap split).
    python -m boxes.codi_ps --out boxes/runs_fsq/s223_ps_s1 --recode --lam 1 --bottleneck [--device cuda:0 ...]"""
import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import boxes.codi_port as CP  # noqa: E402
import boxes.nested_words as NW  # noqa: E402
import boxes.ow_words as OW  # noqa: E402

PAD, BOT, EOT, EOS = CP.PAD, CP.BOT, CP.EOT, CP.EOS; BOS = EOS; ANS_PROMPT = CP.ANS_PROMPT
SEED_TRAIN = 22330
META = ("device", "train_gpu", "train_driver", "torch_version", "transformers_version", "python_version", "cuda_version", "micro", "accumulation", "out", "resume_from")


class CurGate:
    """= boxes/vbank.py CurGate (copied verbatim in behaviour so this module does not import vbank / nested170): stages n_max = ns[0] -> ... ->
    ns[-1]; a stage advances when `need` consecutive completed `bin_`-step bins (from the stage's first step) of the per-step answer CE average
    < thr, or after `cap` steps in the stage; the final stage runs to --steps."""
    def __init__(self, ns, thr=0.3, need=2, cap=2500, bin_=1000):
        self.ns, self.thr, self.need, self.cap, self.bin = list(ns), thr, need, cap, bin_; self.stage = 0; self.start = 1; self.cur = []; self.bins = []; self.hist = []

    def n_max(self):
        return self.ns[self.stage]

    def update(self, step, ans_ce):
        if self.stage == len(self.ns) - 1:
            return None
        if ans_ce is not None:
            self.cur.append(ans_ce)
        reason = None
        if (step - self.start + 1) % self.bin == 0:
            self.bins.append(float(np.mean(self.cur)) if self.cur else float("inf")); self.cur = []
            if len(self.bins) >= self.need and all(b_ < self.thr for b_ in self.bins[-self.need:]):
                reason = "threshold: number-token CE < %.2f for %d bins" % (self.thr, self.need)
        if reason is None and step - self.start + 1 >= self.cap:
            reason = "cap %d steps" % self.cap
        if reason is None:
            return None
        self.hist.append({"step": step, "from": self.ns[self.stage], "to": self.ns[self.stage + 1], "reason": reason, "bins": self.bins[-4:]})
        self.stage += 1; self.start = step + 1; self.cur = []; self.bins = []
        return self.hist[-1]

    def state(self):
        return {k: getattr(self, k) for k in ("stage", "start", "cur", "bins", "hist")}

    def load(self, st):
        for k, v in st.items():
            setattr(self, k, v)


class DHead(nn.Module):
    """M1's state decoder: GPT-2 blocks 0-1 (copied, fp32), wpe (copied), ln_f (copied); the frozen tied lm_head is passed in at call time."""
    def __init__(self, m):
        super().__init__(); tr = m.transformer
        self.blocks = nn.ModuleList([copy.deepcopy(_base_block(tr.h[i])).float() for i in (0, 1)]); self.wpe = copy.deepcopy(tr.wpe).float(); self.ln = copy.deepcopy(tr.ln_f).float()
        for p_ in self.parameters():
            p_.requires_grad_(True)

    def forward(self, x, key_mask):
        """x (N, L, 768) fp32 inputs (position 0 = z_j); key_mask (N, L) 1 = real -> hidden (N, L, 768)."""
        N, L, _ = x.shape; h = x + self.wpe(torch.arange(L, device=x.device))[None]
        causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril(); am = causal[None, None] & key_mask.bool()[:, None, None, :]
        bias = torch.zeros(N, 1, L, L, device=x.device, dtype=h.dtype).masked_fill(~am, torch.finfo(h.dtype).min)
        for blk in self.blocks:
            h = _block_fwd(blk, h, bias)
        return self.ln(h)


def _base_block(blk):
    """a GPT2Block with LoRA wrappers unwrapped to the base Conv1D (D copies the pretrained weights, not the adapters)."""
    b = copy.deepcopy(blk)
    for parent, name in ((b.attn, "c_attn"), (b.attn, "c_proj"), (b.mlp, "c_fc"), (b.mlp, "c_proj")):
        mod = getattr(parent, name)
        if isinstance(mod, CP.LoRAConv1D):
            setattr(parent, name, mod.base)
    return b


def _block_fwd(blk, h, bias):
    """one GPT-2 block with an explicit additive attention bias (version-independent: no HF mask plumbing)."""
    a = blk.ln_1(h); qkv = blk.attn.c_attn(a); q, k, v = qkv.split(768, dim=2); N, L, _ = q.shape
    sh = lambda t: t.view(N, L, 12, 64).transpose(1, 2)  # noqa: E731
    o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=bias).transpose(1, 2).reshape(N, L, 768)
    h = h + blk.attn.c_proj(o); return h + blk.mlp(blk.ln_2(h))


HAS_TOK = 468                                                               # " has" (GPT-2); a step sentence is "<name> has <v> <unit>."


def text_target(it, enc):
    """§227: the decoder tokens [EOT] + tok(" " + R_step[0]) + ... + tok(" " + R_step[-1]) + tok(" The answer is <v_d>.") + [EOS] (piecewise encode ==
    the whole-string encode: every piece starts with a space), the value-token indices (one per step: the token after the first " has") and the
    answer-number index. -> (dec, vidx, nidx)."""
    dec = [EOT]; vidx = []
    for r in it["R_step"]:
        tk = enc.encode(" " + r); h = tk.index(HAS_TOK); vidx.append(len(dec) + h + 1); dec += tk
    tk = enc.encode(" The answer is %d." % it["answer"]); nidx = len(dec) + 3; dec += tk + [EOS]
    return dec, vidx, nidx


def tok_item(it, enc, text=False):
    """-> {q (BOS + question tokens), dec, R (list of token lists, R_1..R_d each + EOS), d, ans}; text (§227): dec = text_target + "vidx" / "nidx"."""
    q = [BOS] + enc.encode(it["text"])
    if text:
        dec, vidx, nidx = text_target(it, enc)
        return {"q": q, "dec": dec, "vidx": vidx, "nidx": nidx, "vals": it["values"][1:], "R": [enc.encode(r) + [EOS] for r in it["R"][1:]], "d": it["d"], "ans": it["answer"],
                **({"wrap": it["wrap_final"]} if "wrap_final" in it else {})}
    dec = [EOT] + enc.encode("The answer is: %d" % it["answer"]) + [EOS]
    assert dec[1:5] == ANS_PROMPT and len(enc.encode(" %d" % it["answer"])) == 1, (dec, it["answer"])
    return {"q": q, "dec": dec, "R": [enc.encode(r) + [EOS] for r in it["R"][1:]], "d": it["d"], "ans": it["answer"], **({"wrap": it["wrap_final"]} if "wrap_final" in it else {})}


def collate(its, K, dev, compat=False):
    """-> dict of tensors for one bucket (all items share d; K latents)."""
    B = len(its); qs = [it["q"][1:] if compat else it["q"] for it in its]; Tq = max(len(q) + 1 for q in qs); Td = max(len(it["dec"]) for it in its)
    enc = torch.full((B, Tq), PAD); em = torch.zeros(B, Tq, dtype=torch.long); qm = torch.zeros(B, Tq, dtype=torch.long)   # qm: question content + <bot>
    dec = torch.full((B, Td), PAD); dl = torch.full((B, Td), -100); npos = []
    for b, (it, q) in enumerate(zip(its, qs)):
        s = q + [BOT]; enc[b, Tq - len(s):] = torch.tensor(s); em[b, Tq - len(s):] = 1
        qm[b, Tq - len(s) + (0 if compat else 1):] = 1                    # every position after BOS (question content and <bot>)
        dec[b, :len(it["dec"])] = torch.tensor(it["dec"]); dl[b, :len(it["dec"])] = torch.tensor(it["dec"]); npos.append(it["nidx"] - 1 if "nidx" in it else CP.ans_pos(it["dec"]) - 1)
    d = its[0]["d"]; R = []
    for j in range(1, K + 1):                                               # z_j's target: R_j (j <= d) or the HOLD target R_d (j > d)
        R.append([it["R"][min(j, d) - 1] for it in its])
    t = lambda x: x.to(dev)  # noqa: E731
    out = {"enc": t(enc), "em": t(em), "qm": t(qm), "dec": t(dec), "dl": t(dl), "npos": t(torch.tensor(npos)), "R": R, "K": K, "d": d,
            "holds": [j for j in range(1, K + 1) if j > d]}
    if "wrap" in its[0]:                                                    # §226: OW items only
        out["wrapf"] = t(torch.tensor([bool(it["wrap"]) for it in its]))
    if "vidx" in its[0]:                                                    # §227: the step value positions (predicting position = index - 1), padded
        nv = max(len(it["vidx"]) for it in its); vp = torch.zeros(B, nv, dtype=torch.long); vm = torch.zeros(B, nv, dtype=torch.bool)
        for b, it in enumerate(its):
            vp[b, :len(it["vidx"])] = torch.tensor(it["vidx"]) - 1; vm[b, :len(it["vidx"])] = True
        out["vpos"] = t(vp); out["vmask"] = t(vm)
    return out


def _ones(B, n, dev):
    return torch.ones(B, n, dtype=torch.long, device=dev)


def _past_kv(pkv):
    """the cached per-layer (K, V) (B, H, L, 64) across transformers versions (5.x Cache.layers[i].keys / .values; 4.4x DynamicCache key_cache /
    value_cache; legacy tuples), as _leaf_cache reads them."""
    return [(l.keys, l.values) for l in pkv.layers] if hasattr(pkv, "layers") else (list(zip(pkv.key_cache, pkv.value_cache)) if hasattr(pkv, "key_cache") else list(pkv))


def soft_decoder(m, pkv, x, vis, qf, mval, capture=None):
    """§223 Note 223-8 R3 (--m2_stage; Theory 10-22 conditions a-d): the decoder call as a self-contained GPT-2 forward from the cached K / V
    (transformers 4.46 reshapes any attention_mask to 2-D, so an additive soft mask cannot go through the HF call). It runs the model's OWN modules
    (the LoRA-wrapped c_attn / c_proj / c_fc, ln_1 / ln_2 / ln_f, wpe, the tied lm_head) so LoRA and prj train through it, with GPT-2's train-mode
    dropouts (embd: transformer.drop; attention probabilities: attn.attn_dropout.p inside SDPA; resid: attn.resid_dropout; the MLP's own dropout)
    and HF's default positions (past length .. past length + L - 1). Implementation difference (PS cell only): the dropout RNG draws differ from
    the HF call's order (same distribution, not bitwise).
    x (B, Ld, 768) decoder input embeddings; vis (B, Lp) 1 = a visible cached key (padding 0); qf (B, Lp) 1 = question content + <bot> keys;
    mval: m < 1 -> + log(max(1 - m, 1e-6)) on the qf keys' logits (each question key's unnormalised weight x (1 - m)); m >= 1 -> qf keys masked (the
    registered M2 key mask). Own tokens: causal. capture (gates only): a list -> per layer the attention probabilities (B, H, Ld, Lp + Ld), computed
    explicitly in fp32 (fp64 in fp64 gates). -> logits (B, Ld, V)."""
    tr = m.transformer; B, Ld, _ = x.shape; PK = _past_kv(pkv); Lp = PK[0][0].shape[2]; dev = x.device
    h = tr.drop(x + tr.wpe(torch.arange(Lp, Lp + Ld, device=dev))[None])
    vis_ = vis.bool() & ~(qf.bool() if mval >= 1 else torch.zeros_like(qf, dtype=torch.bool))
    bias_fn = None
    for i, blk in enumerate(tr.h):
        a = blk.ln_1(h); q, k, v = blk.attn.c_attn(a).split(768, dim=2)
        sh = lambda t: t.view(B, Ld, 12, 64).transpose(1, 2)  # noqa: E731
        q, k, v = sh(q), sh(k), sh(v); K_ = torch.cat([PK[i][0].to(k.dtype), k], 2); V_ = torch.cat([PK[i][1].to(v.dtype), v], 2)
        if bias_fn is None:                                                 # one additive bias for every layer, in the attention dtype
            mn = torch.finfo(q.dtype).min; pb = torch.zeros(B, Lp, dtype=torch.float64, device=dev)
            if mval < 1:
                pb = pb + qf.to(torch.float64) * math.log(max(1.0 - mval, 1e-6))   # 0 exactly at m = 0; the 1e-6 floor only near m = 1
            pb = pb.masked_fill(~vis_, mn); own = torch.zeros(Ld, Ld, dtype=torch.float64, device=dev).masked_fill(torch.ones(Ld, Ld, dtype=torch.bool, device=dev).triu(1), mn)
            bias_fn = torch.cat([pb[:, None, None, :].expand(B, 1, Ld, Lp), own[None, None].expand(B, 1, Ld, Ld)], -1).to(q.dtype)
        o = F.scaled_dot_product_attention(q, K_, V_, attn_mask=bias_fn, dropout_p=blk.attn.attn_dropout.p if m.training else 0.0)
        if capture is not None:
            cd = torch.float64 if q.dtype == torch.float64 else torch.float32
            capture.append(torch.softmax((q.to(cd) @ K_.to(cd).transpose(-1, -2)) / 8.0 + bias_fn.to(cd), -1).detach())
        h = h + blk.attn.resid_dropout(blk.attn.c_proj(o.transpose(1, 2).reshape(B, Ld, 768)))
        h = h + blk.mlp(blk.ln_2(h))
    return m.lm_head(tr.ln_f(h))


def forward(m, prj, D, b, a, probe=None, m2=None, capture=None):
    """the student path -> (loss, parts dict, extras). probe (gradflow only): {"enc_kv": True} makes the decoder call's encoder-position past K/V
    leaf tensors (and the latent rows' K/V) so their grads can be read. m2 (the trainer with --m2_stage only): the staged-M2 value m -> the
    decoder call is soft_decoder (capture: its per-layer attention probabilities, gates only); None = the HF decoder call (unchanged)."""
    dev = b["enc"].device; B = b["enc"].shape[0]; compat = a.codi_compat; K = 0 if a.direct else b["K"]
    am_enc = b["em"]                                                      # codi_port masks the encoder call too
    o = m(input_ids=b["enc"], attention_mask=am_enc, use_cache=True, output_hidden_states=True); pkv = o.past_key_values
    zs = []; z = prj(o.hidden_states[-1][:, -1:, :]); am = b["em"]
    for j in range(K):
        zs.append(z); am = torch.cat([am, _ones(B, 1, dev)], 1)
        o = m(inputs_embeds=z, attention_mask=None if compat else am, use_cache=True, output_hidden_states=True, past_key_values=pkv); pkv = o.past_key_values
        z = prj(o.hidden_states[-1][:, -1:, :])
    dm = am.clone()
    if a.bottleneck and not a.direct:
        dm[:, :b["em"].shape[1]] = dm[:, :b["em"].shape[1]] * (1 - b["qm"])   # M2: question content + <bot> keys invisible to every decoder query
    ex = {}
    if probe:
        pkv, ex = _leaf_cache(pkv, b["em"].shape[1])
    if m2 is not None and a.bottleneck and not a.direct:                 # §223 R3 (--m2_stage): the staged mask through soft_decoder
        qf = torch.zeros_like(am); qf[:, :b["em"].shape[1]] = b["qm"]
        logits = soft_decoder(m, pkv, m.transformer.wte(b["dec"]), am, qf, float(m2), capture)
    else:
        dmf = torch.cat([dm, _ones(B, b["dec"].shape[1], dev)], 1)
        so = m(inputs_embeds=m.transformer.wte(b["dec"]), attention_mask=None if (compat and not a.bottleneck) else dmf, use_cache=True, past_key_values=pkv)
        logits = so.logits
    lg = logits[:, :-1]; tg = b["dl"][:, 1:]; lgc = lg if compat else lg.float()                 # compat: codi_port's CE dtype (plain-path smoke)
    ce = F.cross_entropy(lgc.reshape(-1, lg.size(-1)), tg.reshape(-1), ignore_index=-100)
    nl = lg.gather(1, b["npos"][:, None, None].expand(-1, 1, lg.size(-1))).squeeze(1).float(); nt = b["dec"].gather(1, (b["npos"] + 1)[:, None]).squeeze(1)
    nce = F.cross_entropy(nl, nt); parts = {"ans_ce": float(ce.detach()), "num_ce": float(nce.detach()), "num_acc": float((nl.argmax(-1) == nt).float().mean())}
    if "vpos" in b:                                                         # §227: the step-value CE / acc (teacher forced; the CurGate signal)
        vl = lg.gather(1, b["vpos"][:, :, None].expand(-1, -1, lg.size(-1))).float(); vt = b["dec"].gather(1, b["vpos"] + 1); vm_ = b["vmask"]
        vce = F.cross_entropy(vl[vm_], vt[vm_]); parts.update(step_value_ce=float(vce.detach()), step_value_acc=float((vl[vm_].argmax(-1) == vt[vm_]).float().mean()))
    if "wrapf" in b:                                                        # §226: the wrap split as counts (window means of counts -> pooled accuracy)
        ok_ = nl.argmax(-1) == nt; w_ = b["wrapf"]
        parts.update(num_ok_wrap=float((ok_ & w_).sum()), n_wrap=float(w_.sum()), num_ok_nowrap=float((ok_ & ~w_).sum()), n_nowrap=float((~w_).sum()))
    loss = ce
    if a.recode and not a.direct:
        rce, rv, nv = recode_ce(m, D, zs, b, dev); loss = loss + a.lam * rce; ex["rce"] = rce
        parts.update({"recode_ce": float(rce.detach()), "recode_val_ce": rv, "recode_val_n": nv})
    ex.update({"zs": zs, "nce": nce, "ce": ce, "logits": logits}); return loss, parts, ex


def recode_ce(m, D, zs, b, dev):
    """mean over latents j of the per-token mean CE of D(z_j) -> tok(R_j) + EOS (teacher forced). -> (ce, value-token CE, n value tokens)."""
    W = m.transformer.wte.weight; tot = []; vs, vn = 0.0, 0
    for j, z in enumerate(zs):
        tg = b["R"][j]; L = max(len(r) for r in tg); N = len(tg)
        ids = torch.full((N, L), PAD, device=dev); km = torch.zeros(N, L, dtype=torch.long, device=dev); lab = torch.full((N, L), -100, device=dev)
        for i, r in enumerate(tg):
            ids[i, :len(r)] = torch.tensor(r, device=dev); km[i, :len(r)] = 1; lab[i, :len(r)] = torch.tensor(r, device=dev)
        x = torch.cat([z.float(), m.transformer.wte(ids[:, :-1]).float()], 1); h = D(x, km)       # position p predicts token p (z_j predicts the first)
        lg = F.linear(h, W); ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), lab.reshape(-1), ignore_index=-100); tot.append(ce)   # bf16 logits under autocast
        with torch.no_grad():
            isv = torch.isin(lab, torch.tensor(sorted(VALUE_TOKS), device=dev)) & (lab != -100)
            if isv.any():
                vce = F.cross_entropy(lg[isv].float(), lab[isv], reduction="sum"); vs += float(vce); vn += int(isv.sum())
    return torch.stack(tot).mean(), (vs / vn if vn else None), vn


def _leaf_cache(pkv, n_enc):
    """gradflow probe: rebuild the past cache from leaf K/V tensors (encoder positions and latent rows separately) -> (cache, {enc_k, enc_v, lat_k, lat_v})."""
    from transformers import DynamicCache
    layers = [(l.keys, l.values) for l in pkv.layers] if hasattr(pkv, "layers") else (list(zip(pkv.key_cache, pkv.value_cache)) if hasattr(pkv, "key_cache") else list(pkv))
    ek, ev, lk, lv = [], [], [], []; new = []
    for k, v in layers:
        a_k = k[:, :, :n_enc].detach().clone().requires_grad_(True); a_v = v[:, :, :n_enc].detach().clone().requires_grad_(True)
        b_k = k[:, :, n_enc:].detach().clone().requires_grad_(True); b_v = v[:, :, n_enc:].detach().clone().requires_grad_(True)
        ek.append(a_k); ev.append(a_v); lk.append(b_k); lv.append(b_v); new.append((torch.cat([a_k, b_k], 2), torch.cat([a_v, b_v], 2)))
    c = DynamicCache()
    for i, (k, v) in enumerate(new):
        c.update(k, v, i)
    return c, {"enc_k": ek, "enc_v": ev, "lat_k": lk, "lat_v": lv}


VALUE_TOKS = set()


def value_tokens(enc):
    return {enc.encode(" %d" % n)[0] for n in range(100)} | {enc.encode("%d" % n)[0] for n in range(100) if len(enc.encode("%d" % n)) == 1}


def term_grad_norms(SH, ex, a):
    """Audit 10-21 (logged only): the L2 norms of d(answer CE)/d(shared) and d(lam x recode CE)/d(shared), and their cosine, on one micro-batch."""
    ga = torch.autograd.grad(ex["ce"], SH, retain_graph=True, allow_unused=True)
    va = torch.cat([g.float().flatten() if g is not None else torch.zeros(p_.numel(), device=p_.device) for g, p_ in zip(ga, SH)])
    out = {"gn_ans_shared": float(va.norm())}
    if a.recode and not a.direct and "rce" in ex:
        gr = torch.autograd.grad(a.lam * ex["rce"], SH, retain_graph=True, allow_unused=True)
        vr = torch.cat([g.float().flatten() if g is not None else torch.zeros(p_.numel(), device=p_.device) for g, p_ in zip(gr, SH)])
        out.update({"gn_rec_shared": float(vr.norm()), "cos_ans_rec": float(F.cosine_similarity(va, vr, dim=0))})
    return out


ANS_RE = re.compile(r" The answer is (\d+)\.")
STEP_RE = re.compile(r"\b(\w+) has (\d+) ")


@torch.no_grad()
def greedy_decode(m, its, dev, max_new):
    """§227: batched greedy decoding after [question | <bot>] (the training encoder call) from <eot>, max_new tokens, stop at EOS. -> per item
    (generated token list without EOS, terminated, overflow)."""
    b = collate(its, 0, dev); B = len(its); Tq = b["enc"].shape[1]; npos_max = m.config.n_positions
    o = m(input_ids=b["enc"], attention_mask=b["em"], use_cache=True); pkv = o.past_key_values; am = b["em"]
    cur = torch.full((B, 1), EOT, device=dev); gen = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device=dev)
    steps = min(max_new, npos_max - Tq - 1)
    for _ in range(steps):
        am = torch.cat([am, _ones(B, 1, dev)], 1); o = m(inputs_embeds=m.transformer.wte(cur), attention_mask=am, use_cache=True, past_key_values=pkv); pkv = o.past_key_values
        nxt = o.logits[:, -1].float().argmax(-1)
        for i in range(B):
            if not bool(done[i]):
                if int(nxt[i]) == EOS:
                    done[i] = True
                else:
                    gen[i].append(int(nxt[i]))
        if bool(done.all()):
            break
        cur = nxt[:, None]
    return [(gen[i], bool(done[i]), steps < max_new) for i in range(B)]


def decode_eval(m, prj, ED, enc, dev, d_max, step):
    """§227 primary reading: greedy free decoding on the held-out set for d 1..min(d_max + 1, 8) -> one record (acc, terminated, parsed, per-step value
    acc by step index, wrap split). Unparseable / non-terminating / context overflow = wrong."""
    was = m.training; m.eval(); prj.eval(); rec = {"step": step, "d_max": d_max, "by_d": {}}
    for d in range(1, min(d_max + 1, 8) + 1):
        its = ED[d]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda")):
            res = greedy_decode(m, its, dev, 24 * d + 16)
        ok, term, parsed, ps, wr = [], [], [], np.zeros(d), {"w": [0, 0], "n": [0, 0]}
        for it, (g, t_, ovf) in zip(its, res):
            txt = enc.decode(g); ma = ANS_RE.search(txt); val = int(ma.group(1)) if ma else None
            c = bool(t_ and not ovf and val is not None and val == it["ans"]); ok.append(c); term.append(t_); parsed.append(val is not None)
            sv = [int(v) for _, v in STEP_RE.findall(txt[:ma.start()] if ma else txt)]
            for k in range(d):
                ps[k] += int(k < len(sv) and sv[k] == it["vals"][k])
            key = "w" if it.get("wrap") else "n"; wr[key][0] += int(c); wr[key][1] += 1
        n = len(its)
        rec["by_d"][str(d)] = {"n": n, "acc": float(np.mean(ok)), "terminated": float(np.mean(term)), "parsed": float(np.mean(parsed)), "per_step_value_acc": (ps / n).tolist(),
                               "acc_wrap": wr["w"][0] / wr["w"][1] if wr["w"][1] else None, "n_wrap": wr["w"][1], "acc_nowrap": wr["n"][0] / wr["n"][1] if wr["n"][1] else None}
    if was:
        m.train(); prj.train()
    return rec


def build_full(dev, seed=1):
    """§226 P0d (--full_ft): codi_port.build's base WITHOUT the bf16 cast and WITHOUT LoRA (fp32 master weights, every GPT-2 parameter trainable):
    the same pretrained load, the same seed + 2090 draws for the 3 new token embeddings (transformers' mean resizing), the tied lm_head, and prj after
    torch.manual_seed(seed + 2091) as in codi_port.build (identical init; LoRA's init there uses its own generator, not the global rng)."""
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(); m = GPT2LMHeadModel(cfg)
    sd = torch.load(os.path.join(CP.ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if not k.endswith(("attn.bias", "masked_bias"))}
    miss, unexp = m.transformer.load_state_dict(sd, strict=False); assert not unexp and all(k.endswith(("attn.bias", "masked_bias")) for k in miss), (miss, unexp)
    E = m.transformer.wte.weight.data; mu = E.mean(0); Ec = E - mu; cov = Ec.T @ Ec / E.shape[0]
    gen = torch.Generator().manual_seed(seed + 2090); L_ = torch.linalg.cholesky(1e-9 * cov.double() + 1e-30 * torch.eye(768, dtype=torch.double))
    new = (mu.double() + torch.randn(3, 768, generator=gen, dtype=torch.double) @ L_.T).float()
    W = torch.cat([E, new]); m.transformer.wte = nn.Embedding(CP.V0 + 3, 768); m.transformer.wte.weight.data.copy_(W); m.config.vocab_size = CP.V0 + 3
    m.lm_head = nn.Linear(768, CP.V0 + 3, bias=False); m.tie_weights(); assert m.lm_head.weight.data_ptr() == m.transformer.wte.weight.data_ptr()
    for p_ in m.parameters():
        p_.requires_grad_(True)
    torch.manual_seed(seed + 2091)
    prj = nn.Sequential(nn.Dropout(0.0), nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 768)); prj.add_module("ln", nn.LayerNorm(768))
    return m.to(dev), prj.to(dev)


def build_all(a, dev):
    m, prj = build_full(dev, a.seed) if a.full_ft else CP.build(dev, a.seed); D = DHead(m).to(dev) if (a.recode and not a.direct) else None
    TP = CP.trainable(m, prj) + ([("D." + n, p_) for n, p_ in D.named_parameters()] if D is not None else []); return m, prj, D, TP


def make_opt(a, TP, total):
    nodecay = lambda n: n.endswith("bias") or n.startswith("prj.ln") or ".ln_" in n or n.startswith("D.ln")  # noqa: E731  (HF Trainer: no decay on biases / LayerNorm)
    dec = [p_ for n, p_ in TP if not nodecay(n)]; nod = [p_ for n, p_ in TP if nodecay(n)]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": a.wd}, {"params": nod, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.999), eps=1e-8); warm = math.ceil(a.warmup * total)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(1, warm) if s < warm else max(0.0, 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))))
    return opt, sch, warm


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--eff_bs", type=int, default=64); ap.add_argument("--micro", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--wd", type=float, default=0.1); ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--clip", type=float, default=2.0); ap.add_argument("--k_extra", type=int, default=2); ap.add_argument("--k_cap", type=int, default=10)
    ap.add_argument("--recode", action="store_true"); ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--bottleneck", action="store_true")
    ap.add_argument("--direct", action="store_true"); ap.add_argument("--codi_compat", action="store_true", help="smokes only (plain-path comparison with codi_port)")
    ap.add_argument("--d_stages", default="1,2,3,4,5,6,7,8"); ap.add_argument("--cg_thr", type=float, default=0.3); ap.add_argument("--cg_need", type=int, default=2)
    ap.add_argument("--cg_cap", type=int, default=2500); ap.add_argument("--cg_bin", type=int, default=1000)
    ap.add_argument("--stream_seed", type=int, default=None, help="default: 22330 (nested) / ow_words.SEED_TRAIN 22930 (ow)")
    ap.add_argument("--task", default="nested", choices=["nested", "ow"], help="§226: ow = boxes/ow_words.py items")
    ap.add_argument("--order_mode", default="forward", choices=["forward", "topo"], help="§226 (ow): premise layout")
    ap.add_argument("--dag", default="on", choices=["on", "off"], help="§226 (ow): off = a strict previous-state chain (no add_s, no forced mul_s)")
    ap.add_argument("--full_ft", action="store_true", help="§226 P0d: no LoRA, every GPT-2 parameter trainable (fp32 master weights)")
    ap.add_argument("--text_cot", action="store_true", help="§227: serial text CoT target (needs --direct)"); ap.add_argument("--decode_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=100); ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--args_only", action="store_true"); ap.add_argument("--args_ref", default=""); ap.add_argument("--time_steps", type=int, default=0)
    ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--auto_micro", action="store_true"); ap.add_argument("--mem_budget", type=float, default=21.0)
    ap.add_argument("--stop_at", type=int, default=0, help="resume test only: exit 9 after this step")
    ap.add_argument("--m2_stage", action="store_true", help="Note 223-8 R3: staged M2 (needs --bottleneck)"); ap.add_argument("--m2_ramp", type=int, default=2000)
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True); dev = a.device; torch.manual_seed(a.seed)
    if a.stream_seed is None:                                               # §226: per-task default (nested unchanged: 22330)
        a.stream_seed = OW.SEED_TRAIN if a.task == "ow" else SEED_TRAIN
    assert a.direct or not a.text_cot, "--text_cot needs --direct"; assert a.task == "ow" or not a.text_cot, "--text_cot needs --task ow (R_step)"; assert a.bottleneck or not a.m2_stage, "--m2_stage needs --bottleneck"; assert not (a.direct and (a.recode or a.bottleneck)), "DIRECT has no latent flags"; assert a.recode == (a.lam > 0), "--recode needs --lam > 0 (and only then)"
    import tiktoken
    enc = tiktoken.get_encoding("gpt2"); VALUE_TOKS.update(value_tokens(enc))
    if a.task == "ow":                                                      # §226: OW test set / pair candidates (same layout / dag as the stream)
        okw = {"order_mode": a.order_mode, "dag": a.dag == "on"}; TE = OW.test_set(**okw); PC = OW.pair_candidates(**okw); D_ = OW
    else:
        TE = NW.test_set(); PC = NW.pair_candidates(); D_ = NW
    excl = {it["sig"] for d, its in TE.items() for it in its}
    EDraw = None
    if a.text_cot:                                                          # §227: the sha-pinned held-out decode set (train frames 0-5), excluded from the stream
        es = (OW.Stream(22950, excl, frames=NW.FRAME_TRAIN, order_mode=a.order_mode, dag=a.dag == "on") if a.task == "ow" else NW.Stream(22950, excl, frames=NW.FRAME_TRAIN))
        EDraw = {d: [es.draw(d) for _ in range(200)] for d in range(1, 9)}; excl |= {it["sig"] for d in EDraw for it in EDraw[d]}
    excl |= {x["sig"] for d, ps in PC.items() for pr in ps for x in pr}  # the pair candidates (recipient + donor; a superset of the eval's pairs)
    info = {**vars(a), "cell": "DIRECT" if a.direct else ("PS" if a.recode and a.bottleneck else "M1-only" if a.recode else "M2-only" if a.bottleneck else "PLAIN"),
            "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
            "train_driver": os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader").read().strip().split("\n")[0] if dev.startswith("cuda") else None,
            "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__, "python_version": sys.version.split()[0],
            "cuda_version": torch.version.cuda, "generator_sha256": sha_file(os.path.join(ROOT, "boxes", "nested_words.py")),
            "module_sha256": sha_file(os.path.abspath(__file__)), "codi_port_sha256": sha_file(os.path.join(ROOT, "boxes", "codi_port.py")),
            "test_items_sha": D_.set_sha([it for d in sorted(TE) for it in TE[d]]), "pair_candidates_sha": D_.set_sha([x for d in sorted(PC) for pr in PC[d] for x in pr]), "excl_n": len(excl),
            **({"ow_words_sha256": sha_file(os.path.join(ROOT, "boxes", "ow_words.py"))} if a.task == "ow" else {}),
            **({"cg_signal": "step_value_ce", "text_eval_sha": D_.set_sha([it for d in sorted(EDraw) for it in EDraw[d]]), "text_eval_n_per_d": 200} if a.text_cot else {})}
    info["accumulation"] = a.eff_bs // a.micro
    if a.args_only or a.args_ref:
        ref = json.load(open(a.args_ref)) if a.args_ref else None
        if ref is not None:
            dif = {k: [ref.get(k), info.get(k)] for k in sorted(set(ref) | set(info)) if k not in META and ref.get(k) != info.get(k)}
            json.dump({"ref": a.args_ref, "diff": dif}, open(os.path.join(a.out, "args_diff.json"), "w", newline="\n"), indent=1); print("ARGS-DIFF", json.dumps(dif), flush=True)
    json.dump(info, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    if a.args_only:
        print("ARGS_ONLY", a.out, flush=True); return
    m, prj, D, TP = build_all(a, dev); rng = np.random.default_rng(a.seed + 22331)
    stream = OW.Stream(a.stream_seed, excl, order_mode=a.order_mode, dag=a.dag == "on") if a.task == "ow" else NW.Stream(a.stream_seed, excl)
    cg = CurGate([int(x) for x in a.d_stages.split(",")], a.cg_thr, a.cg_need, a.cg_cap, a.cg_bin)

    def draw_step():
        d = int(rng.integers(1, cg.n_max() + 1)); K = 0 if a.direct else min(a.k_cap, d + int(rng.integers(0, a.k_extra + 1)))
        return d, K, [tok_item(stream.draw(d), enc, a.text_cot) for _ in range(a.eff_bs)]
    if a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.resume):
        deep = [tok_item(stream.draw(8), enc, a.text_cot) for _ in range(256)]; deep = sorted(deep, key=lambda it: -(len(it["q"]) + sum(len(r) for r in it["R"])))[:1]
        mc = []
        while True:
            torch.cuda.reset_peak_memory_stats(dev)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l_, _, _ = forward(m, prj, D, collate(deep * a.micro, 0 if a.direct else a.k_cap, dev), a, m2=0.5 if a.m2_stage else None)
                l_.backward(); pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30 + 3 * sum(p_.numel() * 4 for _, p_ in TP) / 2 ** 30
            except torch.OutOfMemoryError:
                pk = float("inf")
            for _, p_ in TP:
                p_.grad = None
            torch.cuda.empty_cache(); mc.append({"micro": a.micro, "peak_gib": pk}); print("MEM", a.micro, pk, flush=True)
            if pk <= a.mem_budget or a.mem_check or a.micro <= 4:
                break
            a.micro //= 2
        json.dump({"checks": mc, "micro": a.micro, "accumulation": a.eff_bs // a.micro, "shape": {"d": 8, "K": a.k_cap, "q_tokens": len(deep[0]["q"]), "R_tokens": [len(r) for r in deep[0]["R"]]},
                   "gpu": info["train_gpu"]}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return
        info["micro"] = a.micro; info["accumulation"] = a.eff_bs // a.micro; json.dump(info, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    acc_steps = a.eff_bs // a.micro; assert a.eff_bs % a.micro == 0
    opt, sch, warm = make_opt(a, TP, a.steps); step = 0; t0 = time.time(); entry = None; ck = os.path.join(a.out, "ckpt_last.pt"); m2_start = None
    SH = [p_ for n, p_ in TP if ".lora_" in n or n.startswith("prj.")]      # the params both loss terms train
    if a.resume and os.path.exists(ck):
        c_ = torch.load(ck, map_location=dev, weights_only=False)
        for n, p_ in TP:
            p_.data.copy_(c_["tp"][n])
        opt.load_state_dict(c_["opt"]); sch.load_state_dict(c_["sched"]); step = c_["step"]; cg.load(c_["cg"]); entry = c_["entry"]
        rng.bit_generator.state = c_["rng"]; stream.rng.bit_generator.state = c_["stream_rng"]; stream.stats = c_["stream_stats"]
        torch.set_rng_state(c_["torch_rng"].cpu()); torch.cuda.set_rng_state(c_["cuda_rng"].cpu(), dev) if dev.startswith("cuda") else None
        t0 = time.time() - c_["elapsed"]; m2_start = c_.get("m2_start"); print("RESUME from step", step, flush=True)
    ED = {d: [tok_item(it, enc, True) for it in EDraw[d]] for d in EDraw} if a.text_cot else None
    log = open(os.path.join(a.out, "log.jsonl"), "a", newline="\n"); hist = []; m.train(); prj.train(); tl = []
    if D is not None:
        D.train()
    total = a.time_steps or a.steps
    while step < total:
        d, K, its = draw_step(); opt.zero_grad(set_to_none=True); P = []; t1 = time.time(); mv = None
        if a.m2_stage:                                                      # R3: m for this step (step + 1) from the stage the step trains at
            if m2_start is None and cg.stage >= 2:
                m2_start = step + 1
            mv = 0.0 if m2_start is None else min(1.0, (step + 1 - m2_start) / a.m2_ramp)
        for q0 in range(0, a.eff_bs, a.micro):
            b = collate(its[q0:q0 + a.micro], K, dev, a.codi_compat)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                l_, pt, ex = forward(m, prj, D, b, a, m2=mv)
            if q0 == 0 and (step + 1) % a.log_every == 0:                    # Audit 10-21: per-term grad norms on the SHARED params (LoRA + prj), logged only:
                pt.update(term_grad_norms(SH, ex, a))                       # autograd.grad reads gradients without touching .grad, so the update is unchanged
            (l_ / acc_steps).backward(); P.append((float(l_.detach()), pt))
        gn = torch.nn.utils.clip_grad_norm_([p_ for _, p_ in TP], a.clip); opt.step(); sch.step(); step += 1; tl.append(time.time() - t1)
        num = float(np.mean([p["num_ce"] for _, p in P])); hist.append((float(np.mean([x for x, _ in P])), {k: float(np.mean([p[k] for _, p in P if p.get(k) is not None]))
                                                                                                      for k in P[0][1] if k != "recode_val_n" and P[0][1][k] is not None}, float(gn), d, K))
        sig = float(np.mean([p["step_value_ce"] for _, p in P])) if a.text_cot else num   # §227: CurGate on step_value_ce, never num_ce
        adv = cg.update(step, sig)
        if adv:
            adv["entry_step_so_far"] = entry; open(os.path.join(a.out, "stages.jsonl"), "a", newline="\n").write(json.dumps(adv) + "\n"); print("STAGE", json.dumps(adv), flush=True)
        if step % a.log_every == 0 or step == total:
            pm = {k: float(np.mean([h[1][k] for h in hist if k in h[1]])) for k in hist[-1][1]}
            if entry is None and pm.get("step_value_ce" if a.text_cot else "num_ce", 9) < 0.5:
                entry = step
            rec = {"step": step, "loss": float(np.mean([h[0] for h in hist])), "sec": round(time.time() - t0), **pm, "gn_preclip": float(np.mean([h[2] for h in hist])),
                   "lr": sch.get_last_lr()[0], "d_max": cg.n_max(), "stage": cg.stage, "d_hist": np.bincount([h[3] for h in hist], minlength=13).tolist(),
                   "K_mean": float(np.mean([h[4] for h in hist])), "entry_step": entry, "stream_restarts": stream.stats.get("restarts", 0)}
            if a.m2_stage:
                rec.update(m2_m=mv, m2_start=m2_start)
            hist = []; log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if a.text_cot and not a.time_steps and (step % a.decode_every == 0 or (a.stop_at and step >= a.stop_at) or step == total):   # §227 free decoding
            de = decode_eval(m, prj, ED, enc, dev, cg.n_max(), step); open(os.path.join(a.out, "decode_evals.jsonl"), "a", newline="\n").write(json.dumps(de) + "\n")
            print("DECODE", step, "done (ds %s)" % sorted(de["by_d"]), flush=True)       # no values printed (read from decode_evals.jsonl by Theory)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < total and not a.time_steps:
            torch.save({"tp": {n: p_.detach() for n, p_ in TP}, "opt": opt.state_dict(), "sched": sch.state_dict(), "step": step, "cg": cg.state(), "entry": entry,
                        "rng": rng.bit_generator.state, "stream_rng": stream.rng.bit_generator.state, "stream_stats": dict(stream.stats), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else torch.zeros(1), "elapsed": time.time() - t0,
                        **({"m2_start": m2_start} if a.m2_stage else {})}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            json.dump({"step": step, "stage": cg.stage, "entry": entry}, open(os.path.join(a.out, "ckpt_fingerprint.json"), "w", newline="\n"))
        if a.stop_at and step >= a.stop_at:
            print("STOP_AT", step, flush=True); sys.exit(9)
    if a.time_steps:
        w = tl[min(10, len(tl) // 4):]
        json.dump({"steps_timed": step, "sec_per_step": float(np.mean(w)), "est_hours": float(np.mean(w)) * a.steps / 3600, "micro": a.micro, "accumulation": acc_steps,
                   "gpu": info["train_gpu"], "cell": info["cell"], "d_max_reached": cg.n_max()}, open(os.path.join(a.out, "timing.json"), "w", newline="\n"), indent=1)
        print("TIMING", float(np.mean(w)), flush=True); return
    torch.save({n: p_.detach().cpu() for n, p_ in TP}, os.path.join(a.out, "trainable.pt"))
    json.dump({"steps": step, "train_sec": round(time.time() - t0), "entry_step": entry, "stages": cg.hist, "final_d_max": cg.n_max(), "trainable_sha256": sha_file(os.path.join(a.out, "trainable.pt")),
               "stream_restarts": stream.stats.get("restarts", 0), "stream_items": stream.stats.get("items", 0), **({"m2_start": m2_start} if a.m2_stage else {})},
              open(os.path.join(a.out, "train_done.json"), "w", newline="\n"), indent=1)
    if os.path.exists(ck):
        os.remove(ck)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
