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
  * outputs: args.json (--args_only stops there; --args_ref -> args_diff.json + an ARGS-DIFF line), log.jsonl every --log_every (step / loss / sec
    adjacent for cost_guard), stages.json (every advance), a full-state ckpt_last.pt + ckpt_fingerprint.json every --ckpt_every (LoRA / prj / D,
    optimiser, scheduler, CurGate, the item stream and the step-draw generator states, torch / cuda RNG, elapsed), trainable.pt + train_done.json
    at the end. --time_steps N -> timing.json; --mem_check -> mem_check.json (the deepest shape: d 8, K 10, the longest R). No eval here (a separate
    eval module).
    python -m boxes.codi_ps --out boxes/runs_fsq/s223_ps_s1 --recode --lam 1 --bottleneck [--device cuda:0 ...]"""
import argparse
import copy
import hashlib
import json
import math
import os
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


def tok_item(it, enc):
    """-> {q (BOS + question tokens), dec, R (list of token lists, R_1..R_d each + EOS), d, ans}."""
    q = [BOS] + enc.encode(it["text"]); dec = [EOT] + enc.encode("The answer is: %d" % it["answer"]) + [EOS]
    assert dec[1:5] == ANS_PROMPT and len(enc.encode(" %d" % it["answer"])) == 1, (dec, it["answer"])
    return {"q": q, "dec": dec, "R": [enc.encode(r) + [EOS] for r in it["R"][1:]], "d": it["d"], "ans": it["answer"]}


def collate(its, K, dev, compat=False):
    """-> dict of tensors for one bucket (all items share d; K latents)."""
    B = len(its); qs = [it["q"][1:] if compat else it["q"] for it in its]; Tq = max(len(q) + 1 for q in qs); Td = max(len(it["dec"]) for it in its)
    enc = torch.full((B, Tq), PAD); em = torch.zeros(B, Tq, dtype=torch.long); qm = torch.zeros(B, Tq, dtype=torch.long)   # qm: question content + <bot>
    dec = torch.full((B, Td), PAD); dl = torch.full((B, Td), -100); npos = []
    for b, (it, q) in enumerate(zip(its, qs)):
        s = q + [BOT]; enc[b, Tq - len(s):] = torch.tensor(s); em[b, Tq - len(s):] = 1
        qm[b, Tq - len(s) + (0 if compat else 1):] = 1                    # every position after BOS (question content and <bot>)
        dec[b, :len(it["dec"])] = torch.tensor(it["dec"]); dl[b, :len(it["dec"])] = torch.tensor(it["dec"]); npos.append(CP.ans_pos(it["dec"]) - 1)
    d = its[0]["d"]; R = []
    for j in range(1, K + 1):                                               # z_j's target: R_j (j <= d) or the HOLD target R_d (j > d)
        R.append([it["R"][min(j, d) - 1] for it in its])
    t = lambda x: x.to(dev)  # noqa: E731
    return {"enc": t(enc), "em": t(em), "qm": t(qm), "dec": t(dec), "dl": t(dl), "npos": t(torch.tensor(npos)), "R": R, "K": K, "d": d,
            "holds": [j for j in range(1, K + 1) if j > d]}


def _ones(B, n, dev):
    return torch.ones(B, n, dtype=torch.long, device=dev)


def forward(m, prj, D, b, a, probe=None):
    """the student path -> (loss, parts dict, extras). probe (gradflow only): {"enc_kv": True} makes the decoder call's encoder-position past K/V
    leaf tensors (and the latent rows' K/V) so their grads can be read."""
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
    dmf = torch.cat([dm, _ones(B, b["dec"].shape[1], dev)], 1)
    so = m(inputs_embeds=m.transformer.wte(b["dec"]), attention_mask=None if (compat and not a.bottleneck) else dmf, use_cache=True, past_key_values=pkv)
    lg = so.logits[:, :-1]; tg = b["dl"][:, 1:]; lgc = lg if compat else lg.float()                 # compat: codi_port's CE dtype (plain-path smoke)
    ce = F.cross_entropy(lgc.reshape(-1, lg.size(-1)), tg.reshape(-1), ignore_index=-100)
    nl = lg.gather(1, b["npos"][:, None, None].expand(-1, 1, lg.size(-1))).squeeze(1).float(); nt = b["dec"].gather(1, (b["npos"] + 1)[:, None]).squeeze(1)
    nce = F.cross_entropy(nl, nt); parts = {"ans_ce": float(ce.detach()), "num_ce": float(nce.detach()), "num_acc": float((nl.argmax(-1) == nt).float().mean())}
    loss = ce
    if a.recode and not a.direct:
        rce, rv, nv = recode_ce(m, D, zs, b, dev); loss = loss + a.lam * rce
        parts.update({"recode_ce": float(rce.detach()), "recode_val_ce": rv, "recode_val_n": nv})
    ex.update({"zs": zs, "nce": nce, "ce": ce}); return loss, parts, ex


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


def build_all(a, dev):
    m, prj = CP.build(dev, a.seed); D = DHead(m).to(dev) if (a.recode and not a.direct) else None
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
    ap.add_argument("--cg_cap", type=int, default=2500); ap.add_argument("--cg_bin", type=int, default=1000); ap.add_argument("--stream_seed", type=int, default=SEED_TRAIN)
    ap.add_argument("--log_every", type=int, default=100); ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--args_only", action="store_true"); ap.add_argument("--args_ref", default=""); ap.add_argument("--time_steps", type=int, default=0)
    ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--auto_micro", action="store_true"); ap.add_argument("--mem_budget", type=float, default=21.0)
    ap.add_argument("--stop_at", type=int, default=0, help="resume test only: exit 9 after this step")
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True); dev = a.device; torch.manual_seed(a.seed)
    assert not (a.direct and (a.recode or a.bottleneck)), "DIRECT has no latent flags"; assert a.recode == (a.lam > 0), "--recode needs --lam > 0 (and only then)"
    import tiktoken
    enc = tiktoken.get_encoding("gpt2"); VALUE_TOKS.update(value_tokens(enc))
    TE = NW.test_set(); PC = NW.pair_candidates(); excl = {it["sig"] for d, its in TE.items() for it in its}
    excl |= {x["sig"] for d, ps in PC.items() for pr in ps for x in pr}  # the pair candidates (recipient + donor; a superset of the eval's pairs)
    info = {**vars(a), "cell": "DIRECT" if a.direct else ("PS" if a.recode and a.bottleneck else "M1-only" if a.recode else "M2-only" if a.bottleneck else "PLAIN"),
            "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
            "train_driver": os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader").read().strip().split("\n")[0] if dev.startswith("cuda") else None,
            "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__, "python_version": sys.version.split()[0],
            "cuda_version": torch.version.cuda, "generator_sha256": sha_file(os.path.join(ROOT, "boxes", "nested_words.py")),
            "module_sha256": sha_file(os.path.abspath(__file__)), "codi_port_sha256": sha_file(os.path.join(ROOT, "boxes", "codi_port.py")),
            "test_items_sha": NW.set_sha([it for d in sorted(TE) for it in TE[d]]), "pair_candidates_sha": NW.set_sha([x for d in sorted(PC) for pr in PC[d] for x in pr]), "excl_n": len(excl)}
    info["accumulation"] = a.eff_bs // a.micro
    if a.args_only or a.args_ref:
        ref = json.load(open(a.args_ref)) if a.args_ref else None
        if ref is not None:
            dif = {k: [ref.get(k), info.get(k)] for k in sorted(set(ref) | set(info)) if k not in META and ref.get(k) != info.get(k)}
            json.dump({"ref": a.args_ref, "diff": dif}, open(os.path.join(a.out, "args_diff.json"), "w", newline="\n"), indent=1); print("ARGS-DIFF", json.dumps(dif), flush=True)
    json.dump(info, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    if a.args_only:
        print("ARGS_ONLY", a.out, flush=True); return
    m, prj, D, TP = build_all(a, dev); stream = NW.Stream(a.stream_seed, excl); rng = np.random.default_rng(a.seed + 22331)
    cg = CurGate([int(x) for x in a.d_stages.split(",")], a.cg_thr, a.cg_need, a.cg_cap, a.cg_bin)

    def draw_step():
        d = int(rng.integers(1, cg.n_max() + 1)); K = 0 if a.direct else min(a.k_cap, d + int(rng.integers(0, a.k_extra + 1)))
        return d, K, [tok_item(stream.draw(d), enc) for _ in range(a.eff_bs)]
    if a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.resume):
        deep = [tok_item(stream.draw(8), enc) for _ in range(256)]; deep = sorted(deep, key=lambda it: -(len(it["q"]) + sum(len(r) for r in it["R"])))[:1]
        mc = []
        while True:
            torch.cuda.reset_peak_memory_stats(dev)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l_, _, _ = forward(m, prj, D, collate(deep * a.micro, 0 if a.direct else a.k_cap, dev), a)
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
    opt, sch, warm = make_opt(a, TP, a.steps); step = 0; t0 = time.time(); entry = None; ck = os.path.join(a.out, "ckpt_last.pt")
    if a.resume and os.path.exists(ck):
        c_ = torch.load(ck, map_location=dev, weights_only=False)
        for n, p_ in TP:
            p_.data.copy_(c_["tp"][n])
        opt.load_state_dict(c_["opt"]); sch.load_state_dict(c_["sched"]); step = c_["step"]; cg.load(c_["cg"]); entry = c_["entry"]
        rng.bit_generator.state = c_["rng"]; stream.rng.bit_generator.state = c_["stream_rng"]; stream.stats = c_["stream_stats"]
        torch.set_rng_state(c_["torch_rng"].cpu()); torch.cuda.set_rng_state(c_["cuda_rng"].cpu(), dev) if dev.startswith("cuda") else None
        t0 = time.time() - c_["elapsed"]; print("RESUME from step", step, flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a", newline="\n"); hist = []; m.train(); prj.train(); tl = []
    if D is not None:
        D.train()
    total = a.time_steps or a.steps
    while step < total:
        d, K, its = draw_step(); opt.zero_grad(set_to_none=True); P = []; t1 = time.time()
        for q0 in range(0, a.eff_bs, a.micro):
            b = collate(its[q0:q0 + a.micro], K, dev, a.codi_compat)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                l_, pt, _ = forward(m, prj, D, b, a)
            (l_ / acc_steps).backward(); P.append((float(l_.detach()), pt))
        gn = torch.nn.utils.clip_grad_norm_([p_ for _, p_ in TP], a.clip); opt.step(); sch.step(); step += 1; tl.append(time.time() - t1)
        num = float(np.mean([p["num_ce"] for _, p in P])); hist.append((float(np.mean([x for x, _ in P])), {k: float(np.mean([p[k] for _, p in P if p.get(k) is not None]))
                                                                                                      for k in P[0][1] if k != "recode_val_n" and P[0][1][k] is not None}, float(gn), d, K))
        adv = cg.update(step, num)
        if adv:
            adv["entry_step_so_far"] = entry; open(os.path.join(a.out, "stages.jsonl"), "a", newline="\n").write(json.dumps(adv) + "\n"); print("STAGE", json.dumps(adv), flush=True)
        if step % a.log_every == 0 or step == total:
            pm = {k: float(np.mean([h[1][k] for h in hist if k in h[1]])) for k in hist[-1][1]}
            if entry is None and pm.get("num_ce", 9) < 0.5:
                entry = step
            rec = {"step": step, "loss": float(np.mean([h[0] for h in hist])), "sec": round(time.time() - t0), **pm, "gn_preclip": float(np.mean([h[2] for h in hist])),
                   "lr": sch.get_last_lr()[0], "d_max": cg.n_max(), "stage": cg.stage, "d_hist": np.bincount([h[3] for h in hist], minlength=13).tolist(),
                   "K_mean": float(np.mean([h[4] for h in hist])), "entry_step": entry, "stream_restarts": stream.stats.get("restarts", 0)}
            hist = []; log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < total and not a.time_steps:
            torch.save({"tp": {n: p_.detach() for n, p_ in TP}, "opt": opt.state_dict(), "sched": sch.state_dict(), "step": step, "cg": cg.state(), "entry": entry,
                        "rng": rng.bit_generator.state, "stream_rng": stream.rng.bit_generator.state, "stream_stats": dict(stream.stats), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else torch.zeros(1), "elapsed": time.time() - t0}, ck + ".tmp")
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
               "stream_restarts": stream.stats.get("restarts", 0), "stream_items": stream.stats.get("items", 0)}, open(os.path.join(a.out, "train_done.json"), "w", newline="\n"), indent=1)
    if os.path.exists(ck):
        os.remove(ck)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
