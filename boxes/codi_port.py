"""§209 CODI-AT-OUR-BUDGET (registered; note 209-1): CODI's GPT-2 small recipe (github zhenyi4/codi: src/model.py CODI.forward, train.py, scripts/
train_gpt2_gsm8k-aug.sh) ported into our pipeline, trained on our GSM8K-Aug train split for --epochs (4.0) epochs counted by examples seen.
NEW module (imported by nothing else). Faithful to CODI's code paths (fix_attn_mask False, remove_eos True, use_prj, distill smooth_l1 / teacher std,
include_last_cot False, no decoder), with these explicit choices:
  * model: HF GPT2LMHeadModel loaded from prosqa/ckpt/gpt2_pretrained.bin (= HF gpt2), base weights in bf16 (CODI: torch_dtype bf16), vocab resized
    50257 -> 50260 (pad / <bot> / <eot>) with transformers' mean resizing (new rows = mean + N(0, 1e-9 cov) sample, seeded), lm_head tied, base frozen.
  * LoRA (peft 0.15 equivalent, own wrapper on the HF Conv1D modules c_attn / c_proj / c_fc of every block): r 128, alpha 32 (scale 0.25), dropout 0.1
    on the LoRA input, A kaiming_uniform(a = sqrt 5), B = 0, fp32 adapters; projection prj = Dropout(0) - Linear - GELU - Linear - LayerNorm (fp32).
  * per example: encoder = question + <bot> (left padded, key mask); 6 latents (prj of the last hidden state, fed as embeddings, past KV, no mask);
    decoder = <eot> + "The answer is: N" + EOS (no mask); teacher = question + CoT minus its last step + "The answer is: N" + EOS (right padded, no mask;
    one no-grad pass for the targets, one grad pass for the teacher CE). Loss = student CE + distill + gamma x teacher CE, distill = mean over the 13
    hidden states of smooth_l1(student[':'], sg(teacher[':'])) / std(teacher[':']). gamma = --gamma (2: "CODI recipe, gamma 2 (4-epoch prior)").
  * optimiser as HF Trainer: AdamW lr 3e-3, wd 0.1 except biases / LayerNorm, betas (0.9, 0.999), eps 1e-8; cosine with 3% linear warm-up; clip 2.0;
    effective batch 128 = micro x accumulation (loss / accumulation); bf16 autocast; train mode (GPT-2 dropout 0.1). Data: CODI's filters (token count
    <= 1000, answer starts with a digit), reshuffled every epoch (seed).
  * eval = theory/codi_probe.py's recipe (the bench harness, G1) on an fp32 copy with LoRA merged: Aug valid every --valid_every steps (valid_hist),
    the 1,319 test items at the end (acc, by_steps) and the 0-vs-6-latent ablation.
    python -m boxes.codi_port --out boxes/runs_fsq/gsm_codi_4ep_s1 [--epochs 4.0 --micro 64 --device cuda:0 ...]"""
import argparse
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
import boxes.gsm8k_data as G  # noqa: E402

V0, PAD, BOT, EOT = 50257, 50257, 50258, 50259; EOS = 50256; ANS_PROMPT = [464, 3280, 318, 25]   # "The answer is:"
LABEL = "CODI recipe, gamma 2 (4-epoch prior), not CODI's published gamma 1"


class LoRAConv1D(nn.Module):
    """peft LoRA on an HF Conv1D (weight (in, out)): base(x) + B(A(dropout(x))) * alpha / r, adapters fp32."""
    def __init__(self, base, r, alpha, p, gen):
        super().__init__(); self.base = base; fin, fout = base.weight.shape; self.s = alpha / r; self.drop = nn.Dropout(p)
        self.lora_A = nn.Linear(fin, r, bias=False); self.lora_B = nn.Linear(r, fout, bias=False)
        with torch.no_grad():
            bound = 1.0 / math.sqrt(fin)                                    # kaiming_uniform_(a = sqrt 5) on (r, fin): U(-1/sqrt(fin), 1/sqrt(fin))
            self.lora_A.weight.copy_((torch.rand(r, fin, generator=gen) * 2 - 1) * bound); self.lora_B.weight.zero_()

    def forward(self, x):
        res = self.base(x); xl = x.to(self.lora_A.weight.dtype)
        return res + (self.lora_B(self.lora_A(self.drop(xl))) * self.s).to(res.dtype)


def build(dev, seed=1, r=128, alpha=32, p=0.1):
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(); m = GPT2LMHeadModel(cfg)
    sd = torch.load(os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if not k.endswith(("attn.bias", "masked_bias"))}   # the causal-mask buffers (present in transformers 4.x only)
    miss, unexp = m.transformer.load_state_dict(sd, strict=False); assert not unexp and all(k.endswith(("attn.bias", "masked_bias")) for k in miss), (miss, unexp)
    E = m.transformer.wte.weight.data; mu = E.mean(0); Ec = E - mu; cov = Ec.T @ Ec / E.shape[0]           # transformers' mean resizing
    gen = torch.Generator().manual_seed(seed + 2090); L_ = torch.linalg.cholesky(1e-9 * cov.double() + 1e-30 * torch.eye(768, dtype=torch.double))
    new = (mu.double() + torch.randn(3, 768, generator=gen, dtype=torch.double) @ L_.T).float()
    W = torch.cat([E, new]); m.transformer.wte = nn.Embedding(V0 + 3, 768); m.transformer.wte.weight.data.copy_(W); m.config.vocab_size = V0 + 3
    m.lm_head = nn.Linear(768, V0 + 3, bias=False); m.tie_weights(); assert m.lm_head.weight.data_ptr() == m.transformer.wte.weight.data_ptr()
    m = m.to(torch.bfloat16)
    for p_ in m.parameters():
        p_.requires_grad_(False)
    for blk in m.transformer.h:
        blk.attn.c_attn = LoRAConv1D(blk.attn.c_attn, r, alpha, p, gen); blk.attn.c_proj = LoRAConv1D(blk.attn.c_proj, r, alpha, p, gen)
        blk.mlp.c_fc = LoRAConv1D(blk.mlp.c_fc, r, alpha, p, gen); blk.mlp.c_proj = LoRAConv1D(blk.mlp.c_proj, r, alpha, p, gen)
    torch.manual_seed(seed + 2091)
    prj = nn.Sequential(nn.Dropout(0.0), nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 768)); prj.add_module("ln", nn.LayerNorm(768))
    return m.to(dev), prj.to(dev)


def trainable(m, prj):
    return [(n, p_) for n, p_ in m.named_parameters() if p_.requires_grad] + [("prj." + n, p_) for n, p_ in prj.named_parameters()]


def load_items(split, enc):
    """CODI's icot formatting on our GSM8K-Aug lines (q||<<..>> .. #### a): CoT minus its last step, "The answer is: N" + EOS; CODI's filters."""
    R, _ = G.load_aug(split, filter_bad=False); out = []; drop = {"tokens": 0, "answer": 0}
    for r in R:
        body = r["_line"].split("||", 1)[1]; cot_s, ans_s = body.rsplit("####", 1); cot = cot_s.strip().split(" "); ans = ans_s.strip().split(" ")[-1]
        if len(enc.encode(r["q"] + cot_s.strip() + "#### " + ans)) > 1000:
            drop["tokens"] += 1; continue
        if not ans[:1].isdigit():
            drop["answer"] += 1; continue
        q = enc.encode(r["q"]); c = enc.encode(" ".join(cot[:-1])); a = enc.encode("The answer is: " + ans.replace("####", "")) + [EOS]
        out.append({"q": q, "ref": q + c + a, "nq": len(q), "dec": [EOT] + a, "m": len(r["eqs"])})
    return out, drop


def ans_pos(seq):
    for i in range(len(seq) - 3):
        if seq[i:i + 4] == ANS_PROMPT:
            return i + 4
    raise AssertionError("CODI: 'The answer is:' not found")


def collate(its, dev):
    Tq = max(len(it["q"]) + 1 for it in its); Tr = max(len(it["ref"]) for it in its); Td = max(len(it["dec"]) for it in its); B = len(its)
    enc = torch.full((B, Tq), PAD); em = torch.zeros(B, Tq, dtype=torch.long); ref = torch.full((B, Tr), PAD); rl = torch.full((B, Tr), -100)
    dec = torch.full((B, Td), PAD); dl = torch.full((B, Td), -100); rp, mp = [], []
    for b, it in enumerate(its):
        s = it["q"] + [BOT]; enc[b, Tq - len(s):] = torch.tensor(s); em[b, Tq - len(s):] = 1
        ref[b, :len(it["ref"])] = torch.tensor(it["ref"]); rl[b, it["nq"]:len(it["ref"])] = torch.tensor(it["ref"][it["nq"]:])
        dec[b, :len(it["dec"])] = torch.tensor(it["dec"]); dl[b, :len(it["dec"])] = torch.tensor(it["dec"])
        rp.append(ans_pos(it["ref"]) - 1); mp.append(ans_pos(it["dec"]) - 1)        # the ':' position (CODI: answer position - 1)
    t = lambda x: x.to(dev)  # noqa: E731
    return t(enc), t(em), t(ref), t(rl), t(dec), t(dl), t(torch.tensor(rp)), t(torch.tensor(mp))


def codi_loss(m, prj, batch, n_latent=6, gamma=2.0, parts=False):
    """CODI.forward (fix_attn_mask False, use_prj, smooth_l1 / std distill, no decoder) -> (loss, {ce, distill, ref_ce})."""
    enc, em, ref, rl, dec, dl, rp, mp = batch
    o = m(input_ids=enc, attention_mask=em, use_cache=True, output_hidden_states=True); pkv = o.past_key_values
    lat = prj(o.hidden_states[-1][:, -1:, :])
    with torch.no_grad():
        ro = m(input_ids=ref, output_hidden_states=True)
    rog = m(input_ids=ref, output_hidden_states=True)
    for i in range(n_latent):
        o = m(inputs_embeds=lat, use_cache=True, output_hidden_states=True, past_key_values=pkv); pkv = o.past_key_values
        lat = prj(o.hidden_states[-1][:, -1:, :])
    so = m(inputs_embeds=m.transformer.wte(dec), use_cache=True, output_hidden_states=True, past_key_values=pkv)
    dist = 0.0; ix = lambda h, pos: h.gather(1, pos[:, None, None].expand(-1, 1, h.size(-1)))  # noqa: E731
    for out, rout in zip(so.hidden_states, ro.hidden_states):
        rs = ix(rout, rp); ss = ix(out, mp); dist = dist + F.smooth_l1_loss(ss, rs.detach()) / rs.std()
    dist = dist / len(so.hidden_states)
    lg = so.logits[:, :-1]; ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), dl[:, 1:].reshape(-1), ignore_index=-100)
    rlg = rog.logits[:, :-1]; rce = F.cross_entropy(rlg.reshape(-1, rlg.size(-1)), rl[:, 1:].reshape(-1), ignore_index=-100) * gamma
    return ce + dist + rce, {"ce": float(ce.detach()), "distill": float(dist.detach()), "ref_ce": float(rce.detach())}


def merged_fp32(m, prj, dev):
    """an fp32 GPT2LMHeadModel with LoRA merged (W + s (B A)^T on the Conv1D (in, out) weight) + prj, for codi_probe.run (G1's recipe)."""
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=V0 + 3); cfg._attn_implementation = "eager"; mm = GPT2LMHeadModel(cfg); sd = {}
    for n, mod in m.named_modules():
        if isinstance(mod, LoRAConv1D):
            sd[n + ".weight"] = mod.base.weight.float() + mod.s * (mod.lora_B.weight.float() @ mod.lora_A.weight.float()).T; sd[n + ".bias"] = mod.base.bias.float()
    for n, p_ in m.named_parameters():
        if ".lora_" not in n and ".base." not in n:
            sd[n] = p_.float()
    miss, unexp = mm.load_state_dict(sd, strict=False); assert not unexp and all(k.endswith(("attn.bias", "masked_bias")) or k == "lm_head.weight" for k in miss), (miss, unexp)
    mm.tie_weights(); assert torch.equal(mm.lm_head.weight, mm.transformer.wte.weight); pj = nn.Sequential(nn.Dropout(0.0), nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 768)); pj.add_module("ln", nn.LayerNorm(768))
    pj.load_state_dict(prj.state_dict()); return mm.to(dev).eval(), pj.to(dev).eval()


def codi_tok():
    from transformers import GPT2Tokenizer
    p = os.path.join(ROOT, "prosqa", "ckpt_ext", "hf", "simcot_codi"); tok = GPT2Tokenizer(os.path.join(p, "vocab.json"), os.path.join(p, "merges.txt"))
    tok.add_special_tokens({"pad_token": "[PAD]"}); return tok


def cp_mod():
    os.environ.setdefault("CODI_SP", os.path.join(ROOT, "prosqa", "ckpt_ext")); sys.path.insert(0, os.path.join(ROOT, "theory"))
    import codi_probe as cp
    return cp


def evaluate(m, prj, data, dev, n_latent=6, bs=16):
    cp = cp_mod(); mm, pj = merged_fp32(m, prj, dev); o = cp.run(mm, pj, codi_tok(), data, n_latent, dev=dev, bs=bs); del mm
    by = {}
    for r in o["rows"]:
        if r["nsteps"] is not None:
            by.setdefault(r["nsteps"], []).append(r["ok"])
    return o["acc"], {str(k): [sum(v) / len(v), len(v)] for k, v in sorted(by.items())}, o["rows"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=4.0); ap.add_argument("--eff_bs", type=int, default=128); ap.add_argument("--micro", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--wd", type=float, default=0.1); ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--clip", type=float, default=2.0); ap.add_argument("--gamma", type=float, default=2.0); ap.add_argument("--n_latent", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=50); ap.add_argument("--valid_every", type=int, default=0, help="0 = every ~0.5 epoch"); ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--train_n", type=int, default=0); ap.add_argument("--test_n", type=int, default=0); ap.add_argument("--valid_n", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0, help="stop after N optimiser steps (smokes / timing); the schedule still spans --epochs")
    ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--auto_micro", action="store_true"); ap.add_argument("--mem_budget", type=float, default=21.0)
    ap.add_argument("--eval_only", action="store_true"); ap.add_argument("--no_ablation", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = a.device; torch.manual_seed(a.seed)
    import tiktoken
    enc = tiktoken.get_encoding("gpt2"); Dtr, drop = load_items("train", enc); Dtr = Dtr[:a.train_n] if a.train_n else Dtr
    cp = cp_mod(); test = cp.gsm8k(10_000); test = test[:a.test_n] if a.test_n else test
    Rva, _ = G.load_aug("valid", filter_bad=False); valid = [(r["q"].strip().replace("  ", " "), float(r["ans"].replace(",", "")), len(r["eqs"])) for r in Rva]
    valid = valid[:a.valid_n] if a.valid_n else valid
    m, prj = build(dev, a.seed); TP = trainable(m, prj)
    nl = sum(p_.numel() for n, p_ in TP if ".lora_" in n); npj = sum(p_.numel() for n, p_ in TP if n.startswith("prj."))
    acc_steps = a.eff_bs // a.micro; total = math.ceil(a.epochs * len(Dtr) / a.eff_bs); warm = math.ceil(a.warmup * total)
    if a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.eval_only):
        longest = sorted(Dtr, key=lambda it: -(len(it["ref"]) + len(it["q"])))[:1]; mc = []
        while True:
            torch.cuda.reset_peak_memory_stats(dev)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l_, _ = codi_loss(m, prj, collate(longest * a.micro, dev), a.n_latent, a.gamma)
                l_.backward(); pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30 + 3 * sum(p_.numel() * 4 for _, p_ in TP) / 2 ** 30
            except torch.OutOfMemoryError:
                pk = float("inf")
            for _, p_ in TP:
                p_.grad = None
            torch.cuda.empty_cache(); mc.append({"micro": a.micro, "peak_gib": pk}); print("MEM", a.micro, pk, flush=True)
            if pk <= a.mem_budget or a.mem_check or a.micro <= 8:
                break
            a.micro //= 2
        acc_steps = a.eff_bs // a.micro
        json.dump({"checks": mc, "micro": a.micro, "accumulation": acc_steps, "longest_ref_tokens": len(longest[0]["ref"])}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return
    assert a.eff_bs % a.micro == 0
    info = {**vars(a), "label": LABEL, "train_items": len(Dtr), "train_filter_dropped": drop, "steps_total": total, "warmup_steps": warm, "accumulation": acc_steps,
            "effective_bs": a.micro * acc_steps, "lora_params": nl, "prj_params": npj, "lora": {"r": 128, "alpha": 32, "dropout": 0.1, "targets": ["c_attn", "c_proj", "c_fc"]},
            "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "codi_source": "github zhenyi4/codi (src/model.py, train.py, scripts/train_gpt2_gsm8k-aug.sh)"}
    json.dump(info, open(os.path.join(a.out, "args_eval.json" if a.eval_only else "args.json"), "w", newline="\n"), indent=1)
    dec = [p_ for n, p_ in TP if not (n.endswith("bias") or n.startswith("prj.ln"))]; nod = [p_ for n, p_ in TP if n.endswith("bias") or n.startswith("prj.ln")]   # HF Trainer: no decay on biases / LayerNorm
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": a.wd}, {"params": nod, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(1, warm) if s < warm else max(0.0, 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))))
    ve = a.valid_every or max(1, round(0.5 * len(Dtr) / a.eff_bs)); vhist = []; step = 0; seen = 0; t0 = time.time(); rng = np.random.default_rng(a.seed)
    ck = os.path.join(a.out, "ckpt_last.pt"); order, pos = rng.permutation(len(Dtr)), 0; ep = 0
    if os.path.exists(ck) and not a.eval_only:
        c_ = torch.load(ck, map_location=dev, weights_only=False)
        for (n, p_) in TP:
            p_.data.copy_(c_["tp"][n])
        opt.load_state_dict(c_["opt"]); sched.load_state_dict(c_["sched"]); step, seen, pos, ep, vhist = c_["step"], c_["seen"], c_["pos"], c_["ep"], c_["vhist"]
        rng.bit_generator.state = c_["rng"]; order = c_["order"]; torch.set_rng_state(c_["torch_rng"].cpu()); torch.cuda.set_rng_state(c_["cuda_rng"].cpu(), dev)
        t0 = time.time() - c_.get("elapsed", 0.0)                           # sec keeps counting across a resume (the cost guard's s/step)
    log = open(os.path.join(a.out, "log.jsonl"), "a", newline="\n"); hist = []
    if a.eval_only:
        c_ = torch.load(os.path.join(a.out, "trainable.pt"), map_location=dev, weights_only=False)
        for (n, p_) in TP:
            p_.data.copy_(c_[n])
        total = step = 0
    if step == 0 and not a.eval_only:                                       # preflight: the eval path on this host's stack before any spend on training
        m.eval(); prj.eval(); tv = time.time(); pa, _, _ = evaluate(m, prj, test[:32], dev)
        log.write(json.dumps({"step": 0, "preflight_eval32_acc": pa, "sec": round(time.time() - tv, 1), "transformers": __import__("transformers").__version__}) + "\n"); log.flush()
    m.train(); prj.train()
    while step < total and not (a.max_steps and step >= a.max_steps):
        opt.zero_grad(set_to_none=True); parts = []
        for _ in range(acc_steps):
            if pos + a.micro > len(order):                                  # epoch end: the remainder, then a fresh permutation (Trainer: no drop_last)
                idx = order[pos:]; order, pos, ep = rng.permutation(len(Dtr)), 0, ep + 1
                need = a.micro - len(idx); idx = np.concatenate([idx, order[:need]]); pos = need
            else:
                idx = order[pos:pos + a.micro]; pos += a.micro
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                l_, pt = codi_loss(m, prj, collate([Dtr[i] for i in idx], dev), a.n_latent, a.gamma)
            (l_ / acc_steps).backward(); parts.append((float(l_.detach()), pt)); seen += len(idx)
        gn = torch.nn.utils.clip_grad_norm_([p_ for _, p_ in TP], a.clip); opt.step(); sched.step(); step += 1
        hist.append((np.mean([x for x, _ in parts]), {k: np.mean([p[k] for _, p in parts]) for k in ("ce", "distill", "ref_ce")}, float(gn)))
        if step % a.log_every == 0 or step == total:
            rec = {"step": step, "loss": float(np.mean([h[0] for h in hist])), "sec": round(time.time() - t0),       # step / loss / sec adjacent: cost_guard's regex
                   **{k: float(np.mean([h[1][k] for h in hist])) for k in ("ce", "distill", "ref_ce")},
                   "gn_preclip": float(np.mean([h[2] for h in hist])), "lr": sched.get_last_lr()[0], "seen": seen, "epochs": seen / len(Dtr)}
            hist = []; log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if (step % ve == 0 or step == total) and not a.max_steps:
            m.eval(); prj.eval(); tv = time.time(); va, _, _ = evaluate(m, prj, valid, dev); m.train(); prj.train(); vhist.append([step, va])
            log.write(json.dumps({"step": step, "valid_acc": va, "epochs": seen / len(Dtr), "valid_sec": round(time.time() - tv, 1)}) + "\n"); log.flush()
        if a.ckpt_every and step % a.ckpt_every == 0 and step < total:
            torch.save({"tp": {n: p_.detach() for n, p_ in TP}, "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "seen": seen, "pos": pos, "ep": ep,
                        "vhist": vhist, "rng": rng.bit_generator.state, "order": order, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(dev),
                        "elapsed": time.time() - t0}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    if not a.eval_only:
        torch.save({n: p_.detach().cpu() for n, p_ in TP}, os.path.join(a.out, "trainable.pt"))
    if a.max_steps and not a.eval_only:
        sps = (time.time() - t0) / max(1, step); m.eval(); prj.eval(); te = time.time(); sa, _, _ = evaluate(m, prj, test[:32], dev)   # the eval path on this host's stack
        json.dump({"steps_timed": step, "sec": round(time.time() - t0), "sec_per_step": sps, "steps_total": total, "est_hours": sps * total / 3600, "micro": a.micro,
                   "accumulation": acc_steps, "eval32_acc": sa, "eval32_sec": round(time.time() - te, 1), "valid_every": ve,
                   "transformers": __import__("transformers").__version__, "torch": torch.__version__}, open(os.path.join(a.out, "timing.json"), "w", newline="\n"), indent=1)
        json.dump({"timing": True, "sec_per_step": sps}, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
        print("TIMING", (time.time() - t0) / max(1, step), flush=True); return
    m.eval(); prj.eval(); te = time.time(); acc, by, rows = evaluate(m, prj, test, dev, a.n_latent)
    res = {"acc": acc, "n": len(test), "epochs": seen / len(Dtr) if seen else None, "examples_seen": seen, "steps": step, "valid_hist": vhist, "by_steps": by,
           "train_items": len(Dtr), "label": LABEL, "gamma": a.gamma, "effective_bs": a.micro * acc_steps, "train_sec": round(te - t0),
           "eval_recipe": "theory/codi_probe.py run() on the LoRA-merged fp32 model (bs 16), CODI's GSM8K test json", "eval_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"}
    json.dump([{"i": i, "pred": r["pred"], "gold": r["gold"], "ok": bool(r["ok"])} for i, r in enumerate(rows)], open(os.path.join(a.out, "eval_items.json"), "w", newline="\n"), indent=0)
    json.dump({**res, "eval_partial": True}, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    if not a.no_ablation:
        res["k_latent_ablation"] = {"6": acc, "0": evaluate(m, prj, test, dev, 0)[0]}
    res["eval_sec"] = round(time.time() - te); res["eval_partial"] = False
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    if os.path.exists(ck):
        os.remove(ck)
    print("EVAL", json.dumps({k: res[k] for k in ("acc", "n", "epochs")}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
