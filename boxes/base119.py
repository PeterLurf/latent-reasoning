"""THEORY §119: published latent-CoT methods on our S5 chunked host (§97's harness), scored with §104's donor protocol and
our length protocol. One 4-op chunk: question = [start text (6 tokens, " > A B C D E")] [op_1 .. op_4 texts (4 x 6)];
CoT steps = the state texts after ops 1-3 (§97's format); answer = the state text after op 4 (+ EOS in training). Training
chunks are cut at random positions of the S5=128 training sequences with the TRUE start (latent86.batch). Test: gen97's
restart protocol (start = identity; every 4 ops the decoded answer's 6 tokens are re-embedded as the next start text, the
history cut). The LM head is GPT-2's tied head over its 50,257 tokens (new marker rows are inputs only).
Methods (--method; recipes read from the released code on 2026-10-05):
  coconut  Coconut 2412.06769 (facebookresearch/coconut; the copy in InternLM/SIM-CoT): question <bot> n thoughts <eot>
           rest; a thought's input is the previous position's last hidden (after ln_f); the question stays visible to every
           thought. Stage s = 0..3 replaces the first s steps with s*c thoughts (c = 1); CE (mean over tokens) on the
           remaining steps, the answer and EOS; AdamW (lr, wd 0.01), constant lr, no clipping, optimiser reset at every
           stage; <bot>/<eot> rows initialised from "<<" (trainable). --stage_steps per stage 0-2, the rest at stage 3.
  simcot   SIM-CoT 2509.20317 (InternLM/SIM-CoT Coconut/coconut.py, CoconutGPT_Same_Word_Embedding; gsm_simcot.yaml):
           Coconut with c = 2 from the trained S5 Coconut model (--init), the stage curriculum rerun from stage 0, plus an
           auxiliary decoder phi = a separate GPT-2 from the pretrained base. For each latent step k it sees ONLY step k's c
           thoughts, then step k's text embedded with theta's wte ("same word embedding") and EOS, positions from 1, and
           predicts the text + EOS with its own LM head. loss = CE_main + 1.0 x (sum over latent steps of the per-row summed
           decoder CE, mean over rows) / (number of latent steps). Full training (theta and phi); phi is unused at test.
  codi     CODI 2502.21074 (zhenyi4/codi src/model.py, train.py, scripts/train_gpt2_gsm8k-aug.sh): one model, LoRA r 128 /
           alpha 32 / dropout 0.1 on c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj (base weights frozen; <bot>/<eot> frozen at
           the wte mean, as resized rows outside LoRA); prj = Linear-GELU-Linear-LayerNorm (768). Student: question <bot>,
           --n_lat latents (z_1 = prj(h at <bot>), z_(i+1) = prj(h at z_i)), then <eot> " The answer is:" answer EOS, CE on
           the prompt, answer and EOS. Teacher: question, the 3 step texts, " The answer is:", answer, EOS, CE on all but the
           question. KD: at ":" (the token before the answer), SmoothL1(student, sg(teacher)) / std(teacher) per hidden
           state, averaged over the 13 (embeddings, blocks 1-11, ln_f), teacher from a separate no-grad pass. alpha = beta
           = gamma = 1. AdamW (lr, wd 0.1), warmup 3 % + cosine, clip 2.0.
Eval (eval.json; the keys theory/score_s119.py reads): intact_acc (§104: final-state accuracy on n = 2000 fresh random
chunks, unpatched), donor (§104: slot j's thought(s) <- a donor chunk B's; donor_consistent = decode == B's state after j
ops composed with A's ops j+1..4, recipient_consistent = decode == A's true final state; coconut slot j = thought j,
simcot slot j = thoughts 2j-1, 2j jointly, codi block = every latent, donor target = B's final state), cumulative
{128, 256, 512} on §71's test set and extension_2048 (first 500 of S5=2048) - CHUNK ENDS ONLY (in-chunk states are latent,
not decoded), chunk_check (20k fresh chunks), removal {zero, mean} of every thought (descriptive), train_split_final_acc
(5,000 chunks of the training sequences: the registered lr-sweep criterion), trainable_params, faithful (+ reference).
    python -m boxes.base119 --method coconut --lr 1e-4 --seed 1 --out boxes/runs_s5/s119_coconut_s1
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
from boxes.gen97 import NST, STTOK, amp_ctx, decode_to_el, true_chain  # noqa: E402
from boxes.latent86 import OPTOK, batch  # noqa: E402

K = 4                                                                     # ops per chunk; steps 1..3 + answer (op 4)
EOS = 50256
LT = bc.tok("<<")[0]                                                      # Coconut's init row for its new tokens
PROMPT = bc.tok(" The answer is:")                                        # CODI's answer prompt; KD at its last token ":"
FAITHFUL = {"coconut": "ProsQA, official recipe through our port: 97.6 vs published 97.0 (bench, 0ef4a78)",
            "simcot": "SIM-CoT + Coconut GPT-2 (HF checkpoint_28) GSM8K through our port: 44.8 vs published 44.8 (docs/GSM8K_REPRO.md)",
            "codi": "CODI GPT-2 GSM8K through our port: 42.5 vs published 43.7 (docs/GSM8K_REPRO.md; within 2 points)"}
LORA_MODS = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


class GPT2X(bc.GPT2KV):
    """GPT2KV's forward (no inject / kv_mask) with optional LoRA (CODI) and cap = an index into x: also return the 13
    hidden states at that position, as HF's hidden_states (input embeddings + positions, blocks 1..11, ln_f output)."""
    def __init__(self, sd, lora_r=0, lora_alpha=32, lora_drop=0.1):
        super().__init__(sd)
        self.lora_r = lora_r
        if lora_r:
            for p_ in self.p.values():
                p_.requires_grad_(False)
            self.la = nn.ParameterDict(); self.lb = nn.ParameterDict(); self.ls = lora_alpha / lora_r; self.ldrop = lora_drop
            for i in range(self.L):
                for nm in LORA_MODS:
                    W = self.w(f"h.{i}.{nm}.weight"); key = f"h_{i}_{nm.replace('.', '_')}"
                    A = torch.empty(lora_r, W.shape[0]); nn.init.kaiming_uniform_(A, a=math.sqrt(5))   # peft: lora_A, B = 0
                    self.la[key] = nn.Parameter(A.T.contiguous()); self.lb[key] = nn.Parameter(torch.zeros(lora_r, W.shape[1]))

    def lin(self, pre, nm, a):
        y = a @ self.w(pre + nm + ".weight") + self.w(pre + nm + ".bias")
        if self.lora_r:
            key = pre.replace(".", "_") + nm.replace(".", "_")
            y = y + (F.dropout(a, self.ldrop, self.training) @ self.la[key] @ self.lb[key]) * self.ls
        return y

    def forward(self, x, past=None, pos0=None, cap=None):
        B, T, D = x.shape; P0 = 0 if past is None else past[0][0].shape[2]; hd = D // self.H
        p0 = P0 if pos0 is None else pos0
        h = x + self.w("wpe.weight")[p0:p0 + T][None]; new = []; hs = [h[:, cap]] if cap is not None else None
        for i in range(self.L):
            pre = f"h.{i}."
            a = F.layer_norm(h, (D,), self.w(pre + "ln_1.weight"), self.w(pre + "ln_1.bias"))
            q, k, v = self.lin(pre, "attn.c_attn", a).split(D, -1)
            q, k, v = (t.view(B, T, self.H, hd).transpose(1, 2) for t in (q, k, v))
            if past is not None:
                k = torch.cat([past[i][0], k], 2); v = torch.cat([past[i][1], v], 2)
            new.append((k, v))
            if P0 == 0:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                mask = torch.ones(T, P0 + T, dtype=torch.bool, device=x.device).tril(P0)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            h = h + self.lin(pre, "attn.c_proj", o.transpose(1, 2).reshape(B, T, D))
            a = F.layer_norm(h, (D,), self.w(pre + "ln_2.weight"), self.w(pre + "ln_2.bias"))
            h = h + self.lin(pre, "mlp.c_proj", F.gelu(self.lin(pre, "mlp.c_fc", a), approximate="tanh"))
            if cap is not None and i < self.L - 1:
                hs.append(h[:, cap])
        out = F.layer_norm(h, (D,), self.w("ln_f.weight"), self.w("ln_f.bias"))
        if cap is not None:
            hs.append(out[:, cap]); return out, new, hs
        return out, new


def load_gpt2x(dev, lora_r=0):
    sd = torch.load(os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), map_location="cpu", weights_only=True)
    return GPT2X(sd, lora_r=lora_r).to(dev)


class Base119(nn.Module):
    def __init__(self, g, method, c=1, n_lat=4, dec=None):
        super().__init__()
        self.g = g; self.method = method; self.c = c; self.dec = dec
        self.register_buffer("sttok", torch.tensor(STTOK)); self.register_buffer("optok", torch.tensor([OPTOK[e] for e in range(120)]))
        self.register_buffer("prompt", torch.tensor(PROMPT))
        if method == "codi":
            mu = g.w("wte.weight").mean(0).detach()
            self.register_buffer("special", torch.stack([mu, mu]).clone())            # <bot>, <eot>: frozen
            self.prj = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 768), nn.LayerNorm(768))
            self.n_final = n_lat
        else:
            e = g.w("wte.weight")[LT].detach().clone()
            self.special = nn.Parameter(torch.stack([e, e.clone()]))                   # <bot>, <eot>: from "<<"
            self.prj = None; self.n_final = 3 * c

    def question(self, stt, ops):
        B = ops.shape[0]; g = self.g
        return torch.cat([g.embed(stt), g.embed(self.optok[ops]).reshape(B, K * NST, -1), self.special[0].expand(B, 1, -1)], 1)

    def thoughts(self, x, n, override=None):
        """x = question + <bot>; n latent inputs (Coconut: the last hidden; CODI: prj of it). override {i: (B, D)} replaces
        thought i's input (0-based). Returns the cache after the last thought and the thought inputs."""
        h, past = self.g(x); hl = h[:, -1]; zs = []
        for i in range(n):
            z = self.prj(hl) if self.prj is not None else hl
            if override is not None and i in override:
                z = override[i].to(z.dtype)
            zs.append(z); h, past = self.g(z[:, None], past); hl = h[:, -1]
        return past, zs

    def loss(self, stt, ops, tgt, stage=None):
        return self.loss_codi(stt, ops, tgt) if self.method == "codi" else self.loss_coconut(stt, ops, tgt, stage)

    def loss_coconut(self, stt, ops, tgt, stage):
        B = ops.shape[0]; g = self.g; n = stage * self.c
        past, zs = self.thoughts(self.question(stt, ops), n)
        rest = torch.cat([self.sttok[tgt[:, stage:]].reshape(B, -1), torch.full((B, 1), EOS, device=ops.device)], 1)
        x2 = torch.cat([self.special[1].expand(B, 1, -1), g.embed(rest[:, :-1])], 1)
        h, _ = g(x2, past); lg = g.logits(h).float()
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), rest.reshape(-1)); parts = {"ce": ce}; loss = ce
        if self.dec is not None and stage > 0:                                   # SIM-CoT step-level supervision
            Z = torch.stack(zs, 1).view(B, stage, self.c, -1).transpose(0, 1).reshape(stage * B, self.c, -1)
            T = self.sttok[tgt[:, :stage]].transpose(0, 1).reshape(stage * B, NST)
            lab = torch.cat([T, torch.full((stage * B, 1), EOS, device=ops.device)], 1)
            hd, _ = self.dec(torch.cat([Z.to(h.dtype), g.embed(T).to(h.dtype)], 1), pos0=1)
            lgd = self.dec.logits(hd[:, self.c - 1:]).float()
            st = F.cross_entropy(lgd.reshape(-1, lgd.shape[-1]), lab.reshape(-1), reduction="sum") / (B * stage)
            parts["step"] = st; loss = loss + st
        return loss, parts

    def loss_codi(self, stt, ops, tgt):
        B = ops.shape[0]; g = self.g; dev = ops.device
        Q = torch.cat([stt, self.optok[ops].reshape(B, -1)], 1); P = self.prompt.expand(B, -1)
        eos = torch.full((B, 1), EOS, device=dev); ans = self.sttok[tgt[:, 3]]
        ref = torch.cat([Q, self.sttok[tgt[:, :3]].reshape(B, -1), P, ans, eos], 1)
        ci = Q.shape[1] + 3 * NST + P.shape[1] - 1                               # ":" in the teacher sequence
        with torch.no_grad():
            _, _, hT = g(g.embed(ref), cap=ci)                                    # KD target (separate pass, as CODI)
        hR, _ = g(g.embed(ref)); lgR = g.logits(hR[:, Q.shape[1] - 1:-1]).float()
        teach = F.cross_entropy(lgR.reshape(-1, lgR.shape[-1]), ref[:, Q.shape[1]:].reshape(-1))
        past, _ = self.thoughts(torch.cat([g.embed(Q), self.special[0].expand(B, 1, -1)], 1), self.n_final)
        dids = torch.cat([P, ans, eos], 1)
        hS, _, hs = g(torch.cat([self.special[1].expand(B, 1, -1), g.embed(dids)], 1), past, cap=P.shape[1])
        lgS = g.logits(hS[:, :-1]).float()
        stud = F.cross_entropy(lgS.reshape(-1, lgS.shape[-1]), dids.reshape(-1))
        kd = torch.stack([F.smooth_l1_loss(s.float(), t.float().detach()) / t.float().std() for s, t in zip(hs, hT)]).mean()
        return stud + teach + kd, {"student": stud, "teacher": teach, "kd": kd}

    @torch.no_grad()
    def infer(self, stt, ops, override=None, n=None):
        """Greedy 6-token answer after the final stage's thoughts. Returns (tokens (B, 6), thought inputs)."""
        B = ops.shape[0]; g = self.g; n = self.n_final if n is None else n
        if self.method == "codi":
            past, zs = self.thoughts(torch.cat([g.embed(torch.cat([stt, self.optok[ops].reshape(B, -1)], 1)),
                                                self.special[0].expand(B, 1, -1)], 1), n, override)
            x = torch.cat([self.special[1].expand(B, 1, -1), g.embed(self.prompt.expand(B, -1))], 1)
        else:
            past, zs = self.thoughts(self.question(stt, ops), n, override)
            x = self.special[1].expand(B, 1, -1)
        h, past = g(x, past); t = g.logits(h[:, -1]).argmax(-1); out = [t]
        for _ in range(NST - 1):
            h, past = g(g.embed(t)[:, None], past); t = g.logits(h[:, -1]).argmax(-1); out.append(t)
        return torch.stack(out, 1), zs


# ---------------------------------------------------------------- eval
def dec_chunks(m, stt, ops, dev, amp, override=None):
    with amp_ctx(amp):
        toks, zs = m.infer(torch.as_tensor(stt, device=dev), torch.as_tensor(ops, device=dev), override)
    toks = toks.cpu().numpy(); el, _ = decode_to_el(toks)
    return el, toks, zs


def final_acc(m, S, O, dev, amp, bs, override_fn=None):
    ok = 0
    for i in range(0, len(S), bs):
        ov = override_fn(i, i + bs) if override_fn else None
        el, _, _ = dec_chunks(m, STTOK[S[i:i + bs]], O[i:i + bs], dev, amp, ov)
        ok += int((el == true_chain(S[i:i + bs], O[i:i + bs])[:, -1]).sum())
    return ok / len(S)


@torch.no_grad()
def rollout(m, X, dev, amp, bs):
    """Chunk-end predictions (N, L / 4): start = identity; snap = the decoded answer's tokens, history cut."""
    N, L = X.shape; pred = np.full((N, L // K), -1, dtype=np.int64); snaps = valid = 0
    for s in range(0, N, bs):
        n = min(bs, N - s); cur = STTOK[np.zeros(n, dtype=np.int64)]
        for c in range(0, L, K):
            el, toks, _ = dec_chunks(m, cur, X[s:s + bs, c:c + K], dev, amp)
            pred[s:s + bs, c // K] = el; cur = toks; snaps += n; valid += int((el >= 0).sum())
    return pred, {"snaps": snaps, "valid": valid, "rate": valid / max(1, snaps)}


def chunk_cum(pred, Y):
    ok = pred == Y[:, K - 1::K]
    return np.cumprod(ok, axis=1).mean(0)                                 # index i: all chunk ends 1..i+1 correct


def slot_ids(m, j):
    """§104 slot j (1-based) -> the thought indices patched jointly."""
    return [j - 1] if m.c == 1 else list(range((j - 1) * m.c, j * m.c))


@torch.no_grad()
def donor(m, dev, amp, bs, n=2000, seed=11904):
    rng = np.random.default_rng(seed); S = rng.integers(0, 120, n); O = rng.integers(0, 120, (n, K))
    SB = rng.integers(0, 120, n); OB = rng.integers(0, 120, (n, K)); TA = true_chain(S, O)[:, -1]; TB = true_chain(SB, OB)
    zB = []
    for i in range(0, n, bs):
        _, _, zs = dec_chunks(m, STTOK[SB[i:i + bs]], OB[i:i + bs], dev, amp); zB.append(torch.stack(zs, 1))
    zB = torch.cat(zB); res = {}; intact = []
    for i in range(0, n, bs):
        el, _, _ = dec_chunks(m, STTOK[S[i:i + bs]], O[i:i + bs], dev, amp); intact.append(el)
    intact = np.concatenate(intact)
    slots = {"block": list(range(m.n_final))} if m.method == "codi" else {f"slot{j}": slot_ids(m, j) for j in range(1, K)}
    for name, ids in slots.items():
        if name == "block":
            tgt = TB[:, -1]
        else:
            j = int(name[4:]); tgt = TB[:, j - 1]
            for t in range(j, K):
                tgt = s5.COMP[tgt, O[:, t]]
        out = []
        for i in range(0, n, bs):
            el, _, _ = dec_chunks(m, STTOK[S[i:i + bs]], O[i:i + bs], dev, amp, {t: zB[i:i + bs, t] for t in ids})
            out.append(el)
        out = np.concatenate(out)
        res[name] = {"thoughts": ids, "donor_consistent": float((out == tgt).mean()), "recipient_consistent": float((out == TA).mean()),
                     "chance_overlap": float((tgt == TA).mean())}
    return float((intact == TA).mean()), res


@torch.no_grad()
def removal(m, Xtr, Ytr, dev, amp, bs, n=10_000, mean_n=2000):
    rng = np.random.default_rng(11905); st, o, _ = batch(rng, Xtr, Ytr, K, mean_n); zs = []
    for i in range(0, mean_n, bs):
        _, _, z = dec_chunks(m, STTOK[st[i:i + bs]], o[i:i + bs], dev, amp); zs.append(torch.stack(z, 1).float())
    mu = torch.cat(zs).mean(0)
    S = rng.integers(0, 120, n); O = rng.integers(0, 120, (n, K)); res = {}
    for ab in ("zero", "mean"):
        def ov(i0, i1, ab=ab):
            b = min(i1, n) - i0
            return {t: (torch.zeros_like(mu[t]) if ab == "zero" else mu[t])[None].expand(b, -1) for t in range(m.n_final)}
        res[ab] = final_acc(m, S, O, dev, amp, bs, ov)
    return res


def trainable(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


def build(a, dev):
    if a.method == "codi":
        return Base119(load_gpt2x(dev, lora_r=128), "codi", n_lat=a.n_lat).to(dev)
    m = Base119(load_gpt2x(dev), a.method, c=2 if a.method == "simcot" else 1,
                dec=load_gpt2x(dev) if a.method == "simcot" else None).to(dev)
    if a.method == "simcot" and a.init:
        sd = torch.load(a.init, map_location=dev, weights_only=True)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith("dec.") for k in missing), (unexpected, [k for k in missing if not k.startswith("dec.")])
    return m


def stage_of(a, step):
    return None if a.method == "codi" else min(3, (step - 1) // a.stage_steps)


def make_opt(m, a):
    return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.1 if a.method == "codi" else 0.01)


def evaluate(m, a, dev, amp, Xtr, Ytr):
    m.eval(); t1 = time.time()
    Xte, Yte = s5.load_csv(os.path.join(a.data_dir, "S5=512.csv"), limit=a.test_n); s5.check_labels(Xte, Yte)
    res = {"method": a.method, "seed": a.seed, "lr": a.lr, "c": m.c, "n_thoughts": m.n_final, "eval_dtype": str(amp),
           "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "trainable_params": trainable(m),
           "faithful": True, "faithful_ref": FAITHFUL[a.method], "cumulative_definition": "chunk ends only (in-chunk states latent)"}
    res["intact_acc"], res["donor"] = donor(m, dev, amp, a.eval_bs, n=a.donor_n)
    pred, val = rollout(m, Xte, dev, amp, a.eval_bs); np.save(os.path.join(a.out, "preds_test.npy"), pred)
    cum = chunk_cum(pred, Yte)
    res["cumulative"] = {str(t): float(cum[t // K - 1]) for t in (128, 256, 512) if t <= Xte.shape[1]}
    res["final_state_acc_512"] = float((pred[:, -1] == Yte[:, -1]).mean()); res["checkpoint_validity"] = val
    if a.ext_n:
        Xx, Yx = s5.load_csv(os.path.join(a.data_dir, "S5=2048.csv"), limit=a.ext_n); s5.check_labels(Xx, Yx)
        Xx, Yx = Xx[:, :a.ext_len], Yx[:, :a.ext_len]; px, _ = rollout(m, Xx, dev, amp, a.eval_bs); cx = chunk_cum(px, Yx)
        res["extension_2048"] = {"n": len(Xx), "len": Xx.shape[1], "cumulative": {str(t): float(cx[t // K - 1]) for t in (512, 1024, 2048) if t <= Xx.shape[1]}}
    rng = np.random.default_rng(11903); S = rng.integers(0, 120, a.check_n); O = rng.integers(0, 120, (a.check_n, K))
    res["chunk_check"] = {"n": a.check_n, "per_chunk_acc": final_acc(m, S, O, dev, amp, a.eval_bs)}
    st, o, _ = batch(np.random.default_rng(11902), Xtr, Ytr, K, a.split_n)
    res["train_split_final_acc"] = final_acc(m, st, o, dev, amp, a.eval_bs)
    res["removal"] = removal(m, Xtr, Ytr, dev, amp, a.eval_bs, n=min(10_000, a.check_n), mean_n=a.mean_n)
    res["eval_sec"] = round(time.time() - t1)
    return res


def train(a):
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    s5.check_interface()
    Xtr, Ytr = s5.load_csv(os.path.join(a.data_dir, "S5=128.csv")); s5.check_labels(Xtr, Ytr)
    os.makedirs(a.out, exist_ok=True)
    m = build(a, dev)
    if not a.eval_only:
        json.dump({**vars(a), "train_sequences": len(Xtr), "trainable_params": trainable(m),
                   "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"},
                  open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
        log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []; parts_h = {}
        opt = None; cur = "init"; warm = max(1, int(0.03 * a.steps))
        for step in range(1, a.steps + 1):
            stage = stage_of(a, step)
            if stage != cur:                                                      # Coconut / SIM-CoT: reset per stage
                opt = make_opt(m, a); cur = stage
            if a.method == "codi":                                                # CODI: warmup 3 % + cosine
                for gr in opt.param_groups:
                    gr["lr"] = a.lr * (step / warm if step <= warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (a.steps - warm))))
            m.train(); st, o, tg = batch(rng, Xtr, Ytr, K, a.bs)
            with amp_ctx(amp):
                loss, parts = m.loss(torch.tensor(STTOK[st], device=dev), torch.tensor(o, device=dev), torch.tensor(tg, device=dev), stage)
            opt.zero_grad(); loss.backward()
            if a.method == "codi":
                torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
            opt.step(); hist.append(loss.item())
            for k_, v_ in parts.items():
                parts_h.setdefault(k_, []).append(float(v_.detach()))
            if step % 100 == 0 or step == a.steps:
                rec = {"step": step, "stage": stage, "loss": float(np.mean(hist)), **{k_: float(np.mean(v_)) for k_, v_ in parts_h.items()},
                       "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0)}; hist = []; parts_h = {}
                if a.log_every and step % a.log_every == 0 and (stage is None or stage == 3):
                    r2 = np.random.default_rng(11906); S = r2.integers(0, 120, a.log_n); O = r2.integers(0, 120, (a.log_n, K))
                    rec["final_acc"] = final_acc(m, S, O, dev, amp, a.eval_bs)
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        train_sec = round(time.time() - t0)
        torch.save({k_: v_ for k_, v_ in m.state_dict().items() if not k_.startswith("dec.")}, os.path.join(a.out, "model.pt"))
    else:
        m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev, weights_only=True), strict=False); train_sec = None
    res = evaluate(m, a, dev, amp, Xtr, Ytr); res["train_sec"] = train_sec; res["updates"] = a.steps
    res["train_gpu"] = json.load(open(os.path.join(a.out, "args.json"))).get("train_gpu")
    json.dump(res, open(os.path.join(a.out, "eval_only.json" if a.eval_only else "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({k_: res[k_] for k_ in ("intact_acc", "donor", "cumulative", "train_split_final_acc")}), flush=True)
    print("DONE", a.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["coconut", "simcot", "codi"])
    ap.add_argument("--init", default=None, help="simcot: the trained S5 Coconut model.pt")
    ap.add_argument("--n_lat", type=int, default=4, help="codi: latents per chunk (registered 4)")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--stage_steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--eval_bs", type=int, default=500)
    ap.add_argument("--test_n", type=int, default=s5.TEST_N); ap.add_argument("--ext_n", type=int, default=500)
    ap.add_argument("--ext_len", type=int, default=2048, help="< 2048: smoke only")
    ap.add_argument("--check_n", type=int, default=20_000); ap.add_argument("--donor_n", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=1000); ap.add_argument("--log_n", type=int, default=1000)
    ap.add_argument("--split_n", type=int, default=5000, help="< 5000: smoke only"); ap.add_argument("--mean_n", type=int, default=2000, help="< 2000: smoke only")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "bench", "data", "s5"))
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
