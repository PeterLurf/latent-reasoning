"""THEORY §135 TEACHER (registered f1899abd; Audit-approved stage A): an S5 text-CoT GPT-2 small, frozen afterwards, whose hidden states
at each step's state text become the per-hand-off targets of the §135 TSD student (fsq120 --tsd teacher).
Format (fsq120's own S5 token tables): [start state text] + per step [op text (OPTOK, 6 tokens), state text after it (STTOK,
" >" + 5 letters)] + [" a:", final state text]. Problems = fsq120.make_batch(Family("s5"), ...) (random start state, uniform ops), so
a student chain and its teacher sequence are the same problem. Training: L ~ U(1, 32), 8k steps, bs 32, AdamW lr 1e-4 (warmup 100 +
cosine), teacher-forced CE on the step state texts and the answer (ops are given inputs, not predicted).
Gate A1 (eval.json acc_L32): greedy interleaved generation on 1,000 held-out problems at L 32 (ops given, every state text and the
answer generated), exact final answer. Gate A2 (teacher_probe.json): a linear 120-way state probe on the hidden state at the LAST
token of each step's state text, per layer 0..12 (0 = embeddings; c = after c blocks, pre ln_f), held-out problems (fit / score
halves); probe_acc_by_layer, chosen_layer = argmax (ties -> later).
    python -m boxes.teacher135 --device cuda:0 --out boxes/runs_fsq/s135_teacher_s5 [--steps 8000 --bs 32]"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BOXES_NB", "5")
from boxes import core as bc  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.fsq120 import STTOK, Family, driver_version, env_info, make_batch  # noqa: E402
from boxes.latent86 import OPTOK  # noqa: E402

OPT = np.array([OPTOK[e] for e in range(120)])                             # (120, 6)
ANS = np.array(bc.tok(" a:"))
NT = STTOK.shape[1]; NO = OPT.shape[1]; BLK = NO + NT                       # 6 + 6 per step


def build(s0, x, lab):
    """(B,) start states, (B, L) items, (B, L) states -> ids (B, T), target mask (B, T) (predict ids[t] from t-1), state-end
    positions (B, L) (index of the last token of step j's state text)."""
    B, L = x.shape; parts = [STTOK[s0]]
    for j in range(L):
        parts += [OPT[x[:, j]], STTOK[lab[:, j]]]
    parts += [np.repeat(ANS[None], B, 0), STTOK[lab[:, -1]]]
    ids = np.concatenate(parts, 1); T = ids.shape[1]; msk = np.zeros((B, T), bool)
    for j in range(L):
        a = NT + j * BLK + NO; msk[:, a:a + NT] = True                     # step j's state text
    msk[:, T - NT:] = True                                                  # the answer
    ends = np.array([NT + j * BLK + NO + NT - 1 for j in range(L)])
    return ids, msk, np.repeat(ends[None], B, 0)


def hidden_layers(g, ids, upto=None):
    """All positions' residual stream after each block (pre ln_f): [h0 = embeddings + positions, h1, ..., hL]; upto: stop after
    that many blocks. (core.GPT2KV.forward's block, no cache.)"""
    x = g.embed(ids); B, T, D = x.shape; hd = D // g.H; h = x + g.w("wpe.weight")[:T][None]; out = [h]
    for i in range(g.L if upto is None else upto):
        pre = f"h.{i}."
        a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
        q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
        q, k, v = (t.view(B, T, g.H, hd).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
        a = F.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
        h = h + F.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh") @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")
        out.append(h)
    return out


@torch.no_grad()
def teacher_targets(g, layer, s0, x, lab, dev, amp):
    """§135 TSD targets: the frozen teacher's hidden at `layer` at the last token of each step's state text, same problem. (B, L, 768) fp32."""
    ids, _, ends = build(s0, x, lab)
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        h = hidden_layers(g, torch.tensor(ids, device=dev), upto=layer)[layer]
    return h.float()[torch.arange(len(ids), device=dev)[:, None], torch.tensor(ends, device=dev)]


def load_teacher(d, dev):
    g = s5.load_gpt2(dev); g.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev)); g.eval()
    for p_ in g.parameters():
        p_.requires_grad_(False)
    return g


@torch.no_grad()
def greedy_acc(g, dev, amp, n, L, seed, bs=100):
    """A1: ops given, every state text and the answer generated greedily (KV cache); exact final answer."""
    fam = Family("s5"); s0, x, lab = make_batch(fam, np.random.default_rng(seed), n, L); ok = 0; step_ok = 0
    for i in range(0, n, bs):
        S0, X, LB = s0[i:i + bs], x[i:i + bs], lab[i:i + bs]; B = len(S0)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            h, past = g(g.embed(torch.tensor(STTOK[S0], device=dev)))
            def gen(h, past, k):
                toks = []; t = g.logits(h[:, -1]).argmax(-1)
                for r in range(k):
                    toks.append(t); h, past = g(g.embed(t)[:, None], past)
                    if r < k - 1:
                        t = g.logits(h[:, -1]).argmax(-1)
                return torch.stack(toks, 1), h, past
            for j in range(L):
                h, past = g(g.embed(torch.tensor(OPT[X[:, j]], device=dev)), past)
                st, h, past = gen(h, past, NT); step_ok += int((st.cpu().numpy() == STTOK[LB[:, j]]).all(1).sum())
            h, past = g(g.embed(torch.tensor(np.repeat(ANS[None], B, 0), device=dev)), past)
            an, _, _ = gen(h, past, NT)
        ok += int((an.cpu().numpy() == STTOK[LB[:, -1]]).all(1).sum())
    return ok / n, step_ok / (n * L)


def probe_layers(g, dev, amp, n=1024, L=8, seed=13502, iters=300):
    """A2: per layer, a linear 120-way probe (fit half the problems, score the other half) on the last token of each step's state text."""
    fam = Family("s5"); s0, x, lab = make_batch(fam, np.random.default_rng(seed), n, L); ids, _, ends = build(s0, x, lab)
    H = []
    with torch.no_grad():
        for i in range(0, n, 128):
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                hl = hidden_layers(g, torch.tensor(ids[i:i + 128], device=dev))
            e = torch.tensor(ends[i:i + 128], device=dev); r = torch.arange(len(e), device=dev)[:, None]
            H.append(torch.stack([h_.float()[r, e] for h_ in hl], 0).cpu())    # (layers, b, L, D)
    H = torch.cat(H, 1); y = torch.tensor(lab).reshape(-1); half = n // 2; acc = {}
    for li in range(H.shape[0]):
        Z = H[li].reshape(-1, H.shape[-1]).to(dev); mu, sd = Z[: half * L].mean(0), Z[: half * L].std(0) + 1e-5; Z = (Z - mu) / sd
        W = torch.nn.Linear(Z.shape[1], 120).to(dev); opt = torch.optim.Adam(W.parameters(), lr=1e-2); yy = y.to(dev)
        for _ in range(iters):
            opt.zero_grad(); F.cross_entropy(W(Z[: half * L]), yy[: half * L]).backward(); opt.step()
        with torch.no_grad():
            acc[str(li)] = float((W(Z[half * L:]).argmax(-1) == yy[half * L:]).float().mean())
    best = max(acc.values()); chosen = max(int(k) for k, v in acc.items() if v == best)   # ties -> the later layer
    return {"probe_acc_by_layer": acc, "chosen_layer": chosen, "chosen_acc": best, "n": n, "L": L, "position": "last token of each step's state text"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--Lmax", type=int, default=32); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval_n", type=int, default=1000); ap.add_argument("--eval_L", type=int, default=32); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--probe_n", type=int, default=1024); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = a.device; os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    json.dump({**vars(a), "format": "start STTOK + per step (OPTOK, STTOK) + ' a:' + STTOK", "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
               "train_driver": driver_version(), **env_info()}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    g = s5.load_gpt2(dev); g.train(); opt = torch.optim.AdamW(g.parameters(), lr=a.lr); fam = Family("s5")
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / 100) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))))
        for gr in opt.param_groups:
            gr["lr"] = lr
        L = int(rng.integers(1, a.Lmax + 1)); s0, x, lab = make_batch(fam, rng, a.bs, L); ids, msk, _ = build(s0, x, lab)
        I = torch.tensor(ids, device=dev); M = torch.tensor(msk, device=dev)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            h, _ = g(g.embed(I[:, :-1])); lg = g.logits(h).float()
        loss = F.cross_entropy(lg[M[:, 1:]], I[:, 1:][M[:, 1:]])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0); opt.step(); hist.append(float(loss))
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)), "L": L, "lr": lr, "sec": round(time.time() - t0)}; hist = []
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    torch.save(g.state_dict(), os.path.join(a.out, "model.pt")); g.eval(); t1 = time.time()
    acc, step_acc = greedy_acc(g, dev, amp, a.eval_n, a.eval_L, seed=13501)
    res = {"acc_L32": acc, "step_acc_L32": step_acc, "eval_n": a.eval_n, "eval_L": a.eval_L, "gate_A1": acc >= 0.99, "train_sec": round(t1 - t0),
           "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "eval_dtype": str(amp)}
    pr = probe_layers(g, dev, amp, n=a.probe_n); pr["gate_A2"] = pr["chosen_acc"] >= 0.95; res["eval_sec"] = round(time.time() - t1)
    json.dump(pr, open(os.path.join(a.out, "teacher_probe.json"), "w", newline="\n"), indent=1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps(res), "PROBE", json.dumps({k: pr[k] for k in ("chosen_layer", "chosen_acc", "gate_A2")}), flush=True)


if __name__ == "__main__":
    main()
