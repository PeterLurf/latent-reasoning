"""THEORY §86 (latent restarts; Literature M1): §71's written state replaced by one latent slot per op, snapped to a
120-entry codebook every k ops, the history cut between chunks.
Model: GPT-2 small (pretrained base, fine-tuned), a codebook of 120 learned embeddings (one per S5 element, init from
wte statistics as §71's tagger) and a linear head 768 -> 120. A chunk is
    [code_emb(start)] [op_1 tokens " perm p1 .. p5"] [slot_1] [op_2 tokens] [slot_2] ... [op_k tokens] [slot_k]
where slot j's INPUT embedding is GPT-2's final hidden state at the last op-j token (Coconut-style continuous thought;
computed segment by segment with the KV cache, gradients flow through the chain), and the head on slot j's OUTPUT hidden
state is trained with cross-entropy against the true state after op j (all positions; the only loss; no text state
tokens). Snap: the next chunk starts from code_emb(argmax head at slot k); history cut.
Training (§71's budget and schedule): chunks cut at random positions from the S5=128 training sequences, start code =
the true label before the chunk (identity at position 0); 8k steps, bs 64, AdamW lr 1e-4 wd 0.01, 100-step warmup +
cosine; bf16 autocast on compute capability >= 8 (fp32 otherwise), as boxes/s5.py.
Eval (eval.json): all-position rollout on the §71 test set (2,000 x 512) -> cumulative accuracy at 128/256/512, final
state; the 2,048 extension (first 500 of S5=2048.csv); k = 1: certificate over all 14,400 (code, op); k = 4: chunk check
on 100k fresh (state, chunk) inputs (seed 7104 as §71); C1 removal at k = 4 on 10k fresh chunks: the inputs of the
intermediate slots 1..k-1 replaced by zeros, and separately by the dataset mean of slot inputs (2,000 real chunks),
and (note 86-2, descriptive) segment isolation (each later segment sees only its slot, fresh cache), reporting per-slot
accuracy and the accuracy of states after the first ablated slot (slots 2..k); C3: per-512-problem
tokens processed, FLOPs (boxes.core.Meter accounting, head 768 x 120) and sequential passes, beside §71's WC0_k.
    python -m boxes.latent86 train --k 4 --seed 1 --steps 8000 --bs 64 --out boxes/runs_s5/l86_k4_s1 [--device cuda:0]
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

OPTOK = {e: bc.tok(" " + s5.s_perm(e)) for e in range(120)}             # " perm p1 p2 p3 p4 p5" -> 6 tokens
assert len({len(v) for v in OPTOK.values()}) == 1
NOP = len(OPTOK[0])


class LatentRestart(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g
        mu, sd = g.w("wte.weight").mean(0).detach(), g.w("wte.weight").std(0).detach()
        self.code = nn.Parameter(mu + sd * torch.randn(120, 768, device=mu.device))
        self.head = nn.Linear(768, 120)
        self.optok = None

    def _ops(self, ops, dev):
        if self.optok is None or self.optok.device != torch.device(dev):
            self.optok = torch.tensor([OPTOK[e] for e in range(120)], device=dev)
        return self.optok[ops]                                            # (B, NOP)

    def chunk(self, starts, ops, ablate=None, mean_in=None, meter=None):
        """starts (B,) element ids, ops (B, k) -> head logits (B, k, 120), slot inputs (B, k, D).
        ablate: None | "zero" | "mean" applied to the inputs of slots 1..k-1; "isolate" (note 86-2, descriptive): every
        segment after the first ([slot_j, op_(j+1) tokens], and the last slot alone) runs with a fresh cache, so it sees
        only slot_j (no earlier op tokens): a continuous restart at every op, without snapping."""
        B, k = ops.shape; dev = ops.device
        x = torch.cat([self.code[starts][:, None], self.g.embed(self._ops(ops[:, 0], dev))], 1)
        past = None; logits, ins = [], []
        for j in range(k):
            if ablate == "isolate":
                past = None
            ctx = 0 if past is None else past[0][0].shape[2]
            h, past = self.g(x, past)
            if meter:
                meter.step(x.shape[1], ctx, 0)
            if j > 0:
                logits.append(self.head(h[:, 0]))                         # output at slot j (the first position of x)
            s_in = h[:, -1]                                               # hidden at the last op-(j+1) token -> slot j+1 input
            if ablate in ("zero", "mean") and j < k - 1:
                s_in = torch.zeros_like(s_in) if ablate == "zero" else mean_in[j][None].expand_as(s_in)
            ins.append(s_in)
            nxt = [s_in[:, None]]
            if j + 1 < k:
                nxt.append(self.g.embed(self._ops(ops[:, j + 1], dev)))
            x = torch.cat(nxt, 1)
        if ablate == "isolate":
            past = None
        ctx = 0 if past is None else past[0][0].shape[2]
        h, past = self.g(x, past)                                         # the last slot alone
        if meter:
            meter.step(1, ctx, 0); meter.flops += 2 * 768 * 120 * k
        logits.append(self.head(h[:, 0]))
        return torch.stack(logits, 1), torch.stack(ins, 1)


def batch(rng, X, Y, k, B):
    s = rng.integers(0, len(X), B); j = rng.integers(0, X.shape[1] - k + 1, B)
    starts = np.array([0 if jj == 0 else Y[ss, jj - 1] for ss, jj in zip(s, j)])
    ops = np.stack([X[ss, jj:jj + k] for ss, jj in zip(s, j)]); tgt = np.stack([Y[ss, jj:jj + k] for ss, jj in zip(s, j)])
    return starts, ops, tgt


@torch.no_grad()
def chunks_pred(m, starts, ops, dev, amp, ablate=None, mean_in=None, meter=None):
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        lg, _ = m.chunk(torch.tensor(starts, device=dev), torch.tensor(ops, device=dev), ablate, mean_in, meter)
    return lg.argmax(-1).cpu().numpy()


@torch.no_grad()
def rollout(m, X, k, dev, amp, bs):
    N, L = X.shape; pred = np.full((N, L), -1, dtype=np.int64); meter = bc.Meter(m.g)
    for s in range(0, N, bs):
        cur = np.zeros(min(bs, N - s), dtype=np.int64)
        for c in range(0, L, k):
            o = X[s:s + bs, c:c + k]
            p = chunks_pred(m, cur, o, dev, amp, meter=meter if s == 0 else None)
            pred[s:s + bs, c:c + o.shape[1]] = p; cur = p[:, -1]          # snap to the decoded element
    d = meter.as_dict(); d["latent_slots"] = L; d["written_tokens"] = 0; d["gen_tokens"] = 0
    return pred, d                                                     # Meter counts per problem (one row's calls)


@torch.no_grad()
def certificate_k1(m, dev, amp, bs):
    S, O = [a.ravel() for a in np.meshgrid(np.arange(120), np.arange(120), indexing="ij")]; bad = 0
    for i in range(0, len(S), bs):
        p = chunks_pred(m, S[i:i + bs], O[i:i + bs, None], dev, amp)
        bad += int((p[:, 0] != s5.COMP[S[i:i + bs], O[i:i + bs]]).sum())
    return {"k": 1, "inputs": len(S), "errors": bad, "certified": bad == 0}


@torch.no_grad()
def chunk_check(m, k, dev, amp, bs, n=100_000, seed=7104, ablate=None, mean_in=None):
    rng = np.random.default_rng(seed); S = rng.integers(0, 120, n); O = rng.integers(0, 120, (n, k))
    ok = 0; per = np.zeros(k)
    for i in range(0, n, bs):
        p = chunks_pred(m, S[i:i + bs], O[i:i + bs], dev, amp, ablate, mean_in)
        w = S[i:i + bs].copy(); good = np.ones(len(w), bool)
        for t in range(k):
            w = s5.COMP[w, O[i:i + bs, t]]; c = p[:, t] == w; per[t] += c.sum(); good &= c
        ok += int(good.sum())
    r = {"k": k, "n": n, "per_chunk_acc": ok / n, "per_slot_acc": (per / n).tolist(), "pred_seq_acc_512": (ok / n) ** (512 / k)}
    if ablate:
        r["ablate"] = ablate; r["acc_after_first_ablated_slot"] = float(per[1:].sum() / (n * (k - 1)))
    return r


@torch.no_grad()
def mean_slot_inputs(m, X, Y, k, dev, amp, n=2000, seed=7105):
    rng = np.random.default_rng(seed); st, o, _ = batch(rng, X, Y, k, n); out = []
    for i in range(0, n, 250):
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            _, ins = m.chunk(torch.tensor(st[i:i + 250], device=dev), torch.tensor(o[i:i + 250], device=dev))
        out.append(ins.float())
    return torch.cat(out).mean(0)                                         # (k, D)


def train(a):
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    s5.check_interface()
    Xtr, Ytr = s5.load_csv(os.path.join(a.data_dir, "S5=128.csv")); s5.check_labels(Xtr, Ytr)
    Xte, Yte = s5.load_csv(os.path.join(a.data_dir, "S5=512.csv"), limit=a.test_n); s5.check_labels(Xte, Yte)
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), "train_sequences": len(Xtr), "op_tokens": NOP}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    m = LatentRestart(s5.load_gpt2(dev)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []
    for step in range(1, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        st, o, tg = batch(rng, Xtr, Ytr, a.k, a.bs)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            lg, _ = m.chunk(torch.tensor(st, device=dev), torch.tensor(o, device=dev))
            loss = F.cross_entropy(lg.reshape(-1, 120).float(), torch.tensor(tg, device=dev).reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(loss.item())
        if step % 100 == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)), "sec": round(time.time() - t0)}; hist = []
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    train_sec = round(time.time() - t0)
    torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    m.eval(); res = {"k": a.k, "seed": a.seed, "train_sec": train_sec, "eval_dtype": str(amp), "test_sequences": len(Xte)}
    t1 = time.time(); pred, comp = rollout(m, Xte, a.k, dev, amp, a.eval_bs)
    np.save(os.path.join(a.out, "preds_test.npy"), pred); cum = s5.cumulative(pred, Yte)
    res["cumulative"] = {str(t): float(cum[t - 1]) for t in (128, 256, 512)}
    res["final_state_acc_512"] = float((pred[:, -1] == Yte[:, -1]).mean()); res["compute_512_per_problem"] = comp
    if a.ext_n:
        Xx, Yx = s5.load_csv(os.path.join(a.data_dir, "S5=2048.csv"), limit=a.ext_n); s5.check_labels(Xx, Yx)
        px, _ = rollout(m, Xx, a.k, dev, amp, a.eval_bs); cx = s5.cumulative(px, Yx)
        res["extension_2048"] = {"n": len(Xx), "cumulative": {str(t): float(cx[t - 1]) for t in (512, 1024, 2048)}}
    if a.k == 1:
        res["certificate"] = certificate_k1(m, dev, amp, a.eval_bs)
    else:
        res["chunk_check"] = chunk_check(m, a.k, dev, amp, a.eval_bs, n=a.check_n)
        mu = mean_slot_inputs(m, Xtr, Ytr, a.k, dev, amp)
        res["removal"] = {ab: chunk_check(m, a.k, dev, amp, a.eval_bs, n=min(10_000, a.check_n), seed=7106, ablate=ab, mean_in=mu) for ab in ("zero", "mean", "isolate")}
    res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({kk: res[kk] for kk in ("cumulative", "final_state_acc_512")}), flush=True); print("DONE", a.out, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--k", type=int, default=1); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--eval_bs", type=int, default=500)
    ap.add_argument("--test_n", type=int, default=s5.TEST_N); ap.add_argument("--ext_n", type=int, default=500); ap.add_argument("--check_n", type=int, default=100_000, help="< 100k: smoke only")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "bench", "data", "s5"))
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
