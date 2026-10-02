"""THEORY §71: S5 under DeltaProduct's protocol (P1) with GPT-2's written-state chunked loop.

Elements: DeltaProduct's generator (abstract_algebra generate_symmetric_group(5)) indexes the 120 elements as the
permutations of (1..5) in itertools order (index 0 = identity); its label at position t is op(label_{t-1}, x_t) with
op(a, b)[i] = a[b[i]] (1-based one-line notation). Text: the state is the arrangement of letters A-E in 5 boxes
(identity "ABCDE"); an op "perm p1 .. p5" means box i receives what box p_i held. Then the arrangement after ops
x_1..x_t equals letters[label_t[i] - 1]; `check_interface` proves this for all 14,400 (state, op) pairs and
`check_labels` asserts it on every training position (required before training).

Arms (this file): WC0_k all-position (`wc0all`), the GPT-2 tagger (`tagger`), P3 final-only WC0_4 / WC_4 (`p3`).
    python -m boxes.s5 train --arm wc0all --k 1 --seed 1 --steps S --bs B --out boxes/runs_s5/<name> [--device cuda:0]
    python -m boxes.s5 train --arm tagger --seed 1 ...
    python -m boxes.s5 train --arm p3 --latent 0|4 --k 4 --seed 1 ...
Each run writes model.pt, args.json, log.jsonl, preds_test.npy (2,000 x 512 element predictions, -1 = invalid
arrangement), eval.json (cumulative accuracy at 128/256/512, final-state accuracy at 512, compute at 512), and for
wc0all the certificate (k = 1, 2) or the per-chunk check (k = 4). Data: S5=128.csv / S5=512.csv / S5=2048.csv from
DeltaProduct's generator (--data_dir).
"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import core as bc  # noqa: E402

assert bc.NB == 5 and bc.LET == "ABCDE", "boxes.core must be imported with BOXES_NB=5 (import boxes.s5 first)"
LET = bc.LET
ELEMS = list(itertools.permutations(range(1, 6)))                     # DeltaProduct / abstract_algebra order
INDEX = {e: i for i, e in enumerate(ELEMS)}
COMP = np.array([[INDEX[tuple(a[b[i] - 1] for i in range(5))] for b in ELEMS] for a in ELEMS])   # COMP[a, b] = op(a, b)
GT = bc.tok(" >")[0]
TEST_N = 2000


def arr_of(e):
    return "".join(LET[x - 1] for x in ELEMS[e])


ARR = [arr_of(e) for e in range(120)]
ARR_INDEX = {a: i for i, a in enumerate(ARR)}


def apply_perm(arr, e):
    return "".join(arr[p - 1] for p in ELEMS[e])


def s_arr(a):
    return " ".join(a)


def s_perm(e):
    return "perm " + " ".join(map(str, ELEMS[e]))


def chunk_prompt(start, ops, think=False):
    return f"start: {s_arr(start)}. ops: {', '.join(s_perm(o) for o in ops)}." + (" think:" if think else " a:")


def chunk_target(states):
    return "".join(" > " + s_arr(s) for s in states)


def check_interface():
    """Exhaustive: the text op applied to element a's arrangement equals the arrangement of op(a, b), for all 14,400
    pairs; identity first; token layout of a target is [' >', 5 letters] per state."""
    assert ELEMS[0] == (1, 2, 3, 4, 5) and ARR[0] == "ABCDE"
    for a in range(120):
        for b in range(120):
            assert apply_perm(ARR[a], b) == ARR[COMP[a, b]], (a, b)
    t = bc.tok(chunk_target(["ABCDE", "EDCBA"]))
    assert t == [GT] + bc.LET_IDS + [GT] + bc.LET_IDS[::-1], t
    return True


def load_csv(path, limit=None):
    X, Y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if limit and i >= limit:
                break
            X.append([int(v) for v in r["input"].split()]); Y.append([int(v) for v in r["target"].split()])
    return np.array(X, dtype=np.int64), np.array(Y, dtype=np.int64)


def check_labels(X, Y):
    """Required before training (§71): replaying the ops reproduces DeltaProduct's label at every position."""
    acc = np.zeros(len(X), dtype=np.int64)
    for t in range(X.shape[1]):
        acc = COMP[acc, X[:, t]]
        assert (acc == Y[:, t]).all(), f"label mismatch at position {t}"
    return True


def cumulative(pred, Y):
    """DeltaProduct's cumulative sequence accuracy: at t, the share of sequences whose positions 1..t are all correct."""
    ok = (pred == Y)
    return np.cumprod(ok, axis=1).mean(0)


# ---------------------------------------------------------------- model
def load_gpt2(dev):
    sd = torch.load(os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), map_location="cpu", weights_only=True)
    return bc.GPT2KV(sd).to(dev)


class Tagger(nn.Module):
    """Li et al.'s interface: 120 new element tokens, a 120-way label at every position (pretrained GPT-2 body)."""
    def __init__(self, g):
        super().__init__()
        self.g = g
        mu, sd = g.w("wte.weight").mean(0).detach(), g.w("wte.weight").std(0).detach()
        self.emb = nn.Parameter(mu + sd * torch.randn(120, 768))
        self.head = nn.Linear(768, 120)

    def forward(self, x):
        h, _ = self.g(self.emb[x])
        return self.head(h)


# ---------------------------------------------------------------- WC0 all-position: batches, inference, certificates
def wc_batch(rng, X, Y, k, B):
    s = rng.integers(0, len(X), B); j = rng.integers(0, X.shape[1] - k + 1, B)
    pre, tgt = [], []
    for si, ji in zip(s, j):
        start = 0 if ji == 0 else Y[si, ji - 1]
        pre.append(bc.tok(chunk_prompt(ARR[start], X[si, ji:ji + k])))
        tgt.append(bc.tok(chunk_target([ARR[e] for e in Y[si, ji:ji + k]])))
    return pre, tgt


def wc_loss(m, pre, tgt, dev):
    ids = torch.tensor([a + b for a, b in zip(pre, tgt)], device=dev)
    h, _ = m(m.embed(ids[:, :-1])); n = len(pre[0])
    lg = m.logits(h[:, n - 1:])
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, n:].reshape(-1))


@torch.no_grad()
def wc_chunks(m, starts, ops, dev, amp, meter=None, think=0):
    """One chunk per row: starts (B,) element ids, ops (B, k). Returns (B, k) predicted element ids (-1 = invalid);
    greedy, letters only, the ' >' separators fed as given."""
    B, k = ops.shape
    pre = torch.tensor([bc.tok(chunk_prompt(ARR[s], o, think=think > 0)) for s, o in zip(starts, ops)], device=dev)
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        h, past = bc.run_prefix(m, pre, meter=meter); hl = h[:, -1]
        if think:
            hl, past = bc.latents(m, hl, past, think, meter)
            mid = torch.tensor([bc.tok(" done. a:")] * B, device=dev); h, past = bc.run_prefix(m, mid, past, meter); hl = h[:, -1]
        out = []
        for _ in range(k):
            ctx = past[0][0].shape[2]; h, past = m(m.embed(torch.full((B,), GT, device=dev))[:, None, :], past)
            if meter:
                meter.step(1, ctx, 1); meter.written += 1
            hl = h[:, -1]
            ids, hl, past = bc.gen_feed(m, hl, past, 5, meter, bc.LET_IDS)
            out.append([ARR_INDEX.get(bc.dec_arr(r), -1) for r in ids.tolist()])
    return np.array(out).T


@torch.no_grad()
def wc_rollout(m, X, k, dev, amp, bs, think=0):
    """All-position chunked inference from the identity: (N, L) predictions; the history is cut between chunks."""
    N, L = X.shape; pred = np.full((N, L), -1, dtype=np.int64); meter = bc.Meter(m)
    for s in range(0, N, bs):
        cur = np.zeros(min(bs, N - s), dtype=np.int64)
        for c in range(0, L, k):
            o = X[s:s + bs, c:c + k]
            p = wc_chunks(m, np.where(cur < 0, 0, cur), o, dev, amp, meter if s == 0 else None, think)
            pred[s:s + bs, c:c + o.shape[1]] = p
            cur = p[:, -1]                                            # an invalid state restarts from A..E (counted wrong)
    return pred, meter.as_dict()


@torch.no_grad()
def certificate(m, k, dev, amp, bs):
    """k = 1: all 14,400 (state, op) inputs; k = 2: all 1,728,000 two-op chunks + 14,400 one-op tails. Certified iff
    every output equals the true arrangement(s)."""
    res = {"k": k, "batch": bs, "dtype": str(amp)}
    states = np.arange(120)
    if k == 1:
        S, O = np.meshgrid(states, np.arange(120), indexing="ij"); S, O = S.ravel(), O.ravel()
        bad = 0
        for i in range(0, len(S), bs):
            p = wc_chunks(m, S[i:i + bs], O[i:i + bs, None], dev, amp)
            bad += int((p[:, 0] != COMP[S[i:i + bs], O[i:i + bs]]).sum())
        res.update(inputs=len(S), errors=bad, certified=bad == 0)
        return res
    S, A, B2 = [a.ravel() for a in np.meshgrid(states, np.arange(120), np.arange(120), indexing="ij")]
    bad = 0; t0 = time.time()
    for i in range(0, len(S), bs):
        s, a, b = S[i:i + bs], A[i:i + bs], B2[i:i + bs]
        p = wc_chunks(m, s, np.stack([a, b], 1), dev, amp)
        w1 = COMP[s, a]; w2 = COMP[w1, b]
        bad += int(((p[:, 0] != w1) | (p[:, 1] != w2)).sum())
    S1, O1 = [a.ravel() for a in np.meshgrid(states, np.arange(120), indexing="ij")]; tail_bad = 0
    for i in range(0, len(S1), bs):
        p = wc_chunks(m, S1[i:i + bs], O1[i:i + bs, None], dev, amp)
        tail_bad += int((p[:, 0] != COMP[S1[i:i + bs], O1[i:i + bs]]).sum())
    res.update(inputs=len(S), errors=bad, tail_inputs=len(S1), tail_errors=tail_bad, certified=bad == 0 and tail_bad == 0,
               sec=round(time.time() - t0))
    return res


@torch.no_grad()
def chunk_check(m, k, dev, amp, bs, n=100_000, seed=7104, think=0):
    """k = 4: per-chunk accuracy p on n fresh random (state, chunk) inputs; prediction p^(512/k)."""
    rng = np.random.default_rng(seed); S = rng.integers(0, 120, n); O = rng.integers(0, 120, (n, k)); ok = 0
    for i in range(0, n, bs):
        p = wc_chunks(m, S[i:i + bs], O[i:i + bs], dev, amp, think=think)
        w = S[i:i + bs].copy(); good = np.ones(len(w), bool)
        for t in range(k):
            w = COMP[w, O[i:i + bs, t]]; good &= p[:, t] == w
        ok += int(good.sum())
    p = ok / n
    return {"k": k, "n": n, "per_chunk_acc": p, "pred_seq_acc_512": p ** (512 / k), "batch": bs, "dtype": str(amp)}


# ---------------------------------------------------------------- P3 final-only (random start, k' <= k ops, final arrangement)
def p3_batch(rng, k, B):
    kk = int(rng.integers(1, k + 1)); s = rng.integers(0, 120, B); o = rng.integers(0, 120, (B, kk))
    pre = [bc.tok(chunk_prompt(ARR[a], b)) for a, b in zip(s, o)]
    fin = s.copy()
    for t in range(kk):
        fin = COMP[fin, o[:, t]]
    return s, o, fin


def p3_loss(m, s, o, fin, think, dev):
    B = len(s)
    pre = torch.tensor([bc.tok(chunk_prompt(ARR[a], b, think=think > 0)) for a, b in zip(s, o)], device=dev)
    h, past = m(m.embed(pre)); hl = h[:, -1]
    if think:
        hl, past = bc.latents(m, hl, past, min(think, o.shape[1]))
        mid = torch.tensor([bc.tok(" done. a:")] * B, device=dev); h2, past = m(m.embed(mid), past); hl = h2[:, -1]
    tgt = torch.tensor([bc.tok(" > " + s_arr(ARR[f])) for f in fin], device=dev)
    h3, _ = m(m.embed(tgt[:, :-1]), past)
    lg = m.logits(torch.cat([hl[:, None], h3], 1))
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))


@torch.no_grad()
def p3_rollout(m, X, k, dev, amp, bs, think):
    """Final-only chunked inference: each chunk writes only its end state (§66 WC/WC0); final state at L."""
    N, L = X.shape; fin = np.full(N, -1)
    for s in range(0, N, bs):
        cur = np.zeros(min(bs, N - s), dtype=np.int64)
        for c in range(0, L, k):
            o = X[s:s + bs, c:c + k]
            p = wc_chunks_final(m, np.where(cur < 0, 0, cur), o, dev, amp, think)
            cur = p
        fin[s:s + bs] = cur
    return fin


@torch.no_grad()
def wc_chunks_final(m, starts, ops, dev, amp, think):
    B, k = ops.shape
    pre = torch.tensor([bc.tok(chunk_prompt(ARR[s], o, think=think > 0)) for s, o in zip(starts, ops)], device=dev)
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        h, past = bc.run_prefix(m, pre); hl = h[:, -1]
        if think:
            hl, past = bc.latents(m, hl, past, min(think, k))
            mid = torch.tensor([bc.tok(" done. a:")] * B, device=dev); h, past = bc.run_prefix(m, mid, past); hl = h[:, -1]
        h, past = m(m.embed(torch.full((B,), GT, device=dev))[:, None, :], past); hl = h[:, -1]
        ids = bc.greedy(m, hl, past, 5, None, bc.LET_IDS)
    return np.array([ARR_INDEX.get(bc.dec_arr(r), -1) for r in ids.tolist()])


# ---------------------------------------------------------------- train / evaluate
def train(a):
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.is_bf16_supported() else None
    check_interface()
    Xtr, Ytr = load_csv(os.path.join(a.data_dir, "S5=128.csv"))
    check_labels(Xtr, Ytr)
    Xte, Yte = load_csv(os.path.join(a.data_dir, "S5=512.csv"), limit=a.test_n)
    check_labels(Xte, Yte)
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), "train_sequences": len(Xtr)}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    g = load_gpt2(dev)
    m = Tagger(g).to(dev) if a.arm == "tagger" else g
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []; states_seen = 0
    for step in range(1, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            if a.arm == "wc0all":
                pre, tgt = wc_batch(rng, Xtr, Ytr, a.k, a.bs); loss = wc_loss(m, pre, tgt, dev); states_seen += a.bs * a.k
            elif a.arm == "tagger":
                idx = rng.integers(0, len(Xtr), a.bs); x = torch.tensor(Xtr[idx], device=dev); y = torch.tensor(Ytr[idx], device=dev)
                loss = F.cross_entropy(m(x).reshape(-1, 120), y.reshape(-1)); states_seen += a.bs * Xtr.shape[1]
            else:
                s, o, fin = p3_batch(rng, a.k, a.bs); loss = p3_loss(m, s, o, fin, a.latent, dev); states_seen += a.bs
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        hist.append(loss.item())
        if step % 100 == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)), "labelled_states": states_seen, "sec": round(time.time() - t0)}; hist = []
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    train_sec = round(time.time() - t0)
    torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    m.eval(); res = {"arm": a.arm, "k": a.k, "latent": a.latent, "train_sec": train_sec, "labelled_states_seen": states_seen,
                     "eval_batch": a.eval_bs, "eval_dtype": str(amp), "test_sequences": len(Xte)}
    t1 = time.time()
    if a.arm == "wc0all":
        pred, comp = wc_rollout(m, Xte, a.k, dev, amp, a.eval_bs); res["compute_512_per_problem"] = comp
    elif a.arm == "tagger":
        pred = np.concatenate([m(torch.tensor(Xte[i:i + 50], device=dev)).argmax(-1).cpu().numpy() for i in range(0, len(Xte), 50)])
    else:
        fin = p3_rollout(m, Xte, a.k, dev, amp, a.eval_bs, a.latent)
        pred = np.full_like(Yte, -1); pred[:, -1] = fin
    res["eval_sec"] = round(time.time() - t1)
    np.save(os.path.join(a.out, "preds_test.npy"), pred)
    cum = cumulative(pred, Yte)
    res["cumulative"] = {str(t): float(cum[t - 1]) for t in (128, 256, 512)}
    res["final_state_acc_512"] = float((pred[:, -1] == Yte[:, -1]).mean())
    res["invalid_share"] = float((pred < 0).mean())
    if a.arm == "wc0all" and not a.no_cert:
        if a.k in (1, 2):
            res["certificate"] = certificate(m, a.k, dev, amp, a.eval_bs)
        else:
            res["chunk_check"] = chunk_check(m, a.k, dev, amp, a.eval_bs)
    if a.arm == "p3":
        res["chunk_check_final"] = None
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({k: res[k] for k in ("cumulative", "final_state_acc_512")}), res.get("certificate"), res.get("chunk_check"), flush=True)
    print("DONE", a.out, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "check"])
    ap.add_argument("--arm", choices=["wc0all", "tagger", "p3"], default="wc0all")
    ap.add_argument("--k", type=int, default=1); ap.add_argument("--latent", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_bs", type=int, default=500)
    ap.add_argument("--test_n", type=int, default=TEST_N, help="smoke only; §71 uses the first 2,000")
    ap.add_argument("--no_cert", action="store_true", help="smoke only")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "bench", "data", "s5"))
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "check":
        check_interface(); X, Y = load_csv(os.path.join(a.data_dir, "S5=128.csv")); check_labels(X, Y)
        Xt, Yt = load_csv(os.path.join(a.data_dir, "S5=512.csv")); check_labels(Xt, Yt)
        print("CHECK OK", X.shape, Xt.shape); return
    train(a)


if __name__ == "__main__":
    main()
