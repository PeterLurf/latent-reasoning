"""THEORY §97 (note 97-1): a general latent thinker, step 1. §92's latent restarts (boxes/latent86.py, 86-4 recipe) with
the hand-built state removed: the per-op target is the TEXT of the CoT step, decoded with GPT-2's tied LM head over the
full vocabulary from a forked KV cache, and the restart is the decoded text re-embedded with wte.
Arms (--arm, --k):
  gen  GEN-k: chunk = [wte(start text " > A B C D E")] [op_1 tokens] [slot_1] ... [op_k tokens] [slot_k]; slot j's input is
       the final hidden at op j's last token (as §92). Loss (the only one): at every slot j, fork the cache after slot j,
       teacher-force the 6 step-text tokens of the state after op j, CE with the tied LM head (summed over the 6 tokens,
       mean over rows and slots, as 86-4 sums over boxes). The fork is a side branch; the chain continues from slot j.
       Test: greedy-decode 6 tokens at every slot (per-position prediction: exact token match to the true state's
       text); snap every k: the 6 tokens decoded at slot k are re-embedded with wte as the next start, history cut, no
       validity check. Training starts from the true state's text (identity at position 0).
  hyb  HYB-k (note 97-1): GEN's text target (fork, tied head) with §92's structured codebook start (86-4's W[box, letter]
       sum); training uses the true start code; at test the codebook letters come from the decoded text (tokens 2..6;
       a non-letter token maps to letter A and the position is already wrong; a non-permutation still restarts, as 86-4).
Segments run [start, op_1] [slot_1] [op_2] [slot_2] ... (the same token sequence as latent86, split so the cache after
slot j ends at slot j for the fork).
Eval (eval.json; key names as latent86 where shared): cumulative 128/256/512 on §71's test set; extension_2048 (first
500 of S5=2048); chunk_check (100k fresh chunks, random valid start texts) per-slot decode accuracy; k = 1: certificate
over all 14,400 (state text, op); k > 1: removal.{zero,mean} (slot inputs 1..k-1 ablated, ops and start visible; decode
accuracy after the first ablated slot = also the fork-bypass control); checkpoint_validity (share of snapped texts in
the test rollout that are a valid arrangement); wte_drift (cosine, initial vs final wte rows of " >" and " A".." E");
edit_iia (descriptive: after a first chunk, its decoded checkpoint is replaced by another valid state's text and one
more chunk is run; share whose checkpoint equals the true composition from the edited state; with the history cut this
equals the chunk accuracy from that state); compute_512_per_problem: the method's FLOPs (every processed token incl.
the start re-embedding, the slots, and the snap decodes: 6 LM-head rows + 5 decode tokens per snap); the decodes at
non-snap slots are measurement only, reported as measurement_decode_flops.
    python -m boxes.gen97 train --arm gen --k 4 --seed 1 --steps 24000 --bs 64 --out boxes/runs_s5/g97_gen4_s1
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
from boxes.latent86 import CODE2EL, LETIDX, OPTOK, batch  # noqa: E402

STTOK = np.array([bc.tok(s5.chunk_target([s5.ARR[e]])) for e in range(120)])  # (120, 6): " >" + 5 letters
NST = STTOK.shape[1]
assert NST == 6 and len({tuple(r) for r in STTOK}) == 120 and (STTOK[:, 0] == s5.GT).all()
LETTOK = [bc.tok(" " + c)[0] for c in "ABCDE"]
TOK2LET = np.full(max(50257, max(LETTOK) + 1), -1, dtype=np.int64)
for _i, _t in enumerate(LETTOK):
    TOK2LET[_t] = _i


def decode_to_el(toks):
    """toks (B, 6) int -> element ids (-1 = not exactly a valid state's text), letters (B, 5) (non-letters -> 0)."""
    let = TOK2LET[toks[:, 1:]]; ok = (toks[:, 0] == s5.GT) & (let >= 0).all(1)
    L = np.where(let >= 0, let, 0); el = CODE2EL[(L * (5 ** np.arange(5))).sum(1)]
    return np.where(ok, el, -1), L


class Gen97(nn.Module):
    def __init__(self, g, arm):
        super().__init__()
        self.g = g; self.arm = arm
        self.register_buffer("sttok", torch.tensor(STTOK)); self.register_buffer("optok", torch.tensor([OPTOK[e] for e in range(120)]))
        if arm == "hyb":                                                   # 86-4's structured codebook (start only)
            mu, sd = g.w("wte.weight").mean(0).detach(), g.w("wte.weight").std(0).detach()
            self.W = nn.Parameter(mu / 5 + sd / math.sqrt(5) * torch.randn(5, 5, 768, device=mu.device))
            self.register_buffer("letidx", torch.tensor(LETIDX))

    def start_emb(self, start):
        """gen: start (B, 6) token ids -> (B, 6, D). hyb: (B,) element ids or (B, 5) letters -> (B, 1, D)."""
        if self.arm == "gen":
            return self.g.embed(start)
        L = self.letidx[start] if start.dim() == 1 else start
        return self.W[torch.arange(5, device=L.device)[None], L].sum(1)[:, None]

    def greedy(self, hslot, past, meter=None):
        """6 greedy tokens from the slot's output hidden and the cache ending at the slot (a fork; past is not changed)."""
        t = self.g.logits(hslot).argmax(-1); out = [t]
        if meter:
            meter.step(0, 0, 1)
        past, pos = self.fork_past(past)
        for _ in range(NST - 1):
            ctx = past[0][0].shape[2]; h, past = self.g(self.g.embed(t)[:, None], past, pos0=pos)
            if meter:
                meter.step(1, ctx, 1)
            pos = None if pos is None else pos + 1
            t = self.g.logits(h[:, 0]).argmax(-1); out.append(t)
        return torch.stack(out, 1)

    def fork_past(self, past):
        """§105 (--fork mask, §98's mask): a fork sees only the slot's own cached K/V (all layers) and its own tokens, at
        their true positions. Default: the whole cache."""
        if not getattr(self, "mask", False):
            return past, None
        P = past[0][0].shape[2]
        return [(k[:, :, -1:], v[:, :, -1:]) for k, v in past], P

    def chunk(self, start, ops, tgt=None, decode="all", ablate=None, mean_in=None, meter=None, meter_x=None):
        """start: see start_emb; ops (B, k) element ids; tgt (B, k) element ids (fork CE) or None. decode: "all" | "last"
        | None. Returns (loss or None, decoded tokens (B, k', 6) or None, slot inputs (B, k, D))."""
        B, k = ops.shape; g = self.g
        x = torch.cat([self.start_emb(start), g.embed(self.optok[ops[:, 0]])], 1)
        past = None; ins, losses, decs = [], [], []
        for j in range(k):
            ctx = 0 if past is None else past[0][0].shape[2]
            h, past = g(x, past)                                             # op (j+1) tokens (with the start at j = 0)
            if meter:
                meter.step(x.shape[1], ctx, 0)
            s_in = h[:, -1]
            if ablate in ("zero", "mean") and j < k - 1:
                s_in = torch.zeros_like(s_in) if ablate == "zero" else mean_in[j][None].expand_as(s_in)
            ov = getattr(self, "override", None)                             # §104 donor patch (j, (B, D)); default off
            if ov is not None and ov[0] == j:
                s_in = ov[1].to(s_in.dtype)
            ins.append(s_in)
            ctx = past[0][0].shape[2]; hs, past = g(s_in[:, None], past)     # slot j+1
            if meter:
                meter.step(1, ctx, 0)
            hslot = hs[:, 0]
            if tgt is not None:                                              # fork: teacher-forced step text, side branch
                st = self.sttok[tgt[:, j]]
                fp, p0 = self.fork_past(past); hf, _ = g(g.embed(st[:, :-1]), fp, pos0=p0)
                lg = g.logits(torch.cat([hslot[:, None], hf], 1)).float()
                losses.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), st.reshape(-1), reduction="none").view(B, NST).sum(-1).mean())
            if decode == "all" or (decode == "last" and j == k - 1):
                decs.append(self.greedy(hslot, past, meter if j == k - 1 else meter_x))
            if j + 1 < k:
                x = g.embed(self.optok[ops[:, j + 1]])
        loss = torch.stack(losses).mean() if losses else None
        return loss, (torch.stack(decs, 1) if decs else None), torch.stack(ins, 1)

    def start_from(self, els=None, toks=None, lets=None, dev=None):
        """The next start: from element ids (true states / edits) or from a decode (tokens for gen, letters for hyb)."""
        if els is not None:
            return torch.tensor(STTOK[els] if self.arm == "gen" else els, device=dev)
        return torch.tensor(toks if self.arm == "gen" else lets, device=dev)


def amp_ctx(amp):
    return torch.autocast("cuda", dtype=amp, enabled=amp is not None)


@torch.no_grad()
def chunks_dec(m, start, ops, dev, amp, decode="all", ablate=None, mean_in=None, meter=None, meter_x=None):
    with amp_ctx(amp):
        _, d, _ = m.chunk(start, torch.tensor(ops, device=dev), None, decode, ablate, mean_in, meter, meter_x)
    toks = d.cpu().numpy(); B, kk, _ = toks.shape
    el, let = decode_to_el(toks.reshape(-1, NST))
    return el.reshape(B, kk), let.reshape(B, kk, 5), toks


@torch.no_grad()
def rollout(m, X, k, dev, amp, bs):
    N, L = X.shape; pred = np.full((N, L), -1, dtype=np.int64); meter = bc.Meter(m.g); meter_x = bc.Meter(m.g)
    snaps = valid = 0
    for s in range(0, N, bs):
        n = min(bs, N - s); cur = m.start_from(els=np.zeros(n, dtype=np.int64), dev=dev)       # identity at position 0
        for c in range(0, L, k):
            o = X[s:s + bs, c:c + k]
            el, let, toks = chunks_dec(m, cur, o, dev, amp, meter=meter if s == 0 else None, meter_x=meter_x if s == 0 else None)
            pred[s:s + bs, c:c + o.shape[1]] = el
            cur = m.start_from(toks=toks[:, -1], lets=let[:, -1], dev=dev)                     # snap: decoded text, history cut
            snaps += n; valid += int((el[:, -1] >= 0).sum())
    d = meter.as_dict(); d["latent_slots"] = L; d["measurement_decode_flops"] = meter_x.flops
    return pred, d, {"snaps": snaps, "valid": valid, "rate": valid / max(1, snaps)}


def true_chain(S, O):
    w = S.copy(); out = []
    for t in range(O.shape[1]):
        w = s5.COMP[w, O[:, t]]; out.append(w)
    return np.stack(out, 1)


@torch.no_grad()
def chunk_check(m, k, dev, amp, bs, n=100_000, seed=7104, ablate=None, mean_in=None):
    rng = np.random.default_rng(seed); S = rng.integers(0, 120, n); O = rng.integers(0, 120, (n, k))
    ok = 0; per = np.zeros(k)
    for i in range(0, n, bs):
        el, _, _ = chunks_dec(m, m.start_from(els=S[i:i + bs], dev=dev), O[i:i + bs], dev, amp, ablate=ablate, mean_in=mean_in)
        c = el == true_chain(S[i:i + bs], O[i:i + bs]); per += c.sum(0); ok += int(c.all(1).sum())
    r = {"k": k, "n": n, "per_chunk_acc": ok / n, "per_slot_acc": (per / n).tolist(), "pred_seq_acc_512": (ok / n) ** (512 / k)}
    if ablate:
        r["ablate"] = ablate; r["acc_after_first_ablated_slot"] = float(per[1:].sum() / (n * (k - 1)))
    return r


@torch.no_grad()
def certificate_k1(m, dev, amp, bs):
    S, O = [a.ravel() for a in np.meshgrid(np.arange(120), np.arange(120), indexing="ij")]; bad = 0
    for i in range(0, len(S), bs):
        el, _, _ = chunks_dec(m, m.start_from(els=S[i:i + bs], dev=dev), O[i:i + bs, None], dev, amp)
        bad += int((el[:, 0] != s5.COMP[S[i:i + bs], O[i:i + bs]]).sum())
    return {"k": 1, "inputs": len(S), "total": len(S), "errors": bad, "certified": bad == 0}


@torch.no_grad()
def mean_slot_inputs(m, X, Y, k, dev, amp, n=2000, seed=7105):
    rng = np.random.default_rng(seed); st, o, _ = batch(rng, X, Y, k, n); out = []
    for i in range(0, n, 250):
        with amp_ctx(amp):
            _, _, ins = m.chunk(m.start_from(els=st[i:i + 250], dev=dev), torch.tensor(o[i:i + 250], device=dev), None, None)
        out.append(ins.float())
    return torch.cat(out).mean(0)


@torch.no_grad()
def edit_iia(m, k, dev, amp, bs, n=5000, seed=7110):
    rng = np.random.default_rng(seed); S = rng.integers(0, 120, n); O = rng.integers(0, 120, (n, 2 * k)); ok = 0
    for i in range(0, n, bs):
        s, o = S[i:i + bs], O[i:i + bs]
        el1, _, _ = chunks_dec(m, m.start_from(els=s, dev=dev), o[:, :k], dev, amp, decode="last")
        E = (el1[:, 0].clip(0) + rng.integers(1, 120, len(s))) % 120          # another valid state (!= a valid decode)
        el2, _, _ = chunks_dec(m, m.start_from(els=E, dev=dev), o[:, k:], dev, amp, decode="last")
        ok += int((el2[:, 0] == true_chain(E, o[:, k:])[:, -1]).sum())
    return {"n": n, "iia": ok / n}


def wte_rows(m):
    return m.g.w("wte.weight")[torch.tensor([s5.GT] + LETTOK, device=m.g.w("wte.weight").device)].detach().float().clone()


def train(a):
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    s5.check_interface()
    Xtr, Ytr = s5.load_csv(os.path.join(a.data_dir, "S5=128.csv")); s5.check_labels(Xtr, Ytr)
    Xte, Yte = s5.load_csv(os.path.join(a.data_dir, "S5=512.csv"), limit=a.test_n); s5.check_labels(Xte, Yte)
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), "train_sequences": len(Xtr), "op_tokens": len(OPTOK[0]), "state_tokens": NST},
              open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    m = Gen97(s5.load_gpt2(dev), a.arm).to(dev); m.mask = a.fork == "mask"; w0 = wte_rows(m)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []
    for step in range(1, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        st, o, tg = batch(rng, Xtr, Ytr, a.k, a.bs)
        with amp_ctx(amp):
            loss, _, _ = m.chunk(m.start_from(els=st, dev=dev), torch.tensor(o, device=dev), torch.tensor(tg, device=dev), None)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(loss.item())
        if step % 100 == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)), "sec": round(time.time() - t0)}; hist = []
            if a.log_every and step % a.log_every == 0:
                m.eval(); cc = chunk_check(m, a.k, dev, amp, a.eval_bs, n=a.log_n, seed=7107)
                rec["per_slot_acc"] = cc["per_slot_acc"]; rec["per_chunk_acc"] = cc["per_chunk_acc"]; m.train()
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    train_sec = round(time.time() - t0)
    torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    m.eval(); res = {"arm": a.arm, "fork": a.fork, "k": a.k, "seed": a.seed, "train_sec": train_sec, "eval_dtype": str(amp), "test_sequences": len(Xte)}
    res["gpu"] = torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"      # Audit: arms run on 3090 / 4090
    t1 = time.time(); pred, comp, val = rollout(m, Xte, a.k, dev, amp, a.eval_bs)
    np.save(os.path.join(a.out, "preds_test.npy"), pred); cum = s5.cumulative(pred, Yte)
    res["cumulative"] = {str(t): float(cum[t - 1]) for t in (128, 256, 512) if t <= Xte.shape[1]}
    res["final_state_acc_512"] = float((pred[:, -1] == Yte[:, -1]).mean()); res["compute_512_per_problem"] = comp
    res["checkpoint_validity"] = val
    if a.ext_n:
        Xx, Yx = s5.load_csv(os.path.join(a.data_dir, "S5=2048.csv"), limit=a.ext_n); s5.check_labels(Xx, Yx)
        Xx, Yx = Xx[:, :a.ext_len], Yx[:, :a.ext_len]
        px, _, _ = rollout(m, Xx, a.k, dev, amp, a.eval_bs); cx = s5.cumulative(px, Yx)
        res["extension_2048"] = {"n": len(Xx), "len": Xx.shape[1], "cumulative": {str(t): float(cx[t - 1]) for t in (512, 1024, 2048) if t <= Xx.shape[1]}}
    res["chunk_check"] = chunk_check(m, a.k, dev, amp, a.eval_bs, n=a.check_n)
    if a.k == 1:
        res["certificate"] = certificate_k1(m, dev, amp, a.eval_bs)
    else:
        mu = mean_slot_inputs(m, Xtr, Ytr, a.k, dev, amp)
        res["removal"] = {ab: chunk_check(m, a.k, dev, amp, a.eval_bs, n=min(10_000, a.check_n), seed=7106, ablate=ab, mean_in=mu) for ab in ("zero", "mean")}
    w1 = wte_rows(m); cs = F.cosine_similarity(w0, w1, dim=-1).tolist()
    res["wte_drift"] = {"tokens": [" >"] + [" " + c for c in "ABCDE"], "cosine": cs, "min_cosine": min(cs)}
    res["edit_iia"] = edit_iia(m, a.k, dev, amp, a.eval_bs, n=min(5000, a.check_n))
    res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({kk: res[kk] for kk in ("cumulative", "final_state_acc_512", "checkpoint_validity")}), flush=True)
    print("DONE", a.out, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--arm", required=True, choices=["gen", "hyb"])
    ap.add_argument("--fork", default="full", choices=["full", "mask"], help="§105: mask = each fork sees only its slot's K/V and its own tokens")
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--eval_bs", type=int, default=500)
    ap.add_argument("--test_n", type=int, default=s5.TEST_N); ap.add_argument("--ext_n", type=int, default=500)
    ap.add_argument("--ext_len", type=int, default=2048, help="< 2048: smoke only"); ap.add_argument("--check_n", type=int, default=100_000, help="< 100k: smoke only")
    ap.add_argument("--log_every", type=int, default=0); ap.add_argument("--log_n", type=int, default=5000)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "bench", "data", "s5"))
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
