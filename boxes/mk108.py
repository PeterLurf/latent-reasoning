"""THEORY §108: forced latent carriage on S5. §97's GEN recipe (text step target through the tied LM head via a fork; text
snap every k with a history cut) with MARKOV segments: each op step is a fresh-cache segment
    [<|endoftext|>, slot j-1's latent (or the chunk's start text for the first op), op j's tokens] -> slot j
with positions re-indexed from 0 (EOT at position 0 = GPT-2's attention sink, so the latent is never at the sink). Slot j's
input is the final hidden at op j's last token; that latent is what the next segment carries. Forks see only their own
segment, so the state can reach slot j only through slot j-1's latent.
Training (24k steps, bs 64, lr 1e-4, §97's warmup + cosine): K-CURR chunk length 1 (steps 1-2k), 2 (2k-6k), then the arm's
k (4 or 8); chunks cut at random positions of the S5=128 training sequences with the TRUE start state's text. Teacher-forced
segment inputs: one coin per batch per segment position j >= 1 with probability rho replaces slot j-1's latent with the GOLD
previous step text (the true state after op j-1, as at a chunk start); rho = 1 - step / 12k, 0 from 12k (logged).
Test (rho = 0): gen97's rollout (snap every k), chunk_check, removal (zero / mean of the carried latents 1..k-1; secondary),
checkpoint validity, edit IIA, compute; plus
  per_handoff_acc: on 20k fresh chunks with true starts, P(slot j+1 correct | slot j correct) per in-chunk j, and pooled;
  donor (§104's protocol, primary C1): n = 2000 recipient chunks A and donor chunks B (fresh random starts and ops); at each
    in-chunk slot j = 1..k-1, slot j's input (= the latent carried into segment j+1) <- B's slot-j input; scored at the
    chunk's last slot: donor_consistent = decode == (B's state after j ops) composed with A's ops j+1..k;
    recipient_consistent = decode == A's own true final state.
§111 (note 111-1): --arm dag (after rho = 0, half the batches are two chunks, the second from the model's own greedy snap;
targets = the true chain from the snapped state; unparseable snaps dropped); --arm read (auxiliary reader at every slot
from step 0: a fresh segment [EOT, carried latent] teacher-forced to state j's text, CE weight 1; eval reader_acc);
MK-4-2x = --steps 48000 (schedule as §108). All runs log main_loss, null = (k-1)/k (1-rho) ln 120 and ratio per 100
steps; eval.json adds train_handoff_err (128-op rollouts of 20k training rows), early_escape / escape_step / ratio_per_1k
/ final_train_loss. --eval_only [--fp32] re-evaluates model.pt (eval.json / eval_fp32.json); --backfill writes the §111
keys for a finished §108 run to s111_backfill.json.
    python -m boxes.mk108 --k 4 --seed 1 --out boxes/runs_s5/s108_mk4_s1
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
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes.gen97 import (NST, Gen97, amp_ctx, chunk_check, decode_to_el, edit_iia, mean_slot_inputs, rollout,  # noqa: E402
                         true_chain)
from boxes.latent86 import batch  # noqa: E402

EOT = 50256


class MK108(Gen97):
    def __init__(self, g):
        super().__init__(g, "gen"); self.gold = None; self.override = None
        self.reader = False; self.parts = None                              # §111 MK-4-READ auxiliary reader (default off)

    def read_greedy(self, s_in):
        """§111 reader: a fresh segment [EOT, carried latent] (positions 0..1), 6 greedy tokens of the state text."""
        g = self.g; B = s_in.shape[0]; eot = g.embed(torch.full((B, 1), EOT, device=s_in.device))
        h, past = g(torch.cat([eot, s_in[:, None].to(eot.dtype)], 1), None); t = g.logits(h[:, -1]).argmax(-1); out = [t]
        for _ in range(NST - 1):
            h, past = g(g.embed(t)[:, None], past); t = g.logits(h[:, 0]).argmax(-1); out.append(t)
        return torch.stack(out, 1)

    def chunk(self, start, ops, tgt=None, decode="all", ablate=None, mean_in=None, meter=None, meter_x=None):
        """Markov-segment chunk with gen97's signature. start (B, 6) state-text tokens. self.gold: set of in-chunk
        positions j >= 1 whose segment input is the gold previous step text (training; needs tgt). self.override:
        (j, (B, D)) replaces slot j's input (0-based j < c-1; the donor patch)."""
        B, k = ops.shape; g = self.g; eot = g.embed(torch.full((B, 1), EOT, device=ops.device))
        ins, losses, decs, rlosses, alosses = [], [], [], [], []; prev = g.embed(start); fed_latent = False
        for j in range(k):
            x = torch.cat([eot, prev, g.embed(self.optok[ops[:, j]])], 1)
            h, past = g(x, None)                                             # a fresh segment, positions from 0
            if meter:
                meter.step(x.shape[1], 0, 0)
            s_in = h[:, -1]
            if getattr(self, "anchor", False) and tgt is not None and fed_latent:   # §118 ANCHOR: latent-fed segment j vs the text path from gold state j-1
                with torch.no_grad():
                    hb, _ = g(torch.cat([eot, g.embed(self.sttok[tgt[:, j - 1]]), g.embed(self.optok[ops[:, j]])], 1), None); b = hb[:, -1].float()
                a_ = s_in.float(); cos = F.cosine_similarity(a_, b, dim=-1)
                alosses.append(((1 - cos) + (a_.norm(dim=-1).log() - b.norm(dim=-1).log()).abs()).mean())
            if ablate == "shuffle" and j < k - 1:                    # note 114-2 / 111-2: the carried latent of another row (in distribution)
                s_in = s_in[self.perm]
            if ablate in ("zero", "mean") and j < k - 1:
                s_in = torch.zeros_like(s_in) if ablate == "zero" else mean_in[j][None].expand_as(s_in).to(s_in.dtype)
            if self.override is not None and self.override[0] == j:
                s_in = self.override[1].to(s_in.dtype)
            ins.append(s_in)
            ctx = past[0][0].shape[2]; hs, past = g(s_in[:, None], past)
            if meter:
                meter.step(1, ctx, 0)
            hslot = hs[:, 0]
            if tgt is not None:
                st = self.sttok[tgt[:, j]]
                hf, _ = g(g.embed(st[:, :-1]), past)
                lg = g.logits(torch.cat([hslot[:, None], hf], 1)).float()
                losses.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), st.reshape(-1), reduction="none").view(B, NST).sum(-1).mean())
                if self.reader:                                              # [EOT, the carried latent] -> state j's text (CE, weight 1)
                    hr, _ = g(torch.cat([eot, s_in[:, None].to(eot.dtype), g.embed(st[:, :-1])], 1), None)
                    lr_ = g.logits(hr[:, 1:]).float()
                    rlosses.append(F.cross_entropy(lr_.reshape(-1, lr_.shape[-1]), st.reshape(-1), reduction="none").view(B, NST).sum(-1).mean())
            if decode == "all" or (decode == "last" and j == k - 1):
                decs.append(self.greedy(hslot, past, meter if j == k - 1 else meter_x))
            if j + 1 < k:
                if self.gold is not None and (j + 1) in self.gold and tgt is not None:
                    prev = g.embed(self.sttok[tgt[:, j]]); fed_latent = False   # gold previous step text (teacher-forced input)
                else:
                    prev = s_in[:, None].to(eot.dtype); fed_latent = True
        loss = torch.stack(losses).mean() if losses else None
        self.parts = (None if loss is None else float(loss.detach()), float(torch.stack(rlosses).mean().detach()) if rlosses else None)
        if rlosses:
            loss = loss + torch.stack(rlosses).mean()
        self.anchor_val = float(torch.stack(alosses).mean().detach()) if alosses else None
        if alosses:
            loss = loss + torch.stack(alosses).mean()
        return loss, (torch.stack(decs, 1) if decs else None), torch.stack(ins, 1)


@torch.no_grad()
def handoff(m, k, dev, amp, bs, n=20000, seed=10801):
    rng = np.random.default_rng(seed); S = rng.integers(0, 120, n); O = rng.integers(0, 120, (n, k)); C = []
    for i in range(0, n, bs):
        with amp_ctx(amp):
            _, d, _ = m.chunk(m.start_from(els=S[i:i + bs], dev=dev), torch.tensor(O[i:i + bs], device=dev))
        el, _ = decode_to_el(d.cpu().numpy().reshape(-1, NST)); C.append(el.reshape(-1, k) == true_chain(S[i:i + bs], O[i:i + bs]))
    C = np.concatenate(C); per = [float(C[C[:, j], j + 1].mean()) if C[:, j].any() else None for j in range(k - 1)]
    pv = np.concatenate([C[C[:, j], j + 1] for j in range(k - 1)]); pooled = float(pv.mean()) if len(pv) else None
    return {"per_position": per, "pooled": pooled, "first_slot": float(C[:, 0].mean()), "n": n}


@torch.no_grad()
def donor(m, k, dev, amp, bs, n=2000, seed=10802):
    rng = np.random.default_rng(seed); S = rng.integers(0, 120, n); O = rng.integers(0, 120, (n, k))
    SB = rng.integers(0, 120, n); OB = rng.integers(0, 120, (n, k)); TA = true_chain(S, O)[:, -1]; res = {}
    for j in range(k - 1):                                                  # 0-based j: slot j+1 (1-based) patched
        tgt = true_chain(SB, OB[:, :j + 1])[:, -1]
        for t in range(j + 1, k):
            tgt = s5.COMP[tgt, O[:, t]]
        out = []
        for i in range(0, n, bs):
            with amp_ctx(amp):
                _, _, insB = m.chunk(m.start_from(els=SB[i:i + bs], dev=dev), torch.tensor(OB[i:i + bs], device=dev), decode=None)
                m.override = (j, insB[:, j]); _, d, _ = m.chunk(m.start_from(els=S[i:i + bs], dev=dev), torch.tensor(O[i:i + bs], device=dev), decode="last")
                m.override = None
            el, _ = decode_to_el(d.cpu().numpy().reshape(-1, NST)); out.append(el)
        out = np.concatenate(out)
        res[f"slot{j + 1}"] = {"donor_consistent": float((out == tgt).mean()), "recipient_consistent": float((out == TA).mean())}
    return res


def k_train(step, k, sc):
    return 1 if step <= 2000 * sc else (2 if step <= 6000 * sc else k)


def k_long(step, sc):
    """§118 LONG: K-CURR 1 / 2 / 4 / 8 / 16 / 32 from steps 0 / 2k / 6k / 10k / 14k / 18k."""
    for lim, kk in ((2000, 1), (6000, 2), (10000, 4), (14000, 8), (18000, 16)):
        if step <= lim * sc:
            return kk
    return 32


LN120 = math.log(120)


def null_loss(L, rho):
    """Note 111-1: the training loss when every latent-fed slot is at chance, (k - 1)/k x (1 - rho) x ln 120."""
    return (L - 1) / L * (1 - rho) * LN120


def log_stats(recs, sc=1.0):
    """§111 from the 100-step log records: loss/null ratio per 1k (main loss), early_escape (ratio < 0.9 within 1k steps
    after k = 2 starts), escape_step (first record at k >= 2 with ratio < 0.9), final_train_loss (mean main loss over the
    last 1k steps)."""
    rows = []
    for r in recs:
        ml = r.get("main_loss", r["loss"]); nl = null_loss(r["k_train"], r["rho"])
        rows.append((r["step"], ml, ml / nl if nl > 0 else None, r["k_train"]))
    k2 = 2000 * sc
    win = [q for s, _, q, _ in rows if k2 < s <= k2 + 1000 * sc and q is not None]
    esc = [s for s, _, q, k in rows if k >= 2 and q is not None and q < 0.9]
    last = max(s for s, *_ in rows) if rows else 0
    return {"ratio_per_1k": [[s, q] for s, _, q, _ in rows if s % 1000 == 0],
            "early_escape": bool(win) and min(win) < 0.9, "early_window_min_ratio": min(win) if win else None,
            "escape_step": esc[0] if esc else None,
            "final_train_loss": float(np.mean([ml for s, ml, _, _ in rows if s > last - 1000])) if rows else None,
            **rel_stats(recs, sc)}


def rel_stats(recs, sc=1.0):
    """Note 111-2 (post-hoc, descriptive): early_escape_rel = main loss / model-relative null < 0.9 at the first log at or
    after step 6k (k = 4 starts); rel_ratio_per_1k. None when the log has no rel_ratio (runs started before 111-2)."""
    rr = [(r["step"], r["rel_ratio"]) for r in recs if r.get("rel_ratio") is not None]
    first = next(((s, q) for s, q in rr if s >= 6000 * sc), None)
    return {"early_escape_rel": None if first is None else bool(first[1] < 0.9), "rel_ratio_at_6k": None if first is None else first[1],
            "rel_ratio_per_1k": [[s, q] for s, q in rr if s % 1000 == 0]}


@torch.no_grad()
def train_handoff_err(m, X, Y, k, dev, amp, bs, n=20000):
    """§111 selection criterion: per-hand-off error on greedy rollouts (snap every k) of n TRAINING-split sequences. The
    training rows are 128 ops long, so these are 128-op rollouts (not 512). Error at position t >= 1 = P(wrong at t |
    right at t-1), split into in-chunk (latent) hand-offs and snap hand-offs (t multiple of k)."""
    X, Y = X[:n], Y[:n]; pred, _, _ = rollout(m, X, k, dev, amp, bs); C = pred == Y
    t = np.arange(1, X.shape[1]); prev, cur = C[:, :-1], C[:, 1:]; inch = (t % k) != 0

    def err(mask):
        sel = prev & mask[None]
        return float(1 - cur[sel].mean()) if sel.any() else None
    return {"n": len(X), "len": int(X.shape[1]), "note": "training rows are 128 ops: 128-op rollouts",
            "in_chunk_err": err(inch), "snap_err": err(~inch), "pooled_err": err(np.ones_like(inch)),
            "first_pos_err": float(1 - C[:, 0].mean()), "cumulative_128": float(C.all(1).mean())}


@torch.no_grad()
def reader_acc(m, k, dev, amp, bs, n=20000, seed=11101):
    """§111 MK-4-READ: the reader's own decode ([EOT, carried latent] -> state text) per slot on fresh chunks."""
    rng = np.random.default_rng(seed); S = rng.integers(0, 120, n); O = rng.integers(0, 120, (n, k)); C = []
    for i in range(0, n, bs):
        with amp_ctx(amp):
            _, _, ins = m.chunk(m.start_from(els=S[i:i + bs], dev=dev), torch.tensor(O[i:i + bs], device=dev), decode=None)
            d = torch.stack([m.read_greedy(ins[:, j]) for j in range(k)], 1)
        el, _ = decode_to_el(d.cpu().numpy().reshape(-1, NST)); C.append(el.reshape(-1, k) == true_chain(S[i:i + bs], O[i:i + bs]))
    C = np.concatenate(C)
    return {"n": n, "per_slot_acc": C.mean(0).tolist()}


def evaluate(m, a, dev, amp, Xtr, Ytr, Xte, Yte):
    res = {}
    pred, comp, val = rollout(m, Xte, a.k, dev, amp, a.eval_bs); np.save(os.path.join(a.out, "preds_test" + ("_fp32" if a.fp32 else "") + ".npy"), pred)
    cum = s5.cumulative(pred, Yte)
    res["cumulative"] = {str(t): float(cum[t - 1]) for t in (128, 256, 512) if t <= Xte.shape[1]}
    res["final_state_acc_512"] = float((pred[:, -1] == Yte[:, -1]).mean()); res["compute_512_per_problem"] = comp; res["checkpoint_validity"] = val
    if a.ext_n:
        Xx, Yx = s5.load_csv(os.path.join(a.data_dir, "S5=2048.csv"), limit=a.ext_n); s5.check_labels(Xx, Yx); Xx, Yx = Xx[:, :a.ext_len], Yx[:, :a.ext_len]
        px, _, _ = rollout(m, Xx, a.k, dev, amp, a.eval_bs); cx = s5.cumulative(px, Yx)
        res["extension_2048"] = {"n": len(Xx), "len": Xx.shape[1], "cumulative": {str(t): float(cx[t - 1]) for t in (512, 1024, 2048) if t <= Xx.shape[1]}}
    res["chunk_check"] = chunk_check(m, a.k, dev, amp, a.eval_bs, n=a.check_n)
    res["per_handoff_acc"] = handoff(m, a.k, dev, amp, a.eval_bs, n=min(a.handoff_n, a.check_n))
    res["donor"] = donor(m, a.k, dev, amp, a.eval_bs, n=a.donor_n)
    mu = mean_slot_inputs(m, Xtr, Ytr, a.k, dev, amp)
    res["removal"] = {ab: chunk_check(m, a.k, dev, amp, a.eval_bs, n=min(10_000, a.check_n), seed=7106, ablate=ab, mean_in=mu) for ab in ("zero", "mean")}
    res["edit_iia"] = edit_iia(m, a.k, dev, amp, a.eval_bs, n=min(5000, a.check_n))
    res["train_handoff_err"] = train_handoff_err(m, Xtr, Ytr, a.k, dev, amp, a.eval_bs, n=min(a.handoff_n, len(Xtr)))
    res["rel_final"] = rel_ratio_final(m, Xtr, Ytr, a.k, dev, amp)
    res["latent_geometry"] = latent_geometry(m, a.k, dev, amp, a.eval_bs)
    if a.arm == "read":
        res["reader_acc"] = reader_acc(m, a.k, dev, amp, a.eval_bs, n=min(a.handoff_n, a.check_n))
    return res


@torch.no_grad()
def rel_ratio_final(m, X, Y, k, dev, amp, n_batches=20, bs=64, seed=11102):
    """Note 111-2 (theory's (a), "reads at the end"): at the final model, mean main loss / mean main loss with the carried
    latents 1..k-1 zeroed, over n_batches training chunks of k ops (true starts, no gold)."""
    rng = np.random.default_rng(seed); l1 = l0 = ls = 0.0; T = lambda x: torch.tensor(x, device=dev)  # noqa: E731
    for _ in range(n_batches):
        st, o, tg = batch(rng, X, Y, k, bs); m.perm = torch.from_numpy(rng.permutation(bs)).to(dev)
        with amp_ctx(amp):
            m.chunk(m.start_from(els=st, dev=dev), T(o), T(tg), decode=None); l1 += m.parts[0]
            m.chunk(m.start_from(els=st, dev=dev), T(o), T(tg), decode=None, ablate="zero"); l0 += m.parts[0]
            m.chunk(m.start_from(els=st, dev=dev), T(o), T(tg), decode=None, ablate="shuffle"); ls += m.parts[0]
    return {"rel_ratio_final": l1 / max(1e-9, l0), "rel_ratio_final_shuffled": l1 / max(1e-9, ls), "main_loss": l1 / n_batches,
            "zeroed_loss": l0 / n_batches, "shuffled_loss": ls / n_batches, "n_batches": n_batches, "bs": bs}


@torch.no_grad()
def latent_geometry(m, k, dev, amp, bs, n=2000, seed=11801):
    """§118 eval: effective rank (exp of the entropy of the normalised singular values) of n carried latents from fresh
    k-chunks (slots 1..k-1), and per in-chunk position j >= 1 the cosine / |log-norm gap| between the latent-path
    segment output (fed by latent j-1) and the text-path output [EOT, TRUE state j-1 text, op_j] (truth: eval only)."""
    rng = np.random.default_rng(seed); S = rng.integers(0, 120, n); O = rng.integers(0, 120, (n, k)); g = m.g; lat = []; cs = [[] for _ in range(k)]; gp = [[] for _ in range(k)]
    for i in range(0, n, bs):
        s_, o_ = S[i:i + bs], O[i:i + bs]; B = len(s_); tc = true_chain(s_, o_)
        with amp_ctx(amp):
            _, _, ins = m.chunk(m.start_from(els=s_, dev=dev), torch.tensor(o_, device=dev), decode=None)
            eot = g.embed(torch.full((B, 1), EOT, device=dev))
            for j in range(1, k):
                hb, _ = g(torch.cat([eot, g.embed(m.sttok[torch.tensor(tc[:, j - 1], device=dev)]), g.embed(m.optok[torch.tensor(o_[:, j], device=dev)])], 1), None)
                a_, b = ins[:, j].float(), hb[:, -1].float()
                cs[j].append(F.cosine_similarity(a_, b, dim=-1).cpu().numpy()); gp[j].append((a_.norm(dim=-1).log() - b.norm(dim=-1).log()).abs().cpu().numpy())
        lat.append(ins[:, :k - 1].float().reshape(-1, ins.shape[-1]).cpu())
    Lm = torch.cat(lat)[:n]; sv = torch.linalg.svdvals(Lm - Lm.mean(0)); pr = sv / sv.sum()
    return {"n": int(Lm.shape[0]), "effective_rank": float(torch.exp(-(pr * pr.clamp_min(1e-12).log()).sum())),
            "anchor_cos_by_pos": {str(j): float(np.concatenate(cs[j]).mean()) for j in range(1, k)},
            "anchor_lognorm_gap_by_pos": {str(j): float(np.concatenate(gp[j]).mean()) for j in range(1, k)}}


def read_log(lp):
    return [json.loads(r) for r in open(lp, encoding="utf-8") if r.strip()] if os.path.exists(lp) else []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, required=True, choices=[4, 8]); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arm", default="mk", choices=["mk", "dag", "read"], help="§111: dag = DAgger-style 2-chunk batches after rho = 0; read = auxiliary reader")
    ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_bs", type=int, default=200); ap.add_argument("--test_n", type=int, default=s5.TEST_N); ap.add_argument("--ext_n", type=int, default=500)
    ap.add_argument("--ext_len", type=int, default=2048, help="< 2048: smoke only"); ap.add_argument("--check_n", type=int, default=100_000, help="< 100k: smoke only")
    ap.add_argument("--donor_n", type=int, default=2000); ap.add_argument("--handoff_n", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=2000); ap.add_argument("--log_n", type=int, default=2000)
    ap.add_argument("--sched_scale", type=float, default=1.0, help="smoke only"); ap.add_argument("--data_dir", default=os.path.join(ROOT, "bench", "data", "s5"))
    ap.add_argument("--ckpt_every", type=int, default=2000, help="training checkpoint <out>/ckpt.pt (model, optimiser, RNGs); resumed if present")
    ap.add_argument("--eval_only", action="store_true", help="evaluate <out>/model.pt (writes eval.json, or eval_fp32.json with --fp32)")
    ap.add_argument("--fp32", action="store_true", help="evaluation without autocast (fp32 throughout, the carry included)")
    ap.add_argument("--anchor", action="store_true", help="§118 ANCHOR: (1 - cos) + |log-norm gap| between each latent-fed carried latent and the stop-grad text-path latent from the gold previous state")
    ap.add_argument("--long", action="store_true", help="§118 LONG: K-CURR up to 32 (k_long); micro-batches of bs / max(1, k_train // 8)")
    ap.add_argument("--s115", action="store_true", help="§118: run snap115 (A)/(B) in-process after the eval -> eval_s115.json")
    ap.add_argument("--rel_only", action="store_true", help="with --backfill: add only rel_final to an existing s111_backfill.json")
    ap.add_argument("--backfill", action="store_true", help="§111 keys only for a finished run: train_handoff_err + log stats -> <out>/s111_backfill.json")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); sc = a.sched_scale
    amp = None if a.fp32 else (torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None)
    s5.check_interface()
    Xtr, Ytr = s5.load_csv(os.path.join(a.data_dir, "S5=128.csv")); s5.check_labels(Xtr, Ytr)
    Xte, Yte = s5.load_csv(os.path.join(a.data_dir, "S5=512.csv"), limit=a.test_n); s5.check_labels(Xte, Yte)
    os.makedirs(a.out, exist_ok=True); lp = os.path.join(a.out, "log.jsonl")
    m = MK108(s5.load_gpt2(dev)).to(dev); m.reader = a.arm == "read"; m.anchor = a.anchor
    if a.eval_only or a.backfill:
        m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.eval(); t1 = time.time()
        if a.backfill and a.rel_only:
            bp = os.path.join(a.out, "s111_backfill.json"); r = json.load(open(bp)) if os.path.exists(bp) else {}
            r["rel_final"] = rel_ratio_final(m, Xtr, Ytr, a.k, dev, amp); r.update(rel_stats(read_log(lp), sc))
            json.dump(r, open(bp, "w", newline="\n"), indent=1); print("REL", json.dumps(r["rel_final"]), flush=True)
            return
        if a.backfill:
            r = {"train_handoff_err": train_handoff_err(m, Xtr, Ytr, a.k, dev, amp, a.eval_bs, n=min(a.handoff_n, len(Xtr))), **log_stats(read_log(lp), sc),
                 "eval_dtype": str(amp), "sec": round(time.time() - t1)}
            json.dump(r, open(os.path.join(a.out, "s111_backfill.json"), "w", newline="\n"), indent=1); print("BACKFILL", json.dumps(r)[:400], flush=True)
            return
        res = {"arm": f"mk{a.k}" + ("" if a.arm == "mk" else "_" + a.arm) + ("_anchor" if a.anchor else "") + ("_long" if a.long else ""), "k": a.k, "seed": a.seed, "eval_only": True, "eval_dtype": str(amp),
               "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", **evaluate(m, a, dev, amp, Xtr, Ytr, Xte, Yte),
               **log_stats(read_log(lp), sc)}
        res["eval_sec"] = round(time.time() - t1)
        json.dump(res, open(os.path.join(a.out, "eval_fp32.json" if a.fp32 else "eval.json"), "w", newline="\n"), indent=1)
        print("EVAL", json.dumps({kk: res[kk] for kk in ("cumulative", "donor")}), flush=True); print("DONE", a.out, flush=True)
        return
    json.dump({**vars(a), "train_sequences": len(Xtr)}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    t0 = time.time(); hist = []; mains = []; reads = []; dag = []; anchors = []; seg_types = [0, 0]; step0 = 0; ck = os.path.join(a.out, "ckpt.pt")
    if os.path.exists(ck):                                                  # resume after a host restart (k1, 2026-10-04)
        c = torch.load(ck, map_location=dev, weights_only=False); m.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        rng.bit_generator.state = c["np_rng"]; torch.set_rng_state(c["torch_rng"].cpu()); step0 = c["step"]; seg_types = c["seg_types"]; t0 -= c["sec"]
        if dev.startswith("cuda"):
            torch.cuda.set_rng_state(c["cuda_rng"].cpu(), dev)
        keep = [r for r in open(lp, encoding="utf-8") if r.strip() and json.loads(r)["step"] <= step0] if os.path.exists(lp) else []
        open(lp, "w", newline="\n").writelines(keep); print("RESUMED", step0, flush=True); del c
    log = open(lp, "a", newline="\n"); T = lambda x: torch.tensor(x, device=dev)  # noqa: E731
    for step in range(step0 + 1, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        L = k_long(step, sc) if a.long else k_train(step, a.k, sc); rho = max(0.0, 1.0 - step / (12000 * sc)); done_bwd = False
        if a.arm == "dag" and rho == 0 and L == a.k and rng.random() < 0.5:
            # §111 MK-4-DAG: two chunks; chunk 2 starts from the model's OWN greedy snap of chunk 1 (DAgger: its targets are
            # the true chain from the snapped state; rows whose snap does not parse are dropped from chunk 2)
            st, o, tg = batch(rng, Xtr, Ytr, 2 * L, a.bs)
            with amp_ctx(amp):
                l1, d1, _ = m.chunk(m.start_from(els=st, dev=dev), T(o[:, :L]), T(tg[:, :L]), decode="last")
            p1 = m.parts[0]; toks = d1[:, -1].detach().cpu().numpy(); el, _ = decode_to_el(toks); ok = el >= 0
            dag.append((float(ok.mean()), float((el == tg[:, L - 1]).mean()))); loss = l1; mp = p1
            if ok.any():
                ii = np.where(ok)[0]
                with amp_ctx(amp):
                    l2, _, _ = m.chunk(T(toks[ii]), T(o[ii, L:]), T(true_chain(el[ii], o[ii, L:])), decode=None)
                loss = (l1 + l2) / 2; mp = (p1 + m.parts[0]) / 2
            seg_types[1] += 2 * (L - 1)
        else:
            st, o, tg = batch(rng, Xtr, Ytr, L, a.bs)
            m.gold = {j for j in range(1, L) if rng.random() < rho} or None
            seg_types[0] += 1 + (len(m.gold) if m.gold else 0); seg_types[1] += L - 1 - (len(m.gold) if m.gold else 0)
            micro = max(1, L // 8) if a.long else 1
            if micro > 1:                                                   # §118 LONG: gradient accumulation over micro-batches
                opt.zero_grad(); lt = mt = 0.0; rt = []; at = []
                for sl in np.array_split(np.arange(a.bs), micro):
                    with amp_ctx(amp):
                        li, _, _ = m.chunk(m.start_from(els=st[sl], dev=dev), T(o[sl]), T(tg[sl]), decode=None)
                    (li / micro).backward(); lt += li.item() / micro; mt += m.parts[0] / micro
                    if m.parts[1] is not None:
                        rt.append(m.parts[1])
                    if m.anchor_val is not None:
                        at.append(m.anchor_val)
                loss_val = lt; mp = mt; done_bwd = True
                if rt:
                    reads.append(float(np.mean(rt)))
                if at:
                    anchors.append(float(np.mean(at)))
            else:
                with amp_ctx(amp):
                    loss, _, _ = m.chunk(m.start_from(els=st, dev=dev), T(o), T(tg), decode=None)
                mp = m.parts[0]
                if m.parts[1] is not None:
                    reads.append(m.parts[1])
                if m.anchor_val is not None:
                    anchors.append(m.anchor_val)
        if done_bwd:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(loss_val); mains.append(mp)
        else:
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(loss.item()); mains.append(mp)
        gold_last = m.gold; m.gold = None
        if step % 100 == 0 or step == a.steps:
            ml = float(np.mean(mains)); nl = null_loss(L, rho)
            rec = {"step": step, "loss": float(np.mean(hist)), "main_loss": ml, "null": nl, "ratio": ml / nl if nl > 0 else None,
                   "rho": rho, "k_train": L, "sec": round(time.time() - t0)}
            if L >= 2:                                                      # note 111-2: model-relative null (same batch, carried latents zeroed)
                m.eval(); m.gold = gold_last if o.shape[1] == L else None
                with torch.no_grad(), amp_ctx(amp):
                    m.chunk(m.start_from(els=st, dev=dev), T(o[:, :L]), T(tg[:, :L]), decode=None); p1 = m.parts[0]
                    m.chunk(m.start_from(els=st, dev=dev), T(o[:, :L]), T(tg[:, :L]), decode=None, ablate="zero"); p0 = m.parts[0]
                    m.perm = torch.from_numpy(np.random.default_rng(step).permutation(o.shape[0])).to(dev)
                    m.chunk(m.start_from(els=st, dev=dev), T(o[:, :L]), T(tg[:, :L]), decode=None, ablate="shuffle"); ps = m.parts[0]
                m.gold = None; m.train(); rec["rel_main"] = p1; rec["rel_null"] = ps; rec["rel_null_zero"] = p0
                rec["rel_ratio"] = p1 / max(1e-9, ps); rec["rel_ratio_zero"] = p1 / max(1e-9, p0)
            if reads:
                rec["reader_loss"] = float(np.mean(reads))
            if anchors:
                rec["anchor_loss"] = float(np.mean(anchors)); anchors.clear()
            if dag:
                rec["dag_two_chunk_batches"] = len(dag); rec["dag_snap_valid"] = float(np.mean([v for v, _ in dag]))
                rec["dag_snap_correct"] = float(np.mean([c_ for _, c_ in dag]))
            hist, mains, reads, dag = [], [], [], []
            if a.log_every and step % a.log_every == 0:
                m.eval(); cc = chunk_check(m, a.k, dev, amp, a.eval_bs, n=a.log_n, seed=7107)
                rec["per_slot_acc"] = cc["per_slot_acc"]; rec["per_chunk_acc"] = cc["per_chunk_acc"]; m.train()
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            c = {"model": m.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None, "step": step, "seg_types": seg_types,
                 "sec": round(time.time() - t0)}
            torch.save(c, ck + ".tmp"); os.replace(ck + ".tmp", ck); del c
    train_sec = round(time.time() - t0)
    torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    if os.path.exists(ck):
        os.remove(ck)
    del opt; m.zero_grad(set_to_none=True); m.eval(); t1 = time.time()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    res = {"arm": f"mk{a.k}" + ("" if a.arm == "mk" else "_" + a.arm) + ("_anchor" if a.anchor else "") + ("_long" if a.long else ""), "k": a.k, "seed": a.seed, "steps": a.steps, "train_sec": train_sec,
           "eval_dtype": str(amp), "test_sequences": len(Xte), "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "segment_types": {"start_or_gold_text": seg_types[0], "latent": seg_types[1]}}
    res.update(evaluate(m, a, dev, amp, Xtr, Ytr, Xte, Yte)); res.update(log_stats(read_log(lp), sc)); res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({kk: res[kk] for kk in ("cumulative", "donor")}), flush=True)
    if a.s115:                                                              # §118: snap115 (A)/(B) in-process -> eval_s115.json
        from boxes.snap115 import run_markov
        run_markov(m, a.out, dev, amp, a.eval_bs, a.test_n, a.ext_n, a.data_dir, a.k, "A,B", a.ext_len if a.ext_len < 2048 else 0)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
