"""THEORY §110: self-training by length v2 (S5 from the identity, FINAL ANSWERS ONLY), one independent seed per process.
Base: boxes/self103.py (§97's GEN-4 recipe, snap every 4, history cut, R0 + rounds by length, adaptive insertion) with:
  * §105's masked fork (gen97 fork_past "mask") for every step fork and every decode (training and test);
  * one seed per process, its own fresh 50k pool per round labelled with its own greedy rollouts (no vote, no lockstep);
  * filters per problem, in order: validity (every chunk-end checkpoint parses) -> final answer correct ->
    self-consistency (at every position t, the chained decode equals the decode of a k = 1 chunk restarted from the
    text of the chained decode at t-1 (the identity at t = 0); an unparseable decode at t-1 fails). Kept problems give
    per-slot targets (the chained decodes) on every chunk; the others give the final-answer target only (as §103);
  * label-free gate: among final-correct problems, the self-consistency rate must be >= 0.995. If it is not (or §103's
    rule fires: kept share < 0.20), the midpoint length is inserted and a fresh pool is labelled; at most 2 insertions
    before each round that trains (after 2, the round trains on what it kept);
  * final relabelling: the last --final_steps (4,000) updates train on a fresh pool at lengths 3..L_max labelled by the
    current model (no insertion). R0 (1,200 updates, lengths 1-2) and the rounds share the rest (as §103).
Logged per round (truth for logging only, never used in training): filtered_by {validity, final, self_consistency},
kept_share, gate (self-consistency among final-correct), truth_precision_kept, truth_rate_final_correct,
agreement_vs_truth_gap (gate - truth_rate_final_correct), op_hist {all, kept} (120 bins over all positions), op_tv.
Eval (<out>/eval.json): §103's keys + donor (§104 single patch, slots 1..3, n = 2000) and removal.{zero,mean} (gen97
chunk_check incl. acc_after_first_ablated_slot). <out>/pool_final.npz: the final relabelled pool (rows into S5=128,
lens, keep, targets (N, L_max) chained decodes, -1 padded, before_last (N, 6) tokens) for the conditional STUDENT arm.
§110b (--random_start, note 110-1): R0 shorts start from the identity; every problem of length >= 3 (rounds and the
final relabelling) starts from a uniform random given state written as state text (the start is input, not a label);
the self-consistency restart at t = 0 uses that start; truth is the chain from that start. R0 logs
short_transition_coverage {direct, incl_2step} (distinct (state, op) pairs over 14,400). eval.json arm "self110b".
§112 (--cov f, note 112-1 + addendum): R0 shorts restricted to T_f (all (e, a); for every op b a fraction g of the 119
non-identity states; length-2 shorts need (e.a, b) in T_f); R0 count fixed. coverage_drawn (transitions labelled by the
shorts actually drawn) defines "labelled"; eval adds k1_labelled_acc / k1_never_labelled_acc / k1_certified (all 14,400
k = 1 restarts, masked fork), k1_never_labelled_in_pool / _not_in_pool (true transitions of kept problems), and
per_round_split (k = 1 accuracy split after each training round); --save_rounds keeps an fp16 model per round in rounds/.
§117 (notes 117-1): --markov (MK: Markov segments [EOT, carried latent / start text, op] with own-segment forks from step 0)
or --st_switch (ST: §105 masked-fork KV chunks until final-answer accuracy >= 0.5 on 5k fresh length 5-6 problems at a
round end, or step 12,000 exactly, then Markov; 12k Markov steps incl. the rest of that round + the 4k final relabel; the
cosine horizon becomes switch_step + 16k), --kcurr (training chunks cut at k_train 1 / 2 / 4 at 2k / 6k via items_for),
--reader (§113 reader from step 0). No gold teacher forcing, no rho.
WHERE TRUE STATES ENTER (Theory / Audit rule: in-run decisions use final answers only):
  §116 NONE+COPY copy rows (Gen110.copy_loss) use only the input: the start state's own text is its target.
  in-run decisions (final answer only): label_round's final filter (true FINAL answer of each chain via true_states(...)[-1]),
    r0_items / r0_items_cov final-answer targets, items_for / items_from_pool final-only targets (true_final), the ST switch
    check (final answers of length 5-6 problems); the self-consistency gate, kept_share and insertion use no truth.
  logging / eval only (never feed a decision): label_round's truth_precision_kept / truth_rate_final_correct /
    agreement_vs_truth_gap, kept_transitions, the reader_target_precision log, round_end's k1_table split and
    foothold_yield / acc_by_length, and the final eval (rollout, chunk_check, removal, donor, edit_iia, handoff,
    rel_ratio_final, k1 keys).
Checkpoint <out>/ckpt.pt at each round boundary (model, optimiser, RNGs, schedule state); resumed if present.
    python -m boxes.self110 --seed 1 --out boxes/runs_s5/s110_s1
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
from boxes.gen97 import (NST, STTOK, amp_ctx, chunk_check, decode_to_el, edit_iia, mean_slot_inputs, rollout,  # noqa: E402
                         true_chain)
from boxes.mk108 import donor  # noqa: E402
from boxes.self103 import Gen103, draw, make_pool, r0_items  # noqa: E402

K = 4
GATE = 0.995


def build_tf(f, seed):
    """§112 (note 112-1) transition subset T_f (state, op) -> bool: all 120 first-step transitions (e, a); for every op
    b, a random fraction g = (f 14400 - 120) / (14400 - 120) of the 119 non-identity states (stratified by op)."""
    r = np.random.default_rng(12000 + seed); T = np.zeros((120, 120), bool); T[0, :] = True
    g = (f * 14400 - 120) / (14400 - 120); nb = int(round(g * 119))
    for b in range(120):
        T[1 + r.choice(119, nb, replace=False), b] = True
    return T, g


def r0_items_cov(rng, T, n=50000):
    """R0 shorts restricted to T_f (count fixed): length 1 (e; a) always allowed (T[0] is all True); length 2 (e; a, b)
    allowed iff (e.a, b) in T_f. Ops drawn uniformly over the allowed sets (uniform ops, as the data's)."""
    lens = rng.integers(1, 3, n); g = {}
    a1 = np.where(T[0])[0]; pairs = np.array([(a, b) for a in a1 for b in np.where(T[s5.COMP[0, a]])[0]])
    for L in (1, 2):
        m_ = int((lens == L).sum())
        ops = a1[rng.integers(0, len(a1), m_)][:, None] if L == 1 else pairs[rng.integers(0, len(pairs), m_)]
        tg = np.full((m_, L), -1); tg[:, -1] = true_chain(np.zeros(m_, np.int64), ops)[:, -1]
        g[L] = (np.tile(STTOK[0], (m_, 1)), ops, tg)
    return g


def r0_labelled(r0):
    """(120, 120) transitions labelled by R0's shorts: (e, a) of every short, and (e.a, b) of length-2 shorts."""
    T = np.zeros((120, 120), bool)
    for L, (_, ops, _) in r0.items():
        T[0, ops[:, 0]] = True
        if L == 2:
            T[s5.COMP[0, ops[:, 0]], ops[:, 1]] = True
    return T


@torch.no_grad()
def k1_table(m, dev, amp, bs=1000, return_dec=False):
    """k = 1 restart from every state's text with every op (all 14,400, masked fork): (120, 120) correct (and the decodes)."""
    S_, O_ = [x.ravel() for x in np.meshgrid(np.arange(120), np.arange(120), indexing="ij")]; ok = np.zeros(len(S_), bool); dec = np.zeros(len(S_), np.int64)
    for i in range(0, len(S_), bs):
        with amp_ctx(amp):
            _, d, _ = m.chunk(torch.tensor(STTOK[S_[i:i + bs]], device=dev), torch.tensor(O_[i:i + bs, None], device=dev), decode="all")
        e, _ = decode_to_el(d.cpu().numpy().reshape(-1, NST)); ok[i:i + bs] = e == s5.COMP[S_[i:i + bs], O_[i:i + bs]]; dec[i:i + bs] = e
    return (ok.reshape(120, 120), dec.reshape(120, 120)) if return_dec else ok.reshape(120, 120)


def kept_transitions(X, pool, starts_true):
    """(120, 120) true transitions (state before, op) inside the kept problems of a labelled pool."""
    T = np.zeros((120, 120), bool)
    for i in np.where(pool["keep"])[0]:
        L = int(pool["lens"][i]); ops = X[pool["rows"][i], :L]; st = starts_true[i]
        T[np.concatenate([[pool["starts"][i]], st[:-1]]), ops] = True
    return T


class Gen110(Gen103):
    def __init__(self, g):
        super().__init__(g, "gen"); self.mask = True; self.override = None; self.reader = False; self.parts = None; self.markov = False

    def set_markov(self, on):
        """§117: Markov segments [EOT, carried latent (or start text), op] with own-segment (full) forks; off = §105's
        masked-fork KV chunk."""
        self.markov = on; self.mask = not on

    def chunk(self, start, ops, tgt=None, decode="all", ablate=None, mean_in=None, meter=None, meter_x=None):
        if not self.markov:
            return super().chunk(start, ops, tgt, decode, ablate, mean_in, meter, meter_x)
        B, k = ops.shape; g = self.g; eot = g.embed(torch.full((B, 1), 50256, device=ops.device))
        ins, losses, decs = [], [], []; prev = g.embed(start)
        for j in range(k):
            x = torch.cat([eot, prev, g.embed(self.optok[ops[:, j]])], 1); h, past = g(x, None)
            if meter:
                meter.step(x.shape[1], 0, 0)
            s_in = h[:, -1]
            if ablate == "shuffle" and j < k - 1:
                s_in = s_in[self.perm]
            if ablate in ("zero", "mean") and j < k - 1:
                s_in = torch.zeros_like(s_in) if ablate == "zero" else mean_in[j][None].expand_as(s_in).to(s_in.dtype)
            if self.override is not None and self.override[0] == j:
                s_in = self.override[1].to(s_in.dtype)
            ins.append(s_in); ctx = past[0][0].shape[2]; hs, past = g(s_in[:, None], past)
            if meter:
                meter.step(1, ctx, 0)
            hslot = hs[:, 0]
            if tgt is not None:
                st = self.sttok[tgt[:, j]]; hf, _ = g(g.embed(st[:, :-1]), past)
                lg = g.logits(torch.cat([hslot[:, None], hf], 1)).float()
                losses.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), st.reshape(-1), reduction="none").view(B, NST).sum(-1).mean())
            if decode == "all" or (decode == "last" and j == k - 1):
                decs.append(self.greedy(hslot, past, meter if j == k - 1 else meter_x))
            prev = s_in[:, None].to(eot.dtype)
        loss = torch.stack(losses).mean() if losses else None
        self.parts = (None if loss is None else float(loss.detach()), None)
        return loss, (torch.stack(decs, 1) if decs else None), torch.stack(ins, 1)

    def markov_masked(self, start, ops, tgt):
        """§117 training chunk (Markov segments): fork CE (+ the reader) only where tgt >= 0 (-1 = no target)."""
        B, c = ops.shape; g = self.g; eot1 = g.embed(torch.full((B, 1), 50256, device=ops.device)); prev = g.embed(start)
        tot = rtot = 0.0; n = 0
        for j in range(c):
            x = torch.cat([eot1, prev, g.embed(self.optok[ops[:, j]])], 1); h, past = g(x, None); s_in = h[:, -1]
            hs, past = g(s_in[:, None], past); hslot = hs[:, 0]; sel = tgt[:, j] >= 0
            if bool(sel.any()):
                st = self.sttok[tgt[sel, j]]; hf, _ = g(g.embed(st[:, :-1]), [(k_[sel], v_[sel]) for k_, v_ in past])
                lg = g.logits(torch.cat([hslot[sel][:, None], hf], 1)).float()
                tot = tot + F.cross_entropy(lg.reshape(-1, lg.shape[-1]), st.reshape(-1), reduction="none").view(-1, NST).sum(-1).sum(); n += int(sel.sum())
                if self.reader:
                    e_ = eot1[sel]; hr, _ = g(torch.cat([e_, s_in[sel][:, None].to(e_.dtype), g.embed(st[:, :-1])], 1), None)
                    lr_ = g.logits(hr[:, 1:]).float()
                    rtot = rtot + F.cross_entropy(lr_.reshape(-1, lr_.shape[-1]), st.reshape(-1), reduction="none").view(-1, NST).sum(-1).sum()
            prev = s_in[:, None].to(eot1.dtype)
        main = tot / max(1, n); rd = rtot / max(1, n) if self.reader else None
        self.parts = (float(main.detach()) if torch.is_tensor(main) else main, None if rd is None else (float(rd.detach()) if torch.is_tensor(rd) else rd))
        return main + rd if self.reader else main

    def copy_loss(self, start, states):
        """§116 NONE+COPY (note 116-2): a length-0 copy problem - the start text's last hidden is fed as a slot (as at an op
        position, no marker token) and the masked fork decodes it; target = the start state's own text (input only)."""
        g = self.g; h, past = g(g.embed(start)); s_in = h[:, -1]; hs, past = g(s_in[:, None], past); hslot = hs[:, 0]
        st = self.sttok[states]; fp, p0 = self.fork_past(past); hf, _ = g(g.embed(st[:, :-1]), fp, pos0=p0)
        lg = g.logits(torch.cat([hslot[:, None], hf], 1)).float()
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), st.reshape(-1), reduction="none").view(-1, NST).sum(-1).mean()

    @torch.no_grad()
    def read_greedy(self, s_in):
        """§113 reader: a fresh segment [EOT, slot latent] (positions 0..1) -> 6 greedy state-text tokens."""
        g = self.g; B = s_in.shape[0]; eot = g.embed(torch.full((B, 1), 50256, device=s_in.device))
        h, past = g(torch.cat([eot, s_in[:, None].to(eot.dtype)], 1), None); t = g.logits(h[:, -1]).argmax(-1); out = [t]
        for _ in range(NST - 1):
            h, past = g(g.embed(t)[:, None], past); t = g.logits(h[:, 0]).argmax(-1); out.append(t)
        return torch.stack(out, 1)

    def chunk_masked(self, start, ops, tgt):
        """Gen103.chunk_masked with §105's masked fork (the fork sees only the slot's own K/V, true positions)."""
        if self.markov:
            return self.markov_masked(start, ops, tgt)
        B, c = ops.shape; g = self.g
        x = torch.cat([g.embed(start), g.embed(self.optok[ops[:, 0]])], 1); past = None; tot = 0.0; rtot = 0.0; n = 0
        for j in range(c):
            h, past = g(x, past); s_in = h[:, -1]
            hs, past = g(s_in[:, None], past); hslot = hs[:, 0]
            sel = tgt[:, j] >= 0
            if bool(sel.any()):
                st = self.sttok[tgt[sel, j]]; fp, p0 = self.fork_past([(k_[sel], v_[sel]) for k_, v_ in past])
                hf, _ = g(g.embed(st[:, :-1]), fp, pos0=p0)
                lg = g.logits(torch.cat([hslot[sel][:, None], hf], 1)).float()
                tot = tot + F.cross_entropy(lg.reshape(-1, lg.shape[-1]), st.reshape(-1), reduction="none").view(-1, NST).sum(-1).sum()
                n += int(sel.sum())
                if self.reader:                                             # §113: [EOT, slot latent] -> the same (filtered) target text
                    eot = g.embed(torch.full((int(sel.sum()), 1), 50256, device=st.device))
                    hr, _ = g(torch.cat([eot, s_in[sel][:, None].to(eot.dtype), g.embed(st[:, :-1])], 1), None)
                    lr_ = g.logits(hr[:, 1:]).float()
                    rtot = rtot + F.cross_entropy(lr_.reshape(-1, lr_.shape[-1]), st.reshape(-1), reduction="none").view(-1, NST).sum(-1).sum()
            if j + 1 < c:
                x = g.embed(self.optok[ops[:, j + 1]])
        main = tot / max(1, n); rd = rtot / max(1, n) if self.reader else None
        self.parts = (float(main.detach()) if torch.is_tensor(main) else main, None if rd is None else (float(rd.detach()) if torch.is_tensor(rd) else rd))
        return main + rd if self.reader else main


@torch.no_grad()
def restart_k1(m, X, rows, lens, decs, ii, dev, amp, starts, bs=1000):
    """For problems ii: True iff at every position t the chained decode equals a k = 1 restart from the chained decode at
    t-1 (the problem's start state at t = 0). Unparseable previous decodes fail."""
    pr, pt = [], []
    for i in ii:
        d = decs[i]; L = lens[i]
        pr.extend([i] * L); pt.extend(range(L))
    pr = np.array(pr, np.int64); pt = np.array(pt, np.int64)
    prev = np.array([starts[i] if t == 0 else decs[i][t - 1] for i, t in zip(pr, pt)], np.int64)
    cur = np.array([decs[i][t] for i, t in zip(pr, pt)], np.int64); ops = X[rows[pr], pt]
    agree = np.zeros(len(pr), bool); okp = prev >= 0
    s = 0
    while s < len(pr):                                                      # halves the batch on CUDA OOM (3 seeds per card; e1)
        sl = slice(s, min(len(pr), s + bs)); st = np.where(okp[sl], prev[sl], 0)
        try:
            with amp_ctx(amp):
                _, d, _ = m.chunk(torch.tensor(STTOK[st], device=dev), torch.tensor(ops[sl, None], device=dev), decode="all")
        except torch.OutOfMemoryError:
            if bs <= 50:
                raise
            torch.cuda.empty_cache(); bs //= 2; print("restart_k1 OOM: bs", bs, flush=True); continue
        e, _ = decode_to_el(d.cpu().numpy().reshape(-1, NST)); agree[sl] = (e == cur[sl]) & okp[sl] & (cur[sl] >= 0); s = sl.stop
    out = {}
    for i in ii:
        out[i] = True
    for i, a in zip(pr, agree):
        if not a:
            out[i] = False
    return np.array([out[i] for i in ii], bool)


@torch.no_grad()
def rollout_var(m, X, rows, lens, starts, dev, amp, bs=250):
    """self103.rollout_var with a per-problem start state (§110b; all zeros = the identity, as §110). Returns per problem:
    decoded element per position, chunk-end validity, the final element, and the tokens of the checkpoint before the last
    chunk (the start text if L <= 4)."""
    out_el, ok_ck, fin, before_last = [None] * len(rows), np.zeros(len(rows), bool), np.full(len(rows), -1), np.zeros((len(rows), NST), np.int64)
    for L in sorted(set(lens.tolist())):
        idx = np.where(lens == L)[0]
        s = 0
        while s < len(idx):                                                 # halves the batch on CUDA OOM
            ii = idx[s:s + bs]; n = len(ii); ops = X[rows[ii], :L]
            start = torch.tensor(STTOK[starts[ii]], device=dev); els = np.zeros((n, L), np.int64); valid = np.ones(n, bool)
            try:
                for c in range(0, L, K):
                    if c + K >= L:
                        before_last[ii] = start.cpu().numpy()
                    with amp_ctx(amp):
                        _, d, _ = m.chunk(start, torch.tensor(ops[:, c:c + K], device=dev), decode="all")
                    toks = d.cpu().numpy(); e, _ = decode_to_el(toks.reshape(-1, NST)); e = e.reshape(n, -1)
                    els[:, c:c + e.shape[1]] = e; valid &= e[:, -1] >= 0
                    start = torch.tensor(toks[:, -1], device=dev)
            except torch.OutOfMemoryError:
                if bs <= 25:
                    raise
                torch.cuda.empty_cache(); bs //= 2; print("rollout_var OOM: bs", bs, flush=True); continue
            s += n
            for t, i in enumerate(ii):
                out_el[i] = els[t]
            ok_ck[ii] = valid; fin[ii] = els[:, -1]
    return out_el, ok_ck, fin, before_last


def true_states(X, rows, lens, starts):
    """Truth (logging and the final-answer filter only): the state after every op from each problem's start."""
    out = [None] * len(rows)
    for L in sorted(set(lens.tolist())):
        idx = np.where(lens == L)[0]; tc = true_chain(starts[idx], X[rows[idx], :L])
        for t, i in enumerate(idx):
            out[i] = tc[t]
    return out


def items_from_pool(X, pool):
    """§110 STUDENT: training items from a saved pool_final.npz exactly as label_round builds them (kept problems:
    per-slot pseudo-label targets on every chunk, each chunk from the previous pseudo-label's text; discarded problems:
    the final answer only, on the last chunk, from the teacher's own decoded checkpoint before it)."""
    g = {}

    def add(cl, st, op, tg):
        g.setdefault(cl, ([], [], [])); g[cl][0].append(st); g[cl][1].append(op); g[cl][2].append(tg)
    st0 = pool["starts"] if "starts" in pool else np.zeros(len(pool["rows"]), np.int64)   # pools saved before §110b: identity starts
    for i in range(len(pool["rows"])):
        r, L = int(pool["rows"][i]), int(pool["lens"][i]); ops = X[r, :L]
        if pool["keep"][i]:
            tgt = pool["targets"][i, :L].astype(np.int64)
            for c in range(0, L, K):
                add(min(K, L - c), STTOK[int(st0[i])] if c == 0 else STTOK[tgt[c - 1]], ops[c:c + K], tgt[c:c + K])
        else:
            c = ((L - 1) // K) * K; cl = L - c; tg = np.full(cl, -1); tg[-1] = int(pool["true_final"][i])
            add(cl, pool["before_last"][i].astype(np.int64), ops[c:], tg)
    return {cl: (np.array(a), np.array(b), np.array(t)) for cl, (a, b, t) in g.items()}


def items_for(X, pool, kc, fallback=False):
    """§117 K-CURR items (note 117-1) from a labelled pool: kept problems are cut into chunks of kc, each from the
    pseudo-label text at the cut (the start text at 0); discarded problems give the final answer on their last kc-chunk,
    from the model's own decode at the cut (dropped if that decode does not parse; a label-free validity filter).
    No true intermediate state enters (targets = the model's own filtered decodes + the true FINAL answer)."""
    g = {}

    def add(cl, st, op, tg):
        g.setdefault(cl, ([], [], [])); g[cl][0].append(st); g[cl][1].append(op); g[cl][2].append(tg)
    for i in range(len(pool["rows"])):
        r, L = int(pool["rows"][i]), int(pool["lens"][i]); ops = X[r, :L]; d = pool["targets"][i, :L].astype(np.int64); s0 = int(pool["starts"][i])
        if pool["keep"][i]:
            for c in range(0, L, kc):
                add(min(kc, L - c), STTOK[s0] if c == 0 else STTOK[d[c - 1]], ops[c:c + kc], d[c:c + kc])
        else:
            c = ((L - 1) // kc) * kc
            if c > 0 and d[c - 1] < 0:
                if not fallback:
                    continue
                while c > 0 and d[c - 1] < 0:                                # only when nothing else is trainable: the nearest earlier valid cut
                    c -= kc
            cl = L - c; tg = np.full(cl, -1); tg[-1] = int(pool["true_final"][i])
            add(cl, STTOK[s0] if c == 0 else STTOK[d[c - 1]], ops[c:], tg)
    return {cl: (np.array(a), np.array(b), np.array(t)) for cl, (a, b, t) in g.items()}


def label_round(m, X, Y, rows, lens, starts, dev, amp):
    """One seed's pool: rollouts, filters, items and stats (truth for logging only, except the final answer)."""
    m.eval(); decs, okck, fin, before = rollout_var(m, X, rows, lens, starts, dev, amp)
    N = len(rows); tst = true_states(X, rows, lens, starts); true_fin = np.array([t[-1] for t in tst])
    f_ok = okck & (fin == true_fin); fc = np.where(f_ok)[0]
    sc = np.zeros(N, bool); sc[fc] = restart_k1(m, X, rows, lens, decs, fc, dev, amp, starts) if len(fc) else []
    m.train(); keep = sc
    truth = np.array([np.array_equal(decs[i], tst[i]) for i in range(N)])
    gate = float(sc[fc].mean()) if len(fc) else 0.0
    tr_fc = float(truth[fc].mean()) if len(fc) else None
    oh_all = np.bincount(np.concatenate([X[rows[i], :lens[i]] for i in range(N)]), minlength=120)
    oh_keep = np.bincount(np.concatenate([X[rows[i], :lens[i]] for i in np.where(keep)[0]]), minlength=120) if keep.any() else np.zeros(120, np.int64)
    pa, pk = oh_all / max(1, oh_all.sum()), oh_keep / max(1, oh_keep.sum())
    stats = {"pool": N, "filtered_by": {"validity": float((~okck).mean()), "final": float((okck & ~f_ok).mean()),
                                         "self_consistency": float((f_ok & ~sc).mean())},
             "kept_share": float(keep.mean()), "gate": gate, "acc_new_length_before": float(f_ok.mean()),
             "acc_L5plus": float(f_ok[lens >= 5].mean()) if (lens >= 5).any() else None,
             "truth_precision_kept": float(truth[keep].mean()) if keep.any() else None, "truth_rate_final_correct": tr_fc,
             "agreement_vs_truth_gap": (gate - tr_fc) if tr_fc is not None else None,
             "op_hist": {"all": oh_all.tolist(), "kept": oh_keep.tolist()}, "op_tv": float(0.5 * np.abs(pa - pk).sum()) if keep.any() else None}
    g = {}

    def add(cl, st, op, tg):
        g.setdefault(cl, ([], [], [])); g[cl][0].append(st); g[cl][1].append(op); g[cl][2].append(tg)
    for i in range(N):
        r, L = rows[i], lens[i]; ops = X[r, :L]
        if keep[i]:
            tgt = decs[i]
            for c in range(0, L, K):
                add(min(K, L - c), STTOK[starts[i]] if c == 0 else STTOK[tgt[c - 1]], ops[c:c + K], tgt[c:c + K])
        else:
            c = ((L - 1) // K) * K; cl = L - c; tg = np.full(cl, -1); tg[-1] = true_fin[i]
            add(cl, before[i], ops[c:], tg)
    items = {cl: (np.array(a), np.array(b), np.array(t)) for cl, (a, b, t) in g.items()}
    Lm = int(lens.max()); T = np.full((N, Lm), -1, np.int16)
    for i in range(N):
        T[i, :lens[i]] = decs[i]
    pool = {"rows": rows.astype(np.int32), "starts": starts.astype(np.int16), "lens": lens.astype(np.int16), "keep": keep, "targets": T, "before_last": before.astype(np.int32),
            "true_final": true_fin.astype(np.int16)}
    return items, stats, pool


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--r0_steps", type=int, default=1200)
    ap.add_argument("--final_steps", type=int, default=4000); ap.add_argument("--pool", type=int, default=50000); ap.add_argument("--lengths", default="4,8,16,32")
    ap.add_argument("--max_insert", type=int, default=2)
    ap.add_argument("--random_start", action="store_true", help="§110b: problems of length >= 3 start from a uniform random given state (as text)")
    ap.add_argument("--cov", type=float, default=None, help="§112: R0 shorts restricted to T_f with target coverage f (note 112-1)")
    ap.add_argument("--markov", action="store_true", help="§117 MK: Markov segments + own-segment forks from step 0")
    ap.add_argument("--st_switch", action="store_true", help="§117 ST: masked-fork KV chunks until final-answer acc >= 0.5 on 5k length 5-6 problems (round ends) or step 12000, then Markov for 12k + final relabel")
    ap.add_argument("--switch_n", type=int, default=5000, help="< 5000: smoke only"); ap.add_argument("--switch_at", type=int, default=12000, help="!= 12000: smoke only")
    ap.add_argument("--switch_len", type=int, default=12000, help="Markov steps after the switch; != 12000: smoke only")
    ap.add_argument("--kcurr", action="store_true", help="§117: training chunks cut at k_train = 1 / 2 / 4 (steps <= 2k / <= 6k / later) via items_for")
    ap.add_argument("--opdef", action="store_true", help="§116 OPDEF: T_f = only the 120 (e, a) transitions (g = 0)")
    ap.add_argument("--no_r0", action="store_true", help="§116 NONE: no R0 / no shorts; its updates go to round 1; insertion never below 5")
    ap.add_argument("--foothold_n", type=int, default=5000, help="< 5000: smoke only")
    ap.add_argument("--copy_frac", type=float, default=0.0, help="§116 NONE+COPY (note 116-2): share of each batch's rows that are length-0 copy problems")
    ap.add_argument("--foothold_log", action="store_true", help="§116: per round foothold_yield and acc_by_length on 5k fixed problems (truth, logged only)")
    ap.add_argument("--reader", action="store_true", help="§113: §111-READ's auxiliary reader at every targeted slot (the model's own filtered targets)")
    ap.add_argument("--student_pool", default=None, help="§110 STUDENT: train a fresh model --steps on this pool_final.npz only (no R0 / rounds)")
    ap.add_argument("--save_rounds", action="store_true", help="§112 (Audit): fp16 model at each round boundary -> <out>/rounds/")
    ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=s5.TEST_N); ap.add_argument("--ext_n", type=int, default=500)
    ap.add_argument("--ext_len", type=int, default=2048, help="< 2048: smoke only"); ap.add_argument("--check_n", type=int, default=100_000, help="< 100k: smoke only")
    ap.add_argument("--donor_n", type=int, default=2000); ap.add_argument("--data_dir", default=os.path.join(ROOT, "bench", "data", "s5"))
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = a.device; torch.manual_seed(a.seed)
    rng = np.random.default_rng(11000 + a.seed); drng = np.random.default_rng(a.seed)          # pools / draws, per seed
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    s5.check_interface()
    Xtr, Ytr = s5.load_csv(os.path.join(a.data_dir, "S5=128.csv")); s5.check_labels(Xtr, Ytr)
    Xte, Yte = s5.load_csv(os.path.join(a.data_dir, "S5=512.csv"), limit=a.test_n); s5.check_labels(Xte, Yte)
    Tf, g_inner = build_tf(a.cov, a.seed) if a.cov is not None else (None, None)
    if a.opdef:                                                             # §116 OPDEF: only (e, a); length-2 shorts only with an identity first op
        Tf = np.zeros((120, 120), bool); Tf[0, :] = True; g_inner = 0.0; a.cov = 120 / 14400
    covinfo = {} if Tf is None else {"coverage_target": a.cov, "coverage_tf": float(Tf.mean()), "inner_fraction_g": g_inner}
    os.makedirs(a.out, exist_ok=True); json.dump({**vars(a), **covinfo}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    m = Gen110(s5.load_gpt2(dev)).to(dev); m.reader = a.reader; opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    if a.markov:
        m.set_markov(True)
    lengths = [int(x) for x in a.lengths.split(",")]; ck = os.path.join(a.out, "ckpt.pt"); lp = os.path.join(a.out, "log.jsonl")
    S = {"step": 0, "short": 0, "rounds": [], "prev": 2, "remaining": a.steps - a.r0_steps - a.final_steps, "todo": list(lengths),
         "ins": 0, "phase": "R0", "sec": 0}
    if a.no_r0:                                                             # §116 NONE: R0's updates go to round 1; lengths start at L_max >= 5
        assert min(lengths) >= 5, "NONE needs --lengths with L_max >= 5 (e.g. 6,8,16,32)"
        S.update({"phase": "rounds", "prev": 4, "bonus": a.r0_steps})
    t0 = time.time()
    if os.path.exists(ck):
        c = torch.load(ck, map_location=dev, weights_only=False); m.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        rng.bit_generator.state = c["rng"]; drng.bit_generator.state = c["drng"]; torch.set_rng_state(c["torch_rng"].cpu())
        if dev.startswith("cuda"):
            torch.cuda.set_rng_state(c["cuda_rng"].cpu(), dev)
        S = c["S"]; t0 -= S["sec"]
        if S.get("switched"):
            m.set_markov(True)
        keep = [r for r in open(lp, encoding="utf-8") if r.strip() and json.loads(r)["step"] <= S["step"]] if os.path.exists(lp) else []
        open(lp, "w", newline="\n").writelines(keep); print("RESUMED", S["step"], S["phase"], flush=True); del c
    log = open(lp, "a", newline="\n")

    def save_ck():
        S["sec"] = round(time.time() - t0)
        c = {"model": m.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state, "drng": drng.bit_generator.state,
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None, "S": S}
        torch.save(c, ck + ".tmp"); os.replace(ck + ".tmp", ck)

    def do_switch(left_in_round):
        """§117 ST: switch to Markov segments; 12k Markov steps (incl. the rest of the current round) + the final relabel."""
        m.set_markov(True); S["switched"] = True; S["switch_step"] = S["step"]; S["remaining"] = max(0, a.switch_len - left_in_round)
        S["H"] = S["step"] + a.switch_len + a.final_steps; print("SWITCH at", S["step"], flush=True)

    def kc_now(step):
        return 1 if step <= 2000 else (2 if step <= 6000 else K)

    def train_round(items, n_steps, name, short, pool=None):
        hist = []; rhist = []; chist = []; cache = {}
        for i_ in range(n_steps):
            S["step"] += 1; S["short"] += int(short); step = S["step"]; H = S.get("H", a.steps)
            if a.st_switch and not S.get("switched") and step == a.switch_at:   # note 117-1: forced switch at 12,000 exactly
                do_switch(n_steps - i_ - 1); H = S["H"]
            for gr in opt.param_groups:
                gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * min(step, H) / H))
            if a.kcurr and pool is not None:                               # §117 K-CURR items (rebuilt when k_train changes)
                kc = kc_now(step)
                if kc not in cache:
                    cache[kc] = items_for(Xtr, pool, kc) or items_for(Xtr, pool, kc, fallback=True)
                    if not items_for(Xtr, pool, kc):
                        S.setdefault("fallback_items", []).append([S["step"], name, kc]); print("fallback items", S["step"], name, kc, flush=True)
                items = cache[kc]
            nc = int(round(a.copy_frac * a.bs)) if a.copy_frac else 0     # §116 NONE+COPY: copy rows replace ordinary rows
            st, o_, tg = draw(items, drng, a.bs - nc, dev)
            if short:                                                       # §112 (112-1 addendum): the transitions the drawn shorts label
                oo = o_.cpu().numpy(); drawn[0, oo[:, 0]] = True
                if oo.shape[1] == 2:
                    drawn[s5.COMP[0, oo[:, 0]], oo[:, 1]] = True
            with amp_ctx(amp):
                loss = m.chunk_masked(st, o_, tg)
                if nc:
                    cs = drng.integers(0, 120, nc)
                    lc = m.copy_loss(torch.tensor(STTOK[cs], device=dev), torch.tensor(cs, device=dev)); chist.append(float(lc.detach()))
                    loss = (loss * (a.bs - nc) + lc * nc) / a.bs
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(loss.item())
            if m.reader:
                rhist.append(m.parts[1])
            if step % 100 == 0:
                rec = {"step": step, "round": name, "loss": float(np.mean(hist)), "sec": round(time.time() - t0)}; hist = []
                if rhist:
                    rec["reader_loss"] = float(np.mean(rhist)); rhist.clear()
                if chist:                                                   # §116 NONE+COPY copy CE (Theory 116-4 addendum)
                    rec["copy_loss"] = float(np.mean(chist)); chist.clear()
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)

    def pool_of(Lmax):
        rows, lens = make_pool(rng, Xtr, a.pool, 3, Lmax)
        return rows, lens, (rng.integers(0, 120, a.pool) if a.random_start else np.zeros(a.pool, np.int64))

    drawn = S.setdefault("drawn", np.zeros((120, 120), bool)); S.setdefault("kept_trans", np.zeros((120, 120), bool))
    S.setdefault("per_round_split", [])

    def round_end(name):
        """§112: k = 1 accuracy on the drawn-labelled / never-labelled transitions after each training round (+ fp16 model)."""
        m.eval(); kt, kd = k1_table(m, dev, amp, return_dec=True); m.train(); lab = S["drawn"]
        moves = s5.COMP[np.arange(120)[:, None], np.arange(120)[None]] != np.arange(120)[:, None]   # ops that change the state
        S["per_round_split"].append({"step": S["step"], "round": name, "labelled_acc": float(kt[lab].mean()) if lab.any() else None,
                                     "never_labelled_acc": float(kt[~lab].mean()) if (~lab).any() else None, "all_acc": float(kt.mean()),
                                     "stay_rate": float((kd == np.arange(120)[:, None])[moves].mean())})   # §116-2 (truth, logged only)
        if a.foothold_log:                                                  # §116 (truth, logged only; never used in training)
            fr = np.random.default_rng(11600); rows, lens = make_pool(fr, Xtr, a.foothold_n, 3, 32); st0 = np.zeros(a.foothold_n, np.int64)
            m.eval(); decs, _, fin, _ = rollout_var(m, Xtr, rows, lens, st0, dev, amp); m.train(); tst = true_states(Xtr, rows, lens, st0)
            okf = np.array([d[-1] == t[-1] for d, t in zip(decs, tst)])
            cut_ok = np.array([all(decs[i][c + K - 1] == tst[i][c + K - 1] for c in range(0, lens[i], K) if c + K < lens[i]) for i in range(a.foothold_n)])
            big = lens > 4
            S["per_round_split"][-1]["foothold_yield"] = float(cut_ok[big].mean())
            S["per_round_split"][-1]["acc_by_length"] = {str(L): float(okf[lens == L].mean()) for L in range(3, 33) if (lens == L).any()}
        print("k1 split", {k_: v for k_, v in S["per_round_split"][-1].items() if k_ != "acc_by_length"}, flush=True)
        if a.save_rounds:
            os.makedirs(os.path.join(a.out, "rounds"), exist_ok=True)
            torch.save({k_: v.half() if v.is_floating_point() else v for k_, v in m.state_dict().items()},
                       os.path.join(a.out, "rounds", f"r{len(S['per_round_split']):02d}_{name}_step{S['step']}.pt"))

    if a.student_pool and S["phase"] == "R0":                               # §110 STUDENT (conditional arm)
        pool = dict(np.load(a.student_pool)); items = items_from_pool(Xtr, pool)
        S["rounds"].append({"student_of": a.student_pool, "pool": int(len(pool["rows"])), "kept_share": float(pool["keep"].mean()),
                            "items": {str(cl): int(len(v[0])) for cl, v in items.items()}, "steps": a.steps})
        print("student", S["rounds"][-1], flush=True)
        train_round(items, a.steps, "student", False); S["phase"] = "done"
    if S["phase"] == "R0":
        r0 = r0_items_cov(rng, Tf) if Tf is not None else r0_items(rng, Xtr, Ytr)[0]
        o1 = np.concatenate([r0[1][1][:, 0], r0[2][1][:, 0]])
        direct = {(0, int(x)) for x in o1}; two = direct | {(int(s5.COMP[0, x]), int(y)) for x, y in r0[2][1]}
        train_round(r0, a.r0_steps, "R0", True)
        S["rounds"].append({"length": "1-2", "steps": a.r0_steps, "targets": "final answer only", "start": "identity",
                            "short_transition_coverage": {"direct": len(direct) / 14400, "incl_2step": len(two) / 14400},
                            "coverage_pool": float(r0_labelled(r0).mean()), "coverage_drawn": float(drawn.mean()), **covinfo})
        round_end("R0"); S["phase"] = "rounds"; save_ck()
    while S["phase"] == "rounds" and S["todo"]:
        Lmax = S["todo"][0]; rows, lens, starts = pool_of(Lmax)
        items, st, pl = label_round(m, Xtr, Ytr, rows, lens, starts, dev, amp); prev = S["prev"]; mid = (prev + Lmax) // 2
        why = [w for w, bad in (("gate %.4f < %.3f" % (st["gate"], GATE), st["gate"] < GATE),
                                ("kept share %.3f < 0.20" % st["kept_share"], st["kept_share"] < 0.20)) if bad]
        if why and S["ins"] < a.max_insert and prev < mid < Lmax:
            st.update({"length": Lmax, "action": "; ".join(why) + f": insert length {mid}"}); S["rounds"].append(st)
            S["todo"].insert(0, mid); S["ins"] += 1; print("insert length", mid, why, flush=True); continue
        n_steps = S["remaining"] // len(S["todo"]); S["remaining"] -= n_steps; S["todo"].pop(0)
        if S.get("bonus"):
            n_steps += S["bonus"]; S["bonus"] = 0
        st.update({"length": Lmax, "steps": n_steps}); S["rounds"].append(st); S["ins"] = 0
        print("round", {k: v for k, v in st.items() if k != "op_hist"}, flush=True)
        S["kept_trans"] |= kept_transitions(Xtr, pl, true_states(Xtr, rows, lens, starts))
        if a.kcurr:                                                         # §117 (truth, logged only): reader/fork target precision
            tst_ = true_states(Xtr, rows, lens, starts); kk = np.where(pl["keep"])[0]
            sl_ok = [int((pl["targets"][i, :lens[i]] == tst_[i]).sum()) for i in kk]; sl_n = sum(int(lens[i]) for i in kk)
            nd = int((~pl["keep"]).sum()); st["reader_target_precision"] = (sum(sl_ok) + nd) / max(1, sl_n + nd)
        train_round(items, n_steps, f"L{Lmax}", False, pl); S["prev"] = Lmax; round_end(f"L{Lmax}")
        if a.st_switch and not S.get("switched"):                          # label-free check: final answers on 5k length 5-6
            fr = np.random.default_rng(11700 + S["step"]); r5, l5 = make_pool(fr, Xtr, a.switch_n, 5, 6); z5 = np.zeros(a.switch_n, np.int64)
            m.eval(); _, _, f5, _ = rollout_var(m, Xtr, r5, l5, z5, dev, amp); m.train()
            acc56 = float(np.mean(f5 == np.array([t[-1] for t in true_states(Xtr, r5, l5, z5)]))); st["switch_check_acc56"] = acc56
            if acc56 >= 0.5:
                do_switch(0)
        save_ck()
        if not S["todo"] and S["remaining"] > 0:                            # post-switch budget left: more rounds at L_max
            S["todo"].append(lengths[-1])
    if S["phase"] == "rounds":
        S["phase"] = "final"
    if S["phase"] == "final":
        Lmax = lengths[-1]; rows, lens, starts = pool_of(Lmax)
        items, st, pool = label_round(m, Xtr, Ytr, rows, lens, starts, dev, amp)
        np.savez_compressed(os.path.join(a.out, "pool_final.npz"), **pool)
        st.update({"length": Lmax, "steps": a.final_steps, "action": "final relabelling"}); S["rounds"].append(st)
        print("final", {k: v for k, v in st.items() if k != "op_hist"}, flush=True)
        S["kept_trans"] |= kept_transitions(Xtr, pool, true_states(Xtr, rows, lens, starts))
        train_round(items, a.final_steps, "final", False, pool); round_end("final"); S["phase"] = "done"
    train_sec = round(time.time() - t0)
    torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    if os.path.exists(ck):
        os.remove(ck)
    del opt; m.eval(); t1 = time.time()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    r = {"arm": ("self117_mk" if a.markov else "self117_st") if (a.markov or a.st_switch) else ("self116_opdef" if a.opdef else ("self116_nonecopy" if a.copy_frac else "self116_none")) if (a.opdef or a.no_r0) else "self113" if a.reader else "self110_student" if a.student_pool else "self112" if a.cov is not None else ("self110b" if a.random_start else "self110"), "start_ge3": "random" if a.random_start else "identity", "seed": a.seed, "k": K, "fork": "mask", "train_sec": train_sec, "eval_dtype": str(amp), "test_sequences": len(Xte),
         "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
         "short_fraction": S["short"] / max(1, S["step"]), "updates": S["step"], "rounds": S["rounds"]}
    pred, comp, val = rollout(m, Xte, K, dev, amp, a.eval_bs); cum = s5.cumulative(pred, Yte)
    np.save(os.path.join(a.out, "preds_test.npy"), pred)
    r["cumulative"] = {str(t): float(cum[t - 1]) for t in (128, 256, 512) if t <= Xte.shape[1]}
    r["final_state_acc_512"] = float((pred[:, -1] == Yte[:, -1]).mean()); r["compute_512_per_problem"] = comp; r["checkpoint_validity"] = val
    if a.ext_n:
        Xx, Yx = s5.load_csv(os.path.join(a.data_dir, "S5=2048.csv"), limit=a.ext_n); s5.check_labels(Xx, Yx); Xx, Yx = Xx[:, :a.ext_len], Yx[:, :a.ext_len]
        px, _, _ = rollout(m, Xx, K, dev, amp, a.eval_bs); cx = s5.cumulative(px, Yx)
        r["extension_2048"] = {"n": len(Xx), "len": Xx.shape[1], "cumulative": {str(t): float(cx[t - 1]) for t in (512, 1024, 2048) if t <= Xx.shape[1]}}
    r["chunk_check"] = chunk_check(m, K, dev, amp, a.eval_bs, n=a.check_n)
    mu = mean_slot_inputs(m, Xtr, Ytr, K, dev, amp)
    r["removal"] = {ab: chunk_check(m, K, dev, amp, a.eval_bs, n=min(10_000, a.check_n), seed=7106, ablate=ab, mean_in=mu) for ab in ("zero", "mean")}
    r["donor"] = donor(m, K, dev, amp, a.eval_bs, n=a.donor_n)
    if m.markov:                                                            # §117 (eval only): per-hand-off and the shuffled no-read flag
        from boxes.mk108 import handoff, rel_ratio_final
        r["per_handoff_acc"] = handoff(m, K, dev, amp, a.eval_bs, n=min(20000, a.check_n)); r["rel_final"] = rel_ratio_final(m, Xtr, Ytr, K, dev, amp)
        r["no_read_final"] = bool(r["rel_final"]["rel_ratio_final_shuffled"] >= 0.5)
    r["switch_step"] = S.get("switch_step"); r["markov_at_eval"] = bool(m.markov)
    r["fallback_items"] = S.get("fallback_items", [])                      # note 117-2: [step, round, k_train] where the fallback was used
    r["edit_iia"] = edit_iia(m, K, dev, amp, a.eval_bs, n=min(5000, a.check_n))
    if m.reader:                                                            # §113: the reader's own decode per slot on fresh chunks
        rng_r = np.random.default_rng(11301); nr = min(20000, a.check_n); S_ = rng_r.integers(0, 120, nr); O_ = rng_r.integers(0, 120, (nr, K)); Cr = []
        with torch.no_grad():
            for i in range(0, nr, a.eval_bs):
                with amp_ctx(amp):
                    _, _, ins = m.chunk(m.start_from(els=S_[i:i + a.eval_bs], dev=dev), torch.tensor(O_[i:i + a.eval_bs], device=dev), decode=None)
                    d = torch.stack([m.read_greedy(ins[:, j]) for j in range(K)], 1)
                el, _ = decode_to_el(d.cpu().numpy().reshape(-1, NST)); Cr.append(el.reshape(-1, K) == true_chain(S_[i:i + a.eval_bs], O_[i:i + a.eval_bs]))
        r["reader_acc"] = {"n": nr, "per_slot_acc": np.concatenate(Cr).mean(0).tolist()}
    kt = k1_table(m, dev, amp); lab = S["drawn"]; nl = ~lab; kp = S["kept_trans"]          # §112 keys (drawn set = labelled)
    acc = lambda msk: float(kt[msk].mean()) if msk.any() else None  # noqa: E731
    r.update({**covinfo, "coverage_drawn": float(lab.mean()), "coverage_realised": float(lab.mean()),
              "k1_labelled_acc": acc(lab), "k1_never_labelled_acc": acc(nl), "k1_certified": bool(kt.all()), "k1_all_acc": float(kt.mean()),
              "k1_never_labelled_in_pool": acc(nl & kp), "k1_never_labelled_not_in_pool": acc(nl & ~kp),
              "n_never_labelled": int(nl.sum()), "n_never_labelled_in_pool": int((nl & kp).sum()), "per_round_split": S["per_round_split"]})
    r["eval_sec"] = round(time.time() - t1)
    json.dump(r, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps(r["cumulative"]), json.dumps(r["donor"]), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
