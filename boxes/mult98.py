"""THEORY §98 (notes 98-1, 98-2): §97's general latent thinker on multi-digit multiplication (the implicit-CoT repo's
{4,5}_by_{4,5}_mult data, digits least-significant first, "a * b||<steps> #### <product>"), with Coconut, text-CoT
and no-CoT baselines on the same data and budget (24k steps, bs 64, AdamW lr 1e-4, 100-step warmup + cosine).
Arms (--arm):
  genm  GEN-M (§97's gen arm with a task adapter). Chunk = [E(" a1 .. an *")] [E(start text)] [op_1] [slot_1] ...
        [op_k] [slot_k]; op j = the multiplier digit b_j (one token); slot j's input = the final hidden at op j's token.
        Step text j (the dataset's own CoT segments split on " + "): step 1 = " P_1", step j > 1 = " + P_j ( S_j )", the
        last step " + P_n". Loss: at every slot, fork the cache and teacher-force step j's text; after the last slot of
        the last chunk, fork and teacher-force " #### <product>"; CE through the tied LM head summed over the fork's
        tokens, mean over forks. k = 4: 4x4 = one chunk; 5x5 = ops 1-4, snap (the greedy step-4 text re-embedded as the
        next start, history cut), op 5. Training chunks are the test-time chunks (one chunk index drawn per batch; the
        start = the true previous step text).
  mask  GEN-M-MASK (Audit (a)): as genm, but every fork sees only the slot's own cached K/V (all layers) and its own
        tokens, at their true positions; the chain is unchanged, so the state reaches the decoder only through the slot.
  coco  COCO-M (Audit (b)): Coconut's stage curriculum. Sequence = [question "a * b ||"] [thoughts] [remaining step
        texts] [" #### <product>"]; stage s (n + 1 equal shares of the updates) replaces the first s steps with s
        continuous thoughts (a thought's input = the last hidden state); CE on the remaining text and the answer.
        Eval at the final stage (n thoughts, then the answer); removal = all n thought inputs zeroed / mean-replaced.
  tc    TC-M: text CoT, the question then all step texts and the answer (teacher-forced CE on everything after the
        question); eval greedy-decodes the CoT and the answer at their known token count.
  nc    NC-M (note 98-1): the question then " #### <product>" only.
Decoding uses the format's known token counts (as §97's 6-token state); an answer is correct when its digits after
"####" equal the product's digits (the bench scorer).
Eval (eval.json): arm, task, seed, gpu, acc_test (the 1,000 bench items, bench/data/mult_{n}x{n}_test.txt);
per_slot_decode_acc (genm/mask: exact step text per slot in the test rollout); removal.{zero,mean}.answer_acc
(genm/mask: slot inputs 1..k-1 of each chunk; coco: all thoughts) and, for mask, removal.*.per_slot_decode_acc;
compute.{flops, lm_head_rows, sequential_passes} (boxes.core.Meter per problem, full vocabulary; for genm/mask the
method path only: segments, the snap decode and the answer decode; the non-snap step decodes are measurement,
compute.measurement_decode_flops); 5x5 genm/mask: checkpoint_validity.rate (the snapped step-4 text has the format
" + <P digits> ( <S digits> )"), edit_iia.iia (the snapped text replaced by another test item's true step-4 text: share
whose answer equals the donor's S_4 + a * b_5 * 10^4); training.{epochs, flops} (Meter forward FLOPs per trained
problem x 3 for forward + backward, x steps x bs).
    python -m boxes.mult98 train --arm genm --task 4x4 --seed 1 --out boxes/runs_mult/m98_genm_4x4_s1
"""
import os
os.environ["BOXES_NB"] = "5"                                         # boxes.s5 (load_gpt2) asserts it; no effect here
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

K = 4
TOK = {w: bc.tok(" " + w)[0] for w in "0123456789+()*"}
for _w in "0123456789+()*":
    assert len(bc.tok(" " + _w)) == 1
HASH = bc.tok(" ####")                                                    # 2 tokens
QSEP = bc.tok(" ||")
DIGIT_IDS = np.array([TOK[str(d)] for d in range(10)])


class MeterR(bc.Meter):
    """boxes.core.Meter that also counts LM-head rows (so the restricted-vocabulary convention can be recomputed)."""
    def __init__(self, m):
        super().__init__(m); self.rows = 0

    def step(self, T, ctx_before, logit_rows):
        super().step(T, ctx_before, logit_rows); self.rows += logit_rows

    def as_dict(self):
        d = super().as_dict(); d["lm_head_rows"] = self.rows; d["sequential_passes"] = self.passes; return d


# ---------------------------------------------------------------- data
def parse(line):
    q, rest = line.strip().split("||"); a, b = q.split(" * "); cot, prod = rest.split(" #### ")
    segs = cot.split(" + ")
    steps = [segs[0]] + ["+ " + s for s in segs[1:]]
    return a.split(), b.split(), [s.split() for s in steps], prod.split()


def load(path, limit=None):
    """-> dict of int arrays: A (N, na) digits, B (N, nb) digits, steps: list over j of (N, L_j) token ids, ans (N, La)
    answer token ids (" ####" + digits), prod (N, np) digits."""
    A, B, S, P = [], [], None, []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            a, b, steps, prod = parse(line)
            A.append([int(x) for x in a]); B.append([int(x) for x in b]); P.append([int(x) for x in prod])
            if S is None:
                S = [[] for _ in steps]
            for j, st in enumerate(steps):
                S[j].append([TOK[w] for w in st])
    A, B, P = np.array(A), np.array(B), np.array(P)
    ans = np.concatenate([np.tile(HASH, (len(P), 1)), DIGIT_IDS[P]], 1)
    return {"A": A, "B": B, "steps": [np.array(s) for s in S], "ans": ans, "prod": P, "n": B.shape[1]}


def to_int(d):
    """LSB-first digit rows (N, L) -> python ints."""
    return [int("".join(str(x) for x in row[::-1])) for row in d]


def digits_lsb(v, L):
    return [int(c) for c in str(v).zfill(L)[::-1]][:L]


def answer_correct(dec_tokens, true_prod):
    """dec (B, La) greedy answer tokens; correct when the tokens after '####' are exactly the product's digits."""
    t = dec_tokens[:, len(HASH):]
    return (t == DIGIT_IDS[true_prod]).all(1) & (dec_tokens[:, :len(HASH)] == np.array(HASH)).all(1)


# ---------------------------------------------------------------- GEN-M / GEN-M-MASK
class MultGen(nn.Module):
    def __init__(self, g, arm):
        super().__init__()
        self.g = g; self.mask = arm == "mask"
        self.register_buffer("digtok", torch.tensor(DIGIT_IDS))

    def fork_past(self, past):
        if not self.mask:
            return past, None
        P = past[0][0].shape[2]
        return [(k[:, :, -1:], v[:, :, -1:]) for k, v in past], P              # only the slot's own K/V, true positions

    def fork_tf(self, hslot, past, tgt):
        """Teacher-forced fork: CE (summed over the tgt tokens, mean over rows)."""
        fp, p0 = self.fork_past(past); g = self.g
        hf, _ = g(g.embed(tgt[:, :-1]), fp, pos0=p0) if tgt.shape[1] > 1 else (hslot[:, None][:, :0], None)
        lg = g.logits(torch.cat([hslot[:, None], hf], 1)).float()
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape).sum(-1).mean()

    def fork_greedy(self, hslot, past, L, meter=None):
        fp, p0 = self.fork_past(past); g = self.g
        t = g.logits(hslot).argmax(-1); out = [t]
        if meter:
            meter.step(0, 0, 1)
        pos = p0
        for _ in range(L - 1):
            ctx = fp[0][0].shape[2]; h, fp = g(g.embed(t)[:, None], fp, pos0=pos)
            if meter:
                meter.step(1, ctx, 1)
            pos = None if pos is None else pos + 1
            t = g.logits(h[:, 0]).argmax(-1); out.append(t)
        return torch.stack(out, 1)

    def chunk(self, prefix, start, ops, step_tgt=None, ans_tgt=None, step_len=None, ans_len=None, decode="all",
              ablate=None, mean_in=None, meter=None, meter_x=None, snap=False):
        """prefix (B, Lp) tokens; start (B, Ls) tokens or None; ops (B, c) digits. Training: step_tgt list of (B, L_j)
        (+ ans_tgt on the final chunk) -> loss. Eval: step_len list, ans_len (final chunk) -> greedy decodes. snap: the
        last step decode is the method's (metered); the others are measurement (meter_x)."""
        B, c = ops.shape; g = self.g
        parts = [g.embed(prefix)] + ([g.embed(start)] if start is not None else []) + [g.embed(self.digtok[ops[:, 0]])[:, None]]
        x = torch.cat(parts, 1); past = None; losses, decs, ins = [], [], []; ans = None
        for j in range(c):
            ctx = 0 if past is None else past[0][0].shape[2]
            h, past = g(x, past)
            if meter:
                meter.step(x.shape[1], ctx, 0)
            s_in = h[:, -1]
            if ablate in ("zero", "mean") and j < c - 1:
                s_in = torch.zeros_like(s_in) if ablate == "zero" else mean_in[j][None].expand_as(s_in)
            ov = getattr(self, "override", None)                             # donor patch (j, (B, D)); default off
            if ov is not None and ov[0] == j:
                s_in = ov[1].to(s_in.dtype)
            ins.append(s_in)
            ctx = past[0][0].shape[2]; hs, past = g(s_in[:, None], past)
            if meter:
                meter.step(1, ctx, 0)
            hslot = hs[:, 0]
            final = j == c - 1 and (ans_tgt is not None or ans_len is not None)
            # note 98-3: the final slot has ONE fork, the step text followed by " #### <product>" (the dataset's own CoT
            # continuation); two forks from the same slot with different first tokens cannot both be decoded
            if step_tgt is not None:
                tgt = torch.cat([step_tgt[j], ans_tgt], 1) if (final and ans_tgt is not None) else step_tgt[j]
                losses.append(self.fork_tf(hslot, past, tgt))
            if final and ans_len is not None:
                d = self.fork_greedy(hslot, past, step_len[j] + ans_len, meter)       # the method's decode (step n + answer)
                decs.append(d[:, :step_len[j]]); ans = d[:, step_len[j]:]
            elif step_len is not None and (decode == "all" or j == c - 1):
                decs.append(self.fork_greedy(hslot, past, step_len[j], meter if (snap and j == c - 1) else meter_x))
            if j + 1 < c:
                x = g.embed(self.digtok[ops[:, j + 1]])[:, None]
        loss = torch.stack(losses).mean() if losses else None
        return loss, decs, ans, torch.stack(ins, 1)


def chunks_of(n):
    return [(c0, min(c0 + K, n)) for c0 in range(0, n, K)]


def gen_batch(D, rng, bs, dev):
    """One training batch of the test-time chunk ci (drawn per batch) with true starts."""
    n = D["n"]; ch = chunks_of(n); ci = int(rng.integers(len(ch))); c0, c1 = ch[ci]; idx = rng.integers(0, len(D["A"]), bs)
    T = lambda a: torch.tensor(a, device=dev)  # noqa: E731
    prefix = T(np.concatenate([DIGIT_IDS[D["A"][idx]], np.full((bs, 1), TOK["*"])], 1))
    start = T(D["steps"][c0 - 1][idx]) if c0 > 0 else None
    return prefix, start, T(D["B"][idx, c0:c1]), [T(D["steps"][j][idx]) for j in range(c0, c1)], (T(D["ans"][idx]) if c1 == n else None)


@torch.no_grad()
def gen_rollout(m, D, dev, amp, bs, ablate=None, mean_in=None, override=None, meter=None, meter_x=None):
    """Test rollout over the chunks with snaps. Returns per-slot exact decode (N, n) bool, answer correct (N,) bool,
    snapped texts (list per snap of (N, L) arrays). override: (snap index, (N, L) tokens) replaces a snapped text."""
    N, n = D["B"].shape; ch = chunks_of(n); per = np.zeros((N, n), bool); ok = np.zeros(N, bool); snaps = [[] for _ in ch[:-1]]
    for s in range(0, N, bs):
        e = min(N, s + bs); T = lambda a: torch.tensor(a, device=dev)  # noqa: E731
        prefix = T(np.concatenate([DIGIT_IDS[D["A"][s:e]], np.full((e - s, 1), TOK["*"])], 1)); start = None
        for ci, (c0, c1) in enumerate(ch):
            last = c1 == n
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                _, decs, ans, _ = m.chunk(prefix, start, T(D["B"][s:e, c0:c1]), step_len=[D["steps"][j].shape[1] for j in range(c0, c1)],
                                          ans_len=D["ans"].shape[1] if last else None, ablate=ablate,
                                          mean_in=None if mean_in is None else mean_in[ci], meter=meter if s == 0 else None,
                                          meter_x=meter_x if s == 0 else None, snap=not last)
            for jj, d in enumerate(decs):
                per[s:e, c0 + jj] = (d.cpu().numpy() == D["steps"][c0 + jj][s:e]).all(1)
            if not last:
                st = decs[-1].cpu().numpy(); snaps[ci].append(st)
                if override is not None and override[0] == ci:
                    st = override[1][s:e]
                start = T(st)
            else:
                ok[s:e] = answer_correct(ans.cpu().numpy(), D["prod"][s:e])
    return per, ok, [np.concatenate(x) for x in snaps]


def rollout_safe(m, D, dev, amp, bs, **kw):
    """gen_rollout, halving the batch on CUDA OOM (three runs on one card can finish and evaluate together; y2/y3)."""
    while True:
        try:
            return gen_rollout(m, D, dev, amp, bs, **kw)
        except torch.OutOfMemoryError:
            if bs <= 10:
                raise
            torch.cuda.empty_cache(); bs //= 2; print("eval OOM: retry at bs", bs, flush=True)


@torch.no_grad()
def gen_mean_inputs(m, D, dev, amp, n_rows=2000, seed=9805):
    rng = np.random.default_rng(seed); out = []
    for ci, (c0, c1) in enumerate(chunks_of(D["n"])):
        idx = rng.integers(0, len(D["A"]), n_rows); T = lambda a: torch.tensor(a, device=dev)  # noqa: E731
        prefix = T(np.concatenate([DIGIT_IDS[D["A"][idx]], np.full((n_rows, 1), TOK["*"])], 1))
        start = T(D["steps"][c0 - 1][idx]) if c0 > 0 else None
        acc = 0                                                             # 250-row batches (2,000 at once OOM'd beside 2 evals, u3)
        for b0 in range(0, n_rows, 250):
            sl = slice(b0, b0 + 250)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                _, _, _, ins = m.chunk(prefix[sl], None if start is None else start[sl], T(D["B"][idx[sl], c0:c1]))
            acc = acc + ins.float().sum(0)
        out.append(acc / n_rows)
    return out


def valid_step(tokens, L_P, L_S):
    """' + <L_P digits> ( <L_S digits> )' token pattern."""
    t = tokens; dig = np.isin(t, DIGIT_IDS)
    return (t[:, 0] == TOK["+"]) & dig[:, 1:1 + L_P].all(1) & (t[:, 1 + L_P] == TOK["("]) & dig[:, 2 + L_P:2 + L_P + L_S].all(1) & (t[:, -1] == TOK[")"])


# ---------------------------------------------------------------- COCO-M / TC-M / NC-M (one sequence model)
def question(D, idx):
    return np.concatenate([DIGIT_IDS[D["A"][idx]], np.full((len(idx), 1), TOK["*"]), DIGIT_IDS[D["B"][idx]],
                           np.tile(QSEP, (len(idx), 1))], 1)


def seq_forward(g, q, n_thought, text, ablate=None, mean_in=None, meter=None):
    """[q] [n_thought continuous thoughts] [text (teacher-forced)] -> hidden over the text positions' predictors
    (the position before each text token), plus the cache. Thought t's input = the last hidden state."""
    h, past = g(g.embed(q)); last = h[:, -1]
    if meter:
        meter.step(q.shape[1], 0, 0)
    for t in range(n_thought):
        x = last
        if ablate == "zero":
            x = torch.zeros_like(x)
        elif ablate == "mean":
            x = mean_in[t][None].expand_as(x).to(x.dtype)
        ctx = past[0][0].shape[2]; hh, past = g(x[:, None], past); last = hh[:, -1]
        if meter:
            meter.step(1, ctx, 0)
    if text is None:
        return last, past
    ctx = past[0][0].shape[2]; ht, past2 = g(g.embed(text[:, :-1]), past) if text.shape[1] > 1 else (last[:, None][:, :0], past)
    if meter:
        meter.step(text.shape[1] - 1, ctx, 0)
    return torch.cat([last[:, None], ht], 1), past2


@torch.no_grad()
def seq_greedy(g, last, past, L, meter=None):
    t = g.logits(last).argmax(-1); out = [t]
    if meter:
        meter.step(0, 0, 1)
    for _ in range(L - 1):
        ctx = past[0][0].shape[2]; h, past = g(g.embed(t)[:, None], past)
        if meter:
            meter.step(1, ctx, 1)
        t = g.logits(h[:, 0]).argmax(-1); out.append(t)
    return torch.stack(out, 1)


def gen_items(A, B):
    """§164: the bench's implicit-CoT text for operand rows A (N, na), B (N, nb), digits LSB first: step 1 = P_1, step j > 1 = "+ P_j ( S_j )"
    (the last step "+ P_n"), P_j = (j - 1) zeros + a x b_j over na + 1 digits, S_j = sum_{i<=j} a b_i 10^(i-1) over na + j digits; answer
    " ####" + the na + nb product digits. Same dict as load()."""
    N, na = A.shape; nb = B.shape[1]; a = to_int(A); S = [[] for _ in range(nb)]; P = []
    for r in range(N):
        acc = 0
        for j in range(nb):
            pj = a[r] * int(B[r, j]); acc += pj * 10 ** j; w = [0] * j + digits_lsb(pj, na + 1)
            st = ([] if j == 0 else ["+"]) + [str(x) for x in w] + (["("] + [str(x) for x in digits_lsb(acc, na + j + 1)] + [")"] if 0 < j < nb - 1 else [])
            S[j].append([TOK[t] for t in st])
        P.append(digits_lsb(acc, na + nb))
    P = np.array(P); ans = np.concatenate([np.tile(HASH, (N, 1)), DIGIT_IDS[P]], 1)
    return {"A": np.asarray(A), "B": np.asarray(B), "steps": [np.array(x) for x in S], "ans": ans, "prod": P, "n": nb}


def mix_draw(rng, bs, ns, excl):
    """§164: one n per batch from ns; A 4-digit, B n-digit (multpair's gen_operands); 4x4 rows equal to a bench test item (either order) redrawn."""
    from boxes.multcol import gen_operands
    nb = int(ns[int(rng.integers(0, len(ns)))]); A, B = gen_operands(rng, bs, 4), gen_operands(rng, bs, nb)
    if nb == 4 and excl:
        for _ in range(100):
            bad = np.array([(x.tobytes(), y.tobytes()) in excl for x, y in zip(A, B)])
            if not bad.any():
                break
            A[bad], B[bad] = gen_operands(rng, int(bad.sum()), 4), gen_operands(rng, int(bad.sum()), 4)
    return gen_items(A, B)


@torch.no_grad()
def seq_eval_digits(g, D, dev, amp, bs, arm, meter=None):
    """as seq_eval, returning the decoded answer digits (N, np) (-1 where the token is not a digit) and the exact-answer flags."""
    N = len(D["A"]); n = D["n"]; L_text = D["ans"].shape[1] + (sum(D["steps"][j].shape[1] for j in range(n)) if arm == "tc" else 0)
    lut = {int(t): i for i, t in enumerate(DIGIT_IDS)}; dig = np.full(D["prod"].shape, -1); ok = np.zeros(N, bool)
    for s_ in range(0, N, bs):
        idx = np.arange(s_, min(N, s_ + bs))
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            last, past = seq_forward(g, torch.tensor(question(D, idx), device=dev), 0, None, meter=meter if s_ == 0 else None)
            dec = seq_greedy(g, last, past, L_text, meter if s_ == 0 else None).cpu().numpy()
        at = dec[:, -D["ans"].shape[1]:]; ok[idx] = answer_correct(at, D["prod"][idx]); dig[idx] = np.vectorize(lambda t: lut.get(int(t), -1))(at[:, len(HASH):])
    return dig, ok



@torch.no_grad()
def seq_eval_anchored(g, D, dev, amp, bs, arm, extra=50):
    """§164 (Audit): greedy decode with budget L_text + extra; anchored = the na + nb digits right after the FIRST " ####" in the decode;
    position_locked = the old read (the last answer-length tokens of the first L_text, " ####" required at that offset); cot_len = tokens before
    the first " ####" (the true schedule: sum of the step texts; nc: 0)."""
    N = len(D["A"]); n = D["n"]; La = D["ans"].shape[1]; nd = La - len(HASH); Lcot = sum(D["steps"][j].shape[1] for j in range(n)) if arm == "tc" else 0
    L_text = La + Lcot; budget = L_text + extra; lut = {int(t): i for i, t in enumerate(DIGIT_IDS)}; H = list(HASH)
    anc = np.full((N, nd), -1); pos = np.full((N, nd), -1); cot = np.full(N, -1)
    for s_ in range(0, N, bs):
        idx = np.arange(s_, min(N, s_ + bs))
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            last, past = seq_forward(g, torch.tensor(question(D, idx), device=dev), 0, None)
            dec = seq_greedy(g, last, past, budget).cpu().numpy()
        for r_, row in zip(idx, dec):
            hit = next((i for i in range(len(row) - 1) if row[i] == H[0] and row[i + 1] == H[1]), None)
            if hit is not None:
                cot[r_] = hit; d_ = [lut.get(int(t), -1) for t in row[hit + 2:hit + 2 + nd]]; anc[r_, :len(d_)] = d_
            at = row[L_text - La:L_text]
            if list(at[:len(H)]) == H:
                pos[r_] = [lut.get(int(t), -1) for t in at[len(H):]]
    P = D["prod"]
    def rep(dg):
        C = dg == P; return {"test_acc": float(C.all(1).mean()), "acc_by_column": {"per_column": [round(float(v), 4) for v in C.mean(0)]}}
    return {"anchored": rep(anc), "position_locked": rep(pos), "cot_len_true": int(Lcot), "cot_len_mismatch_share": float((cot != Lcot).mean()),
            "no_hash_share": float((cot < 0).mean()), "cot_len_median": float(np.median(cot[cot >= 0])) if (cot >= 0).any() else None, "budget": int(budget), "n": int(N)}


def acc_report_digits(dig, P):
    C = dig == P; return {"test_acc": float(C.all(1).mean()), "acc_by_column": {"per_column": [round(float(v), 4) for v in C.mean(0)]}, "n": int(len(P))}


def seq_batch(D, rng, bs, dev, arm, stage):
    idx = rng.integers(0, len(D["A"]), bs); q = question(D, idx)
    if arm == "nc":
        text = D["ans"][idx]
    else:
        s0 = stage if arm == "coco" else 0
        text = np.concatenate([D["steps"][j][idx] for j in range(s0, D["n"])] + [D["ans"][idx]], 1)
    return torch.tensor(q, device=dev), torch.tensor(text, device=dev)


@torch.no_grad()
def seq_eval(g, D, dev, amp, bs, arm, ablate=None, mean_in=None, meter=None):
    N = len(D["A"]); n = D["n"]; ok = np.zeros(N, bool)
    nth = n if arm == "coco" else 0
    L_text = D["ans"].shape[1] + (sum(D["steps"][j].shape[1] for j in range(n)) if arm == "tc" else 0)
    for s in range(0, N, bs):
        idx = np.arange(s, min(N, s + bs))
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            last, past = seq_forward(g, torch.tensor(question(D, idx), device=dev), nth, None, ablate, mean_in, meter if s == 0 else None)
            dec = seq_greedy(g, last, past, L_text, meter if s == 0 else None).cpu().numpy()
        ok[idx] = answer_correct(dec[:, -D["ans"].shape[1]:], D["prod"][idx])
    return ok


@torch.no_grad()
def coco_mean_thoughts(g, D, dev, amp, n_rows=2000, seed=9806):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(D["A"]), n_rows); out = []
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        h, past = g(g.embed(torch.tensor(question(D, idx), device=dev))); last = h[:, -1]
        for t in range(D["n"]):
            out.append(last.float().mean(0)); hh, past = g(last[:, None], past); last = hh[:, -1]
    return out


# ---------------------------------------------------------------- train / eval
def train(a):
    global K
    K = a.k                                                                 # §100: chunk length (default 4 = §98)
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    n = int(a.task[0])
    Dtr = load(os.path.join(a.train_dir, f"{a.task}_train.txt"), limit=a.train_limit or None)
    Dte = load(os.path.join(ROOT, "bench", "data", f"mult_{a.task}_test.txt"), limit=a.test_n)
    assert Dtr["n"] == n and Dte["n"] == n
    SHM = [int(x) for x in a.shape_mix.split(",") if x.strip()]             # §164
    assert not SHM or (a.arm in ("tc", "nc") and a.task == "4x4"), "§164: --shape_mix for tc / nc on 4x4"
    EXCL = set()
    if SHM:                                                                 # the full 4x4 bench test (both orders) is never drawn
        Dfull = load(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt"))
        EXCL = {(x.tobytes(), y.tobytes()) for x, y in zip(Dfull["A"], Dfull["B"])} | {(y.tobytes(), x.tobytes()) for x, y in zip(Dfull["A"], Dfull["B"])}
    if a.eval_anchored:                                                     # §164 (Audit): the anchored re-eval of a finished tc / nc run
        from boxes.multcol import gen_operands
        g = s5.load_gpt2(dev); g.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); g.eval(); t1 = time.time()
        R = {"4x4": seq_eval_anchored(g, Dte, dev, amp, a.eval_bs, a.arm, a.anchor_extra)}; seeds = {}
        for nb_, s1_, s2_ in ((6, 14016, 14017), (7, 14018, 14019), (8, 14004, 14005), (16, 14006, 14007)):
            Dx = gen_items(gen_operands(np.random.default_rng(s1_), a.ext_n, 4), gen_operands(np.random.default_rng(s2_), a.ext_n, nb_)); seeds["4x%d" % nb_] = [s1_, s2_]
            R["4x%d" % nb_] = seq_eval_anchored(g, Dx, dev, amp, a.eval_bs, a.arm, a.anchor_extra)
        out = {"arm": a.arm, "seed": a.seed, "acc_test": R["4x4"]["anchored"]["test_acc"], "acc_by_column_4x4": R["4x4"]["anchored"]["acc_by_column"],
               **{"ext_" + k: {**R[k]["anchored"], "seeds": seeds[k], "n": R[k]["n"]} for k in seeds},
               "cot_len_mismatch_share": {k: v["cot_len_mismatch_share"] for k, v in R.items()}, "no_hash_share": {k: v["no_hash_share"] for k, v in R.items()},
               "cot_len_true": {k: v["cot_len_true"] for k, v in R.items()}, "cot_len_median": {k: v["cot_len_median"] for k, v in R.items()},
               "position_locked": {k: v["position_locked"] for k, v in R.items()}, "budget_extra": a.anchor_extra, "budget": {k: v["budget"] for k, v in R.items()},
               "anchor_def": "greedy decode, budget = the schedule's token count + extra; answer = the na + nb digit tokens after the model's FIRST ' ####' (no ' ####' = wrong)",
               "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "eval_dtype": str(amp), "eval_sec": round(time.time() - t1)}
        json.dump(out, open(os.path.join(a.out, "eval_s164_anchor.json"), "w", newline="\n"), indent=1)
        print("ANCHORED", json.dumps({"acc_test": out["acc_test"], **{k: out["ext_" + k]["test_acc"] for k in seeds}, "mismatch": out["cot_len_mismatch_share"]}), flush=True)
        return
    if a.eval_only:                                                         # re-run a finished run's eval (§98 genm 5x5 s3: eval OOM'd beside 2 others)
        g = s5.load_gpt2(dev); m = MultGen(g, a.arm).to(dev) if a.arm in ("genm", "mask") else g
        m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev))
        last = [json.loads(x) for x in open(os.path.join(a.out, "log.jsonl"))][-1]
        assert last["step"] == a.steps, "model.pt is not from a finished run"
        train_sec = last["sec"]
    else:
        os.makedirs(a.out, exist_ok=True)
        json.dump({**vars(a), "train_problems": len(Dtr["A"]), "step_tokens": [s.shape[1] for s in Dtr["steps"]], "answer_tokens": Dtr["ans"].shape[1]},
                  open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
        g = s5.load_gpt2(dev)
        m = MultGen(g, a.arm).to(dev) if a.arm in ("genm", "mask") else g
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
        log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []
        n_stages = n + 1
        for step in range(1, a.steps + 1):
            for gr in opt.param_groups:
                gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
            stage = min(n_stages - 1, (step - 1) * n_stages // a.steps) if a.arm == "coco" else 0
            micro = a.micro or (1 if a.arm in ("genm", "mask") else 2)          # memory: micro-batches, same effective batch
            if a.arm in ("genm", "mask"):
                prefix, start, ops, st, ans = gen_batch(Dtr, rng, a.bs, dev)
            elif SHM:                                                       # §164: a 4 x n batch from the generator (n per batch)
                Dm = mix_draw(rng, a.bs, SHM, EXCL); idx_ = np.arange(a.bs); q = torch.tensor(question(Dm, idx_), device=dev)
                text = torch.tensor(Dm["ans"] if a.arm == "nc" else np.concatenate([Dm["steps"][j] for j in range(Dm["n"])] + [Dm["ans"]], 1), device=dev)
            else:
                q, text = seq_batch(Dtr, rng, a.bs, dev, a.arm, stage)
            opt.zero_grad(); tot = 0.0; mb = a.bs // micro
            for i in range(micro):
                sl = slice(i * mb, (i + 1) * mb)
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    if a.arm in ("genm", "mask"):
                        loss, _, _, _ = m.chunk(prefix[sl], None if start is None else start[sl], ops[sl], step_tgt=[t[sl] for t in st],
                                                ans_tgt=None if ans is None else ans[sl])
                    else:
                        hs, _ = seq_forward(g, q[sl], stage if a.arm == "coco" else 0, text[sl])
                        lg = g.logits(hs).float(); loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), text[sl].reshape(-1))
                (loss / micro).backward(); tot += loss.item() / micro
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(tot)
            if step % 100 == 0 or step == a.steps:
                rec = {"step": step, "loss": float(np.mean(hist)), "sec": round(time.time() - t0)}; hist = []
                if a.arm == "coco":
                    rec["stage"] = stage
                if a.log_every and step % a.log_every == 0:
                    m.eval(); sub = {k: (v[:a.log_n] if isinstance(v, np.ndarray) else ([x[:a.log_n] for x in v] if isinstance(v, list) else v)) for k, v in Dte.items()}
                    if a.arm in ("genm", "mask"):
                        per, ok, _ = gen_rollout(m, sub, dev, amp, a.eval_bs); rec["per_slot_decode_acc"] = per.mean(0).tolist(); rec["acc"] = float(ok.mean())
                    elif a.arm != "coco" or stage == n_stages - 1:
                        rec["acc"] = float(seq_eval(g, sub, dev, amp, a.eval_bs, a.arm).mean())
                    m.train()
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        train_sec = round(time.time() - t0)
        torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
        del opt; m.zero_grad(set_to_none=True)                              # free optimizer state before the final eval
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    m.eval(); t1 = time.time()
    res = {"arm": a.arm, "task": a.task, "seed": a.seed, "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "train_sec": train_sec, "eval_dtype": str(amp), "test_items": len(Dte["A"])}
    # training compute: Meter forward FLOPs of one problem per training input type x 3 (forward + backward)
    tm = []
    with torch.no_grad():
        if a.arm in ("genm", "mask"):
            for c0, c1 in chunks_of(n):                                     # each training chunk shape (drawn uniformly)
                mt = MeterR(g); idx = np.array([0]); T = lambda x: torch.tensor(x, device=dev)  # noqa: E731
                pre = T(np.concatenate([DIGIT_IDS[Dtr["A"][idx]], np.full((1, 1), TOK["*"])], 1))
                stt = T(Dtr["steps"][c0 - 1][idx]) if c0 > 0 else None
                m.chunk(pre, stt, T(Dtr["B"][idx, c0:c1]), meter=mt)
                fork_tokens = sum(Dtr["steps"][j].shape[1] for j in range(c0, c1)) + (Dtr["ans"].shape[1] if c1 == n else 0)
                n_forks = c1 - c0                                    # one fork per slot (note 98-3: the answer continues the last fork)
                ctx_end = pre.shape[1] + (0 if stt is None else stt.shape[1]) + 2 * (c1 - c0)
                for _ in range(fork_tokens - n_forks):                     # teacher-forced fork tokens (approx. ctx)
                    mt.step(1, ctx_end, 0)
                mt.flops += 2 * 768 * 50257 * fork_tokens; tm.append(mt.flops)
        else:
            mt = MeterR(g); q, text = seq_batch(Dtr, np.random.default_rng(0), 1, dev, a.arm, n if a.arm == "coco" else 0)
            seq_forward(g, q, n if a.arm == "coco" else 0, text, meter=mt); mt.flops += 2 * 768 * 50257 * text.shape[1]; tm.append(mt.flops)
    res["training"] = {"epochs": a.steps * a.bs / len(Dtr["A"]), "flops": float(np.mean(tm)) * 3 * a.steps * a.bs,
                       "flops_convention": "Meter forward FLOPs per trained problem (full-vocab LM head) x 3 (fwd+bwd) x steps x bs"}
    if a.arm in ("genm", "mask"):
        meter, meter_x = MeterR(g), MeterR(g)
        per, ok, snaps = rollout_safe(m, Dte, dev, amp, a.eval_bs, meter=meter, meter_x=meter_x)
        res["acc_test"] = float(ok.mean()); res["per_slot_decode_acc"] = per.mean(0).tolist()
        res["k"] = K; res["per_chunk_acc"] = [float(per[:, c1 - 1].mean()) for c0, c1 in chunks_of(n)]   # §100: each chunk's end state
        res["compute"] = {**meter.as_dict(), "measurement_decode_flops": meter_x.flops}
        mu = gen_mean_inputs(m, Dtr, dev, amp); res["removal"] = {}
        for ab in ("zero", "mean"):
            p_, o_, _ = rollout_safe(m, Dte, dev, amp, a.eval_bs, ablate=ab, mean_in=mu)
            res["removal"][ab] = {"answer_acc": float(o_.mean()), "per_slot_decode_acc": p_.mean(0).tolist()}
        if n > K:                                                           # 5x5: the snap after op 4
            L_P, L_S = n + 1 + (K - 1), n + K
            res["checkpoint_validity"] = {"snaps": len(snaps[0]), "rate": float(valid_step(snaps[0], L_P, L_S).mean())}
            perm = np.random.default_rng(9807).permutation(len(Dte["A"])); donor = Dte["steps"][K - 1][perm]
            _, o_e, _ = rollout_safe(m, Dte, dev, amp, a.eval_bs, override=(0, donor))
            S_d = to_int(np.array([[int(DIGIT_IDS.tolist().index(t)) for t in row[2 + L_P:2 + L_P + L_S]] for row in donor]))
            a_int = to_int(Dte["A"])                                        # edited S_K plus a times the remaining ops (j >= K, 0-based)
            want = [S + ai * sum(int(Dte["B"][r, j]) * 10 ** j for j in range(K, n)) for r, (S, ai) in enumerate(zip(S_d, a_int))]
            # decode under the edit, then compare with the edited composition
            Dw = dict(Dte); Dw["prod"] = np.array([digits_lsb(v, Dte["prod"].shape[1]) for v in want])
            _, o_w, _ = rollout_safe(m, Dw, dev, amp, a.eval_bs, override=(0, donor))
            res["edit_iia"] = {"n": len(Dte["A"]), "iia": float(o_w.mean()), "answer_acc_vs_original": float(o_e.mean())}
    else:
        meter = MeterR(g); ok = seq_eval(g, Dte, dev, amp, a.eval_bs, a.arm, meter=meter)
        res["acc_test"] = float(ok.mean()); res["compute"] = meter.as_dict()
        if SHM:                                                             # §164: §161's ext items (gen_operands with its seeds), 1,000 each (--ext_n)
            from boxes.multcol import gen_operands
            for nb_, s1_, s2_ in ((6, 14016, 14017), (7, 14018, 14019), (8, 14004, 14005), (16, 14006, 14007)):
                Dx = gen_items(gen_operands(np.random.default_rng(s1_), a.ext_n, 4), gen_operands(np.random.default_rng(s2_), a.ext_n, nb_)); mx_ = MeterR(g) if nb_ == 8 else None
                dg_, _ = seq_eval_digits(g, Dx, dev, amp, a.eval_bs, a.arm, meter=mx_); res["ext_4x%d" % nb_] = {**acc_report_digits(dg_, Dx["prod"]), "seeds": [s1_, s2_]}
                if mx_ is not None:
                    res["compute_4x8"] = mx_.as_dict()
            res["shape_mix"] = a.shape_mix
        if a.arm == "coco":
            mu = coco_mean_thoughts(g, Dtr, dev, amp)
            res["removal"] = {ab: {"answer_acc": float(seq_eval(g, Dte, dev, amp, a.eval_bs, "coco", ablate=ab, mean_in=mu).mean())} for ab in ("zero", "mean")}
    res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({k: res[k] for k in ("acc_test",)}), flush=True); print("DONE", a.out, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--arm", required=True, choices=["genm", "mask", "coco", "tc", "nc"])
    ap.add_argument("--task", required=True, choices=["4x4", "5x5"]); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--eval_bs", type=int, default=250)
    ap.add_argument("--eval_only", action="store_true", help="evaluate <out>/model.pt of a finished run only (log.jsonl and args.json untouched)")
    ap.add_argument("--ext_n", type=int, default=1000, help="§164 ext items per shape (smokes only change it)")
    ap.add_argument("--eval_anchored", action="store_true", help="§164 (Audit): <out>/model.pt -> eval_s164_anchor.json (answer after the first ' ####'; note 164-1)")
    ap.add_argument("--anchor_extra", type=int, default=50, help="§164 anchored decode budget = L_text + this")
    ap.add_argument("--shape_mix", default="", help="§164: tc / nc training on 4 x n chains, n drawn per batch from this list (e.g. 2,3,4,5)")
    ap.add_argument("--k", type=int, default=4, help="§100: ops per chunk (snap every k; training chunks aligned)")
    ap.add_argument("--micro", type=int, default=0, help="micro-batches per update (0: 1 for genm/mask, 2 for coco/tc/nc: 3 stage-0 jobs OOM a 24 GB card)")
    ap.add_argument("--test_n", type=int, default=None, help="smoke only"); ap.add_argument("--train_limit", type=int, default=0, help="smoke only")
    ap.add_argument("--log_every", type=int, default=2000); ap.add_argument("--log_n", type=int, default=200)
    ap.add_argument("--train_dir", default=os.path.join(ROOT, "bench", "data", "mult_train"))
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
