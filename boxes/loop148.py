"""THEORY §148 LOOP-RETRO (b1d7741c / note 147-1 (3)): recurrent depth retrofitted into GPT-2 for DIRECT-answer 4x4 multiplication (no CoT).
    prompt = a's 4 digit tokens (LSB first) " *" b's 4 digit tokens " ="; answer = the product's 8 digits LSB first (teacher-forced CE in training,
    greedy at test). GPT-2 small = prelude (blocks 0-3) -> loop (blocks 4-7, weight-tied, r iterations) -> coda (blocks 8-11) -> ln_f -> LM head.
    loop: e = prelude output; h_0 = e; iteration 1: h_1 = Loop(h_0 + P(e)); iteration r > 1: h_r = h_{r-1} + eta (Loop(h_{r-1} + P(e)) - h_{r-1}),
    P a zero-initialised linear map (its own AdamW group, 300-step warm-up); at r = 1 the model IS GPT-2 (max |delta logits| = 0).
Training: r = 1 to 2k, then r ~ U{1 .. r_max} per batch; deep supervision (114-1's form): CE at the final r plus one uniform earlier r (mean of
the two). 24k, bs 64, lr 1e-4, AdamW wd 0.01, clip 1.0, the 127-3 guard; one run per seed, no restarts. Arms: loop4 (r_max 4, eta 0.25),
plain (r_max 1). eval.json: test_acc (at r = r_max), acc_by_column, s148 {acc_by_r (r = 1..16), ce_by_r, halt (||h_r - h_{r-1}|| / ||h_{r-1}||
< 1e-3 at the prompt's last position, max 16: acc, mean r), loop_swap (the donor's h after iteration 2 swapped in, r = 4: donor- /
recipient-consistent / chance), rho_loop (first-3 form of the loop map after iteration 2)}.
    python -m boxes.loop148 --arm loop4 --seed 1 --out boxes/runs_fsq/s148_loop4_4x4_s1"""
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
try:
    from boxes import memcap  # noqa: E402
except ImportError:
    memcap = None
from boxes import s5  # noqa: E402
from boxes import core as bc  # noqa: E402
from boxes.fsq120 import driver_version, env_info, lr_at  # noqa: E402
from boxes.multcol import DTOK, dedup, load_bench, to_int  # noqa: E402

MUL, EQ = bc.tok(" *")[0], bc.tok(" =")[0]; NANS = 8; PRE, LOOP, CODA = range(0, 4), range(4, 8), range(8, 12); PROMPT_LEN = 10


def prompt_tokens(A, B):
    return np.c_[np.array(DTOK)[A], np.full(len(A), MUL), np.array(DTOK)[B], np.full(len(A), EQ)]


def answer_digits(A, B):
    p = to_int(A) * to_int(B); return np.stack([(p // 10 ** k) % 10 for k in range(NANS)], 1)


def block(g, i, h):
    """core.GPT.forward's block i (causal self-attention over the whole sequence; no past)."""
    if _SLOT["pos"] is not None:                                            # §156 slot_patch (off by default)
        if _SLOT["cap"] is not None:
            _SLOT["cap"].append(h[:, _SLOT["pos"]].detach().clone())
        elif _SLOT["rep"] is not None:
            h = h.clone(); h[:, _SLOT["pos"]] = _SLOT["rep"][_SLOT["i"]].to(h.dtype)
        _SLOT["i"] += 1
    B, T, D = h.shape; hd = D // g.H; pre = f"h.{i}."
    a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
    q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
    q, k, v = (t.view(B, T, g.H, hd).transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
    a = F.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
    m = F.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh")
    return h + m @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")


class LoopGPT(nn.Module):
    def __init__(self, g, eta=0.25):
        super().__init__()
        self.g = g; self.eta = eta; dev = g.w("wte.weight").device; self.P = nn.Linear(768, 768).to(dev)
        with torch.no_grad():
            self.P.weight.zero_(); self.P.bias.zero_()

    def prelude(self, ids):
        h = self.g.embed(ids) + self.g.w("wpe.weight")[:ids.shape[1]][None]
        for i in PRE:
            h = block(self.g, i, h)
        return h

    def loop_once(self, h, e, first):
        x = h + self.P(e.float()).to(h.dtype)
        for i in LOOP:
            x = block(self.g, i, x)
        return x if first else h + self.eta * (x - h)

    def coda_logits(self, h):
        for i in CODA:
            h = block(self.g, i, h)
        D = h.shape[-1]; return self.g.logits(F.layer_norm(h, (D,), self.g.w("ln_f.weight"), self.g.w("ln_f.bias")))

    def run(self, ids, r_max, want=None, swap=None, keep=None, perturb=None):
        """Loop iterations 1..r_max; logits at the iterations in `want` (default {r_max}). swap: (r, h) replaces h after iteration r; keep: r
        whose h is returned; perturb: (r, delta) added after iteration r. Returns ({r: logits}, kept_h, [h_r])."""
        want = {r_max} if want is None else set(want); e = self.prelude(ids); h = e; out = {}; kept = None; hs = []
        for r in range(1, r_max + 1):
            h = self.loop_once(h, e, r == 1)
            if swap is not None and swap[0] == r:
                h = swap[1].to(h.dtype)
            if perturb is not None and perturb[0] == r:
                h = h + perturb[1].to(h.dtype)
            if keep == r:
                kept = h.detach()
            hs.append(h)
            if r in want:
                out[r] = self.coda_logits(h)
        return out, kept, hs


# ================================================================ THEORY §156 SEP-SLOT (37ac38d1): separator slots in the answer
SEP = {"tok": None, "mode": None}                                           # set in main: --sep_token's id, --answer_sep
_SLOT = {"pos": None, "cap": None, "rep": None, "i": 0}                    # §156 slot_patch: capture / replace h[:, pos] at every block input


def layout(P, Y):
    """Teacher-forced input and the 8 prediction positions. None: [P, d1..d7], d_k at P_len - 1 + (k - 1). interleave: [P, |, d1, |, .., d7, |],
    d_k at the '|' before it. pause: [P, | x 8, d1..d7], d_1 at the 8th '|'. (interleave and pause: the same length P_len + 15.) pos is a slice."""
    n, L = P.shape; S_ = SEP["tok"]
    if SEP["mode"] == "interleave":
        x = np.c_[P, np.stack([np.full(n, S_) if j % 2 == 0 else Y[:, j // 2] for j in range(2 * NANS - 1)], 1)]; pos = slice(L, L + 2 * NANS, 2)
    elif SEP["mode"] == "pause":
        x = np.c_[P, np.full((n, NANS), S_), Y[:, :-1]]; pos = slice(L + NANS - 1, L + 2 * NANS - 1)
    else:
        x = np.c_[P, Y[:, :-1]]; pos = slice(L - 1, L - 1 + NANS)                  # a slice: the same reduction order as before (bit-identical)
    return x, pos


def first_ids(P):
    """The greedy decode's first input (the prompt plus the forced separators before d1)."""
    n = len(P)
    if SEP["mode"] == "interleave":
        return np.c_[P, np.full(n, SEP["tok"])]
    if SEP["mode"] == "pause":
        return np.c_[P, np.full((n, NANS), SEP["tok"])]
    return P


def carries(A, B):
    """carry into each answer digit k (LSB first; 0 for k = 0) and the products-only column sums cs_k."""
    n = len(A); cs = np.zeros((n, NANS), np.int64)
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            cs[:, i + j] += A[:, i] * B[:, j]
    ci = np.zeros((n, NANS), np.int64)
    for k in range(1, NANS):
        ci[:, k] = (cs[:, k - 1] + ci[:, k - 1]) // 10
    return ci, cs


@torch.no_grad()
def s156_eval(m, A, B, dev, amp, bs, r, slot_col=4, seed=15601):
    """§156 descriptive: seq_len; ring_probe (per slot k: held-out ridge R^2 of cos / sin(2 pi carry_k / C), C = max carry + 1, from the final
    residual (pre-ln_f) at the position predicting d_k, teacher-forced); slot_patch (column slot_col: the residual at its slot position at every
    block input replaced by a donor's with a different carry-in: share emitting (recipient's column sum + donor's carry) mod 10 vs the recipient's
    own digit)."""
    from boxes.multcol import ridge_r2
    Pp = prompt_tokens(A, B); G = answer_digits(A, B); Y = np.array(DTOK)[G]; x, pos = layout(Pp, Y); pos = list(range(x.shape[1]))[pos]; ci, cs = carries(A, B); C = int(ci.max()) + 1; n = len(A); h = n // 2
    H = np.zeros((n, NANS, 768), np.float32)
    for s_ in range(0, n, bs):
        sl = slice(s_, s_ + bs); xx = torch.tensor(x[sl], device=dev)
        with amp_ctx(amp):
            _, _, hs = m.run(xx, r, want=set())
            hh = hs[-1]
            for i in CODA:
                hh = block(m.g, i, hh)
        H[sl] = hh[:, pos].float().cpu().numpy()
    ring = {}
    for k in range(1, NANS):
        th = 2 * np.pi * ci[:, k] / C
        ring[str(k + 1)] = {"cos_r2": ridge_r2(H[:h, k], np.cos(th[:h]), H[h:, k], np.cos(th[h:])) if np.cos(th).std() > 0 else None,
                            "sin_r2": ridge_r2(H[:h, k], np.sin(th[:h]), H[h:, k], np.sin(th[h:])) if np.sin(th).std() > 0 else None}
    ring["C"] = C; ring["def"] = "per slot (answer digit k, 1-based): held-out ridge R^2 of cos / sin(2 pi carry_in_k / C) from the final residual at the slot"
    k = slot_col - 1; p = pos[k]; rng = np.random.default_rng(seed); ir = rng.permutation(n)[:min(500, n)]; idn = rng.permutation(n)[:len(ir)]
    for _ in range(100):
        bad = np.nonzero(ci[idn, k] == ci[ir, k])[0]
        if not len(bad):
            break
        idn[bad] = rng.integers(0, n, len(bad))
    keep = ci[idn, k] != ci[ir, k]; ir, idn = ir[keep], idn[keep]; got = np.zeros(len(ir), np.int64); lut = {t: i_ for i_, t in enumerate(DTOK)}
    for s_ in range(0, len(ir), bs):
        a_, d_ = ir[s_:s_ + bs], idn[s_:s_ + bs]
        _SLOT.update({"pos": p, "cap": [], "rep": None, "i": 0})
        with amp_ctx(amp):
            m.run(torch.tensor(x[d_], device=dev), r, want={r})
        cap = _SLOT["cap"]; _SLOT.update({"cap": None, "rep": cap, "i": 0})
        with amp_ctx(amp):
            out, _, _ = m.run(torch.tensor(x[a_], device=dev), r, want={r})
        _SLOT.update({"pos": None, "rep": None, "i": 0}); got[s_:s_ + bs] = [lut.get(int(t), -1) for t in out[r][:, p].float().argmax(-1).tolist()]
    tgt = (cs[ir, k] + ci[idn, k]) % 10; own = G[ir, k]
    patch = {"column": slot_col, "n": int(len(ir)), "donor_carry_share": float((got == tgt).mean()), "own_digit_share": float((got == own).mean()),
             "chance_overlap": float((tgt == own).mean()), "def": "slot residual at every block input from a donor with a different carry-in (teacher-forced prefix)"}
    return {"seq_len": int(x.shape[1] + 1), "ring_probe": ring, "slot_patch": patch, "answer_sep": SEP["mode"], "sep_token_id": SEP["tok"], "slot_positions": pos}


def compute_item(m, r):
    """boxes.core.Meter per item (KV-cache convention; full vocabulary), forced separators counted as processed tokens: the first pass is the
    prompt (+ forced separators before d1), then 7 passes of [d_k (+ the next separator)]; loop iterations r (plain: 1) on blocks 4-7 add their FLOPs."""
    mt = bc.Meter(m.g); L0 = PROMPT_LEN + (1 if SEP["mode"] == "interleave" else NANS if SEP["mode"] == "pause" else 0); step = 2 if SEP["mode"] == "interleave" else 1
    mt.step(L0, 0, 1); ctx = L0
    for _ in range(NANS - 1):
        mt.step(step, ctx, 1); ctx += step
    d = mt.as_dict(); d["sequential_passes"] = d["seq_passes"]; d["loop_iterations"] = r
    if r > 1:
        d["flops"] = d["flops"] * (1 + (r - 1) * len(LOOP) / m.g.L)
    return d


def amp_ctx(amp):
    return torch.autocast("cuda", dtype=amp, enabled=amp is not None)


@torch.no_grad()
def greedy(m, A, B, dev, amp, bs, r, swap_h=None, keep=None, halt=False):
    """Greedy 8-digit answers at r iterations (fresh full pass per token). swap_h: per token pass k, (N, T_k, D) donor h after iteration 2;
    keep: return h after that iteration per pass; halt: iterate up to 16 per pass, each row stopping at the first r with ||h_r - h_{r-1}|| /
    ||h_{r-1}|| < 1e-3 at the last position. Returns digits (N, 8) [, kept list] [, mean r used]."""
    N = len(A); P = first_ids(prompt_tokens(A, B)); dig = np.full((N, NANS), -1); lut = {t: i for i, t in enumerate(DTOK)}; kept = [[] for _ in range(NANS)]; rused = []
    for s_ in range(0, N, bs):
        sl = slice(s_, s_ + bs); ids = torch.tensor(P[sl], device=dev); toks = []
        for k in range(NANS):
            x = torch.cat([ids] + toks, 1) if toks else ids                 # §156: separators are forced inputs, never generated
            with amp_ctx(amp):
                if halt:
                    _, _, hs = m.run(x, 16, want=set()); last = torch.stack([hh[:, -1].float() for hh in hs], 0)   # (16, B, D)
                    rel = (last[1:] - last[:-1]).norm(dim=-1) / last[:-1].norm(dim=-1).clamp_min(1e-9); stop = torch.full((x.shape[0],), 16, device=dev)
                    for rr in range(1, 16):
                        stop = torch.where((stop == 16) & (rel[rr - 1] < 1e-3), torch.full_like(stop, rr + 1), stop)
                    lg = torch.stack([m.coda_logits(hs[int(rr) - 1][j:j + 1])[0, -1] for j, rr in enumerate(stop.tolist())]); rused += stop.tolist()
                else:
                    sw = None if swap_h is None else (2, torch.tensor(swap_h[k][sl], device=dev))
                    out, kh, _ = m.run(x, r, swap=sw, keep=keep); lg = out[r][:, -1]
                    if keep is not None:
                        kept[k].append(kh.float().cpu().numpy())
            nt = lg.float().argmax(-1); toks.append(nt[:, None]); dig[sl, k] = [lut.get(int(t), -1) for t in nt.tolist()]
            if SEP["mode"] == "interleave" and k < NANS - 1:
                toks.append(torch.full((nt.shape[0], 1), SEP["tok"], device=dev))
    res = [dig]
    if keep is not None:
        res.append([np.concatenate(kk) for kk in kept])
    if halt:
        res.append(float(np.mean(rused)))
    return res[0] if len(res) == 1 else tuple(res)


@torch.no_grad()
def ce_at(m, A, B, dev, amp, bs, rs):
    """Teacher-forced answer CE (mean over the 8 tokens) at each r in rs, on (A, B)."""
    P = prompt_tokens(A, B); Y = np.array(DTOK)[answer_digits(A, B)]; tot = {r: 0.0 for r in rs}; n = 0; X, pos = layout(P, Y)
    for s_ in range(0, len(A), bs):
        sl = slice(s_, s_ + bs); x = torch.tensor(X[sl], device=dev); y = torch.tensor(Y[sl], device=dev)
        with amp_ctx(amp):
            out, _, _ = m.run(x, max(rs), want=set(rs))
        for r in rs:
            lg = out[r][:, pos].float(); tot[r] += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum")) / NANS
        n += len(A[sl])
    return {str(r): tot[r] / n for r in rs}


@torch.no_grad()
def rho_loop(m, A, B, dev, amp, n=300, r0=2, K=8, seed=14801):
    """Loop-map contraction: perturb h after iteration r0 (all positions) by eps x per-coordinate std x u; distances to the clean loop states
    over the next K iterations; median ratios and 133-1's first-3 form."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(A))[:n]; x = torch.tensor(prompt_tokens(A[idx], B[idx]), device=dev)
    with amp_ctx(amp):
        _, _, hs = m.run(x, r0 + K, want=set())
    H = [h.float() for h in hs]; sd = torch.stack(H[r0 - 1:]).std(0).clamp_min(1e-6); U = torch.tensor(rng.standard_normal(tuple(H[0].shape)), dtype=torch.float32, device=dev); res = {}
    for eps in (0.01, 0.1):
        with amp_ctx(amp):
            _, _, hp = m.run(x, r0 + K, want=set(), perturb=(r0, eps * sd * U))
        D = np.stack([(hp[r].float() - H[r]).flatten(1).norm(dim=1).cpu().numpy() for r in range(r0 - 1, r0 + K)]); rk = np.median(D[1:] / np.maximum(D[:-1], 1e-9), axis=1)
        dk = np.median(D, axis=1); use = [rk[k] for k in range(min(3, len(rk))) if dk[k] > 1e-6 and rk[k] > 0]
        res[str(eps)] = {"rho_k": rk.round(4).tolist(), "dist_med_k": dk.round(5).tolist(), "rho_k3": float(np.exp(np.mean(np.log(use)))) if use else None}
    return {"r0": r0, "K": K, "n": int(len(idx)), "eps": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="mult_direct", choices=["mult_direct"]); ap.add_argument("--task", default="4x4", choices=["4x4"])
    ap.add_argument("--arm", required=True, choices=["loop4", "plain"]); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--eta", type=float, default=0.25)
    ap.add_argument("--r_switch", type=int, default=2000); ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=1000)
    ap.add_argument("--swap_n", type=int, default=500); ap.add_argument("--sched_scale", type=float, default=1.0); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=0); ap.add_argument("--resume", default=None); ap.add_argument("--stop_after", type=int, default=0)
    ap.add_argument("--train_n", type=int, default=0, help="smoke only"); ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--r_eval_max", type=int, default=16)
    ap.add_argument("--cotenant_runs", type=int, default=1); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    ap.add_argument("--answer_sep", default=None, choices=[None, "interleave", "pause"], help="§156 SEP-SLOT: forced separators in the answer")
    ap.add_argument("--sep_token", default="|")
    a = ap.parse_args(); a.r_max = 4 if a.arm == "loop4" else 1; a.loop = "4-7"; a.deep_sup = True
    if a.answer_sep:                                                        # §156: '|' must be one token (91) and never merge with ' d'
        st_ = bc.tok(a.sep_token); assert len(st_) == 1, "--sep_token must be one GPT-2 token"; SEP.update({"tok": st_[0], "mode": a.answer_sep})
        smp = []
        for d_ in range(10):
            smp += [st_[0], DTOK[d_]]
        assert bc.ENC.encode(bc.ENC.decode(bc.tok(" =") + smp)) == bc.tok(" =") + smp, "§156: separator / digit tokens merge on re-encoding"
        a.sep_token_id = st_[0]
    else:
        a.sep_token = None
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); sc = a.sched_scale; memcap and memcap.apply(dev)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    os.makedirs(a.out, exist_ok=True); dname = lambda: torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"  # noqa: E731
    teA, teB = load_bench(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt")); teA, teB = teA[:a.test_n], teB[:a.test_n]
    if a.train_n:
        from boxes.multcol import load_bench_head
        trA, trB = load_bench_head(os.path.join(ROOT, "bench", "data", "mult_train", "4x4_train.txt"), a.train_n)
    else:
        trA, trB = load_bench(os.path.join(ROOT, "bench", "data", "mult_train", "4x4_train.txt"))
    n_raw = len(trA); trA, trB, n_rem = dedup(trA, trB, teA, teB)
    info = {**vars(a), "train_items": int(len(trA)), "train_items_raw": int(n_raw), "train_removed_test_overlap": n_rem, "prompt": "a(LSB) ' *' b(LSB) ' ='",
            "answer": "8 digits LSB first", "train_gpu": dname(), "train_driver": driver_version(), **env_info()}
    json.dump({**info, "resumed_at": time.strftime("%Y-%m-%d %H:%M:%S")} if a.resume else info,
              open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    m = LoopGPT(s5.load_gpt2(dev), a.eta); m.train(); pid = {id(p_) for p_ in m.P.parameters()}
    opt = torch.optim.AdamW([{"params": [p_ for p_ in m.parameters() if id(p_) not in pid], "mult": 1.0},
                             {"params": list(m.P.parameters()), "mult": 1.0, "warm_from": 0, "name": "P"}], lr=a.lr, weight_decay=0.01)
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time(); hist = []; n_nonfinite = 0; step0 = 1; rs_hist = []
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"].cpu()); n_nonfinite = ck["n_nonfinite"]
        if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        step0 = ck["step"] + 1; t0 = time.time() - ck["elapsed"]; log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    def loss_at(ii, r, r2):
        A, B = trA[ii], trB[ii]; P = prompt_tokens(A, B); Y = np.array(DTOK)[answer_digits(A, B)]; X, pos = layout(P, Y)   # §156: no loss on separators
        x = torch.tensor(X, device=dev); y = torch.tensor(Y, device=dev); rs = {r} | ({r2} if r2 else set())
        with amp_ctx(amp):
            out, _, _ = m.run(x, r, want=rs)
        ce = {rr: F.cross_entropy(out[rr][:, pos].float().reshape(-1, out[rr].shape[-1]), y.reshape(-1)) for rr in rs}
        return ce[r] if not r2 else 0.5 * (ce[r] + ce[r2])
    if a.mem_check:
        ii = rng.integers(0, len(trA), a.bs); loss = loss_at(ii, a.r_max, a.r_max - 1 if a.r_max > 1 else None); loss.backward(); m.eval()
        greedy(m, teA[:a.eval_bs], teB[:a.eval_bs], dev, amp, a.eval_bs, a.r_eval_max)
        pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else None
        json.dump({"peak_gib": pk, "bs": a.bs, "r_train": a.r_max, "r_eval": a.r_eval_max}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1); print("MEM", pk); return
    for step in range(step0, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps, warm=max(1, int(300 * sc)))
        r = 1 if step <= a.r_switch * sc else int(rng.integers(1, a.r_max + 1)); r2 = int(rng.integers(1, r)) if r > 1 else None; rs_hist.append(r)
        ii = rng.integers(0, len(trA), a.bs); opt.zero_grad(); loss = loss_at(ii, r, r2); loss.backward(); tot = float(loss.item())
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):                # 127-3 guard
            log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush(); n_nonfinite += 1; opt.zero_grad()
        else:
            opt.step(); hist.append(tot)
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "r_drawn": sorted(set(rs_hist)), "P_norm": float(m.P.weight.norm()), "sec": round(time.time() - t0)}
            hist, rs_hist = [], []; log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None,
                        "np_rng": rng.bit_generator.state, "step": step, "n_nonfinite": n_nonfinite, "elapsed": time.time() - t0}, cp + ".tmp")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            os.replace(cp + ".tmp", cp); log.write(json.dumps({"step": step, "ckpt": "ckpt_last.pt"}) + "\n"); log.flush()
        if a.stop_after and step >= a.stop_after:
            log.close(); print("STOP_AFTER", step, flush=True); return
    train_sec = round(time.time() - t0); torch.save(m.state_dict(), os.path.join(a.out, "model.pt")); m.eval(); t1 = time.time()
    G = answer_digits(teA, teB); accr = {}
    for r in range(1, a.r_eval_max + 1):
        accr[str(r)] = float((greedy(m, teA, teB, dev, amp, a.eval_bs, r) == G).all(1).mean())
    pred = greedy(m, teA, teB, dev, amp, a.eval_bs, a.r_max); C = pred == G
    res = {"family": a.family, "task": a.task, "arm": a.arm, "seed": a.seed, "test_acc": float(C.all(1).mean()), "long_acc": float(C.all(1).mean()),
           "acc_by_column": {"per_column": [round(float(v), 4) for v in C.mean(0)], "cumulative": [round(float(C[:, :k + 1].all(1).mean()), 4) for k in range(NANS)]},
           "train_sec": train_sec, "n_nonfinite": n_nonfinite, "train_removed_test_overlap": n_rem, "eval_dtype": str(amp), "gpu": dname(),
           "train_gpu": info["train_gpu"], "train_driver": info["train_driver"]}
    hd, hr = greedy(m, teA, teB, dev, amp, a.eval_bs, 16, halt=True)
    srng = np.random.default_rng(14802); N = min(a.swap_n, len(teA)); ir, idn = srng.permutation(len(teA))[:N], srng.permutation(len(teA))[:N]
    dd, dh = greedy(m, teA[idn], teB[idn], dev, amp, a.eval_bs, max(a.r_max, 2), keep=2); got = greedy(m, teA[ir], teB[ir], dev, amp, a.eval_bs, max(a.r_max, 2), swap_h=dh)
    Gd, Gr = G[idn], G[ir]
    res["s148"] = {"acc_by_r": accr, "ce_by_r": ce_at(m, teA, teB, dev, amp, a.eval_bs, list(range(1, a.r_eval_max + 1))),
                   "halt": {"acc": float((hd == G).all(1).mean()), "mean_r": hr, "threshold": 1e-3, "r_cap": 16},
                   "loop_swap": {"after_iteration": 2, "r": max(a.r_max, 2), "n": int(N), "donor_consistent": float((got == Gd).all(1).mean()),
                                 "recipient_consistent": float((got == Gr).all(1).mean()), "chance_overlap": float((Gd == Gr).all(1).mean()),
                                 "donor_self_acc": float((dd == Gd).all(1).mean())},
                   "rho_loop": rho_loop(m, teA, teB, dev, amp), "r_max": a.r_max, "eta": a.eta, "loop": a.loop, "deep_sup": a.deep_sup}
    res["compute"] = compute_item(m, a.r_max)
    if a.answer_sep:                                                        # §156 descriptive block
        res["s156"] = s156_eval(m, teA, teB, dev, amp, a.eval_bs, a.r_max)
    res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({"test_acc": res["test_acc"], "acc_by_r": accr}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
