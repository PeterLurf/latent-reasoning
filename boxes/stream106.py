"""THEORY §106 (note 106-1): bounded-state streaming multiplication, latent chains (mult98's GEN-M / GEN-M-MASK chunk)
on a generated task with a restartable checkpoint.
Data (generator; digits least-significant first): a has n_a digits (4; a5b8 test 5), most significant digit non-zero;
b has n_b digits (train n_b uniform in 1..5; test n_b = 4, 8, 16, 32: 1,000 items each, plus a = 5 digits at n_b = 8).
Step j (op = the multiplier digit b_j): T = C + a * b_j; emit d_j = T mod 10; carry C <- T // 10 (C_0 = 0).
Step text j = " emit d_j , carry c_1 .. c_W" (C_j's digits LSB-first at fixed width W = n_a + 1): 4 + W tokens.
Flush text (trained, Literature (b)) = " flush f_1 .. f_W" (the final carry's digits, LSB-first).
The product is assembled OUTSIDE the chunk (Literature (a)): digits d_1..d_nb followed by the flush digits; correct
when it equals a * b exactly.
Model (mult98.MultGen, note 98-3's single final fork): chunk = [a digits, " *"] [start text] [op_1] [slot_1] ...; the fork
at every slot teacher-forces step text j; the final slot's fork teacher-forces step text n ++ flush text. k = 4: a snap
(the greedy step text at a chunk's last slot, re-embedded, history cut) every 4 steps. Training chunks are the test-time
chunks of a problem (one chunk index and one n_b drawn per batch), starting from the true previous step text.
Arms: gens (GEN-S: full fork), gens_mask (GEN-S-MASK: §98's masked fork). Note 106-3 (k = 2): gens_mask_k2 (GEN-S-MASK
with a snap every 2 steps, one latent hand-off per chunk); wcs_k2 (WC-S, the written comparator: the same step texts
written out after each op token, [a digits, " *"] [start text] [op_1] [step text 1] [op_2] [step text 2 (++ flush)],
history cut every 2 steps with a restart from the last written step text; loss = CE summed over each step text, mean
over steps, as the fork's; removal does not apply (null)).
Eval keys: acc_by_b {"4","8","16","32"}, acc_a5_b8, per_step_decode_acc {b: [per slot]}, removal.{zero,mean}.acc_b8
(slot inputs 1..k-1 of each chunk ablated), compute.flops_by_b (Meter, the method's path: segments, snap decodes, the
final step + flush decode), checkpoint_validity_by_b, arm, seed, gpu, training {epochs: null (generator), flops}.
    python -m boxes.stream106 --arm gens_mask --seed 1 --out boxes/runs_mult/s106_gens_mask_s1
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
from boxes import s5  # noqa: E402,F401
from boxes import core as bc  # noqa: E402
from boxes.mult98 import DIGIT_IDS, TOK, MeterR, MultGen  # noqa: E402

K = 4
W_EMIT, W_COMMA, W_CARRY, W_FLUSH = bc.tok(" emit")[0], bc.tok(" ,")[0], bc.tok(" carry")[0], bc.tok(" flush")[0]


def lsb(v, w):
    return [int(c) for c in str(v).zfill(w)[::-1]][:w]


def gen_problems(rng, n, na, nb):
    """-> dict A (n, na), B (n, nb) LSB-first digits, steps list over j of (n, 4 + W) tokens, flush (n, 1 + W) tokens,
    emits (n, nb), carries (n, nb) ints, prod (n,) python ints."""
    W = na + 1
    hi = rng.integers(1, 10, n); lo = rng.integers(0, 10 ** (na - 1), n); a = hi * 10 ** (na - 1) + lo
    B = rng.integers(0, 10, (n, nb)); A = np.array([lsb(int(x), na) for x in a])
    steps = [np.zeros((n, 4 + W), np.int64) for _ in range(nb)]; flush = np.zeros((n, 1 + W), np.int64)
    emits = np.zeros((n, nb), np.int64); prods = []
    for i in range(n):
        C = 0
        for j in range(nb):
            T = C + int(a[i]) * int(B[i, j]); d, C = T % 10, T // 10; emits[i, j] = d
            steps[j][i] = [W_EMIT, DIGIT_IDS[d], W_COMMA, W_CARRY] + [DIGIT_IDS[x] for x in lsb(C, W)]
        flush[i] = [W_FLUSH] + [DIGIT_IDS[x] for x in lsb(C, W)]
        prods.append(int(a[i]) * int("".join(str(x) for x in B[i][::-1])))
    return {"A": A, "B": B, "steps": steps, "flush": flush, "emits": emits, "prod": prods, "na": na, "nb": nb, "W": W}


TOK2DIG = {int(t): d for d, t in enumerate(DIGIT_IDS)}


class WCS(torch.nn.Module):
    """Note 106-3's written comparator: per chunk, [prefix] [start] then for each op: [op token] [step text] (the final
    step's text ++ the flush text); every step text is the model's own (greedy at test, teacher-forced in training)."""
    def __init__(self, g):
        super().__init__()
        self.g = g; self.register_buffer("digtok", torch.tensor(DIGIT_IDS))

    def chunk(self, prefix, start, ops, step_tgt=None, ans_tgt=None, step_len=None, ans_len=None, ablate=None,
              mean_in=None, meter=None, snap=False):
        B, c = ops.shape; g = self.g; head = [prefix] + ([start] if start is not None else [])
        opt = self.digtok[ops]
        if step_tgt is not None:                                            # training: one teacher-forced pass
            seq = list(head); spans = []; L0 = sum(t.shape[1] for t in head)
            for j in range(c):
                tg = torch.cat([step_tgt[j], ans_tgt], 1) if (j == c - 1 and ans_tgt is not None) else step_tgt[j]
                seq += [opt[:, j:j + 1], tg]; L0 += 1; spans.append((L0, tg)); L0 += tg.shape[1]
            seq = torch.cat(seq, 1); h, _ = g(g.embed(seq[:, :-1]))
            if meter:
                meter.step(seq.shape[1] - 1, 0, sum(t.shape[1] for _, t in spans))
            losses = []
            for p0, tg in spans:
                lg = g.logits(h[:, p0 - 1:p0 - 1 + tg.shape[1]]).float()
                losses.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1), reduction="none").view(tg.shape).sum(-1).mean())
            return torch.stack(losses).mean(), [], None, None
        x = torch.cat(head + [opt[:, :1]], 1); past = None; decs = []; ans = None
        for j in range(c):
            L = step_len[j] + (ans_len if (j == c - 1 and ans_len is not None) else 0); out = []
            for _ in range(L):
                ctx = 0 if past is None else past[0][0].shape[2]; h, past = g(g.embed(x), past)
                if meter:
                    meter.step(x.shape[1], ctx, 1)
                t = g.logits(h[:, -1]).argmax(-1); out.append(t); x = t[:, None]
            d = torch.stack(out, 1); decs.append(d[:, :step_len[j]])
            if j == c - 1 and ans_len is not None:
                ans = d[:, step_len[j]:]
            if j + 1 < c:
                x = torch.cat([x, opt[:, j + 1:j + 2]], 1)                  # feed the last written token, then the next op
        return None, decs, ans, None


def assemble(step_decs, flush_dec, nb, W):
    """Per problem: emitted digits from the step decodes (token 2 of each step) + the flush digits -> int or None."""
    out = []
    for i in range(len(flush_dec)):
        try:
            ds = [TOK2DIG[int(step_decs[j][i][1])] for j in range(nb)]
            if int(flush_dec[i][0]) != W_FLUSH:
                out.append(None); continue
            fs = [TOK2DIG[int(t)] for t in flush_dec[i][1:1 + W]]
            out.append(int("".join(str(x) for x in (ds + fs)[::-1])))
        except KeyError:
            out.append(None)
    return out


def valid_step(t, W):
    dig = np.isin(t, DIGIT_IDS)
    return (t[:, 0] == W_EMIT) & dig[:, 1] & (t[:, 2] == W_COMMA) & (t[:, 3] == W_CARRY) & dig[:, 4:4 + W].all(1)


def subset(D, n):
    return {k: (v[:n] if isinstance(v, np.ndarray) else ([x[:n] for x in v] if k == "steps" else (v[:n] if k == "prod" else v))) for k, v in D.items()}


def chunks_of(n):
    return [(c0, min(c0 + K, n)) for c0 in range(0, n, K)]


def prefix_of(D, idx, dev):
    return torch.tensor(np.concatenate([DIGIT_IDS[D["A"][idx]], np.full((len(idx), 1), TOK["*"])], 1), device=dev)


@torch.no_grad()
def rollout(m, D, dev, amp, bs, ablate=None, mean_in=None, meter=None):
    """k = 4 test rollout with snaps. Returns exact-product correctness (N,), per-slot exact step decode (N, nb),
    checkpoint validity (share of snapped texts that parse)."""
    N, nb, W = len(D["A"]), D["nb"], D["W"]; ok = np.zeros(N, bool); per = np.zeros((N, nb), bool); snaps = valid = 0
    for s in range(0, N, bs):
        idx = np.arange(s, min(N, s + bs)); pre = prefix_of(D, idx, dev); start = None; step_decs = [None] * nb; fl = None
        for ci, (c0, c1) in enumerate(chunks_of(nb)):
            last = c1 == nb
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                _, decs, ans, _ = m.chunk(pre, start, torch.tensor(D["B"][idx, c0:c1], device=dev),
                                          step_len=[4 + W] * (c1 - c0), ans_len=(1 + W) if last else None, ablate=ablate,
                                          mean_in=mean_in, meter=meter if s == 0 else None,
                                          snap=not last)
            for jj, d in enumerate(decs):
                dd = d.cpu().numpy(); step_decs[c0 + jj] = dd; per[idx, c0 + jj] = (dd == D["steps"][c0 + jj][idx]).all(1)
            if not last:
                st = decs[-1].cpu().numpy(); snaps += len(idx); valid += int(valid_step(st, W).sum()); start = torch.tensor(st, device=dev)
            else:
                fl = ans.cpu().numpy()
        got = assemble(step_decs, fl, nb, W); ok[idx] = [g == D["prod"][i] for g, i in zip(got, idx)]
    return ok, per, {"snaps": snaps, "valid": valid, "rate": valid / max(1, snaps)}


@torch.no_grad()
def mean_inputs(m, rng, dev, amp, na=4, n=2000):
    """Mean slot inputs (K, D) over fresh training-distribution 4-op chunks (a first chunk and a restarted chunk with the
    true start text), used for every chunk's slots 1..k-1 under mean removal."""
    D = gen_problems(rng, n, na, 2 * K); out = []
    for c0 in (0, K):
        part = []
        for s in range(0, n, 250):                                          # 250-row batches (2,000 at once OOM'd with 3 runs per card)
            idx = np.arange(s, min(n, s + 250)); start = torch.tensor(D["steps"][K - 1][idx], device=dev) if c0 else None
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                _, _, _, ins = m.chunk(prefix_of(D, idx, dev), start, torch.tensor(D["B"][idx, c0:c0 + K], device=dev))
            part.append(ins.float())
        out.append(torch.cat(part).mean(0))
    return torch.stack(out).mean(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=["gens", "gens_mask", "gens_mask_k2", "wcs_k2"]); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_bs", type=int, default=100); ap.add_argument("--test_n", type=int, default=1000); ap.add_argument("--log_every", type=int, default=2000)
    ap.add_argument("--test_bs", default="4,8,16,32"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    ap.add_argument("--eval_only", action="store_true", help="evaluate <out>/model.pt (training FLOPs replayed from the same draws; the log is kept)")
    a = ap.parse_args(); dev = a.device; global K; K = 2 if a.arm.endswith("_k2") else 4; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args_eval_only.json" if a.eval_only else "args.json"), "w", newline="\n"), indent=1)
    trng = np.random.default_rng(10600)                                     # fixed test sets (independent of the seed)
    tests = {b: gen_problems(trng, a.test_n, 4, b) for b in [int(x) for x in a.test_bs.split(",")]}; t_a5 = gen_problems(trng, a.test_n, 5, 8)
    wcs = a.arm == "wcs_k2"
    m = WCS(s5.load_gpt2(dev)).to(dev) if wcs else MultGen(s5.load_gpt2(dev), "mask" if a.arm.startswith("gens_mask") else "genm").to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    if a.eval_only:
        m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.eval()
    log = None if a.eval_only else open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []; flops = 0.0; fcache = {}
    for step in range(1, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        nb = int(rng.integers(1, 6)); ch = chunks_of(nb); c0, c1 = ch[int(rng.integers(len(ch)))]; last = c1 == nb
        D = gen_problems(rng, a.bs, 4, nb); idx = np.arange(a.bs); T = lambda x: torch.tensor(x, device=dev)  # noqa: E731
        start = T(D["steps"][c0 - 1]) if c0 > 0 else None
        if not a.eval_only:
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                loss, _, _, _ = m.chunk(prefix_of(D, idx, dev), start, T(D["B"][:, c0:c1]), step_tgt=[T(D["steps"][j]) for j in range(c0, c1)],
                                        ans_tgt=T(D["flush"]) if last else None)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(loss.item())
        key = (c1 - c0, c0 > 0, last)
        if key not in fcache:                                               # forward FLOPs per trained problem of this shape (x3 for fwd+bwd)
            mt = MeterR(m.g); m.eval()
            with torch.no_grad():
                if wcs:                                                     # the teacher-forced pass itself, LM-head rows included
                    m.chunk(prefix_of(D, idx[:1], dev), None if start is None else start[:1], T(D["B"][:1, c0:c1]),
                            step_tgt=[T(D["steps"][j][:1]) for j in range(c0, c1)], ans_tgt=T(D["flush"][:1]) if last else None, meter=mt)
                else:
                    m.chunk(prefix_of(D, idx[:1], dev), None if start is None else start[:1], T(D["B"][:1, c0:c1]), meter=mt)
            ft = (c1 - c0) * (4 + D["W"]) + ((1 + D["W"]) if last else 0)
            fcache[key] = mt.flops if wcs else mt.flops + 2 * 768 * 50257 * ft + ft * 2 * m.g.n_nonemb; m.train()
        flops += 3 * fcache[key] * a.bs
        if a.eval_only:
            continue
        if step % 100 == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)), "sec": round(time.time() - t0)}; hist = []
            if a.log_every and step % a.log_every == 0:
                m.eval(); o4, p4, _ = rollout(m, subset(tests[4], 200), dev, amp, a.eval_bs)
                rec["acc_b4_200"] = float(o4.mean()); rec["per_slot_b4_200"] = p4.mean(0).round(3).tolist(); m.train()
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    if a.eval_only:
        train_sec = [json.loads(r) for r in open(os.path.join(a.out, "log.jsonl"), encoding="utf-8") if r.strip()][-1]["sec"]
    else:
        train_sec = round(time.time() - t0)
        torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    del opt; m.zero_grad(set_to_none=True); m.eval(); t1 = time.time()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    res = {"arm": a.arm, "seed": a.seed, "k": K, "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "eval_dtype": str(amp), "train_sec": train_sec, "acc_by_b": {}, "per_step_decode_acc": {}, "checkpoint_validity_by_b": {},
           "compute": {"flops_by_b": {}, "lm_head_rows_by_b": {}, "sequential_passes_by_b": {}},
           "training": {"epochs": None, "flops": flops, "flops_convention": "Meter forward FLOPs per trained problem (shape of that update) x 3 x bs, summed; generated data"}}
    for b, D in tests.items():
        mt = MeterR(m.g); ok, per, val = rollout(m, D, dev, amp, a.eval_bs, meter=mt)
        res["acc_by_b"][str(b)] = float(ok.mean()); res["per_step_decode_acc"][str(b)] = per.mean(0).round(4).tolist(); res["checkpoint_validity_by_b"][str(b)] = val
        d = mt.as_dict(); res["compute"]["flops_by_b"][str(b)] = d["flops"]; res["compute"]["lm_head_rows_by_b"][str(b)] = d["lm_head_rows"]; res["compute"]["sequential_passes_by_b"][str(b)] = d["sequential_passes"]
    ok5, _, _ = rollout(m, t_a5, dev, amp, a.eval_bs); res["acc_a5_b8"] = float(ok5.mean())
    if wcs:
        res["removal"] = None
    elif 8 in tests:
        mu = mean_inputs(m, np.random.default_rng(10601), dev, amp); res["removal"] = {}
        for ab in ("zero", "mean"):
            o_, _, _ = rollout(m, tests[8], dev, amp, a.eval_bs, ablate=ab, mean_in=mu); res["removal"][ab] = {"acc_b8": float(o_.mean())}
    res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps(res["acc_by_b"]), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
