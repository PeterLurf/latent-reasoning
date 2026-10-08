"""THEORY §165 FIXED-CHUNK (60e55135; scorer theory/score_s165.py): §155 R's DeltaMem-R (6 + 6 heads after block 6, top writes, gamma 2k -> 6k,
pass-start reads, W_o fade-in, g_add calibrated at step 0) as the only cross-segment channel, on the bench's problem text cut into fixed chunks:
    input  " a_1 .. a_4 * b_1 .. b_n =" (digit tokens LSB first) cut into chunks of --chunk_k tokens at multiples of k (no alignment to '*' /
           '='); the last chunk padded with " ."; each chunk = one fresh GPT-2 pass [EOT, chunk] that reads the pass-start memory and writes.
    think  --think_k passes [EOT, " ."] (no input), read then write as usual.
    decode [EOT] + the 4 + n answer digits LSB first (teacher-forced in training, greedy at test), READ-ONLY (no write).
Training: FINAL only, CE on the lowest min(c, 4 + n) answer digits (c ~ U(1, c_max), c_max 1 / 2 / 4 / 8 at 0 / 2k / 5k / 9k; c >= 8 = all
4 + n digits, as §161's "full"); 4 x n chains, n ~ U{2..5} per batch, operands from multpair's gen_operands minus the 4x4 bench test items (both
orders). §120's recipe otherwise (48k, bs 64, AdamW 1e-4 wd 0.01, warmup 100 + cosine, clip 1.0); ckpt_last / prev; --auto_micro (memcheck at the
4x5 training chain AND the 4x16 eval chain at K' = 4K).
eval.json: test_acc / acc_report (4x4 bench, K' = K), ext_4x6 / 4x7 / 4x8 / 4x16 (§161's items, K' = K), compute / compute_4x8 (Meter),
s165 {acc_by_kprime {0, 1, K/2, K, 2K, 4K}, ext_4x8_by_kprime {K, 2K, 4K}, skip_pass {t: acc} (think pass t's write skipped), digit_emergence
{after_input, after_pass_P: {digit: acc}} (10-way near-OLS, 155-4's settings, 8k generator 4x4 fit, scored on the bench), donor_think (whole-memory
swap at think pass K/2: donor / recipient consistent), s155-style write-grad spikes (160-1 floor), n_nonfinite}.
    python -m boxes.fixedchunk --chunk_k 4 --think_k 16 --seed 1 --out boxes/runs_fsq/s165_k16_4x4_s1"""
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
import torch.nn.functional as Fn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes import core as bc  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes.fsq120 import EOT, make_opt, lr_at, driver_version, env_info  # noqa: E402
from boxes.multcol import DTOK, columns, gen_operands, load_bench  # noqa: E402
from boxes.deltas5 import fwd_read, write_grad_norm  # noqa: E402

STAR, EQ, PAD = bc.tok(" *")[0], bc.tok(" =")[0], bc.tok(" .")[0]
DT = np.array(DTOK)
assert len({STAR, EQ, PAD}) == 3 and not {STAR, EQ, PAD} & set(DTOK)
CMAX = ((2000, 1), (5000, 2), (9000, 4), (None, 8))


def cmax(step):
    for lim, c in CMAX:
        if lim is None or step <= lim:
            return c


def amp_ctx(amp):
    return torch.autocast("cuda", dtype=amp, enabled=amp is not None)


def problem_tokens(A, B):
    """the bench's problem text as tokens: a digits LSB first, " *", b digits LSB first, " =" -> (N, na + nb + 2)."""
    return np.concatenate([DT[A], np.full((len(A), 1), STAR), DT[B], np.full((len(A), 1), EQ)], 1)


def chunks(tok, k):
    """fixed chunks of k tokens at multiples of k (no task boundary), the last padded with " ." -> list of (N, k)."""
    N, L = tok.shape; nc = -(-L // k); pad = np.concatenate([tok, np.full((N, nc * k - L), PAD)], 1)
    return [pad[:, i * k:(i + 1) * k] for i in range(nc)]


def product_digits(A, B):
    return columns(A, B)[0]                                                  # (N, na + nb) LSB first


class FixedChunk(nn.Module):
    def __init__(self, g, layer=6, write="top"):
        super().__init__(); self.g = g; self.mem = DM.DeltaMem(g, layer, write)

    def write(self, S, ids):
        return DM.gpt_forward_mem(self.g, self.g.embed(ids), self.mem, S)[1]

    def ids(self, body):
        return torch.cat([torch.full((body.shape[0], 1), EOT, device=body.device), body], 1)

    def encode(self, A, B, k, dev, S=None):
        if S is None:
            S = self.mem.zero_state(len(A), dev)
        for ch in chunks(problem_tokens(A, B), k):
            S = self.write(S, self.ids(torch.tensor(ch, device=dev)))
        return S

    def think(self, S, K, skip=(), override=None):
        B = S.shape[0]; ids = self.ids(torch.full((B, 1), PAD, device=S.device))
        for t in range(1, K + 1):
            if t not in skip:
                S = self.write(S, ids)
            if override is not None and override[0] == t:
                S = override[1]
        return S

    def dec_logits(self, S, ans):
        B = S.shape[0]; inp = torch.cat([torch.full((B, 1), EOT, device=S.device), ans[:, :-1]], 1)
        return self.g.logits(fwd_read(self.g, self.g.embed(inp), self.mem, S)).float()

    @torch.no_grad()
    def greedy(self, S, L):
        B = S.shape[0]; ids = torch.full((B, 1), EOT, device=S.device); lut = {int(t): i for i, t in enumerate(DTOK)}
        for _ in range(L):
            nt = self.g.logits(fwd_read(self.g, self.g.embed(ids), self.mem, S)[:, -1]).float().argmax(-1); ids = torch.cat([ids, nt[:, None]], 1)
        return np.vectorize(lambda t: lut.get(int(t), -1))(ids[:, 1:].cpu().numpy())


@torch.no_grad()
def predict(m, A, B, k, K, dev, amp, bs, skip=(), keep=None, override=None):
    """greedy answer digits (N, na + nb) after the chunks and K' = K think passes; keep: {'input'} / think pass numbers -> flattened states."""
    N = len(A); L = A.shape[1] + B.shape[1]; out = np.full((N, L), -1); kept = {}
    for i in range(0, N, bs):
        sl = slice(i, i + bs)
        with amp_ctx(amp):
            S = m.encode(A[sl], B[sl], k, dev)
            if keep and "input" in keep:
                kept.setdefault("input", []).append(S.float().flatten(1).cpu().numpy())
            Bn = S.shape[0]; ids = m.ids(torch.full((Bn, 1), PAD, device=dev))
            for t in range(1, K + 1):
                if t not in skip:
                    S = m.write(S, ids)
                if override is not None and override[0] == t:
                    S = torch.tensor(override[1][sl], device=dev, dtype=S.dtype)
                if keep and t in keep:
                    kept.setdefault(t, []).append(S.float().flatten(1).cpu().numpy())
            out[sl] = m.greedy(S, L)
    return out, {k_: np.concatenate(v) for k_, v in kept.items()}


def acc_report(pred, P):
    C = pred == P; return {"test_acc": float(C.all(1).mean()), "acc_by_column": {"per_column": [round(float(v), 4) for v in C.mean(0)]}, "n": int(len(P))}


def draw(rng, bs, ns, excl):
    nb = int(ns[int(rng.integers(0, len(ns)))]); A, B = gen_operands(rng, bs, 4), gen_operands(rng, bs, nb)
    if nb == 4 and excl:
        for _ in range(100):
            bad = np.array([(x.tobytes(), y.tobytes()) in excl for x, y in zip(A, B)])
            if not bad.any():
                break
            A[bad], B[bad] = gen_operands(rng, int(bad.sum()), 4), gen_operands(rng, int(bad.sum()), 4)
    return A, B


def compute_fc(m, na, nb, k, K):
    """boxes.core.Meter per item (§98 convention, full-vocabulary rows): every chunk pass [EOT, k tokens] and think pass [EOT, " ."] fresh
    (context 0, no rows); the decoder = na + nb KV-cached greedy passes, one token and one LM-head row each (Audit: counted like TC-M generated tokens); plus the
    DeltaMem-R branch (2 x branch params per token, the read and the writes per head and token, as multpair.compute_pair)."""
    mt = bc.Meter(m.g); nc = -(-(na + nb + 2) // k); tok = 0
    for _ in range(nc):
        mt.step(k + 1, 0, 0); tok += k + 1
    for _ in range(K):
        mt.step(2, 0, 0); tok += 2
    Ld = na + nb; tok_r = Ld
    for t in range(Ld):                                                     # Audit (3539b83e): the greedy decoder is na + nb sequential passes (KV-cached)
        mt.step(1, t, 1)
    nb_ = sum(p_.numel() for n_, p_ in m.mem.named_parameters() if not n_.startswith("s_")); D2 = DM.DH ** 2
    fb = 2 * nb_ * (tok + tok_r) + (DM.H_ADD + DM.H_DEL) * 2 * D2 * (tok + tok_r) + (DM.H_ADD * 2 + DM.H_DEL * 8) * D2 * tok; d = mt.as_dict()
    return {"flops": d["flops"] + fb, "flops_trunk": d["flops"], "flops_branch": float(fb), "sequential_passes": d["seq_passes"], "lm_head_rows": Ld,
            "tokens_processed": tok + tok_r, "chunks": nc, "think_passes": K, "convention": "§98 (boxes.core.Meter); every chunk / think pass fresh; decoder = na + nb KV-cached passes"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_k", type=int, required=True); ap.add_argument("--think_k", type=int, required=True); ap.add_argument("--shape_mix", default="2,3,4,5")
    ap.add_argument("--steps", type=int, default=48000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--micro", type=int, default=1); ap.add_argument("--mem_write", default="top", choices=["top", "block9"])
    ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000)
    ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", default=None); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=1000); ap.add_argument("--ext_n", type=int, default=1000)
    ap.add_argument("--probe_n_fit", type=int, default=8000); ap.add_argument("--donor_n", type=int, default=1000)
    ap.add_argument("--auto_micro", action="store_true"); ap.add_argument("--mem_budget", type=float, default=21.0); ap.add_argument("--mem_check", action="store_true")
    ap.add_argument("--sched_scale", type=float, default=1.0, help="smokes only"); ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); a.arm = "fixedchunk"; os.makedirs(a.out, exist_ok=True); dev = a.device; sc = a.sched_scale; k, K = a.chunk_k, a.think_k
    SHM = [int(x) for x in a.shape_mix.split(",") if x.strip()]; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    teA, teB = load_bench(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt"))
    EXCL = {(x.tobytes(), y.tobytes()) for x, y in zip(teA, teB)} | {(y.tobytes(), x.tobytes()) for x, y in zip(teA, teB)}; teA, teB = teA[:a.test_n], teB[:a.test_n]
    m = FixedChunk(s5.load_gpt2(dev), 6, a.mem_write).to(dev)

    def train_loss(A, B, c):
        S = m.think(m.encode(A, B, k, dev), K); P = product_digits(A, B); ans = torch.tensor(DT[P], device=dev); lg = m.dec_logits(S, ans); ce = P.shape[1] if c >= 8 else min(c, P.shape[1])
        return Fn.cross_entropy(lg[:, :ce].reshape(-1, lg.shape[-1]), ans[:, :ce].reshape(-1))

    def mem_check(micro):
        torch.cuda.reset_peak_memory_stats(dev); r_ = np.random.default_rng(0); A, B = gen_operands(r_, a.bs // micro, 4), gen_operands(r_, a.bs // micro, max(SHM)); m.train()
        with amp_ctx(amp):
            l_ = train_loss(A, B, 8)
        l_.backward(); m.zero_grad(set_to_none=True); m.eval()
        predict(m, gen_operands(r_, a.eval_bs, 4), gen_operands(r_, a.eval_bs, 16), k, 4 * K, dev, amp, a.eval_bs); m.train()   # the 4x16 eval chain at K' = 4K
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30
    if a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.resume):
        mc = []
        while True:
            pk = mem_check(a.micro); mc.append({"micro": a.micro, "peak_gib": pk}); print("MEM", a.micro, pk, flush=True)
            if pk <= a.mem_budget or a.micro >= a.bs or a.mem_check:
                break
            a.micro *= 2
        json.dump({"checks": mc, "micro": a.micro, "peak_gib": mc[-1]["peak_gib"], "train_shape": "4x%d" % max(SHM), "eval_shape": "4x16 at K' = %d" % (4 * K)},
                  open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return
        assert mc[-1]["peak_gib"] <= a.mem_budget, "memory %.1f GiB at micro %d" % (mc[-1]["peak_gib"], a.micro)
    info = {**vars(a), "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info(),
            "tokens": {"star": " *", "eq": " =", "pad": " ."}, "think_input": "[EOT, ' .']"}
    json.dump(info, open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    opt = make_opt(m, a.lr); step0 = 1; n_nonfinite = 0; wg_hist = []; wwin = []; wlast = []; hist = []
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time()
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu()); step0 = ck["step"] + 1; n_nonfinite = ck["n_nonfinite"]; wg_hist = ck["wg_hist"]; t0 = time.time() - ck["elapsed"]
        if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    else:                                                                   # note 147-2: g_add = median ||read_del|| / ||read_add|| at the chunk / think hand-offs
        r_ = np.random.default_rng(16501); A0, B0 = gen_operands(r_, 128, 4), gen_operands(r_, 128, 4); m.mem.norms = []
        with torch.no_grad(), amp_ctx(amp):
            m.think(m.encode(A0, B0, k, dev), min(K, 4))
        nr = m.mem.norms[1:]; m.mem.norms = None; r0 = float(torch.stack([d_ / a_.clamp_min(1e-12) for d_, a_ in nr]).median()) if nr else 1.0
        m.mem.s_add.data.fill_(math.log(max(r0, 1e-12))); json.dump({"g_add_init": r0}, open(os.path.join(a.out, "s165_init.json"), "w", newline="\n"))
    for step in range(step0, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        m.train(); m.mem.fade.fill_(min(1.0, step / max(1, int(500 * sc))))
        m.mem.gamma.fill_(0.0 if step <= a.write_mix_start * sc else min(1.0, (step - a.write_mix_start * sc) / max(1.0, (a.write_mix_end - a.write_mix_start) * sc)))
        c = int(rng.integers(1, cmax(step / sc) + 1)); A, B = draw(rng, a.bs, SHM, EXCL); opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro
        for k_ in range(a.micro):
            sl = slice(k_ * ch, (k_ + 1) * ch)
            with amp_ctx(amp):
                lk = train_loss(A[sl], B[sl], c)
            (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
        wwin.append(write_grad_norm(m)); gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush(); n_nonfinite += 1; opt.zero_grad()
        else:
            opt.step(); hist.append(tot)
        if step % a.log_every == 0 or step == a.steps:
            md_ = float(np.median(wlast)) if wlast else 0.0; mx_ = float(np.max(wwin)); late = step > a.write_mix_start * sc
            w_rec = {"step": step, "mean": float(np.mean(wwin)), "max": mx_, "median_prev_1k": md_, "spike_100x": bool(late and mx_ > 100 * md_ > 0),
                     "spike_100x_floor": bool(late and mx_ > 100 * md_ > 0 and mx_ > 1.0)}
            wlast.extend(wwin); del wlast[:-int(1000 * sc) or None]; wwin = []; wg_hist.append(w_rec)
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "c_max": cmax(step / sc), "sec": round(time.time() - t0), "g_add": float(torch.exp(m.mem.s_add)),
                   "g_del": float(torch.exp(m.mem.s_del)), "fade": float(m.mem.fade), "gamma": float(m.mem.gamma), "write_grad_norm": w_rec["mean"], "write_grad_spike_100x_floor": w_rec["spike_100x_floor"]}
            hist = []; log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None, "step": step, "n_nonfinite": n_nonfinite, "wg_hist": wg_hist,
                        "elapsed": time.time() - t0}, cp)
    train_sec = round(time.time() - t0); torch.save(m.state_dict(), os.path.join(a.out, "model.pt")); m.eval(); t1 = time.time(); bs = a.eval_bs
    Pte = product_digits(teA, teB); pred, _ = predict(m, teA, teB, k, K, dev, amp, bs); res = acc_report(pred, Pte)
    res.update({"arm": "fixedchunk", "chunk_k": k, "think_k": K, "seed": a.seed, "train_sec": train_sec, "n_nonfinite": n_nonfinite, "eval_dtype": str(amp), "micro": a.micro,
                "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_gpu": info["train_gpu"], "train_driver": info["train_driver"]})
    xs = {}
    for nb_, s1_, s2_ in ((6, 14016, 14017), (7, 14018, 14019), (8, 14004, 14005), (16, 14006, 14007)):
        xA, xB = gen_operands(np.random.default_rng(s1_), a.ext_n, 4), gen_operands(np.random.default_rng(s2_), a.ext_n, nb_); xs[nb_] = (xA, xB)
        res["ext_4x%d" % nb_] = {**acc_report(predict(m, xA, xB, k, K, dev, amp, bs)[0], product_digits(xA, xB)), "seeds": [s1_, s2_]}
    res["compute"] = compute_fc(m, 4, 4, k, K); res["compute_4x8"] = compute_fc(m, 4, 8, k, K)
    s = {"acc_by_kprime": {}, "ext_4x8_by_kprime": {}, "skip_pass": {}, "write_grad_norm": wg_hist, "write_grad_spikes_floor": [x_ for x_ in wg_hist if x_["spike_100x_floor"]],
         "n_nonfinite": n_nonfinite, "gamma_final": float(m.mem.gamma), "g_add": float(torch.exp(m.mem.s_add)), "g_del": float(torch.exp(m.mem.s_del))}
    for kp in sorted({0, 1, K // 2, K, 2 * K, 4 * K}):
        s["acc_by_kprime"][str(kp)] = float((predict(m, teA, teB, k, kp, dev, amp, bs)[0] == Pte).all(1).mean())
    for kp in sorted({K, 2 * K, 4 * K}):
        s["ext_4x8_by_kprime"][str(kp)] = float((predict(m, *xs[8], k, kp, dev, amp, bs)[0] == product_digits(*xs[8])).all(1).mean())
    pts = sorted({t for t in (1, K // 4, K // 2, 3 * K // 4, K) if t >= 1}) if K else []
    for t in pts:
        s["skip_pass"][str(t)] = float((predict(m, teA, teB, k, K, dev, amp, bs, skip=(t,))[0] == Pte).all(1).mean())
    if K:                                                                   # donor_think (sanity): the donor's whole memory after think pass K/2
        r_ = np.random.default_rng(16502); N = min(a.donor_n, len(teA)); ir, idn = r_.permutation(len(teA))[:N], r_.permutation(len(teA))[:N]; tk = max(1, K // 2)
        _, kd = predict(m, teA[idn], teB[idn], k, tk, dev, amp, bs, keep={tk}); Sd = kd[tk].reshape((N, DM.H_ADD + DM.H_DEL, DM.DH, DM.DH))
        got, _ = predict(m, teA[ir], teB[ir], k, K, dev, amp, bs, override=(tk, Sd))
        s["donor_think"] = {"pass": tk, "donor_consistent": float((got == Pte[idn]).all(1).mean()), "recipient_consistent": float((got == Pte[ir]).all(1).mean()),
                            "chance_overlap": float((Pte[idn] == Pte[ir]).all(1).mean()), "n": int(N)}
    from boxes.multpair import _dual_fit_pred                               # digit_emergence: 10-way near-OLS (155-4) of the product digits
    fA, fB = draw(np.random.default_rng(16503), a.probe_n_fit, [4], EXCL); keep = {"input"} | set(pts)
    _, Kf = predict(m, fA, fB, k, K, dev, amp, bs, keep=keep); _, Ks = predict(m, teA, teB, k, K, dev, amp, bs, keep=keep); Pf = product_digits(fA, fB); de = {}
    for key in ["input"] + pts:
        de["after_input" if key == "input" else "after_pass_%d" % key] = {str(d + 1): float((_dual_fit_pred(Kf[key], np.eye(10)[Pf[:, d]], Ks[key], dev, cls=True)[0].argmax(1) == Pte[:, d]).mean())
                                                                          for d in range(Pf.shape[1])}
    s["digit_emergence"] = de; s["probe_n_fit"] = int(a.probe_n_fit); res["s165"] = s; res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({"test_acc": res["test_acc"], "acc_by_kprime": s["acc_by_kprime"]}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
