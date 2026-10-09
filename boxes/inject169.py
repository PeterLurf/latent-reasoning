"""THEORY §169 INJECT (5fffefd1; scorer theory/score_s169.py): a task-agnostic latent harness. §155 R's DeltaMem-R (6 + 6 heads after block 6, top
writes, gamma 2k -> 6k, pass-start reads, W_o fade-in, g_add calibrated at step 0) is the ONLY cross-pass state.
    think pass t (t = 1..K) = [EOT, the FULL problem text " a_1 .. a_4 * b_1 .. b_n", " ."] - a fresh cache, the same text every pass; it reads the
                              memory then writes (no chunking, no schedule, no flags)
    decoder                 = [EOT, problem text, " ="] -> the 4 + n answer digits LSB first (teacher-forced in training, greedy at test); it reads the
                              memory and writes NOTHING. K = 0 is the same-harness single pass.
Training: FINAL only, §161's 4 x n mix (n ~ U{2..5} per batch; multpair's gen_operands minus the 4x4 bench test, both orders), 165-1's curriculum (CE on the
lowest min(c, 4 + n) digits, c >= 8 -> all), 48k, AdamW 1e-4 wd 0.01, warm-up 100 + cosine, clip 1.0; ckpts / resume; --auto_micro (4x5 training + the
4x16 eval at K' = 4K). Optional (§168 step 2): --micro (gradient accumulation), --ckpt_passes (torch.utils.checkpoint around each think pass).
eval.json: test_acc / acc_by_column (4x4 bench, K' = K), ext_4x6 / 4x7 / 4x8 / 4x16 (§161's seeds), compute / compute_4x8 (Meter), s169 {acc_by_kprime
{0, 1, K/2, K, 2K, 4K}, ext_4x8_by_kprime {K, 2K, 4K}, acc_by_size_kprime {4x4 / 4x6 / 4x8 / 4x16: {K': acc}}, skip_pass {t: acc}, inprog_by_pass
{pass: {col: {units, carry}}} (155-4's colsum_cls on the memory after think pass t; 8k generator 4x4 fit, scored on the bench), spikes, n_nonfinite}.
    python -m boxes.inject169 --think_k 16 --seed 1 --out boxes/runs_fsq/s169_k16_4x4_s1"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes import core as bc  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes.fsq120 import EOT, make_opt, lr_at, driver_version, env_info  # noqa: E402
from boxes.multcol import DTOK, gen_operands, load_bench  # noqa: E402
from boxes.deltas5 import fwd_read, write_grad_norm  # noqa: E402
from boxes.fixedchunk import FixedChunk, DT, STAR, EQ, PAD, product_digits, acc_report, draw, cmax, amp_ctx  # noqa: E402


def think_text(A, B):
    """the think-pass input after EOT: a digits, " *", b digits (LSB first), " ." -> (N, na + nb + 2)."""
    return np.concatenate([DT[A], np.full((len(A), 1), STAR), DT[B], np.full((len(A), 1), PAD)], 1)


def dec_prefix(A, B):
    """the decoder's prefix after EOT: the problem text and " =" -> (N, na + nb + 2)."""
    return np.concatenate([DT[A], np.full((len(A), 1), STAR), DT[B], np.full((len(A), 1), EQ)], 1)


class Inject(FixedChunk):
    def think(self, S, A, B, K, skip=(), override=None, ckpt=False):
        ids = self.ids(torch.tensor(think_text(A, B), device=S.device))
        for t in range(1, K + 1):
            if t not in skip:
                S = checkpoint(self.write, S, ids, use_reentrant=False) if (ckpt and S.requires_grad is not None and torch.is_grad_enabled()) else self.write(S, ids)
            if override is not None and override[0] == t:
                S = override[1]
        return S

    def dec_logits(self, S, A, B, ans):
        """teacher-forced: [EOT, prefix, ans[:-1]] -> logits predicting the answer digits (positions from " =" on)."""
        pre = torch.tensor(dec_prefix(A, B), device=S.device); x = self.ids(torch.cat([pre, ans[:, :-1]], 1)); P = pre.shape[1] + 1
        h = fwd_read(self.g, self.g.embed(x), self.mem, S); return self.g.logits(h[:, P - 1:]).float()

    @torch.no_grad()
    def greedy_ans(self, S, A, B, L):
        x = self.ids(torch.tensor(dec_prefix(A, B), device=S.device)); lut = {int(t): i for i, t in enumerate(DTOK)}; out = []
        for _ in range(L):
            nt = self.g.logits(fwd_read(self.g, self.g.embed(x), self.mem, S)[:, -1]).float().argmax(-1); out.append(nt); x = torch.cat([x, nt[:, None]], 1)
        return np.vectorize(lambda t: lut.get(int(t), -1))(torch.stack(out, 1).cpu().numpy())


@torch.no_grad()
def predict(m, A, B, K, dev, amp, bs, skip=(), keep=None):
    N = len(A); L = A.shape[1] + B.shape[1]; out = np.full((N, L), -1); kept = {}
    for i in range(0, N, bs):
        sl = slice(i, i + bs); S = m.mem.zero_state(len(A[sl]), dev)
        with amp_ctx(amp):
            ids = m.ids(torch.tensor(think_text(A[sl], B[sl]), device=dev))
            for t in range(1, K + 1):
                if t not in skip:
                    S = m.write(S, ids)
                if keep and t in keep:
                    kept.setdefault(t, []).append(S.float().flatten(1).cpu().numpy())
            out[sl] = m.greedy_ans(S, A[sl], B[sl], L)
    return out, {k_: np.concatenate(v) for k_, v in kept.items()}


def compute_inj(m, na, nb, K):
    """Meter per item (§98 convention): K think passes of 1 + na + nb + 2 tokens (fresh, no rows); the decoder's prefix pass (1 + na + nb + 2 tokens,
    1 row) then na + nb - 1 KV-cached single-token passes (1 row each); plus the DeltaMem-R branch (as fixedchunk.compute_fc)."""
    mt = bc.Meter(m.g); Tt = 1 + na + nb + 2; tok = 0
    for _ in range(K):
        mt.step(Tt, 0, 0); tok += Tt
    mt.step(Tt, 0, 1); tok_r = Tt
    for j in range(na + nb - 1):
        mt.step(1, Tt + j, 1); tok_r += 1
    nb_ = sum(p_.numel() for n_, p_ in m.mem.named_parameters() if not n_.startswith("s_")); D2 = DM.DH ** 2
    fb = 2 * nb_ * (tok + tok_r) + (DM.H_ADD + DM.H_DEL) * 2 * D2 * (tok + tok_r) + (DM.H_ADD * 2 + DM.H_DEL * 8) * D2 * tok; d = mt.as_dict()
    return {"flops": d["flops"] + fb, "flops_trunk": d["flops"], "flops_branch": float(fb), "sequential_passes": d["seq_passes"], "lm_head_rows": na + nb,
            "tokens_processed": tok + tok_r, "think_passes": K, "convention": "§98 (boxes.core.Meter); think passes fresh; decoder prefix + (na + nb - 1) KV-cached passes"}


def colsum_cls(Kf, Af, Bf, Ks, As, Bs, dev):
    """155-4's colsum_cls: per column k the full accumulator c^_k (products + carry-in), 10-way units and the carry class, from the memory."""
    from boxes.multpair import _dual_fit_pred
    def acc_cols(A, B):
        n = len(A); L = A.shape[1] + B.shape[1] - 1; cs = np.zeros((n, L), np.int64)
        for i in range(A.shape[1]):
            for j in range(B.shape[1]):
                cs[:, i + j] += A[:, i] * B[:, j]
        ac = np.zeros_like(cs); c_ = np.zeros(n, np.int64)
        for k in range(L):
            ac[:, k] = cs[:, k] + c_; c_ = ac[:, k] // 10
        return ac
    af, as_ = acc_cols(Af, Bf), acc_cols(As, Bs); out = {}
    for k in range(af.shape[1]):
        r = {}
        for nm, yf, ys in (("units", af[:, k] % 10, as_[:, k] % 10), ("carry", af[:, k] // 10, as_[:, k] // 10)):
            kk = int(max(yf.max(), ys.max())) + 1; pr, _ = _dual_fit_pred(Kf, np.eye(kk)[yf], Ks, dev, cls=True); r[nm] = float((pr.argmax(1) == ys).mean())
        out[str(k + 1)] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think_k", type=int, required=True); ap.add_argument("--shape_mix", default="2,3,4,5"); ap.add_argument("--steps", type=int, default=48000)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--micro", type=int, default=1); ap.add_argument("--ckpt_passes", action="store_true", help="activation checkpointing around each think pass (§168 step 2)")
    ap.add_argument("--mem_write", default="top", choices=["top", "block9"]); ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000)
    ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", default=None); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=1000); ap.add_argument("--ext_n", type=int, default=1000)
    ap.add_argument("--probe_n_fit", type=int, default=8000); ap.add_argument("--auto_micro", action="store_true"); ap.add_argument("--eval_at", type=int, default=0, help="also evaluate test_acc / ext_4x8 at this step -> eval_<k>k.json"); ap.add_argument("--fast_add", action="store_true", help="deltamem.FAST_ADD: the additive heads' pass writes as one sum (Audit 2026-10-09; scratch/gate_fastadd.py)"); ap.add_argument("--mem_budget", type=float, default=21.0)
    ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--sched_scale", type=float, default=1.0, help="smokes only"); ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--bench", default=os.path.join("bench", "data", "mult_4x4_test.txt")); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); a.arm = "inject"; os.makedirs(a.out, exist_ok=True); dev = a.device; sc = a.sched_scale; K = a.think_k; DM.FAST_ADD = bool(a.fast_add)
    SHM = [int(x) for x in a.shape_mix.split(",") if x.strip()]; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    teA, teB = load_bench(a.bench if os.path.isabs(a.bench) else os.path.join(ROOT, a.bench))
    EXCL = {(x.tobytes(), y.tobytes()) for x, y in zip(teA, teB)} | {(y.tobytes(), x.tobytes()) for x, y in zip(teA, teB)}; teA, teB = teA[:a.test_n], teB[:a.test_n]
    m = Inject(s5.load_gpt2(dev), 6, a.mem_write).to(dev)

    def train_loss(A, B, c):
        S = m.think(m.mem.zero_state(len(A), dev), A, B, K, ckpt=a.ckpt_passes); P = product_digits(A, B); ans = torch.tensor(DT[P], device=dev)
        lg = m.dec_logits(S, A, B, ans); ce = P.shape[1] if c >= 8 else min(c, P.shape[1])
        return Fn.cross_entropy(lg[:, :ce].reshape(-1, lg.shape[-1]), ans[:, :ce].reshape(-1))

    def mem_check(micro):
        torch.cuda.reset_peak_memory_stats(dev); r_ = np.random.default_rng(0); A, B = gen_operands(r_, a.bs // micro, 4), gen_operands(r_, a.bs // micro, max(SHM)); m.train()
        with amp_ctx(amp):
            l_ = train_loss(A, B, 8)
        l_.backward(); m.zero_grad(set_to_none=True); m.eval()
        predict(m, gen_operands(r_, a.eval_bs, 4), gen_operands(r_, a.eval_bs, 16), 4 * K, dev, amp, a.eval_bs); m.train()
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
        assert mc[-1]["peak_gib"] <= a.mem_budget
    info = {**vars(a), "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info(),
            "think_input": "[EOT, a * b, ' .'] (the full problem text, every pass)", "decoder_input": "[EOT, a * b, ' ='] + answer digits (read-only)"}
    json.dump(info, open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    opt = make_opt(m, a.lr); step0 = 1; n_nonfinite = 0; wg_hist = []; wwin = []; wlast = []; hist = []
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time()
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu()); step0 = ck["step"] + 1; n_nonfinite = ck["n_nonfinite"]; wg_hist = ck["wg_hist"]; t0 = time.time() - ck["elapsed"]
        if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    elif K > 0:                                                             # note 147-2: g_add = median ||read_del|| / ||read_add|| at the think hand-offs
        r_ = np.random.default_rng(16901); A0, B0 = gen_operands(r_, 128, 4), gen_operands(r_, 128, 4); m.mem.norms = []
        with torch.no_grad(), amp_ctx(amp):
            m.think(m.mem.zero_state(128, dev), A0, B0, min(K, 4))
        nr = m.mem.norms[1:]; m.mem.norms = None; r0 = float(torch.stack([d_ / a_.clamp_min(1e-12) for d_, a_ in nr]).median()) if nr else 1.0
        m.mem.s_add.data.fill_(math.log(max(r0, 1e-12))); json.dump({"g_add_init": r0}, open(os.path.join(a.out, "s169_init.json"), "w", newline="\n"))
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
        if a.eval_at and step == a.eval_at:                                 # Theory 7caa3a82 (§174's matched budget): accuracy at this step, no checkpoint kept
            m.eval(); P_ = product_digits(teA, teB); xa, xb = gen_operands(np.random.default_rng(14004), a.ext_n, 4), gen_operands(np.random.default_rng(14005), a.ext_n, 8)
            e24 = {"step": step, "test_acc": float((predict(m, teA, teB, K, dev, amp, a.eval_bs)[0] == P_).all(1).mean()),
                   "ext_4x8": {"test_acc": float((predict(m, xa, xb, K, dev, amp, a.eval_bs)[0] == product_digits(xa, xb)).all(1).mean()), "seeds": [14004, 14005]}}
            json.dump(e24, open(os.path.join(a.out, "eval_%dk.json" % (step // 1000)), "w", newline="\n"), indent=1); m.train()
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
    Pte = product_digits(teA, teB); pred, _ = predict(m, teA, teB, K, dev, amp, bs); res = acc_report(pred, Pte)
    res.update({"arm": "inject", "think_k": K, "seed": a.seed, "train_sec": train_sec, "n_nonfinite": n_nonfinite, "eval_dtype": str(amp), "micro": a.micro,
                "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_gpu": info["train_gpu"], "train_driver": info["train_driver"]})
    xs = {4: (teA, teB)}
    for nb_, s1_, s2_ in ((6, 14016, 14017), (7, 14018, 14019), (8, 14004, 14005), (16, 14006, 14007)):
        xA, xB = gen_operands(np.random.default_rng(s1_), a.ext_n, 4), gen_operands(np.random.default_rng(s2_), a.ext_n, nb_); xs[nb_] = (xA, xB)
        res["ext_4x%d" % nb_] = {**acc_report(predict(m, xA, xB, K, dev, amp, bs)[0], product_digits(xA, xB)), "seeds": [s1_, s2_]}
    res["compute"] = compute_inj(m, 4, 4, K); res["compute_4x8"] = compute_inj(m, 4, 8, K)
    kps = sorted({0, 1, K // 2, K, 2 * K, 4 * K}); s = {"acc_by_kprime": {}, "ext_4x8_by_kprime": {}, "acc_by_size_kprime": {}, "skip_pass": {}, "write_grad_norm": wg_hist,
                                                         "write_grad_spikes_floor": [x_ for x_ in wg_hist if x_["spike_100x_floor"]], "n_nonfinite": n_nonfinite,
                                                         "gamma_final": float(m.mem.gamma), "g_add": float(torch.exp(m.mem.s_add)), "g_del": float(torch.exp(m.mem.s_del))}
    for nb_ in (4, 6, 8, 16):
        Pn = product_digits(*xs[nb_]); s["acc_by_size_kprime"]["4x%d" % nb_] = {str(kp): float((predict(m, *xs[nb_], kp, dev, amp, bs)[0] == Pn).all(1).mean()) for kp in kps}
    s["acc_by_kprime"] = dict(s["acc_by_size_kprime"]["4x4"]); s["ext_4x8_by_kprime"] = {str(kp): s["acc_by_size_kprime"]["4x8"][str(kp)] for kp in sorted({K, 2 * K, 4 * K})}
    pts = sorted({t for t in (1, K // 4, K // 2, 3 * K // 4, K) if t >= 1}) if K else []
    for t in pts:
        s["skip_pass"][str(t)] = float((predict(m, teA, teB, K, dev, amp, bs, skip=(t,))[0] == Pte).all(1).mean())
    ipts = sorted({t for t in (K // 4, K // 2, 3 * K // 4, K) if t >= 1}) if K else []
    if ipts:
        fA, fB = draw(np.random.default_rng(16903), a.probe_n_fit, [4], EXCL); _, Kf = predict(m, fA, fB, K, dev, amp, bs, keep=set(ipts)); _, Ks = predict(m, teA, teB, K, dev, amp, bs, keep=set(ipts))
        s["inprog_by_pass"] = {str(t): colsum_cls(Kf[t], fA, fB, Ks[t], teA, teB, dev) for t in ipts}; s["probe_n_fit"] = int(a.probe_n_fit)
    if K:                                                                   # Audit (descriptive): the memory's norms per pass up to 4K (the additive heads grow ~K x)
        mn = {}
        with torch.no_grad():
            nA, nB = teA[:256], teB[:256]; S = m.mem.zero_state(len(nA), dev); ids = m.ids(torch.tensor(think_text(nA, nB), device=dev))
            for t in range(1, 4 * K + 1):
                with amp_ctx(amp):
                    S = m.write(S, ids)
                mn[str(t)] = {"additive": float(S[:, :DM.H_ADD].float().flatten(1).norm(dim=1).median()), "delta": float(S[:, DM.H_ADD:].float().flatten(1).norm(dim=1).median())}
        s["mem_norm_by_pass"] = mn
    res["s169"] = s; res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({"test_acc": res["test_acc"], "acc_by_kprime": s["acc_by_kprime"]}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
