"""THEORY §101: snap-strength dose-response on S5 (§97's HYB-4: text step target via the fork + tied LM head; §92's
structured codebook; k = 4). The next chunk's start embedding is s = (1 - beta) * u + beta * code_emb(state), with u = g(h)
the LAST slot's output hidden mapped by §89's standardisation g(h) = mu_E + sd_E * (h - mu_h) / sd_h (mu_h, sd_h an EMA,
momentum 0.99, over the last-slot hiddens of each update, first update initialises; frozen at the end; mu_E, sd_E =
GPT-2's initial wte statistics). The history cut is kept at every beta.
Training (24k steps, bs 64, lr 1e-4, §97's schedule): each update draws a sequence of a random 1-4 chunks (4-16 ops) from
the S5=128 train set, starting from the TRUE start code; gradients flow through every u. At each chunk boundary one coin
per batch gives beta = 1 with probability q (q = 1 -> 0 linearly over the first 60 % of updates, then 0), else beta_min;
the code at a training boundary is the TRUE state's code (teacher-forced, as HYB-4's true start code). Late noise: u +=
N(0, (0.05 sd_E)^2) in the last third. Loss: §97's per-slot fork CE, mean over the sequence's chunks.
Test (beta = beta_min at every boundary; the code from the decoded step text's letters, as HYB-4): all-position rollout on
§71's test set (2,000 x 512) and the 2,048 extension (first 500).
Keys: gen97's cumulative / extension_2048 / removal (chunk_check slot inputs 1..3 zero / mean) / chunk_check.per_slot_acc
and per_chunk_acc, plus beta_min, acc_vs_length {128, 256, 512, 1024, 2048}, removal_u.{zero,mean}.cumulative_512 (u at
every boundary zeroed / replaced by its mean; the beta * code part kept), drift {chunk_index, u_norm, code_dist} along the
2,048 rollout (u_norm = mean ||u||, code_dist = mean distance from u to the nearest of the 120 code embeddings),
u_code_norm_ratio (mean ||u|| / mean ||code_emb||), checkpoint_validity, compute_512_per_problem.flops.
    python -m boxes.snap101 --beta_min 0.1 --seed 1 --out boxes/runs_s5/s101_b0.1_s1
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import core as bc  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.gen97 import Gen97, amp_ctx, chunk_check, decode_to_el, mean_slot_inputs  # noqa: E402
from boxes.latent86 import LETIDX, batch  # noqa: E402

K = 4


class Snap101(Gen97):
    def __init__(self, g):
        super().__init__(g, "hyb")
        E = g.w("wte.weight").detach()
        self.register_buffer("muE", E.mean(0).clone()); self.register_buffer("sdE", E.std(0).clone())
        self.register_buffer("muH", torch.zeros(768)); self.register_buffer("sdH", torch.ones(768)); self.register_buffer("seen", torch.tensor(0))
        self._raw = []; self._hlast = None

    def start_emb(self, start):
        if torch.is_floating_point(start):                                  # a soft-snap start embedding (B, D)
            return start.view(start.shape[0], 1, -1)
        return super().start_emb(start)

    def code_emb(self, start):
        """(B,) element ids or (B, 5) letters -> (B, D)."""
        return super().start_emb(start)[:, 0]

    def u_of(self, h):
        return self.muE + self.sdE * (h.float() - self.muH) / self.sdH

    @torch.no_grad()
    def update_stats(self):
        if self._raw:
            H = torch.cat(self._raw); hm, hs = H.mean(0), H.std(0) + 1e-5
            if int(self.seen) == 0:
                self.muH.copy_(hm); self.sdH.copy_(hs)
            else:
                self.muH.mul_(0.99).add_(0.01 * hm); self.sdH.mul_(0.99).add_(0.01 * hs)
            self.seen += 1
        self._raw = []

    def chunk(self, start, ops, tgt=None, decode="all", ablate=None, mean_in=None, meter=None, meter_x=None, collect=False):
        """Gen97.chunk, keeping the last slot's output hidden in self._hlast (and its detached copy for the EMA)."""
        import torch.nn.functional as F
        from boxes.gen97 import NST
        B, k = ops.shape; g = self.g
        x = torch.cat([self.start_emb(start).to(g.embed(self.optok[ops[:, 0]]).dtype), g.embed(self.optok[ops[:, 0]])], 1)
        past = None; ins, losses, decs = [], [], []
        for j in range(k):
            ctx = 0 if past is None else past[0][0].shape[2]
            h, past = g(x, past)
            if meter:
                meter.step(x.shape[1], ctx, 0)
            s_in = h[:, -1]
            if ablate in ("zero", "mean") and j < k - 1:
                s_in = torch.zeros_like(s_in) if ablate == "zero" else mean_in[j][None].expand_as(s_in)
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
            if decode == "all" or (decode == "last" and j == k - 1):
                decs.append(self.greedy(hslot, past, meter if j == k - 1 else meter_x))
            if j + 1 < k:
                x = g.embed(self.optok[ops[:, j + 1]])
        self._hlast = hslot
        if collect:
            self._raw.append(hslot.detach().float())
        loss = torch.stack(losses).mean() if losses else None
        return loss, (torch.stack(decs, 1) if decs else None), torch.stack(ins, 1)


@torch.no_grad()
def rollout_soft(m, X, beta, dev, amp, bs, ablate_u=None, u_mean=None, drift_every=0, meter=None):
    """All-position rollout with soft snaps. Returns pred (N, L) element ids, drift records, mean ||u||, validity."""
    N, L = X.shape; pred = np.full((N, L), -1, dtype=np.int64); codes = m.code_emb(torch.arange(120, device=dev))   # (120, D)
    drift = {}; un = []; snaps = valid = 0
    for s in range(0, N, bs):
        n = min(bs, N - s); start = m.code_emb(torch.zeros(n, dtype=torch.long, device=dev))   # the identity's code
        for ci, c in enumerate(range(0, L, K)):
            o = torch.tensor(X[s:s + bs, c:c + K], device=dev)
            with amp_ctx(amp):
                _, d, _ = m.chunk(start, o, decode="all", meter=meter if (s == 0 and meter is not None) else None)
            toks = d.cpu().numpy(); el, let = decode_to_el(toks.reshape(-1, toks.shape[-1]))
            el = el.reshape(n, -1); let = let.reshape(n, -1, 5); pred[s:s + bs, c:c + o.shape[1]] = el
            snaps += n; valid += int((el[:, -1] >= 0).sum())
            u = m.u_of(m._hlast)
            un.append(u.norm(dim=-1).mean().item())
            if drift_every and ci % drift_every == 0:
                dd = torch.cdist(u.float(), codes.float()).min(-1).values
                r = drift.setdefault(ci, [0.0, 0.0, 0]); r[0] += u.norm(dim=-1).sum().item(); r[1] += dd.sum().item(); r[2] += n
            if ablate_u == "zero":
                u = torch.zeros_like(u)
            elif ablate_u == "mean":
                u = u_mean[None].expand_as(u)
            code = m.code_emb(torch.tensor(let[:, -1], device=dev))
            start = (1 - beta) * u + beta * code.float()
    dr = {"chunk_index": sorted(drift), "u_norm": [drift[i][0] / drift[i][2] for i in sorted(drift)],
          "code_dist": [drift[i][1] / drift[i][2] for i in sorted(drift)]}
    return pred, dr, float(np.mean(un)), {"snaps": snaps, "valid": valid, "rate": valid / max(1, snaps)}


@torch.no_grad()
def mean_u(m, X, Y, dev, amp, n=2000, seed=10105):
    rng = np.random.default_rng(seed); st, o, _ = batch(rng, X, Y, K, n); out = []
    for i in range(0, n, 250):
        with amp_ctx(amp):
            m.chunk(torch.tensor(st[i:i + 250], device=dev), torch.tensor(o[i:i + 250], device=dev), decode=None)
        out.append(m.u_of(m._hlast))
    return torch.cat(out).mean(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--beta_min", type=float, required=True); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=s5.TEST_N); ap.add_argument("--ext_n", type=int, default=500)
    ap.add_argument("--ext_len", type=int, default=2048, help="< 2048: smoke only"); ap.add_argument("--check_n", type=int, default=100_000, help="< 100k: smoke only")
    ap.add_argument("--log_every", type=int, default=2000); ap.add_argument("--log_n", type=int, default=2000)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "bench", "data", "s5"))
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    s5.check_interface()
    Xtr, Ytr = s5.load_csv(os.path.join(a.data_dir, "S5=128.csv")); s5.check_labels(Xtr, Ytr)
    Xte, Yte = s5.load_csv(os.path.join(a.data_dir, "S5=512.csv"), limit=a.test_n); s5.check_labels(Xte, Yte)
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), "train_sequences": len(Xtr)}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    m = Snap101(s5.load_gpt2(dev)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []; nb = []
    for step in range(1, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        q = max(0.0, 1.0 - step / (0.6 * a.steps)); late = step > 2 * a.steps / 3
        nch = int(rng.integers(1, 5)); st, o, tg = batch(rng, Xtr, Ytr, K * nch, a.bs)
        start = torch.tensor(st, device=dev); tot = 0.0
        with amp_ctx(amp):
            for c in range(nch):
                oc = torch.tensor(o[:, K * c:K * (c + 1)], device=dev); tc = torch.tensor(tg[:, K * c:K * (c + 1)], device=dev)
                loss, _, _ = m.chunk(start, oc, tc, decode=None, collect=True); tot = tot + loss
                if c + 1 < nch:
                    u = m.u_of(m._hlast)
                    if late:
                        u = u + torch.randn_like(u) * (0.05 * m.sdE)
                    beta = 1.0 if rng.random() < q else a.beta_min
                    start = (1 - beta) * u + beta * m.code_emb(tc[:, -1]).float()
            loss = tot / nch
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); m.update_stats()
        hist.append(loss.item()); nb.append(nch)
        if step % 100 == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)), "q": q, "late_noise": late, "mean_chunks": float(np.mean(nb)), "sec": round(time.time() - t0)}; hist = []; nb = []
            if a.log_every and step % a.log_every == 0:
                m.eval(); cc = chunk_check(m, K, dev, amp, a.eval_bs, n=a.log_n, seed=7107)
                rec["per_slot_acc"] = cc["per_slot_acc"]; rec["per_chunk_acc"] = cc["per_chunk_acc"]
                p, _, un, _ = rollout_soft(m, Xte[:200], a.beta_min, dev, amp, 200); rec["cum512_200"] = float(s5.cumulative(p, Yte[:200])[-1]); rec["u_norm"] = un
                m.train()
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    train_sec = round(time.time() - t0)
    torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    del opt; m.zero_grad(set_to_none=True); m.eval(); t1 = time.time()
    res = {"arm": "hyb_snap", "beta_min": a.beta_min, "k": K, "seed": a.seed, "train_sec": train_sec, "eval_dtype": str(amp),
           "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "test_sequences": len(Xte)}
    meter = bc.Meter(m.g)
    pred, _, un, val = rollout_soft(m, Xte, a.beta_min, dev, amp, a.eval_bs, meter=meter)
    np.save(os.path.join(a.out, "preds_test.npy"), pred); cum = s5.cumulative(pred, Yte)
    res["cumulative"] = {str(t): float(cum[t - 1]) for t in (128, 256, 512) if t <= Xte.shape[1]}
    res["final_state_acc_512"] = float((pred[:, -1] == Yte[:, -1]).mean()); res["checkpoint_validity"] = val
    d = meter.as_dict(); d["latent_slots"] = Xte.shape[1]; res["compute_512_per_problem"] = d
    codes = m.code_emb(torch.arange(120, device=dev)); res["u_code_norm_ratio"] = un / float(codes.float().norm(dim=-1).mean())
    acc_len = {str(t): float(cum[t - 1]) for t in (128, 256, 512) if t <= Xte.shape[1]}
    if a.ext_n:
        Xx, Yx = s5.load_csv(os.path.join(a.data_dir, "S5=2048.csv"), limit=a.ext_n); s5.check_labels(Xx, Yx)
        Xx, Yx = Xx[:, :a.ext_len], Yx[:, :a.ext_len]
        px, drift, _, _ = rollout_soft(m, Xx, a.beta_min, dev, amp, a.eval_bs, drift_every=8); cx = s5.cumulative(px, Yx)
        res["extension_2048"] = {"n": len(Xx), "len": Xx.shape[1], "cumulative": {str(t): float(cx[t - 1]) for t in (512, 1024, 2048) if t <= Xx.shape[1]}}
        acc_len.update({str(t): float(cx[t - 1]) for t in (1024, 2048) if t <= Xx.shape[1]}); res["drift"] = drift
    res["acc_vs_length"] = acc_len
    res["chunk_check"] = chunk_check(m, K, dev, amp, a.eval_bs, n=a.check_n)
    mu = mean_slot_inputs(m, Xtr, Ytr, K, dev, amp)
    res["removal"] = {ab: chunk_check(m, K, dev, amp, a.eval_bs, n=min(10_000, a.check_n), seed=7106, ablate=ab, mean_in=mu) for ab in ("zero", "mean")}
    um = mean_u(m, Xtr, Ytr, dev, amp); res["removal_u"] = {}
    for ab in ("zero", "mean"):
        p_, _, _, _ = rollout_soft(m, Xte, a.beta_min, dev, amp, a.eval_bs, ablate_u=ab, u_mean=um)
        res["removal_u"][ab] = {"cumulative_512": float(s5.cumulative(p_, Yte)[-1])}
    res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({k: res[k] for k in ("cumulative", "u_code_norm_ratio")}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
