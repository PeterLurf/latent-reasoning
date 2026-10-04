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

    def chunk(self, start, ops, tgt=None, decode="all", ablate=None, mean_in=None, meter=None, meter_x=None):
        """Markov-segment chunk with gen97's signature. start (B, 6) state-text tokens. self.gold: set of in-chunk
        positions j >= 1 whose segment input is the gold previous step text (training; needs tgt). self.override:
        (j, (B, D)) replaces slot j's input (0-based j < c-1; the donor patch)."""
        B, k = ops.shape; g = self.g; eot = g.embed(torch.full((B, 1), EOT, device=ops.device))
        ins, losses, decs = [], [], []; prev = g.embed(start)
        for j in range(k):
            x = torch.cat([eot, prev, g.embed(self.optok[ops[:, j]])], 1)
            h, past = g(x, None)                                             # a fresh segment, positions from 0
            if meter:
                meter.step(x.shape[1], 0, 0)
            s_in = h[:, -1]
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
            if decode == "all" or (decode == "last" and j == k - 1):
                decs.append(self.greedy(hslot, past, meter if j == k - 1 else meter_x))
            if j + 1 < k:
                if self.gold is not None and (j + 1) in self.gold and tgt is not None:
                    prev = g.embed(self.sttok[tgt[:, j]])                    # gold previous step text (teacher-forced input)
                else:
                    prev = s_in[:, None].to(eot.dtype)
        loss = torch.stack(losses).mean() if losses else None
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, required=True, choices=[4, 8]); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_bs", type=int, default=200); ap.add_argument("--test_n", type=int, default=s5.TEST_N); ap.add_argument("--ext_n", type=int, default=500)
    ap.add_argument("--ext_len", type=int, default=2048, help="< 2048: smoke only"); ap.add_argument("--check_n", type=int, default=100_000, help="< 100k: smoke only")
    ap.add_argument("--donor_n", type=int, default=2000); ap.add_argument("--handoff_n", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=2000); ap.add_argument("--log_n", type=int, default=2000)
    ap.add_argument("--sched_scale", type=float, default=1.0, help="smoke only"); ap.add_argument("--data_dir", default=os.path.join(ROOT, "bench", "data", "s5"))
    ap.add_argument("--ckpt_every", type=int, default=2000, help="training checkpoint <out>/ckpt.pt (model, optimiser, RNGs); resumed if present")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); sc = a.sched_scale
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    s5.check_interface()
    Xtr, Ytr = s5.load_csv(os.path.join(a.data_dir, "S5=128.csv")); s5.check_labels(Xtr, Ytr)
    Xte, Yte = s5.load_csv(os.path.join(a.data_dir, "S5=512.csv"), limit=a.test_n); s5.check_labels(Xte, Yte)
    os.makedirs(a.out, exist_ok=True); json.dump({**vars(a), "train_sequences": len(Xtr)}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    m = MK108(s5.load_gpt2(dev)).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    t0 = time.time(); hist = []; seg_types = [0, 0]; step0 = 0; ck = os.path.join(a.out, "ckpt.pt"); lp = os.path.join(a.out, "log.jsonl")
    if os.path.exists(ck):                                                  # resume after a host restart (k1, 2026-10-04)
        c = torch.load(ck, map_location=dev, weights_only=False); m.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        rng.bit_generator.state = c["np_rng"]; torch.set_rng_state(c["torch_rng"]); step0 = c["step"]; seg_types = c["seg_types"]; t0 -= c["sec"]
        if dev.startswith("cuda"):
            torch.cuda.set_rng_state(c["cuda_rng"], dev)
        keep = [r for r in open(lp, encoding="utf-8") if r.strip() and json.loads(r)["step"] <= step0] if os.path.exists(lp) else []
        open(lp, "w", newline="\n").writelines(keep); print("RESUMED", step0, flush=True); del c
    log = open(lp, "a", newline="\n")
    for step in range(step0 + 1, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        L = k_train(step, a.k, sc); rho = max(0.0, 1.0 - step / (12000 * sc))
        st, o, tg = batch(rng, Xtr, Ytr, L, a.bs)
        m.gold = {j for j in range(1, L) if rng.random() < rho} or None
        seg_types[0] += 1 + (len(m.gold) if m.gold else 0); seg_types[1] += L - 1 - (len(m.gold) if m.gold else 0)
        with amp_ctx(amp):
            loss, _, _ = m.chunk(m.start_from(els=st, dev=dev), torch.tensor(o, device=dev), torch.tensor(tg, device=dev), decode=None)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(loss.item()); m.gold = None
        if step % 100 == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)), "rho": rho, "k_train": L, "sec": round(time.time() - t0)}; hist = []
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
    res = {"arm": f"mk{a.k}", "k": a.k, "seed": a.seed, "train_sec": train_sec, "eval_dtype": str(amp), "test_sequences": len(Xte),
           "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "segment_types": {"start_or_gold_text": seg_types[0], "latent": seg_types[1]}}
    pred, comp, val = rollout(m, Xte, a.k, dev, amp, a.eval_bs); np.save(os.path.join(a.out, "preds_test.npy"), pred); cum = s5.cumulative(pred, Yte)
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
    res["edit_iia"] = edit_iia(m, a.k, dev, amp, a.eval_bs, n=min(5000, a.check_n)); res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({kk: res[kk] for kk in ("cumulative", "donor")}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
