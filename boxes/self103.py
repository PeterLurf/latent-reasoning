"""THEORY §103: filtered self-training by length, S5 from the identity, §97's GEN-4 recipe (text step target via the
fork; text snap every 4; history cut), FINAL ANSWERS ONLY (no ground-truth intermediate state is ever a training target).
Three seeds' models train in lockstep in one process (the cross-seed vote needs all three each round).
Rounds (24k updates total, bs 64 per model, AdamW lr 1e-4, §97's warmup + cosine over the whole 24k):
  R0 (updates 1-1,200, 5 %): lengths 1-2 from the identity; target = the final answer at the last slot only.
  R1-R4 (the remaining 22.8k split evenly; adaptively more rounds): lengths uniform in 3..L_max, L_max = 4, 8, 16, 32
    (lengths <= 2 occur only in R0, so the short share is exactly 1,200 / 24,000 = 0.05).
    Each round first pseudo-labels a fresh pool of 50k problems with every seed's CURRENT model (greedy all-position
    rollout, snap every 4, history cut). Filters, in order, per problem:
      validity: >= 2 seeds' rollouts have every chunk-end checkpoint parse as a state text;
      final:    of those, >= 2 decode the right final answer (the only use of truth in training: final answers);
      vote:     of those, >= 2 have IDENTICAL intermediate decodes; the majority decode is the target.
    Kept problems give per-slot targets (the majority decode) on every chunk, each chunk starting from the target text of
    the previous chunk end. Discarded problems give the final-answer target only, on their last chunk, starting from that
    seed's OWN decoded checkpoint (no target on earlier slots).
    Adaptive rule: if the kept share at a round's new L_max is < 20 %, an intermediate length (the midpoint between the
    previous and new L_max) is inserted before it and the remaining updates are re-split evenly.
Logged per round (truth for logging only): filtered_by {validity, final, vote} (shares of the pool removed at each
stage), kept_share, acc_new_length_before (mean over seeds, final answer), pseudo_label_precision (share of kept targets
equal to the true intermediate states; never used in training), self_consistency (descriptive: kept targets whose k = 1
restart decode agrees, on a 2k subsample).
End (per seed, s103_s<seed>/eval.json): gen97's keys (cumulative 128/256/512 from the identity on §71's test set,
extension_2048, chunk_check per-slot, removal.{zero,mean}, edit_iia, compute_512_per_problem) plus short_fraction and rounds.
    python -m boxes.self103 --out boxes/runs_s5/s103 [--device cuda:0]   (writes <out>/s103_s1..s3 + a summary <out>/eval.json)
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
from boxes.gen97 import (NST, STTOK, Gen97, amp_ctx, chunk_check, decode_to_el, edit_iia, mean_slot_inputs,  # noqa: E402
                         rollout)

K = 4
SEEDS = (1, 2, 3)


class Gen103(Gen97):
    def chunk_masked(self, start, ops, tgt):
        """Fork CE only where tgt >= 0 (tgt (B, c), -1 = no target): sum over the fork's tokens, mean over targeted
        (row, slot) pairs. start: (B, 6) state-text tokens."""
        B, c = ops.shape; g = self.g
        x = torch.cat([g.embed(start), g.embed(self.optok[ops[:, 0]])], 1); past = None; tot = 0.0; n = 0
        for j in range(c):
            h, past = g(x, past); s_in = h[:, -1]
            hs, past = g(s_in[:, None], past); hslot = hs[:, 0]
            sel = tgt[:, j] >= 0
            if bool(sel.any()):
                st = self.sttok[tgt[sel, j]]; pp = [(k_[sel], v_[sel]) for k_, v_ in past]
                hf, _ = g(g.embed(st[:, :-1]), pp)
                lg = g.logits(torch.cat([hslot[sel][:, None], hf], 1)).float()
                tot = tot + F.cross_entropy(lg.reshape(-1, lg.shape[-1]), st.reshape(-1), reduction="none").view(-1, NST).sum(-1).sum()
                n += int(sel.sum())
            if j + 1 < c:
                x = g.embed(self.optok[ops[:, j + 1]])
        return tot / max(1, n)


@torch.no_grad()
def rollout_var(m, X, rows, lens, dev, amp, bs=500):
    """Greedy rollouts of problems (row r, length L) from the identity. Returns, per problem: decoded element per
    position (list of arrays), chunk-end validity (all chunk-end decodes valid), the final element, and the decoded
    tokens of the checkpoint before the last chunk (identity's text if L <= 4)."""
    out_el, ok_ck, fin, before_last = [None] * len(rows), np.zeros(len(rows), bool), np.full(len(rows), -1), np.zeros((len(rows), NST), np.int64)
    for L in sorted(set(lens.tolist())):
        idx = np.where(lens == L)[0]
        for s in range(0, len(idx), bs):
            ii = idx[s:s + bs]; n = len(ii); ops = X[rows[ii], :L]
            start = torch.tensor(np.tile(STTOK[0], (n, 1)), device=dev); els = np.zeros((n, L), np.int64); valid = np.ones(n, bool)
            for c in range(0, L, K):
                if c + K >= L:
                    before_last[ii] = start.cpu().numpy()
                with amp_ctx(amp):
                    _, d, _ = m.chunk(start, torch.tensor(ops[:, c:c + K], device=dev), decode="all")
                toks = d.cpu().numpy(); e, _ = decode_to_el(toks.reshape(-1, NST)); e = e.reshape(n, -1)
                els[:, c:c + e.shape[1]] = e; valid &= e[:, -1] >= 0
                start = torch.tensor(toks[:, -1], device=dev)
            for t, i in enumerate(ii):
                out_el[i] = els[t]
            ok_ck[ii] = valid; fin[ii] = els[:, -1]
    return out_el, ok_ck, fin, before_last


def make_pool(rng, X, n, lo, hi):
    return rng.integers(0, len(X), n), rng.integers(lo, hi + 1, n)


def build_items(X, Y, rows, lens, decs, okck, fin, before, seeds_models, rng):
    """Filters (validity -> final -> vote) and training items. Returns per-seed item dicts {clen: (start, ops, tgt)} and
    the round's stats."""
    N = len(rows); true_fin = Y[rows, lens - 1]
    v_ok = np.stack([okck[s] for s in range(3)], 1)                        # (N, 3)
    f_ok = np.stack([fin[s] == true_fin for s in range(3)], 1) & v_ok
    pass_v = v_ok.sum(1) >= 2; pass_f = pass_v & (f_ok.sum(1) >= 2)
    keep = np.zeros(N, bool); target = [None] * N
    for i in np.where(pass_f)[0]:
        cand = [tuple(decs[s][i]) for s in range(3) if f_ok[i, s]]
        for c in set(cand):
            if cand.count(c) >= 2:
                keep[i] = True; target[i] = np.array(c); break
    true_states = [Y[rows[i], :lens[i]] for i in range(N)]
    prec = float(np.mean([np.array_equal(target[i], true_states[i]) for i in np.where(keep)[0]])) if keep.any() else None
    stats = {"pool": N, "filtered_by": {"validity": float((~pass_v).mean()), "final": float((pass_v & ~pass_f).mean()),
                                         "vote": float((pass_f & ~keep).mean())},
             "kept_share": float(keep.mean()), "pseudo_label_precision": prec,
             "acc_new_length_before": float(np.mean([(fin[s] == true_fin).mean() for s in range(3)]))}
    items = []
    for s in range(3):
        g = {}
        def add(cl, st, op, tg):
            g.setdefault(cl, ([], [], []))
            g[cl][0].append(st); g[cl][1].append(op); g[cl][2].append(tg)
        for i in range(N):
            r, L = rows[i], lens[i]; ops = X[r, :L]
            if keep[i]:
                tgt = target[i]
                for c in range(0, L, K):
                    st = STTOK[0] if c == 0 else STTOK[tgt[c - 1]]
                    add(min(K, L - c), st, ops[c:c + K], tgt[c:c + K])
            else:
                c = ((L - 1) // K) * K; cl = L - c; tg = np.full(cl, -1); tg[-1] = true_fin[i]
                add(cl, before[s][i], ops[c:], tg)
        items.append({cl: (np.array(a), np.array(b), np.array(t)) for cl, (a, b, t) in g.items()})
    return items, stats


def r0_items(rng, X, Y, n=50000):
    rows = rng.integers(0, len(X), n); lens = rng.integers(1, 3, n); g = {}
    for L in (1, 2):
        ii = np.where(lens == L)[0]; tg = np.full((len(ii), L), -1); tg[:, -1] = Y[rows[ii], L - 1]
        g[L] = (np.tile(STTOK[0], (len(ii), 1)), X[rows[ii], :L], tg)
    return [g, g, g]


def draw(items, rng, bs, dev):
    cls = sorted(items); w = np.array([len(items[c][0]) for c in cls], float); cl = cls[int(rng.choice(len(cls), p=w / w.sum()))]
    st, op, tg = items[cl]; ii = rng.integers(0, len(st), bs); T = lambda a: torch.tensor(a, device=dev)  # noqa: E731
    return T(st[ii]), T(op[ii]), T(tg[ii])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--r0_steps", type=int, default=1200)
    ap.add_argument("--pool", type=int, default=50000); ap.add_argument("--lengths", default="4,8,16,32")
    ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=s5.TEST_N); ap.add_argument("--ext_n", type=int, default=500)
    ap.add_argument("--ext_len", type=int, default=2048, help="< 2048: smoke only"); ap.add_argument("--check_n", type=int, default=100_000, help="< 100k: smoke only")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "bench", "data", "s5"))
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = a.device; torch.manual_seed(0); rng = np.random.default_rng(10300)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    s5.check_interface()
    Xtr, Ytr = s5.load_csv(os.path.join(a.data_dir, "S5=128.csv")); s5.check_labels(Xtr, Ytr)
    Xte, Yte = s5.load_csv(os.path.join(a.data_dir, "S5=512.csv"), limit=a.test_n); s5.check_labels(Xte, Yte)
    os.makedirs(a.out, exist_ok=True); outs = [os.path.join(a.out, f"s103_s{s}") for s in SEEDS]   # fetched as one dir; moved to boxes/runs_s5/s103_s<seed> locally
    for s, o in zip(SEEDS, outs):
        os.makedirs(o, exist_ok=True); json.dump({**vars(a), "seed": s, "lockstep_seeds": list(SEEDS)}, open(os.path.join(o, "args.json"), "w", newline="\n"), indent=1)
    models, opts, logs = [], [], []
    for s, o in zip(SEEDS, outs):
        torch.manual_seed(s); mm = Gen103(s5.load_gpt2(dev), "gen").to(dev); models.append(mm)
        opts.append(torch.optim.AdamW(mm.parameters(), lr=a.lr, weight_decay=0.01)); logs.append(open(os.path.join(o, "log.jsonl"), "w", newline="\n"))
    lengths = [int(x) for x in a.lengths.split(",")]; rounds = []; t0 = time.time(); step = 0; short_updates = 0
    seed_rngs = [np.random.default_rng(s) for s in SEEDS]

    def train_round(items, n_steps, name, short):
        nonlocal step, short_updates
        hist = [[] for _ in SEEDS]
        for _ in range(n_steps):
            step += 1; short_updates += int(short)
            lr = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * min(step, a.steps) / a.steps))
            for si, (mm, op) in enumerate(zip(models, opts)):
                for gr in op.param_groups:
                    gr["lr"] = lr
                st, o_, tg = draw(items[si], seed_rngs[si], a.bs, dev)
                with amp_ctx(amp):
                    loss = mm.chunk_masked(st, o_, tg)
                op.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mm.parameters(), 1.0); op.step(); hist[si].append(loss.item())
            if step % 100 == 0:
                for si in range(3):
                    logs[si].write(json.dumps({"step": step, "round": name, "loss": float(np.mean(hist[si])), "sec": round(time.time() - t0)}) + "\n"); logs[si].flush()
                print({"step": step, "round": name, "loss": [round(float(np.mean(h)), 4) for h in hist]}, flush=True); hist = [[] for _ in SEEDS]

    train_round(r0_items(rng, Xtr, Ytr), a.r0_steps, "R0", True)
    rounds.append({"length": "1-2", "steps": a.r0_steps, "targets": "final answer only"})
    prev, remaining, todo = 2, a.steps - a.r0_steps, list(lengths)
    while todo:
        Lmax = todo[0]; rows, lens = make_pool(rng, Xtr, a.pool, 3, Lmax)
        for mm in models:
            mm.eval()
        res = [rollout_var(mm, Xtr, rows, lens, dev, amp) for mm in models]
        for mm in models:
            mm.train()
        decs = [r[0] for r in res]; okck = [r[1] for r in res]; fin = [r[2] for r in res]; before = [r[3] for r in res]
        items, st = build_items(Xtr, Ytr, rows, lens, decs, okck, fin, before, models, rng)
        mid = (prev + Lmax) // 2
        if st["kept_share"] < 0.20 and mid > prev and mid < Lmax and len(todo) < 8:
            st.update({"length": Lmax, "action": f"kept share {st['kept_share']:.3f} < 0.20: insert length {mid}"}); rounds.append(st)
            todo.insert(0, mid); print("insert length", mid, flush=True); continue
        n_steps = remaining // len(todo); remaining -= n_steps; todo.pop(0)
        st.update({"length": Lmax, "steps": n_steps}); rounds.append(st); print("round", st, flush=True)
        train_round(items, n_steps, f"L{Lmax}", False); prev = Lmax
    train_sec = round(time.time() - t0)
    for si, (s, o, mm) in enumerate(zip(SEEDS, outs, models)):
        torch.save(mm.state_dict(), os.path.join(o, "model.pt"))
    del opts
    for si, (s, o, mm) in enumerate(zip(SEEDS, outs, models)):
        mm.eval(); torch.cuda.empty_cache() if dev.startswith("cuda") else None; t1 = time.time()
        r = {"arm": "self103", "seed": s, "k": K, "train_sec": train_sec, "eval_dtype": str(amp), "test_sequences": len(Xte),
             "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
             "short_fraction": short_updates / max(1, step), "updates": step, "rounds": rounds}
        pred, comp, val = rollout(mm, Xte, K, dev, amp, a.eval_bs); cum = s5.cumulative(pred, Yte)
        np.save(os.path.join(o, "preds_test.npy"), pred)
        r["cumulative"] = {str(t): float(cum[t - 1]) for t in (128, 256, 512) if t <= Xte.shape[1]}
        r["final_state_acc_512"] = float((pred[:, -1] == Yte[:, -1]).mean()); r["compute_512_per_problem"] = comp; r["checkpoint_validity"] = val
        if a.ext_n:
            Xx, Yx = s5.load_csv(os.path.join(a.data_dir, "S5=2048.csv"), limit=a.ext_n); s5.check_labels(Xx, Yx); Xx, Yx = Xx[:, :a.ext_len], Yx[:, :a.ext_len]
            px, _, _ = rollout(mm, Xx, K, dev, amp, a.eval_bs); cx = s5.cumulative(px, Yx)
            r["extension_2048"] = {"n": len(Xx), "len": Xx.shape[1], "cumulative": {str(t): float(cx[t - 1]) for t in (512, 1024, 2048) if t <= Xx.shape[1]}}
        r["chunk_check"] = chunk_check(mm, K, dev, amp, a.eval_bs, n=a.check_n)
        mu = mean_slot_inputs(mm, Xtr, Ytr, K, dev, amp)
        r["removal"] = {ab: chunk_check(mm, K, dev, amp, a.eval_bs, n=min(10_000, a.check_n), seed=7106, ablate=ab, mean_in=mu) for ab in ("zero", "mean")}
        r["edit_iia"] = edit_iia(mm, K, dev, amp, a.eval_bs, n=min(5000, a.check_n)); r["eval_sec"] = round(time.time() - t1)
        json.dump(r, open(os.path.join(o, "eval.json"), "w", newline="\n"), indent=1)
        print("EVAL", s, json.dumps(r["cumulative"]), flush=True)
    json.dump({"seeds": list(SEEDS), "dirs": outs, "rounds": rounds}, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)   # summary for the queue / drain
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
