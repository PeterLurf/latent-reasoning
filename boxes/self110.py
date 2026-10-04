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


class Gen110(Gen103):
    def __init__(self, g):
        super().__init__(g, "gen"); self.mask = True; self.override = None

    def chunk_masked(self, start, ops, tgt):
        """Gen103.chunk_masked with §105's masked fork (the fork sees only the slot's own K/V, true positions)."""
        B, c = ops.shape; g = self.g
        x = torch.cat([g.embed(start), g.embed(self.optok[ops[:, 0]])], 1); past = None; tot = 0.0; n = 0
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
            if j + 1 < c:
                x = g.embed(self.optok[ops[:, j + 1]])
        return tot / max(1, n)


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
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    m = Gen110(s5.load_gpt2(dev)).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    lengths = [int(x) for x in a.lengths.split(",")]; ck = os.path.join(a.out, "ckpt.pt"); lp = os.path.join(a.out, "log.jsonl")
    S = {"step": 0, "short": 0, "rounds": [], "prev": 2, "remaining": a.steps - a.r0_steps - a.final_steps, "todo": list(lengths),
         "ins": 0, "phase": "R0", "sec": 0}
    t0 = time.time()
    if os.path.exists(ck):
        c = torch.load(ck, map_location=dev, weights_only=False); m.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        rng.bit_generator.state = c["rng"]; drng.bit_generator.state = c["drng"]; torch.set_rng_state(c["torch_rng"].cpu())
        if dev.startswith("cuda"):
            torch.cuda.set_rng_state(c["cuda_rng"].cpu(), dev)
        S = c["S"]; t0 -= S["sec"]
        keep = [r for r in open(lp, encoding="utf-8") if r.strip() and json.loads(r)["step"] <= S["step"]] if os.path.exists(lp) else []
        open(lp, "w", newline="\n").writelines(keep); print("RESUMED", S["step"], S["phase"], flush=True); del c
    log = open(lp, "a", newline="\n")

    def save_ck():
        S["sec"] = round(time.time() - t0)
        c = {"model": m.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state, "drng": drng.bit_generator.state,
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None, "S": S}
        torch.save(c, ck + ".tmp"); os.replace(ck + ".tmp", ck)

    def train_round(items, n_steps, name, short):
        hist = []
        for _ in range(n_steps):
            S["step"] += 1; S["short"] += int(short); step = S["step"]
            for gr in opt.param_groups:
                gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * min(step, a.steps) / a.steps))
            st, o_, tg = draw(items, drng, a.bs, dev)
            with amp_ctx(amp):
                loss = m.chunk_masked(st, o_, tg)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(loss.item())
            if step % 100 == 0:
                rec = {"step": step, "round": name, "loss": float(np.mean(hist)), "sec": round(time.time() - t0)}; hist = []
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)

    def pool_of(Lmax):
        rows, lens = make_pool(rng, Xtr, a.pool, 3, Lmax)
        return rows, lens, (rng.integers(0, 120, a.pool) if a.random_start else np.zeros(a.pool, np.int64))

    if S["phase"] == "R0":
        r0 = r0_items(rng, Xtr, Ytr)[0]; o1 = np.concatenate([r0[1][1][:, 0], r0[2][1][:, 0]])
        direct = {(0, int(x)) for x in o1}; two = direct | {(int(s5.COMP[0, x]), int(y)) for x, y in r0[2][1]}
        train_round(r0, a.r0_steps, "R0", True)
        S["rounds"].append({"length": "1-2", "steps": a.r0_steps, "targets": "final answer only", "start": "identity",
                            "short_transition_coverage": {"direct": len(direct) / 14400, "incl_2step": len(two) / 14400}})
        S["phase"] = "rounds"; save_ck()
    while S["phase"] == "rounds" and S["todo"]:
        Lmax = S["todo"][0]; rows, lens, starts = pool_of(Lmax)
        items, st, _ = label_round(m, Xtr, Ytr, rows, lens, starts, dev, amp); prev = S["prev"]; mid = (prev + Lmax) // 2
        why = [w for w, bad in (("gate %.4f < %.3f" % (st["gate"], GATE), st["gate"] < GATE),
                                ("kept share %.3f < 0.20" % st["kept_share"], st["kept_share"] < 0.20)) if bad]
        if why and S["ins"] < a.max_insert and prev < mid < Lmax:
            st.update({"length": Lmax, "action": "; ".join(why) + f": insert length {mid}"}); S["rounds"].append(st)
            S["todo"].insert(0, mid); S["ins"] += 1; print("insert length", mid, why, flush=True); continue
        n_steps = S["remaining"] // len(S["todo"]); S["remaining"] -= n_steps; S["todo"].pop(0)
        st.update({"length": Lmax, "steps": n_steps}); S["rounds"].append(st); S["ins"] = 0
        print("round", {k: v for k, v in st.items() if k != "op_hist"}, flush=True)
        train_round(items, n_steps, f"L{Lmax}", False); S["prev"] = Lmax; save_ck()
    if S["phase"] == "rounds":
        S["phase"] = "final"
    if S["phase"] == "final":
        Lmax = lengths[-1]; rows, lens, starts = pool_of(Lmax)
        items, st, pool = label_round(m, Xtr, Ytr, rows, lens, starts, dev, amp)
        np.savez_compressed(os.path.join(a.out, "pool_final.npz"), **pool)
        st.update({"length": Lmax, "steps": a.final_steps, "action": "final relabelling"}); S["rounds"].append(st)
        print("final", {k: v for k, v in st.items() if k != "op_hist"}, flush=True)
        train_round(items, a.final_steps, "final", False); S["phase"] = "done"
    train_sec = round(time.time() - t0)
    torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    if os.path.exists(ck):
        os.remove(ck)
    del opt; m.eval(); t1 = time.time()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    r = {"arm": "self110b" if a.random_start else "self110", "start_ge3": "random" if a.random_start else "identity", "seed": a.seed, "k": K, "fork": "mask", "train_sec": train_sec, "eval_dtype": str(amp), "test_sequences": len(Xte),
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
    r["edit_iia"] = edit_iia(m, K, dev, amp, a.eval_bs, n=min(5000, a.check_n)); r["eval_sec"] = round(time.time() - t1)
    json.dump(r, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps(r["cumulative"]), json.dumps(r["donor"]), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
