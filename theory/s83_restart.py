"""THEORY §83: uncertainty-triggered restarts for §71's written-state solver (S5, DeltaProduct's protocol). Inference only.

The all-position WC0 solver (trained on chunks of k' <= k ops; boxes/s5.py) writes the state after every op of a chunk.
Instead of a hand-fixed chunk size, the adaptive policy offers up to Kmax ops and accepts the written states while the
model's own confidence in each state stays >= tau (confidence = product of the 5 letter probabilities, restricted to the
letters, as in greedy decoding); at the first low-confidence state it restarts (history cut) from the last accepted
state (at least one state is always accepted). tau is calibrated in range only: the 1st percentile of the confidences of
correct states on fresh in-range chunks (random start, k' ~ U{1..k}, uniform ops; seed 8300), so in-range chunks trigger
restarts on <= 1 % of states.
    python theory/s83_restart.py --run boxes/runs_s5/wc0k4_s1 --k 4 --kmax 16 --test_n 2000 --out f.json [--device cuda:0]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
from boxes import s5  # noqa: E402  (must precede boxes.core: sets five boxes)
from boxes import core as bc  # noqa: E402


@torch.no_grad()
def chunk_with_conf(m, starts, ops, dev, amp):
    """starts (B,), ops (B, K) -> predicted element ids (B, K) (-1 invalid) and confidences (B, K)."""
    B, K = ops.shape
    pre = torch.tensor([bc.tok(s5.chunk_prompt(s5.ARR[s], o)) for s, o in zip(starts, ops)], device=dev)
    allowed = torch.tensor(bc.LET_IDS, device=dev)
    preds, confs = [], []
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        h, past = bc.run_prefix(m, pre)
        for _ in range(K):
            h, past = m(m.embed(torch.full((B,), s5.GT, device=dev))[:, None, :], past); hl = h[:, -1]
            ids, lc = [], torch.zeros(B, device=dev)
            for _ in range(5):
                lp = F.log_softmax(m.logits(hl).float()[:, allowed], -1); j = lp.argmax(-1)
                lc = lc + lp.gather(1, j[:, None]).squeeze(1); nxt = allowed[j]; ids.append(nxt)
                h, past = m(m.embed(nxt)[:, None, :], past); hl = h[:, -1]
            ids = torch.stack(ids, 1)
            preds.append([s5.ARR_INDEX.get(bc.dec_arr(r), -1) for r in ids.tolist()]); confs.append(lc.exp().float().cpu().numpy())
    return np.array(preds).T, np.array(confs).T


def s5_apply(s, o):                                   # element id after applying op o to element s (s5's convention)
    return s5.ARR_INDEX[s5.apply_perm(s5.ARR[s], o)]


def calibrate(m, k, dev, amp, n=20000, bs=500, seed=8300):
    bs = min(bs, n)
    rng = np.random.default_rng(seed); conf_ok, conf_bad = [], []
    for b in range(0, n, bs):
        kk = int(rng.integers(1, k + 1)); st = rng.integers(0, 120, bs); ops = rng.integers(0, 120, (bs, kk))
        p, c = chunk_with_conf(m, st, ops, dev, amp)
        for i in range(bs):
            s = int(st[i])
            for j in range(kk):
                s = s5_apply(s, int(ops[i, j])); (conf_ok if p[i, j] == s else conf_bad).append(float(c[i, j]))
    c = np.sort(np.array(conf_ok)); n = len(c)
    def tau_for(alpha):                       # largest tau whose 95 % upper bound on P(conf < tau | correct) <= alpha
        best = float(c[0])
        for i in range(1, n):
            p = i / n
            if p + 1.645 * np.sqrt(p * (1 - p) / n) > alpha:
                break
            best = float(c[i])
        return best
    taus = {str(al): tau_for(al) for al in (0.001, 0.01, 0.05)}
    return taus["0.01"], {"n_states": n + len(conf_bad), "in_range_acc": n / (n + len(conf_bad)), "taus": taus,
                          "conf_ok_p1_p50": [float(np.percentile(c, 1)), float(np.percentile(c, 50))]}


def adaptive_rollout(m, X, Y, kmax, tau, dev, amp, bs):
    N, L = X.shape; pred = np.full((N, L), -1, dtype=np.int64); acc_lens = []; gen_states = 0
    det = {"wrong_flagged": 0, "wrong": 0, "right_flagged": 0, "right": 0}
    for s0 in range(0, N, bs):
        rows = np.arange(s0, min(N, s0 + bs)); cur = np.zeros(len(rows), dtype=np.int64); ptr = np.zeros(len(rows), dtype=np.int64)
        while True:
            act = np.where(ptr < L)[0]
            if not len(act):
                break
            K = np.minimum(kmax, L - ptr[act])
            for k in np.unique(K):
                g = act[K == k]
                ops = np.stack([X[rows[i], ptr[i]:ptr[i] + k] for i in g])
                p, c = chunk_with_conf(m, np.where(cur[g] < 0, 0, cur[g]), ops, dev, amp); gen_states += p.size
                for gi, i in enumerate(g):
                    r, q = rows[i], ptr[i]
                    start_ok = q == 0 or pred[r, q - 1] == Y[r, q - 1]
                    if start_ok:                                       # detection stats on chunks with a correct start
                        for j in range(k):
                            right = p[gi, j] == Y[r, q + j]; flagged = c[gi, j] < tau
                            det["right" if right else "wrong"] += 1
                            det["right_flagged" if right else "wrong_flagged"] += int(flagged)
                            if not right:
                                break                                  # later states inherit the error
                    ok = c[gi] >= tau
                    n = k if ok.all() else max(1, int(np.argmin(ok)))
                    pred[r, q:q + n] = p[gi, :n]; cur[i] = p[gi, n - 1]; ptr[i] += n; acc_lens.append(int(n))
    return pred, acc_lens, gen_states, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--k", type=int, default=4); ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--test_n", type=int, default=2000); ap.add_argument("--ext_n", type=int, default=500)
    ap.add_argument("--fixed", type=int, nargs="*", default=[4, 8, 16]); ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "bench", "data", "s5")); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True); ap.add_argument("--cal_n", type=int, default=20000); ap.add_argument("--sweep_n", type=int, default=500)
    a = ap.parse_args(); dev = a.device
    ev = json.load(open(os.path.join(a.run, "eval.json"))) if os.path.exists(os.path.join(a.run, "eval.json")) else {}
    amp = torch.bfloat16 if "bfloat16" in str(ev.get("eval_dtype", "")) else None
    m = s5.load_gpt2(dev); m.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location=dev)); m.eval()
    t0 = time.time(); tau, cal = calibrate(m, a.k, dev, amp, n=a.cal_n); res = {"tau": tau, "calibration": cal, "amp": str(amp), "run": a.run}
    Xte, Yte = s5.load_csv(os.path.join(a.data_dir, "S5=512.csv"), limit=a.test_n)
    pred, lens, gen, det = adaptive_rollout(m, Xte, Yte, a.kmax, tau, dev, amp, a.bs); cum = s5.cumulative(pred, Yte)
    res["adaptive"] = {"cum": {t: float(cum[t - 1]) for t in (128, 256, 512)}, "final_state": float((pred[:, -1] == Yte[:, -1]).mean()),
                       "mean_accepted": float(np.mean(lens)), "accepted_hist": np.bincount(lens, minlength=a.kmax + 1).tolist(),
                       "generated_states_per_problem": gen / len(Xte), "detection": det}
    for k in a.fixed:
        p, _ = s5.wc_rollout(m, Xte, k, dev, amp, a.bs); c = s5.cumulative(p, Yte)
        res["fixed_k%d" % k] = {"cum": {t: float(c[t - 1]) for t in (128, 256, 512)}}
    res["risk_coverage"] = {}                 # tau sweep on the first sweep_n test sequences (descriptive)
    for al, tv in cal["taus"].items():
        ps, ls, _, ds = adaptive_rollout(m, Xte[:a.sweep_n], Yte[:a.sweep_n], a.kmax, tv, dev, amp, a.bs)
        res["risk_coverage"][al] = {"tau": tv, "cum512": float(s5.cumulative(ps, Yte[:a.sweep_n])[-1]), "mean_accepted": float(np.mean(ls)), "detection": ds}
    Xx, Yx = s5.load_csv(os.path.join(a.data_dir, "S5=2048.csv"), limit=a.ext_n)
    px, lx, _, _ = adaptive_rollout(m, Xx, Yx, a.kmax, tau, dev, amp, a.bs); cx = s5.cumulative(px, Yx)
    res["adaptive_2048"] = {"cum": {t: float(cx[t - 1]) for t in (512, 1024, 2048)}, "mean_accepted": float(np.mean(lx))}
    res["sec"] = round(time.time() - t0, 1)
    d = res["detection"] = det
    print(json.dumps({"tau": tau, "adaptive": res["adaptive"]["cum"], "mean_acc": res["adaptive"]["mean_accepted"],
                      "fixed": {k: res["fixed_k%d" % k]["cum"][512] for k in a.fixed}, "det": d}), flush=True)
    json.dump(res, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
