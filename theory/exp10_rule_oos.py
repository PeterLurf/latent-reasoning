"""Out-of-sample check of the two-factor rule (THEORY §14, frozen before any Pilot model was analysed).

Rule: extrapolates iff (retrieval share of the untrained last thought h_K < 0.5) AND (restoring local map: the local
map advances the pointer, output >= 0.6 of input at the operating point, log-log slope around s = 1 < 0.6).
Applied to every competent Pilot A-C run. Markov layers are reproduced with the rollout's block options; for a model
whose layer 2 is Markov the retrieval share is 0 by construction.
    python theory/exp10_rule_oos.py --out theory/results/exp10_rule_oos.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_arch import load_arch  # noqa: E402
from exp4_shares import shares  # noqa: E402
from exp7_dev import sweep_head  # noqa: E402
from rollout import probe, rollout  # noqa: E402

S = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


@torch.no_grad()
def transfer(m, K, g, br, E, head, markov_layers):
    T = lambda a: torch.as_tensor(a)
    src, dst, start = T(g.src), T(g.dst), T(g.start)
    n = len(g); bi = np.arange(n)[:, None]
    reach = np.concatenate([E["A"][:, 1:], E["B"]], 1); decoy = np.concatenate([E["D"][:, 1:], E["C"]], 1)
    mem = lambda s_: np.nan_to_num(s_[bi, head, reach] - s_[bi, head, decoy]).mean(0)
    # layers that are Markov never see earlier latents; otherwise block layer-2 retrieval at step K (local map only)
    if markov_layers:
        kw = {"block_from": 1, "block_layers": tuple(sorted(set(markov_layers) | {1}))}
    else:
        kw = {"block_from": K, "block_layers": (1,)}
    out = {}
    for s in S:
        def feed(k, h, hist, s=s):
            if k == K:
                c = h.mean(0, keepdim=True)
                return c + s * (h - c)
            return h
        o = rollout(m, K, src, dst, start, K + 1, feed=feed, **kw)
        out[s] = (mem(o["logit1"][K - 1].numpy()), mem(o["logit1"][K].numpy()))
    j_in = int(np.argmax(out[1.0][0])); j_out = int(np.argmax(out[1.0][1]))
    a_in = np.array([out[s][0][j_in] for s in S]); a_out = np.array([out[s][1][j_out] for s in S])
    i1, lo, hi = S.index(1.0), S.index(0.75), S.index(1.25)
    slope = float((np.log(max(a_out[hi], 1e-3)) - np.log(max(a_out[lo], 1e-3))) /
                  (np.log(max(a_in[hi], 1e-3)) - np.log(max(a_in[lo], 1e-3))))
    gain = float(a_out[i1] / max(a_in[i1], 1e-3))
    adv = j_out > j_in
    return {"j_in": j_in + 1, "j_out": j_out + 1, "gain": gain, "slope": slope, "a_in": a_in.tolist(), "a_out": a_out.tolist(),
            "restoring": bool(adv and slope < 0.6 and gain >= 0.6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp10_rule_oos.json")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, E = probe(256)
    for r in sorted(os.listdir("theory/runs")):
        if r[:3] not in ("pa_", "pb_", "pc_") or "stage1fail" in r or r in res:
            continue
        f = f"theory/runs/{r}/eval.json"
        if not os.path.exists(f):
            continue
        ev = json.load(open(f))
        if not ev["gate"]:
            continue
        m, c = load_arch(f"theory/runs/{r}", "cpu")
        K = c["K"]
        ml = tuple(getattr(m, "markov_layers", ()))
        if hasattr(m, "restore") and (m.restore or m.sigread):
            continue                     # restoring/sigmoid variants need their own rollout; reported separately
        with torch.no_grad():
            head, _ = sweep_head(m, K, g, br, E) if not ml else (None, None)
            if ml:                       # sweep head under the Markov rollout
                T = lambda a_: torch.as_tensor(a_)
                o = rollout(m, K, T(g.src), T(g.dst), T(g.start), K, block_from=1, block_layers=ml)
                n = len(g); bi = np.arange(n)[:, None]
                reach = np.concatenate([E["A"][:, 1:], E["B"]], 1); decoy = np.concatenate([E["D"][:, 1:], E["C"]], 1)
                acc = np.zeros(o["attn1"][0].shape[1])
                for k in range(K):
                    at = o["attn1"][k].numpy()
                    for h in range(len(acc)):
                        acc[h] += np.clip((at[bi, h, reach] - at[bi, h, decoy]).mean(0), 0, None).sum()
                head = int(np.argmax(acc))
            retr_K = 0.0 if 1 in ml else (lambda S_: S_[K - 1]["groups"].get("retr", 0) / max(abs(S_[K - 1]["amp"]), 1e-6))(
                shares(m, K, g, br, E, head, Kp=K + 1))
            tr = transfer(m, K, g, br, E, head, ml)
        pred = bool(retr_K < 0.5 and tr["restoring"])
        res[r] = {"extrap": ev["extrap_K12"], "d7": ev["d7_K12"], "far8_10": ev["far8_10_K12"], "retr_K": retr_K, **tr, "pred": pred}
        json.dump(res, open(a.out, "w"), indent=1)
        print("%-12s extrap %-5s pred %-5s | retr_K %.2f restoring %-5s gain %.2f slope %.2f d%d->d%d" % (
            r, ev["extrap_K12"], pred, retr_K, tr["restoring"], tr["gain"], tr["slope"], tr["j_in"], tr["j_out"]), flush=True)
    ys = [v["extrap"] for v in res.values()]; ps = [v["pred"] for v in res.values()]
    tp = sum(y and p for y, p in zip(ys, ps)); tn = sum((not y) and (not p) for y, p in zip(ys, ps))
    print("OUT-OF-SAMPLE: accuracy %d/%d; predicted-extrap %d of which true %d; predicted-fail %d of which true %d" % (
        tp + tn, len(ys), sum(ps), tp, len(ps) - sum(ps), tn))


if __name__ == "__main__":
    main()
