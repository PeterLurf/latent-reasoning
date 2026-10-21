"""Note 216-10 KL-DRIFT ($0, eval-only, descriptive side row on the closed §215 / §216 models): per model r and its lineage base b, on every §170 test item
(N.test_set(1000)) at n 1..16, the answer distribution over the 11 value classes (log_softmax over NUM_IDS of answer_logits) at K' = n + 2 and K' = 24,
from ONE K' 24 rollout per chunk (vbank_diag.part_hist24's method: coda_rows {n + 2, 24}; chunks of the per-n test-order pool, vbank_eval.chunks).
The note reads n 3..6 and 9..16; n 1..2 are a descriptive a_24(n) table; n 7..8 are computed for the GATE only. Local RTX 4060 (cuda:0), fp32, tf32 off,
eval_bs 32. GATE: the intact acc_n2 and acc_k['24'] at n 3..8 == each model's (and base's) committed eval_vb within +-0.005, else UNREADABLE on this card.
Per item (store-items rule): lp_r / lp_b at both K' (theory/score_s216_kl.py's layout). n runs 16 -> 1 (largest shape first). -> <run>/eval_kl_base.json.
Smokes: --per_n 2 --pairs 1 --out_root scratch/_tmp/s216_kl (writes <out_root>/<run>/eval_kl_base.json, never a run dir).
    python -B scratch/s216_kl_base.py --device cuda:0"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402

R = "boxes/runs_fsq/"
PAIRS = [("s215_ftctrl_s1", "s213_d3t_s1"), ("s215_ftself_s1", "s213_d3t_s1"), ("s216_r2_s1", "s213_d3t_s1"), ("s216_r1_s2", "s216_d3t_s2")]
NS = list(range(16, 0, -1))


@torch.no_grad()
def dists(run, TE, dev, bs):
    """-> {n: {"k2": (items, 11) float64 log-probs at K' = n + 2, "k24": ... at K' = 24}} for every n in NS (test order within n)."""
    m, args = VE.load(run, dev); num = torch.tensor(N.NUM_IDS, device=dev); Nn = np.array([it["n"] for it in TE]); out = {}; t0 = time.time()
    for n in NS:
        pool = np.nonzero(Nn == n)[0]; L2, L24 = [], []
        for ch in VE.chunks(pool, bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); rows = torch.arange(len(its), device=dev)
            r_ = m.rollout(x, pad, 24, coda_rows={n + 2: rows, 24: rows})
            L2.append(torch.log_softmax(m.answer_logits(r_["coda"][n + 2][1], lens)[:, num].double(), -1).cpu())
            L24.append(torch.log_softmax(m.answer_logits(r_["coda"][24][1], lens)[:, num].double(), -1).cpu()); del r_
        out[n] = {"k2": torch.cat(L2).numpy(), "k24": torch.cat(L24).numpy()}
        print("KL", run, "n", n, round(time.time() - t0), "s", flush=True)
    del m
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return out, {"mode": args.get("mode"), "sec": round(time.time() - t0)}


def acc_tables(D, Y):
    return {"acc_n2": {str(n): float((D[n]["k2"].argmax(1) == Y[n]).mean()) for n in sorted(D)}, "acc_k24": {str(n): float((D[n]["k24"].argmax(1) == Y[n]).mean()) for n in sorted(D)}}


def gate(run, acc):
    ev = json.load(open(R + run + "/eval_vb.json")); g = {"ref_card": ev.get("card"), "ref_eval_bs": ev.get("eval_bs"), "by_n": {}}
    for n in range(3, 9):
        a2, a24 = acc["acc_n2"][str(n)], acc["acc_k24"][str(n)]; r2, r24 = ev["acc_n2"][str(n)], ev["acc_k"]["24"][str(n)]
        g["by_n"][str(n)] = {"acc_n2": [a2, r2], "acc_k24": [a24, r24], "ok": bool(abs(a2 - r2) <= 0.005 and abs(a24 - r24) <= 0.005)}
    g["PASS"] = all(v["ok"] for v in g["by_n"].values()); return g


def kl(pb, pr):                                                             # KL(p_b || p_r) per item, from log-probs
    return (np.exp(pb) * (pb - pr)).sum(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--per_n", type=int, default=1000); ap.add_argument("--pairs", type=int, default=len(PAIRS)); ap.add_argument("--out_root", default=""); a = ap.parse_args()
    dev = a.device; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    TE = N.test_set(a.per_n); Nn = np.array([it["n"] for it in TE]); Yall = np.array([it["t"] for it in TE])
    Y = {n: Yall[Nn == n] for n in NS}; idx = {n: np.nonzero(Nn == n)[0] for n in NS}
    card = torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"; commit = os.popen("git rev-parse --short HEAD").read().strip(); base_cache = {}
    for run, base in PAIRS[:a.pairs]:
        if base not in base_cache:
            Db, ib = dists(R + base, TE, dev, a.eval_bs); ab = acc_tables(Db, Y); base_cache[base] = (Db, ib, ab, gate(base, ab))
            print("GATE base", base, base_cache[base][3]["PASS"], flush=True)
        Db, ib, ab, gb = base_cache[base]
        Dr, ir = dists(R + run, TE, dev, a.eval_bs); ar = acc_tables(Dr, Y); gr = gate(run, ar); print("GATE", run, gr["PASS"], flush=True)
        rnd = lambda A: [round(float(v), 8) for v in A]  # noqa: E731
        items, klm, agm = [], {}, {}
        for n in sorted(NS):                                                # items in test order (n ascending, test order within n)
            k2, k24 = kl(Db[n]["k2"], Dr[n]["k2"]), kl(Db[n]["k24"], Dr[n]["k24"])
            klm[str(n)] = {"n2": float(k2.mean()), "24": float(k24.mean())}
            agm[str(n)] = {"n2": float((Dr[n]["k2"].argmax(1) == Db[n]["k2"].argmax(1)).mean()), "24": float((Dr[n]["k24"].argmax(1) == Db[n]["k24"].argmax(1)).mean())}
            for q, j in enumerate(idx[n]):
                items.append({"n": n, "idx": int(j), "y": int(Y[n][q]), "lp_r": {"n2": rnd(Dr[n]["k2"][q]), "24": rnd(Dr[n]["k24"][q])},
                              "lp_b": {"n2": rnd(Db[n]["k2"][q]), "24": rnd(Db[n]["k24"][q])}})
        ish = hashlib.sha256(json.dumps([[TE[it["idx"]]["syms"], int(TE[it["idx"]]["t"])] for it in items], separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
        res = {"meta": {"card": card, "dtype": "fp32", "eval_bs": a.eval_bs, "base_run": base, "run": run, "items_sha256": ish,
                        "items_sha256_def": "sha256 of json.dumps([[syms, int(t)] ...], separators (',', ':'), ensure_ascii) over the written items in order",
                        "tf32": False, "per_n": a.per_n, "test": "N.test_set(%d), n 1..16" % a.per_n, "commit": commit, "note": "Note 216-10 KL-DRIFT"},
               "gate": {"acc_n2": {str(n): ar["acc_n2"][str(n)] for n in range(3, 9)}, "acc_k24": {str(n): ar["acc_k24"][str(n)] for n in range(3, 9)}},
               "gate_check": {"model": gr, "base": gb, "PASS": bool(gr["PASS"] and gb["PASS"]),
                              "rule": "intact acc_n2 and acc_k['24'] at n 3..8 == the committed eval_vb within +-0.005 (model and base); else UNREADABLE on this card"},
               "acc": {"model": ar, "base": ab}, "kl_mean_by_n": klm, "agree_by_n": agm, "items": items,
               "def": __doc__ + " Layout = theory/score_s216_kl.py's contract (0a774e9b): items [{n, idx, y, lp_r {n2, 24}, lp_b {n2, 24}}], log-probs over the 11 "
                                "value classes in NUM_IDS order (log_softmax restricted to NUM_IDS, float64, rounded to 1e-8). kl_mean_by_n = KL(p_base || p_model) "
                                "per n (natural log), agree_by_n = argmax agreement; descriptive conveniences, the scorer recomputes."}
        res["sec"] = {"model": ir["sec"], "base": ib["sec"]}
        od = os.path.join(a.out_root, run) if a.out_root else R + run; os.makedirs(od, exist_ok=True); fo = od + "/eval_kl_base.json"; json.dump(res, open(fo + ".tmp", "w", newline="\n")); os.replace(fo + ".tmp", fo); print("DONE", fo, "gate", res["gate_check"]["PASS"], flush=True)


if __name__ == "__main__":
    main()
