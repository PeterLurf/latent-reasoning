"""Note 199-M3 LAG-SPLIT (Theory, registered 539cfab0; reader theory/score_s199_m3.py; descriptive, $0, eval-only, fp32).
Items: test n in {6, 7, 8}, rng 18201 permutation[:500] of that pool. The normal think loop at K = 8 (InjectN.write from M_0 = 0) with the model's own
aux prompt (pass token as trained); at each pass t in {5, .., n} the greedy aux output (InjectN.aux_greedy, as s175 aux_acc_by_pass) is classified
against the item's targets e_2 .. e_{n+1} (aux_of(it)[k - 2] = e_k; pass t's target is e_{t+1}), in this order of precedence:
  correct  = e_{t+1}; lag1 = e_t; lag2plus = e_{t'+1} for some t' <= t - 2 (an earlier step's target); other.
A candidate c (its ids + EOT) matches when the greedy tokens' first len(c) entries == c (the s175 exact-match rule). Greedy length L = the longest
candidate of the batch (greedy is deterministic, so a longer L never changes a prefix).
Memory check first: one batch of the bs longest-input n = 8 items through the whole loop; peak recorded, abort above 7.0 GiB.
    python scratch/lag199.py <run dir> [device, default cuda:2]   -> <run>/eval_s199_lag.json"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402

K, T_LO, BS = 8, 5, 64


@torch.no_grad()
def classify(m, items, dev, by_t):
    x, msk = m.think_inputs(items, dev); S = m.mem.zero_state(len(items), dev)
    cands = [[N.ids(sy) + [N.EOT] for sy, _ in N.aux_of(it)] for it in items]       # cands[i][k] = e_{k+2}
    L = max(len(c) for cs in cands for c in cs)
    for t in range(1, K + 1):
        S = m.write(S, x, msk)
        idx = [i for i, it in enumerate(items) if T_LO <= t <= it["n"]]
        if not idx:
            continue
        gen = m.aux_greedy(S[idx], t, L)
        for r_, i in enumerate(idx):
            g_, cs = list(gen[r_]), cands[i]; hit = lambda c: g_[:len(c)] == c  # noqa: E731
            if hit(cs[t - 1]):
                k_ = "correct"
            elif hit(cs[t - 2]):
                k_ = "lag1"
            elif any(hit(cs[tp - 1]) for tp in range(1, t - 1)):
                k_ = "lag2plus"
            else:
                k_ = "other"
            d = by_t.setdefault(str(t), {"n": 0, "correct": 0, "lag1": 0, "lag2plus": 0, "other": 0}); d["n"] += 1; d[k_] += 1


def main():
    run = sys.argv[1].rstrip("/"); dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:2"; t0 = time.time()
    a = json.load(open(os.path.join(run, "args.json"))); assert a["arm"] == "inject"
    DM.FAST_ADD = bool(a.get("fast_add")); g = N.make_trunk(a, dev); ML, HA, HD = N.mem_geometry(a)
    m = N.InjectN(g, ML, "top", HA, HD, a.get("add_norm") or None).to(dev); m.nopass = bool(a.get("aux_nopass"))
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev, weights_only=False)); m.eval()
    TE = N.test_set(int(a.get("per_n") or 1000)); pool = [it for it in TE if it["n"] in (6, 7, 8)]
    its = [pool[i] for i in np.random.default_rng(18201).permutation(len(pool))[:500]]
    assert all(len(N.aux_of(it)) == it["n"] for it in its), "aux_of length != n"
    big = sorted([it for it in TE if it["n"] == 8], key=lambda it: -len(it["syms"]))[:BS]
    torch.cuda.reset_peak_memory_stats(dev); classify(m, big, dev, {}); peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    assert peak < 7.0, ("memory check", peak)
    by_t = {}
    for i in range(0, len(its), BS):
        classify(m, its[i:i + BS], dev, by_t)
    for d in by_t.values():
        assert d["n"] == d["correct"] + d["lag1"] + d["lag2plus"] + d["other"]
    by_t = dict(sorted(by_t.items(), key=lambda kv: int(kv[0])))
    out = {"by_t": by_t, "items": "test n in {6, 7, 8}, rng 18201 permutation[:500] of that pool (n=6: %d, n=7: %d, n=8: %d)" % tuple(sum(it["n"] == n_ for it in its) for n_ in (6, 7, 8)),
           "think_k": K, "t_range": [T_LO, "n"], "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "dtype": "float32 (no autocast)",
           "nopass": m.nopass, "memcheck_peak_gib": round(peak, 3), "sec": round(time.time() - t0),
           "def": "note 199-M3: greedy aux output at pass t vs the item's e_2..e_{n+1}; precedence correct (e_{t+1}) > lag1 (e_t) > lag2plus (e_{t'+1}, t' <= t-2) > other; "
                  "match = the first len(c) greedy tokens == c (ids + EOT), as s175",
           "sanity_eval_json_aux_acc_by_pass": {t: json.load(open(os.path.join(run, "eval.json")))["s175"]["aux_acc_by_pass"].get(t) for t in by_t}}
    json.dump(out, open(os.path.join(run, "eval_s199_lag.json"), "w", newline="\n"), indent=1)
    print("LAG", run, {t: (d["n"], round(d["correct"] / d["n"], 3), round(d["lag1"] / d["n"], 3), round(d["lag2plus"] / d["n"], 3)) for t, d in by_t.items()},
          "peak %.2f GiB, %d s" % (peak, out["sec"]), flush=True)


if __name__ == "__main__":
    main()
