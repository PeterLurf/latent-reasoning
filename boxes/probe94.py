"""THEORY §94 Step 0 (inference only): is the letter -> box state AVAILABLE linearly in §89's FADE latents? (note 94-1:
a linear letter -> box probe is the transpose of box -> letter, so it shows availability, not direction.)
For each model: 4,000 held-out training-distribution problems (N = 8, both question types, generator seed 9400 + seed;
disjoint from §66's training seeds and test seed 6600), run in pure latent (alpha = 1) with the thought inputs collected
at all 40 thought positions. Probes (linear, on per-coordinate standardised latents; Adam lr 1e-2, 300 full-batch
steps; torch seed 0; train first 75 % / test last 25 %), fitted and tested separately on "where"-question latents,
on "all"-question latents (the question precedes the thoughts), and pooled:
  (i)  letter -> box: for each of the 4 letters, its box index (4-way) in the target arrangement;
  (ii) box -> letter: for each box, its letter (4-way).
Targets: thought position j (1-based) uses the arrangement after op ceil(j / 5); position 40 = final arrangement.
Reported: (mean slot accuracy, all-4-correct) at the last position of every op block (5, 10, ..., 40).
Controls (note 94-1):
  --control pretrained  GPT-2 before fine-tuning (no model.pt), thought inputs built at alpha = 0 (= embeddings of the
                        model's own argmax tokens); same problems and positions.
  --control nc          §66's NC model (boxes/runs/nc_s<seed>, prompt "... q: ... a:", no thoughts): final hidden states
                        at the last token of op i in the prompt (stored as position 5 i, target = state after op i) and
                        at the final " a:" token (stored as "final_a", target = final arrangement).
    python -m boxes.probe94 --run boxes/runs/f89_fade_std_dec_s1 --out theory/results/s94_step0/std_dec_s1.json
    python -m boxes.probe94 --run boxes/runs/f89_fade_std_dec_s1 --control pretrained --out .../pretrained_s1.json
    python -m boxes.probe94 --run boxes/runs/nc_s1 --control nc --seed 1 --out .../nc_s1.json
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes.core import LET, apply, make_problem, prompt, s_arr, s_ops, tok  # noqa: E402
from boxes.fade89 import Fade  # noqa: E402
from boxes.train import load_model  # noqa: E402


def probe(X, Y, steps=300):
    """X (n, D), Y (n, 4) 4-way targets per slot -> (mean slot accuracy, all-4 accuracy) on the last 25 %."""
    torch.manual_seed(0); n_train = int(0.75 * len(X))
    mu, sd = X[:n_train].mean(0), X[:n_train].std(0) + 1e-5; Z = (X - mu) / sd
    lin = nn.Linear(Z.shape[1], 16); op = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(steps):
        lo = F.cross_entropy(lin(Z[:n_train]).view(-1, 4), Y[:n_train].reshape(-1)); op.zero_grad(); lo.backward(); op.step()
    with torch.no_grad():
        p = lin(Z[n_train:]).view(-1, 4, 4).argmax(-1); c = p == Y[n_train:]
    return float(c.float().mean()), float(c.all(-1).float().mean())


def targets(P, n_ops):
    arrs = [apply(p["start"], p["ops"][:n_ops]) for p in P]
    let2box = torch.tensor([[arr.index(c) for c in LET] for arr in arrs])       # (n, 4): box of each letter
    box2let = torch.tensor([[LET.index(c) for c in arr] for arr in arrs])       # (n, 4): letter in each box
    return let2box, box2let


def fade_latents(a, args, probs):
    f = Fade(load_model(os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), a.device), args["arm"])
    alpha = 1.0
    if a.control == "pretrained":
        alpha = 0.0                                                            # no fine-tuning: inputs = E[argmax]
    else:
        f.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location=a.device))
    f.to(a.device).eval(); P, L = [], []
    with torch.no_grad():
        for i in range(0, len(probs), 50):
            b = probs[i:i + 50]
            for qt in ("all", "where"):                                        # batches need one question type
                bb = [p for p in b if p["qtype"] == qt]
                if bb:
                    coll = []; f(bb, a.device, alpha, stage=None, train=False, collect=coll)
                    P += bb; L.append(torch.stack(coll, 1).float())            # (B, 40, D)
    L = torch.cat(L)
    return P, {str(pos): L[:, pos - 1] for pos in range(5, 41, 5)}, {str(pos): (pos + 4) // 5 for pos in range(5, 41, 5)}


def nc_states(a, probs):
    m = load_model(os.path.join(a.run, "model.pt"), a.device).eval(); n_ops = len(probs[0]["ops"])
    H = {str(5 * (i + 1)): [] for i in range(n_ops)}; H["final_a"] = []
    with torch.no_grad():
        for p in probs:
            ids = tok(prompt(p)); h, _ = m(m.embed(torch.tensor([ids], device=a.device)))
            for i in range(n_ops):
                pre = tok(f"start: {s_arr(p['start'])}. ops: {s_ops(p['ops'][:i + 1])}")
                assert ids[:len(pre)] == pre
                H[str(5 * (i + 1))].append(h[0, len(pre) - 1].float())
            H["final_a"].append(h[0, -1].float())
    ops = {k: (int(k) // 5 if k != "final_a" else n_ops) for k in H}
    return probs, {k: torch.stack(v) for k, v in H.items()}, ops


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--control", choices=["none", "pretrained", "nc"], default="none")
    ap.add_argument("--seed", type=int, default=None, help="generator seed offset (default: the run's args.json seed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    args = json.load(open(os.path.join(a.run, "args.json"))) if a.control != "nc" else {"arm": "nc"}
    seed = a.seed if a.seed is not None else args["seed"]
    rng = random.Random(9400 + seed); probs = [make_problem(rng, 8, rng.choice(["all", "where"])) for _ in range(a.n)]
    P, X, ops = nc_states(a, probs) if a.control == "nc" else fade_latents(a, args, probs)
    qt = [p["qtype"] for p in P]
    res = {"run": a.run, "arm": args["arm"], "control": a.control, "seed": seed, "n": len(P),
           "n_by_qtype": {q: qt.count(q) for q in ("where", "all")}, "by_qtype": {}}
    for q in ("where", "all", "pooled"):
        idx = [i for i, t in enumerate(qt) if q == "pooled" or t == q]; out = {}
        for pos, x in X.items():
            l2b, b2l = targets([P[i] for i in idx], ops[pos]); xi = x[idx]
            out[pos] = {"letter_to_box": probe(xi, l2b), "box_to_letter": probe(xi, b2l)}
        res["by_qtype"][q] = out
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w", newline="\n"), indent=1)
    last = "final_a" if a.control == "nc" else "40"
    print(a.run, a.control, {q: (res["by_qtype"][q][last]["letter_to_box"], res["by_qtype"][q][last]["box_to_letter"])
                             for q in res["by_qtype"]}, flush=True)


if __name__ == "__main__":
    main()
