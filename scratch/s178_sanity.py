"""THEORY §178 sanity (Theory / Audit 2026-10-09 ~20:30, after DIRECT-scratch / K0-scratch stayed at chance for 96k): is the scratch path a BUG or the
RECIPE? CPU, nested170's own pieces (make_trunk --scratch 6,384,6, the DIRECT loss of loss_items, make_opt AdamW wd 0.01, lr 5e-4 cosine, warm-up):
  fixed : 64 FIXED mixed-n items (n ~ U{1..8}), trained repeatedly -> loss must head toward 0 (else a bug)
  n1    : online n = 1 only, the recipe's test exclusion kept (~58 distinct trainable n = 1 items; the n = 1 test items are disjoint) -> train / test acc
  n1all : online n = 1 only, no exclusion (all ~470 items trainable) -> train acc (memorisation of the table)
  mix2  : note 178-3 (s178b) gate 1: the full n mix U{1..8}, test items excluded only for n >= 2 (+ the SERIAL pool) -> n = 1 test acc and the loss
          (needs n = 1 test acc >= 0.3 and loss < ln 11 - 0.2 at 4k)
    python scratch/s178_sanity.py <mode> <steps> [warmup]   -> boxes/runs_fsq/s178_sanity/<mode>.json"""
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.nested170 as N  # noqa: E402
from boxes.fsq120 import make_opt, lr_at, EOT  # noqa: E402


def main():
    mode, steps = sys.argv[1], int(sys.argv[2]); warm = int(sys.argv[3]) if len(sys.argv) > 3 else 100; torch.set_num_threads(3); torch.manual_seed(1)
    a = {"scratch": "6,384,6", "seed": 1}; g = N.make_trunk(a, "cpu"); opt = make_opt(g, 5e-4); rng = np.random.default_rng(1)
    TE = N.test_set(1000); EX = {tuple(it["syms"]) for it in TE}; te1 = [it for it in TE if it["n"] == 1]

    def logits(items):
        seqs = [[EOT] + N.ids(it["syms"]) + [N.TID[" ="]] for it in items]; ids_, _ = N.pad(seqs); x = torch.tensor(ids_)
        h, _ = g(g.embed(x), None); pos = torch.tensor([len(q) - 1 for q in seqs]); return g.logits(h[torch.arange(len(seqs)), pos]).float()

    def acc(items):
        with torch.no_grad():
            pr = logits(items)[:, torch.tensor(N.NUM_IDS)].argmax(-1).numpy()
        return float((pr == np.array([it["t"] for it in items])).mean())

    def n1(k, excl):
        out = []
        while len(out) < k:
            r, t = N.generate(rng, 1); s = N.render(r)
            if tuple(s) not in excl:
                out.append({"syms": s, "t": t, "root": r, "n": 1})
        return out
    fixed = N.train_items(np.random.default_rng(7), 64, 8, set()) if mode == "fixed" else None
    if mode == "mix2":                                                      # nested170's exclusion under --excl_min_n 2 (test n >= 2 + the n = 8 SERIAL pool)
        r_sp = np.random.default_rng(17300); EX2 = {tuple(it["syms"]) for it in TE if it["n"] >= 2}
        for _ in range(2200):                                               # nested170 --serial_pool default
            r__, _t = N.generate(r_sp, 8); EX2.add(tuple(N.render(r__)))
    pool1 = n1(4000, EX) if mode == "n1" else None
    log = []; t0 = time.time()
    for step in range(1, steps + 1):
        its = fixed if mode == "fixed" else (N.train_items(rng, 64, 8, EX2) if mode == "mix2" else n1(64, EX if mode == "n1" else set()))
        for gr in opt.param_groups:
            gr["lr"] = lr_at(5e-4, gr, step, steps) / min(1, step / 100) * min(1, step / warm)
        opt.zero_grad(); lg = logits(its); loss = Fn.cross_entropy(lg, torch.tensor(N.NUM_IDS[[it["t"] for it in its]])); loss.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == steps:
            r = {"step": step, "loss": float(loss), "sec": round(time.time() - t0)}
            if mode == "fixed":
                r["train_acc"] = acc(fixed)
            elif mode == "mix2":
                r["test_acc_n1"] = acc(te1); r["test_acc_n2"] = acc([it for it in TE if it["n"] == 2][:500])
            else:
                r["train_acc_pool"] = acc(pool1[:500] if mode == "n1" else n1(500, set())); r["test_acc_n1"] = acc(te1)
            log.append(r); print(r, flush=True)
    od = os.path.join(ROOT, "boxes", "runs_fsq", "s178_sanity"); os.makedirs(od, exist_ok=True)
    n1d = len({tuple(it["syms"]) for it in pool1}) if pool1 else None
    json.dump({"mode": mode, "steps": steps, "warmup": warm, "lr": 5e-4, "wd": 0.01, "bs": 64, "ln11": math.log(11), "distinct_n1_trainable": n1d,
               "n1_test_items": len(te1), "log": log}, open(os.path.join(od, mode + ".json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
