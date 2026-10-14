"""note 191-1 / 198-2a eval-time cost of a trained prosqa168s2 run (§191 A stepaux, B textcot): timed seconds per test item for the answer path
(stepaux: K think passes + the two-candidate margins, the run's cache_lower / chunk_delta; textcot: batched greedy CoT generation + candidate scoring),
on test500's fixed subset (rng 19101 permutation[:500] of p8_na_test), at the run's eval batches (eval_bs; cot_eval_bs for generation), the run's eval
dtype, after one warm-up batch, cuda-synced; plus Meter FLOPs per item at the measured mean lengths (boxes.core.Meter, trunk; §98 convention).
    python -m scratch.eval_timing191 --run <run dir> [--device cuda:0] [--out <ignored>]   -> <run>/eval_timing.json"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.prosqa168s2 as P  # noqa: E402
from boxes import core as bc  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.prosqa168 import candidate_margins  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", default=""); ap.add_argument("--fname", default="eval_timing.json", help="output file name in the run dir (note 201-6: eval_timing_4060.json for the local same-card timing)")
    ap.add_argument("--n_items", type=int, default=500); ap.add_argument("--no_load", action="store_true"); o = ap.parse_args(); dev = o.device
    a = json.load(open(os.path.join(o.run, "args.json"))); SA = a["arm"] == "stepaux"; K = int(a.get("think_k") or 0) if SA else 0
    DM.FAST_ADD = bool(a.get("fast_add")); DM.CHUNK_DELTA = int(a.get("chunk_delta") or 0)
    amp = None if (a.get("fp32") or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    g = s5.load_gpt2(dev); m = P.ProsMem(g).to(dev) if SA else None; params = m if SA else g
    if not o.no_load:
        params.load_state_dict(torch.load(os.path.join(o.run, "model.pt"), map_location=dev, weights_only=False))
    params.eval(); pr = os.path.join(ROOT, "prosqa", "p8data"); Dte = P.load(os.path.join(pr, "p8_na_test.json"))
    its = [Dte[i] for i in np.random.default_rng(19101).permutation(len(Dte))[:o.n_items]]; bs = int(a.get("eval_bs") or 16); cbs = int(a.get("cot_eval_bs") or 64)
    cot_len = []

    @torch.no_grad()
    def run_all(D):
        if SA:
            for i in range(0, len(D), bs):
                ch = D[i:i + bs]; x, msk = m.think_inputs(ch, dev)
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    S = m.think(m.mem.zero_state(len(ch), dev), x, msk, K, cache=bool(a.get("cache_lower")))
                    m.margins(S, ch, dev)
        else:
            Q = P.generate_cot_batched(g, [it["pre"] for it in D], int(a.get("max_new") or 220), dev, amp, cbs)
            cot_len.extend(len(q) - len(it["pre"]) for q, it in zip(Q, D))
            candidate_margins(g, [(q, it["a1"], it["a2"], it["depth"]) for q, it in zip(Q, D)], dev, amp, bs)
    run_all(its[:min(len(its), cbs if not SA else bs)])                     # warm-up
    cot_len.clear()
    if dev.startswith("cuda"):
        torch.cuda.synchronize(dev)
    t0 = time.time(); run_all(its)
    if dev.startswith("cuda"):
        torch.cuda.synchronize(dev)
    sec = time.time() - t0
    qL = float(np.mean([len(it["q"]) + 1 for it in its])); preL = float(np.mean([len(it["pre"]) for it in its])); aL = float(np.mean([len(it["a1"]) for it in its]))
    M_ = bc.Meter(g)
    if SA:                                                                  # K think passes over [EOT, question]; the decoder [EOT, pre, answer] for each of the 2 candidates
        for _ in range(K):
            M_.step(int(round(qL)), 0, 0)
        for _ in range(2):
            M_.step(int(round(1 + preL + aL)), 0, int(round(aL)))
    else:                                                                   # the prompt, then the generated CoT token by token (KV-cached), then 2 candidates after it
        cl = float(np.mean(cot_len)) if cot_len else 0.0; M_.step(int(round(preL)), 0, 1)
        for j in range(int(round(cl)) - 1):
            M_.step(1, int(round(preL)) + j, 1)
        for _ in range(2):
            M_.step(int(round(aL)), int(round(preL + cl)), int(round(aL)))
    out = {"eval_timing": {"s_per_item": sec / len(its), "batch": bs, "cot_eval_bs": None if SA else cbs, "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
                           "n_items": len(its), "arm": a["arm"], "think_k": K, "eval_dtype": str(amp), "total_sec": sec, "cache_lower": bool(a.get("cache_lower")),
                           "chunk_delta": int(a.get("chunk_delta") or 0), "mean_cot_tokens": (float(np.mean(cot_len)) if cot_len else None), "no_load": bool(o.no_load)},
           "flops_per_item": M_.flops, "meter": M_.as_dict(), "mean_lengths": {"question_plus_eot": qL, "pre": preL, "answer": aL},
           "def": "191-1: wall s/item for the answer path on test500's items (rng 19101), the run's eval batch(es) and dtype, one warm-up batch, cuda-synced; "
                  "Meter FLOPs per item at the mean lengths (stepaux: K passes over [EOT, question] + 2 candidate decodes; textcot: prompt + generated CoT + 2 candidates)"}
    json.dump(out, open(os.path.join(o.run, o.fname), "w", newline="\n"), indent=1); print("EVAL_TIMING", o.run, out["eval_timing"]["s_per_item"], out["flops_per_item"], flush=True)


if __name__ == "__main__":
    main()
