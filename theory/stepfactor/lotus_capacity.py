"""LOTUS contrast (docs/THEORY_LOTUS_CONTRAST_2026-10-14.md §2): bits GSM8K's serial content needs per test item vs the usable capacity of §200 A's
memory channel. CPU, no torch. Output lotus_capacity.json here.
needed (per test item, from its gold equations; worst-case digit coding, 3.32 bits / digit + 1 bit if decimal):
  ans_bits   : the answer alone (what f_ans must read if the loop computed serially)
  val_bits   : every step's value v_t
  plan_bits  : per step log2(17) operator + 2 x log2(|C_t|) pointers, |C_t| = distinct question numbers + (t - 1) previous results + 1 constant class
  full_bits  : val_bits + plan_bits;  items_needed = 2 m (one value + one plan entry per step, as key-value items)
capacity (usable, not the float count): delta heads 6 x DH(64) retrievable items (near-orthogonal keys; Schlag 2102.11174); additive heads (phi >= 0)
6 x [1 .. 64] items (dense positive features ~ a bag; disjoint-support keys at best 64); bits per item b in {1, 6, 32}."""
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, "D:/Latent")
import boxes.gsm8k_data as G  # noqa: E402
import tiktoken  # noqa: E402

ENC = tiktoken.get_encoding("gpt2"); HERE = os.path.dirname(os.path.abspath(__file__))
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")


def num_bits(tok):
    t = tok.replace(",", ""); d = sum(ch.isdigit() for ch in t); return d * math.log2(10) + (1.0 if "." in t else 0.0)


def main():
    R, _ = G.load_aug("test", filter_bad=False); rows = []
    for r in R:
        m = len(r["eqs"]); nq = len({float(x.replace(",", "")) for x in G.NUM.findall(r["q"])} if hasattr(G, "NUM") else set())
        vals = []
        for e in r["eqs"]:
            rhs = e.split("=")[-1] if "=" in e else ""
            mm = NUM.findall(rhs); vals.append(num_bits(mm[0]) if mm else 0.0)
        am = NUM.findall(r["ans"]); ans = num_bits(am[0]) if am else 0.0
        plan = sum(math.log2(17) + 2 * math.log2(max(2, nq + t + 1)) for t in range(m))
        rows.append({"m": m, "q_tokens": len(ENC.encode(r["q"])), "ans_bits": ans, "val_bits": sum(vals), "plan_bits": plan, "full_bits": sum(vals) + plan, "items_needed": 2 * m})

    def stats(key, sub):
        x = np.array([r_[key] for r_ in sub]); return {"mean": round(float(x.mean()), 1), "p50": round(float(np.percentile(x, 50)), 1), "p90": round(float(np.percentile(x, 90)), 1),
                                                      "p99": round(float(np.percentile(x, 99)), 1), "max": round(float(x.max()), 1)}
    need = {k: stats(k, rows) for k in ("ans_bits", "val_bits", "plan_bits", "full_bits", "items_needed", "q_tokens")}
    by_m = {str(m): {k: stats(k, [r_ for r_ in rows if r_["m"] == m]) for k in ("full_bits", "items_needed")} for m in sorted({r_["m"] for r_ in rows})}
    cap = {}
    for b in (1, 6, 32):
        cap["b%d" % b] = {"delta_only_bits": 6 * 64 * b, "additive_dense_bits": 6 * 1 * b, "additive_best_bits": 6 * 64 * b, "total_low_bits": 6 * 64 * b + 6 * b,
                          "total_high_bits": 12 * 64 * b}
    mx = need["full_bits"]["max"]
    res = {"n_test": len(rows), "needed": need, "needed_by_m": by_m, "capacity": cap,
           "items_capacity": {"delta": 384, "additive_dense": 6, "additive_best": 384},
           "margin_full_bits_max_vs_delta_b1": round(384 / mx, 2), "margin_full_bits_max_vs_delta_b6": round(6 * 384 / mx, 1),
           "margin_items_needed_max_vs_delta": round(384 / need["items_needed"]["max"], 1),
           "raw_floats": {"ours_state": 12 * 64 * 64, "lotus_gpt2_latent_input": 78 * 768, "lotus_gpt2_kv_seen_by_answer": 78 * 768 * 2 * 12}}
    json.dump(res, open(os.path.join(HERE, "lotus_capacity.json"), "w", newline="\n"), indent=1); print(json.dumps(res, indent=1)[:3500])


if __name__ == "__main__":
    main()
