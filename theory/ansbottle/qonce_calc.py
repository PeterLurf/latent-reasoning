"""QUESTION-ONCE draft (docs/THEORY_QONCE_DRAFT_2026-10-15.md): CPU calculations. No model is loaded.
 (1) copy load per test item: distinct question numbers n_lit (gsm8k_data.NUM), equations m, question tokens L;
 (2) the lossy-copy accuracy cap: cap(rho) = E_x[rho ** (n_lit(x) + m(x))] (each needed literal and each step's plan entry survives
     pass 1 + 9 filler passes independently with probability rho), overall and by n_lit;
 (3) A's test accuracy by n_lit (the difficulty control for a QO-CAPACITY-BOUND reading);
 (4) question-length tails against the addressed-slot count; (5) forward token-block units per item, today vs AB vs AB+QO.
numpy + tiktoken; Idle priority, one thread. Writes theory/ansbottle/qonce_calc.json (LF)."""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = "1"
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
except Exception as e_:                                                   # noqa: BLE001
    print("priority not set:", e_)

import json  # noqa: E402

import numpy as np  # noqa: E402
import tiktoken  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import boxes.gsm8k_data as G  # noqa: E402

ENC = tiktoken.get_encoding("gpt2")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qonce_calc.json")
F_TOK = 16                                                                  # the QO filler length (FILLER-N k16), + EOT
K = 10


def n_distinct_numbers(q):
    vals = []
    for m_ in G.NUM.finditer(q):
        v = G.fnum(m_.group(0))
        if not any(G._near(v, w) for w in vals):
            vals.append(v)
    return len(vals)


def main():
    te, _ = G.load_aug("test", filter_bad=False); tr, _ = G.load_aug("train")
    nl = np.array([n_distinct_numbers(r["q"]) for r in te]); mm = np.array([len(r["eqs"]) for r in te]); L = np.array([len(ENC.encode(r["q"])) for r in te])
    out = {"script": "theory/ansbottle/qonce_calc.py", "n_test": len(te)}
    out["n_lit_test"] = {"mean": float(nl.mean()), "p50": float(np.percentile(nl, 50)), "p90": float(np.percentile(nl, 90)), "max": int(nl.max()),
                         "share_by": {str(k): float(np.mean(np.minimum(nl, 6) == k)) for k in range(0, 7)}, "note": "6 = 6+"}
    out["elements_test"] = {"mean_nlit_plus_m": float((nl + mm).mean()), "p90": float(np.percentile(nl + mm, 90)), "max": int((nl + mm).max())}
    out["question_tokens_test"] = {"mean": float(L.mean()), "p90": float(np.percentile(L, 90)), "max": int(L.max()),
                                   "share_L_plus1_gt_102": float(np.mean(L + 1 > 102)), "share_L_plus1_gt_64": float(np.mean(L + 1 > 64))}
    rhos = (0.6, 0.68, 0.7, 0.8, 0.9, 0.95, 0.99)
    out["copy_cap"] = {"%.2f" % r: float(np.mean(r ** (nl + mm))) for r in rhos}
    out["copy_cap_by_nlit"] = {"%.2f" % r: {str(k): (float(np.mean(r ** (nl + mm)[np.minimum(nl, 6) == k])) if np.any(np.minimum(nl, 6) == k) else None)
                                            for k in range(1, 7)} for r in (0.68, 0.8, 0.9)}
    try:
        ea = json.load(open(os.path.join(ROOT, "boxes", "runs_fsq", "gsm_lat_aug_s1", "eval_items.json")))
        ok = np.array([bool(e["ok"]) for e in sorted(ea, key=lambda e: e["i"])])
        out["A_acc_by_nlit"] = {str(k): {"n": int(np.sum(np.minimum(nl, 6) == k)), "acc": (float(ok[np.minimum(nl, 6) == k].mean()) if np.any(np.minimum(nl, 6) == k) else None)}
                                for k in range(0, 7)}
        out["A_acc_by_nlit_def"] = "A gsm_lat_aug_s1 eval_items.json (3090 bf16, K' = 10); 6 = 6+"
    except Exception as e_:                                                 # noqa: BLE001
        out["A_acc_by_nlit"] = "unavailable: %s" % e_
    rng = np.random.default_rng(7); idx = rng.permutation(len(tr))[:20000]
    Ltr = np.array([len(ENC.encode(tr[i]["q"])) for i in idx]); mtr = np.array([len(tr[i]["eqs"]) for i in idx])
    Lq = float(Ltr.mean()) + 1                                              # [EOT, question]
    aux = float(np.mean(np.minimum(mtr, K))) * 12 * (4 + 8.8)               # aux passes x 12 blocks x (prompt 4 + 8.8 target tokens; STEPFACTOR §1.2)
    dec_plain = 12 * (Lq + 1 + 3.1); dec_ab = 12 * (1 + 3.1)
    think_today = 6 * Lq + K * 6 * Lq                                       # cache_lower: blocks 0-5 once, blocks 6-11 every pass
    think_qo = 12 * Lq + (K - 1) * 6 * (F_TOK + 1)                          # pass 1 full; passes >= 2: blocks 6-11 on the filler (its blocks 0-5 once per batch)
    tb = {"A": think_today + aux + dec_plain, "AB": think_today + aux + dec_ab, "AB_QO": think_qo + aux + dec_ab}
    out["token_blocks_per_item"] = {**{k: round(v, 1) for k, v in tb.items()}, "AB_over_A": tb["AB"] / tb["A"], "ABQO_over_A": tb["AB_QO"] / tb["A"],
                                    "ABQO_over_AB": tb["AB_QO"] / tb["AB"], "mean_train_question_tokens_plus_eot": Lq,
                                    "delta_token_writes_per_item": {"today": K * Lq, "AB_QO": Lq + (K - 1) * (F_TOK + 1)},
                                    "note": "forward token x block units; backward scales alike; the loop is launch-bound, so wall-clock falls less"}
    with open(OUT, "w", newline="\n") as f_:
        json.dump(out, f_, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
