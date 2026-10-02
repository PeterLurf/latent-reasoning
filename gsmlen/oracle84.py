"""THEORY §84 Step 1 (adc0c75): ORACLE-STOP diagnostic on the §78 WS-pad models. ORACLE: a diagnostic upper bound
given the true step count n, never a method result.
* Inference as §78's predict_ws (greedy, history cut, unpadded state), except: the answer marker "####" (GPT-2 token
  4242) is banned until the true n equations have been written; once n values are written, the answer step is forced
  (the state is followed by "####" and the answer is generated). At most 14 calls. A call whose output is not a
  parsable "<<expr=value>>" ends the problem (no answer).
* Per item: the written values and equations, per-position correctness against the reference values, the answer.
    python gsmlen/oracle84.py predict --seed 1 --device cuda:0     -> gsmlen/runs/wspad_s1/ORACLE_pred_test.json
    python gsmlen/oracle84.py summarise                             -> theory/results/s84_step1_ORACLE.json
Summary (seed-pooled, ORACLE-labelled): mid / long accuracy with n and item-bootstrap 95 % CIs; per-position written-value
correctness at positions 3-8; error split over mid + long items: halting (the unconstrained WS-pad run emitted "####"
before writing n values: its pred_test.json values_len < n), step (a wrong value at a position whose earlier values were
all correct), propagation (a wrong value whose own arithmetic is right but uses an earlier wrong value), other; the
recovery share (oracle long - WS-pad long) / (WS-pad id - WS-pad long); the gate (pooled oracle long >= 0.10).
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train78 import EOS, NL, norm, state_text, step_value  # noqa: E402

HASH = 4242
EQ = re.compile(r"<<([^<>=]*)=([^<>]*)>>")


def ref_values(x):
    return [norm(step_value(s)) if step_value(s) is not None else None for s in x["steps"]]


@torch.no_grad()
def predict_oracle(model, tok, items, dev, max_calls=14, max_new=48, bs=64):
    model.eval(); tok.padding_side = "left"; n = len(items)
    vals = [[] for _ in range(n)]; eqs = [[] for _ in range(n)]; ans = [None] * n; done = [False] * n
    qs = [x["question"] for x in items]; need = [x["n"] for x in items]
    for _ in range(max_calls):
        live = [i for i in range(n) if not done[i]]
        if not live:
            break
        step_rows = [i for i in live if len(vals[i]) < need[i]]; ans_rows = [i for i in live if len(vals[i]) >= need[i]]
        for rows, force in ((step_rows, False), (ans_rows, True)):
            for s in range(0, len(rows), bs):
                b = rows[s:s + bs]
                texts = [state_text(qs[i], vals[i]) + ("####" if force else "") for i in b]
                enc = tok(texts, return_tensors="pt", padding=True).to(dev)
                kw = {} if force else {"bad_words_ids": [[HASH]]}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=[NL, EOS], pad_token_id=EOS, **kw)
                for j, i in enumerate(b):
                    g = [t for t in out[j, enc["input_ids"].shape[1]:].tolist() if t != EOS]
                    if NL in g:
                        g = g[:g.index(NL)]
                    text = tok.decode(g).strip()
                    if force:
                        ans[i] = norm(text); done[i] = True
                    else:
                        v = step_value(text)
                        if v is None:
                            done[i] = True
                        else:
                            vals[i].append(v); eqs[i].append(text)
    return ans, vals, eqs


def run_predict(a):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2"); tok.pad_token = tok.eos_token
    run = f"gsmlen/runs/wspad_s{a.seed}"
    model = GPT2LMHeadModel.from_pretrained(f"{run}/model").to(a.device)
    items = json.load(open("gsmlen/d78/test.json", encoding="utf-8"))
    items = items[:a.limit] if a.limit else items                       # smoke only
    t0 = time.time(); ans, vals, eqs = predict_oracle(model, tok, items, a.device)
    rows = []
    for i, (x, p, v, e) in enumerate(zip(items, ans, vals, eqs)):
        ref = ref_values(x)
        rows.append({"i": i, "n": x["n"], "bin": x["bin"], "depth": x["depth"], "pred": p, "gold": norm(x["answer"]),
                     "correct": p is not None and p == norm(x["answer"]), "values": v, "equations": e,
                     "pos_correct": [norm(v[k]) == ref[k] if k < len(ref) else None for k in range(len(v))]})
    json.dump({"ORACLE": "oracle-stop diagnostic (true n given): an upper bound, never a method result", "seed": a.seed,
               "sec": round(time.time() - t0), "rows": rows}, open(f"{run}/ORACLE_pred_test{a.out_suffix}.json", "w", newline="\n"))
    print("ORACLE seed", a.seed, "done", round(time.time() - t0), "s", flush=True)


def arith_ok(eq):
    m = EQ.search(eq)
    if not m:
        return False
    e = m.group(1).replace(",", "")
    if not re.fullmatch(r"[0-9+\-*/(). ]+", e):
        return False
    try:
        return abs(eval(e, {"__builtins__": {}}) - float(m.group(2).replace(",", ""))) <= 1e-6 * max(1.0, abs(float(m.group(2).replace(",", ""))))  # noqa: S307
    except Exception:
        return False


def boot(v, n=10_000, seed=0):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def summarise(a):
    O = {s: json.load(open(f"gsmlen/runs/wspad_s{s}/ORACLE_pred_test.json"))["rows"] for s in (1, 2, 3)}
    W = {s: json.load(open(f"gsmlen/runs/wspad_s{s}/pred_test.json"))["rows"] for s in (1, 2, 3)}
    res = {"ORACLE": "oracle-stop diagnostic (true n given): an upper bound, never a method result"}
    for b in ("id", "mid", "long"):
        idx = [i for i, r in enumerate(O[1]) if r["bin"] == b]
        o = np.array([np.mean([O[s][i]["correct"] for s in O]) for i in idx]); w = np.array([np.mean([W[s][i]["correct"] for s in W]) for i in idx])
        res[f"ORACLE_{b}"] = {"acc": float(o.mean()), "ci95": boot(o), "n_items": len(idx)}
        res[f"wspad_{b}"] = {"acc": float(w.mean()), "ci95": boot(w), "n_items": len(idx)}
    pos = {}
    for k in range(3, 9):
        c = [r["pos_correct"][k - 1] for s in O for r in O[s] if len(r["pos_correct"]) >= k and r["pos_correct"][k - 1] is not None]
        pos[str(k)] = [float(np.mean(c)) if c else None, len(c)]
    res["ORACLE_position_correct"] = pos
    err = {"halting_items": 0, "step_positions": 0, "propagation_positions": 0, "other_positions": 0, "items": 0}
    for s in O:
        for r, w in zip(O[s], W[s]):
            if r["bin"] not in ("mid", "long"):
                continue
            err["items"] += 1
            if (w["values_len"] or 0) < r["n"] and not w["correct"]:
                err["halting_items"] += 1
            pc = r["pos_correct"]
            for k, ok in enumerate(pc):
                if ok is False:
                    if all(pc[:k]):
                        err["step_positions"] += 1
                    elif arith_ok(r["equations"][k]):
                        err["propagation_positions"] += 1
                    else:
                        err["other_positions"] += 1
    res["error_split_mid_long"] = err
    ol, wl, wi = res["ORACLE_long"]["acc"], res["wspad_long"]["acc"], res["wspad_id"]["acc"]
    res["recovery_share"] = (ol - wl) / (wi - wl) if wi != wl else None
    res["gate_step2"] = {"pooled_ORACLE_long": ol, "threshold": 0.10, "passes": ol >= 0.10}
    json.dump(res, open("theory/results/s84_step1_ORACLE.json", "w", newline="\n"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["predict", "summarise"])
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="smoke only"); ap.add_argument("--out_suffix", default="")
    a = ap.parse_args()
    run_predict(a) if a.cmd == "predict" else summarise(a)


if __name__ == "__main__":
    main()
