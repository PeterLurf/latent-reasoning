"""§223 PORT-STATE G-W0 (i) (PS2_v1.md at 54f63381 §1, §8 F1; CPU): the NESTED-WORDS v1 checks before any latent cell, as §218's G-L0.
  * tokens: every name and unit is ONE GPT-2 token with its leading space, names distinct; every number 0..99 is one token with a leading space;
  * invariants: on every drawn item an INDEPENDENT regex solver (boxes.nested_words.solve) re-derives the answer from the text and from every R_t;
    the constraints (>= 1 mul_v, every value in 0..99, R_d = the answer sentence) hold;
  * collisions: the panel's train-stream draws (frames 0-5) vs the test set's rendered signatures (exclusion by rendered match) -> must be 0 after
    exclusion; raw overlap reported;
  * test set: per_d 1,000 at d 1..12 (all 8 frames), sha per d and pooled;
  * shortcut panel: single-feature rules on N draws per d (train stream and test set): first / last / max / min / random number in the text,
    sum of all numbers mod 100, the number in the last statement naming the asked quantity (near_q), and the per-d majority answer of the train
    stream (mode), and (Theory 10-21) the affine-chain solver (S_0 through every stated operand, cross-spine ops ignored) which must be <= 0.10.
    Gate: every rule < 0.6 and affine <= 0.10 at every d in GATE_DS (3..8) on both streams (PS2_v1 §1 G-W0 (i): 'any single-feature rule <= 0.6 at every
    d'; d 1-2 reported, not gated - a d-1 item is one operation, so a rule can BE the computation there).
    python -B scratch/s223_gw0.py [--n 5000] -> scratch/gate_s223/g_w0.json"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402

import boxes.nested_words as NW  # noqa: E402

GATE_DS = range(3, 9); SEED_PANEL_TRAIN = 22320


def nums(text):
    return [int(x) for x in re.findall(r"\b(\d+)\b", text)]


def rules(it, rng, mode):
    t = it["text"]; body = t[:t.rfind("How many")]; ns = nums(body); tgt = it["names"]["spine"][-1]
    sents = [s for s in re.split(r"(?<=\.)\s+", body.strip()) if s]; nq = [nums(s) for s in sents if re.search(r"\b%s\b" % tgt, s)]
    return {"affine": NW.affine_solve(it), "first": ns[0], "last": ns[-1], "max": max(ns), "min": min(ns), "random": ns[int(rng.integers(0, len(ns)))], "sum_mod100": sum(ns) % 100,
            "near_q": (nq[-1][-1] if nq and nq[-1] else None), "mode": mode}


def tok_checks():
    from transformers import GPT2TokenizerFast
    tk = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    one = lambda w: len(tk.encode(" " + w)) == 1
    names = {w: tk.encode(" " + w) for w in NW.NAMES}
    return {"names_one_token": all(one(w) for w in NW.NAMES), "names_tokens_distinct": len({tuple(v) for v in names.values()}) == len(NW.NAMES),
            "names_not_one_token": [w for w in NW.NAMES if not one(w)], "units_one_token": {u: one(u) for u in NW.UNITS},
            "numbers_one_token_0_99": all(one(str(n)) for n in range(100)), "names_possessive_tokens": {w: len(tk.encode(" " + w + "'s")) for w in NW.NAMES[:5]}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5000); ap.add_argument("--per_d", type=int, default=1000); a = ap.parse_args(); t0 = time.time()
    out = {"spec": "PS2_v1 §1 G-W0 (i) at 54f63381; gated rules < 0.6 at every d in 3..8 on train + test streams", "tokens": tok_checks()}
    TE = NW.test_set(a.per_d); excl = {it["sig"] for d in TE for it in TE[d]}
    out["test_items_sha_by_d"] = {str(d): NW.set_sha(TE[d]) for d in TE}; out["test_items_sha"] = NW.set_sha([it for d in sorted(TE) for it in TE[d]])
    out["test_per_d"] = a.per_d; out["test_seed"] = NW.SEED_TEST
    raw = NW.Stream(SEED_PANEL_TRAIN, ()); tr = NW.Stream(SEED_PANEL_TRAIN, excl); rng = np.random.default_rng(22321)
    panel = {}; inv_fail = []; raw_hits = {}
    for d in range(1, 13):
        R = [raw.draw(d) for _ in range(a.n)]; raw_hits[str(d)] = sum(it["sig"] in excl for it in R)
        r0, i0 = tr.stats.get("restarts", 0), tr.stats.get("items", 0)
        T = [tr.draw(d) for _ in range(a.n)]; coll = sum(it["sig"] in excl for it in T)
        rest = (tr.stats["restarts"] - r0) / max(1, tr.stats["items"] - i0)
        mode = int(np.bincount([it["answer"] for it in T], minlength=100).argmax())
        row = {"n_train": len(T), "collisions_after_exclusion": coll, "mode_answer": mode, "restarts_per_item": round(rest, 4),
               "mul_s_t_hist": {str(k): int(v) for k, v in sorted(Counter(it["mul_s_t"] for it in T if it["mul_s_t"]).items())},
               "mul_s_j_hist": {str(k): int(v) for k, v in sorted(Counter(it["src_j"][it["mul_s_t"] - 1] for it in T if it["mul_s_t"]).items())},
               "add_s_share": round(float(np.mean(["add_s" in it["ops"] for it in T])), 4)}
        for nm, S in (("train", T), ("test", TE[d])):
            acc = {}
            for it in S:
                for k, v in rules(it, rng, mode).items():
                    acc.setdefault(k, []).append(v == it["answer"])
            row[nm] = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
            row[nm + "_max"] = max(row[nm].items(), key=lambda kv: kv[1])
            for it in S[:1000]:
                if NW.solve(it["text"]) != it["answer"] or any(NW.solve(it["R"][t]) != it["answer"] for t in range(1, d)) or it["ops"].count("mul_s") != (1 if d >= 2 else 0) \
                        or not all(0 <= v <= 99 for v in it["values"]) or it["R"][-1] != NW.GIVEN.format(A=it["names"]["spine"][-1], v=it["answer"], u=it["unit"]):
                    inv_fail.append([nm, d, it["text"][:200]])
        row["answer_hist_top5"] = [[int(x), int(c)] for x, c in sorted(enumerate(np.bincount([it["answer"] for it in T], minlength=100)), key=lambda kv: -kv[1])[:5]]
        row["T_words"] = {"mean": float(np.mean([len(it["text"].split()) for it in T])), "max": int(max(len(it["text"].split()) for it in T))}
        panel[str(d)] = row; print("G-W0 d", d, "train max", row["train_max"], "test max", row["test_max"], "affine", row["train"]["affine"], row["test"]["affine"], "restarts", row["restarts_per_item"], "coll", coll, round(time.time() - t0), "s", flush=True)
    out["panel"] = panel; out["raw_test_overlap_by_d"] = raw_hits; out["invariant_failures"] = inv_fail[:20]; out["invariant_failures_total"] = len(inv_fail)
    gate = {str(d): max(max(v for k, v in panel[str(d)][nm].items() if k != "affine") for nm in ("train", "test")) for d in GATE_DS}
    gate_aff = {str(d): max(panel[str(d)]["train"]["affine"], panel[str(d)]["test"]["affine"]) for d in GATE_DS}; out["gate_affine_by_d"] = gate_aff
    tk = out["tokens"]
    out["gate_max_by_d"] = gate
    out["pass"] = bool(all(v < 0.6 for v in gate.values()) and all(v <= 0.10 for v in gate_aff.values()) and not inv_fail and all(panel[str(d)]["collisions_after_exclusion"] == 0 for d in range(1, 13))
                       and tk["names_one_token"] and tk["names_tokens_distinct"] and tk["numbers_one_token_0_99"] and all(tk["units_one_token"].values()))
    out["sec"] = round(time.time() - t0); out["generator_sha256"] = __import__("hashlib").sha256(open("boxes/nested_words.py", "rb").read()).hexdigest()
    os.makedirs("scratch/gate_s223", exist_ok=True); json.dump(out, open("scratch/gate_s223/g_w0.json", "w", newline="\n"), indent=1)
    print("G-W0 PASS" if out["pass"] else "G-W0 FAIL", json.dumps(gate), "affine", json.dumps(gate_aff), "tokens", {k: tk[k] for k in ("names_one_token", "names_tokens_distinct", "numbers_one_token_0_99")}, flush=True)


if __name__ == "__main__":
    main()
