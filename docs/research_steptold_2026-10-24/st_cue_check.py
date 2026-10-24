"""STEP-TOLD $0 check G-CUE (CPU, 1 thread, idle priority; no model): the cue names the step and never carries its value; token lengths; d = 2 pairs.

Cue (ST_draft sec. 2): at think pass t the input is [EOT] + enc(" " + text + " " + QUESTION(u, S_min(t, d))); CUED-DIRECT's decoder input is
[EOT] + enc(" " + text + " " + QUESTION(u, S_1) + ... + " " + QUESTION(u, S_d)) + enc(" The answer is:").
  C1  no number: no cue string contains a digit, no cue token is one of the 11 answer tokens (all items, d 1..8, every t <= 8).
  C2  skeleton-only: on same-skeleton redraw pairs (OW.redraw: same names / unit / order / ops, new values) the cue token lists are bitwise equal at
      every t -> the cue is a function of the skeleton, so I(cue; values | skeleton) = 0 [PROVEN by this test + the generator's draw order].
  C3  best lookup from the cue: table predictors (name, t), (unit, t), (name, unit, t) -> majority v_t on a fit half, read on the other half.
  C4  token lengths (plain think, ST think at pass t, CUED-DIRECT dec) by d -> the cost factors in st_oc.py.
  C5  d = 2 STATE-SWAP(1) pairs under §228's kept rule (own, donor-final, cf(a) distinct; cf_b not in {own, donor-final}; exact token length):
      kept share, cf(a) != cf_b share; and duplicate sigs within 200k d = 2 draws (collision risk for a pair set outside the training EXCL).
  C6  generator facts used by the derivation: for every t >= 2, S_t's defining sentence names S_{t-1} (the chain); S_d = the question entity.
Seeds 23242 (items), 23243 / 23244 (pairs / redraw rng), 23245 (collision draws): unused in boxes/ theory/ docs scripts (grep 10-24).
    python -B st_cue_check.py [--n 500]
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
except Exception:  # noqa: BLE001
    pass
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import tiktoken  # noqa: E402

import boxes.ow_words as OW  # noqa: E402

ENC = tiktoken.get_encoding("gpt2")
NUM = {ENC.encode(" %d" % v)[0] for v in range(11)}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "st_cue_check.json")
K = 8


def cue_str(it, t):
    return " " + OW.QUESTION.format(u=it["unit"], A=it["names"]["spine"][min(t, it["d"])])


def cues(it, k=K):
    return [ENC.encode(cue_str(it, t)) for t in range(1, k + 1)]


def cf(rec, don, t, mode):
    S = list(don["values"][:t + 1]) if mode == "a" else list(rec["values"][:t]) + [don["values"][t]]
    for k in range(t + 1, rec["d"] + 1):
        op, b, j = rec["ops"][k - 1], rec["operands"][k - 1], rec["src_j"][k - 1]
        S.append(OW.apply(op, S[k - 1], S[j] if op.endswith("_s") else b))
    return S[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=500); ap.add_argument("--coll", type=int, default=200000); a = ap.parse_args(argv)
    t0 = time.time(); res = {}
    st = OW.Stream(23242, frames=OW.FRAME_TRAIN, order_mode="topo", dag=True); items = {d: [st.draw(d) for _ in range(a.n)] for d in range(1, 9)}
    # C1
    bad_digit = sum(any(ch.isdigit() for t in range(1, K + 1) for ch in cue_str(it, t)) for d in items for it in items[d])
    bad_tok = sum(any(tk in NUM for c in cues(it) for tk in c) for d in items for it in items[d])
    res["C1_no_number"] = {"items": 8 * a.n, "cues_with_digit": bad_digit, "cues_with_answer_token": bad_tok, "pass": bad_digit == 0 and bad_tok == 0}
    # C2
    r = np.random.default_rng(23244); ps = OW.Stream(23243, frames=OW.FRAME_TRAIN, order_mode="topo", dag=True); n_pairs = 0; n_diff = 0; n_valdiff = 0
    for d in range(1, 9):
        for _ in range(a.n // 2):
            rec = ps.draw(d); don = OW.redraw(r, rec)
            if don is None:
                continue
            n_pairs += 1; n_diff += int(cues(rec) != cues(don)); n_valdiff += int(rec["values"] != don["values"])
    res["C2_skeleton_only"] = {"pairs": n_pairs, "pairs_values_differ": n_valdiff, "pairs_cue_differs": n_diff, "pass": n_diff == 0}
    # C3
    rows = [(it["names"]["spine"][t], it["unit"], t, int(it["values"][t])) for d in items for it in items[d] for t in range(1, d + 1)]
    rng = np.random.default_rng(0); idx = rng.permutation(len(rows)); fit = [rows[i] for i in idx[: len(rows) // 2]]; te = [rows[i] for i in idx[len(rows) // 2:]]
    glob = Counter(y for *_, y in fit).most_common(1)[0][0]; c3 = {"n_fit": len(fit), "n_test": len(te), "majority_test": float(np.mean([y == glob for *_, y in te]))}
    for nm, key in (("name_t", lambda x: (x[0], x[2])), ("unit_t", lambda x: (x[1], x[2])), ("name_unit_t", lambda x: (x[0], x[1], x[2]))):
        tab = defaultdict(Counter)
        for x in fit:
            tab[key(x)][x[3]] += 1
        pr = [tab[key(x)].most_common(1)[0][0] if key(x) in tab else glob for x in te]; c3[nm] = float(np.mean([p == x[3] for p, x in zip(pr, te)]))
    c3["best"] = max(c3[k] for k in ("name_t", "unit_t", "name_unit_t")); c3["pass"] = c3["best"] <= c3["majority_test"] + 0.02 + 2 * math.sqrt(0.25 / len(te))
    res["C3_cue_lookup"] = c3
    # C4
    L = {}
    for d in range(1, 9):
        plain = [1 + len(ENC.encode(" " + it["text"])) for it in items[d]]; cl = [np.mean([len(c) for c in cues(it, d)]) for it in items[d]]
        cd = [1 + len(ENC.encode(" " + it["text"] + "".join(cue_str(it, t) for t in range(1, d + 1)))) + 4 for it in items[d]]
        L[str(d)] = {"think_plain_mean": float(np.mean(plain)), "think_plain_max": int(np.max(plain)), "cue_tokens_mean": float(np.mean(cl)),
                     "think_st_mean": float(np.mean(plain) + np.mean(cl)), "cued_direct_dec_mean": float(np.mean(cd)), "cued_direct_dec_max": int(np.max(cd)),
                     "plain_dec_mean": float(np.mean(plain) + 4)}
    res["C4_lengths"] = L
    # C5
    r2 = np.random.default_rng(23244 + 1); s2 = OW.Stream(23243 + 1, frames=OW.FRAME_TRAIN, order_mode="topo", dag=True); tried = kept = lm = ab = adds = 0
    for _ in range(3000):
        rec = s2.draw(2); don = OW.redraw(r2, rec)
        if don is None or don["sig"] == rec["sig"]:
            continue
        tried += 1; own, dfin, ca, cb = rec["answer"], don["answer"], cf(rec, don, 1, "a"), cf(rec, don, 1, "b"); adds += int(rec["ops"][1] == "add_s")
        if len({own, dfin, ca}) == 3 and cb not in (own, dfin):
            if len(ENC.encode(rec["text"])) != len(ENC.encode(don["text"])):
                lm += 1; continue
            kept += 1; ab += int(ca != cb)
    cs = OW.Stream(23245, frames=OW.FRAME_TRAIN, order_mode="topo", dag=True); sig = Counter(cs.draw(2)["sig"] for _ in range(a.coll))
    res["C5_d2_pairs"] = {"t": 1, "candidates": tried, "kept": kept, "kept_share": kept / max(1, tried), "len_mismatch_dropped": lm, "cfa_ne_cfb_in_kept": ab,
                          "add_s_share_step2": adds / max(1, tried), "collision_draws": a.coll, "duplicate_sigs": int(sum(v - 1 for v in sig.values() if v > 1))}
    # C6
    nm_ok = q_ok = n6 = 0
    for d in range(2, 9):
        for it in items[d]:
            sp = it["names"]["spine"]; q_ok += int(it["question"] == OW.QUESTION.format(u=it["unit"], A=sp[d]))
            sents = re.findall(r"[^.?]*[.?]", it["text"])
            for t in range(2, d + 1):
                n6 += 1; defs = [s for s in sents if re.search(r"\b%s\b" % sp[t], s) and re.search(r"\b%s\b" % sp[t - 1], s) and "How many" not in s]
                nm_ok += int(len(defs) >= 1)
    res["C6_chain"] = {"steps_checked": n6, "S_t_sentence_names_S_t-1": nm_ok, "question_is_S_d": q_ok, "items_d2_8": 7 * a.n,
                       "pass": nm_ok == n6 and q_ok == 7 * a.n}
    res["sec"] = round(time.time() - t0, 1); res["all_pass"] = all(res[k]["pass"] for k in ("C1_no_number", "C2_skeleton_only", "C3_cue_lookup", "C6_chain"))
    with open(OUT, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
