"""ANSWER-BOTTLENECK draft (docs/THEORY_ANSBOTTLE_DRAFT_2026-10-15.md): CPU calculations for the design. No model is loaded.
 (1) K'0 chance: an item-independent decoder's best accuracy (the largest answer frequency) and its answer CE (per item; per labelled token of
     "### " + answer + EOS), against the train marginal;
 (2) answer token counts: how often the first answer token fixes the whole answer;
 (3) TEMPLATE-SWAP pairs: recipient x_r and donor x_d share one recomputable template (gsm8k_data.parse_recomputable) with different literals
     (R_x draws, gsm8k_data.draw_literal / render). Hybrid targets y_hyb(t) for a swap after pass t, under the STEP schedule (step s is done at
     pass s) and the LEVEL schedule (every step of level <= t is done by pass t). Kept = y_hyb not in {y_own, y_don}; strict = kept and y_hyb
     renders as a GSM8K-style answer (>= 0, <= 2 decimals). Pools: TEST (test recipients, redrawn donors), TRAIN-REDRAW (both redrawn from Aug
     train templates);
 (4) LITERAL-ONLY-SWAP pairs: the donor is another recomputable train template carrying the recipient's used literal values (same values,
     different operator structure); generability and the kept fraction of the step-schedule hybrid at t = 1 (recipient chain with v_1 := donor v_1);
 (5) literal counts per test item (for the copy-bandwidth estimate).
numpy + tiktoken; Idle priority, one thread. Writes theory/ansbottle/pairs_calc.json (LF)."""
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
import math  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import tiktoken  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import boxes.gsm8k_data as G  # noqa: E402
from bench.tasks import norm_number  # noqa: E402

ENC = tiktoken.get_encoding("gpt2")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairs_calc.json")
D_DRAWS = 16
T0 = time.time()


def tol_eq(a, b):
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))


def render_ans(y):
    """a GSM8K-style answer string for a numeric target, or None (negative, non-finite or > 2 decimals)."""
    if y is None or not math.isfinite(y) or y < -1e-9:
        return None
    if abs(y - round(y)) < 1e-9:
        return str(int(round(y)))
    if abs(y - round(y, 2)) < 1e-9:
        return ("%.2f" % y).rstrip("0").rstrip(".")
    return None


def chain(tpl, qv, fixed=None):
    """the template's step values with question values qv {q index: value}; fixed {step index: value} overrides a step (hybrids). None on an
    evaluation error. Unrounded (a hybrid need not be representable)."""
    vals = []
    for s_i, t in enumerate(tpl["eqt"]):
        if fixed is not None and s_i in fixed:
            vals.append(float(fixed[s_i])); continue
        ex = []
        for s, e, _tok, kind, ref in t["ops"]:
            v = qv[ref] if kind == "q" else (vals[ref] if kind == "r" else ref)
            ex.append((s, e, repr(float(v))))
        expr = "".join(t["lhs"][a:b] if k is None else k for a, b, k in G._segments(t["lhs"], ex))
        try:
            vals.append(G.safe_eval(expr.replace(",", "")))
        except G.Bad:
            return None
    return vals


def levels(tpl):
    lv = []
    for t in tpl["eqt"]:
        refs = [ref for _, _, _, kind, ref in t["ops"] if kind == "r"]
        lv.append(1 + max((lv[r] for r in refs), default=0))
    return lv


def own_qv(tpl):
    return {i: tpl["qvals"][i] for i in tpl["used_q"]}


def ok_values(tpl, nv_map):
    """R_x's distinctness rule (gsm8k_data.resample): new values distinct, distinct from the question's other numbers and from the constants."""
    other = [v for _, _, _, v in tpl["qn"]]; nv = list(nv_map.values())
    if len(set(nv)) != len(nv):
        return False
    if any(G._near(v, w) for v in nv for w in other if not any(G._near(w, tpl["qvals"][i]) for i in nv_map)):
        return False
    return not any(G._near(v, c) for v in nv for c in tpl["consts"])


def redraw(raw, tpl, rng):
    """gsm8k_data.resample's loop without its train-only assert (eval-time pairs only; nothing here trains) -> (newq, vals) or None."""
    for _ in range(G.TRIES):
        newq = {}
        for i in tpl["used_q"]:
            tok = next(tk for _, _, tk, v in tpl["qn"] if G._near(v, tpl["qvals"][i])); newq[i] = G.draw_literal(tok, rng)
        if not ok_values(tpl, newq):
            continue
        res, vals = G.render(raw, tpl, newq)
        if res[0] is None:
            continue
        return newq, vals
    return None


def op_sig(tpl):
    return tuple("".join(c for c in G.NUMTOK.sub("#", t["lhs"]) if c in "+-*/%()#") for t in tpl["eqt"])


def hybrid_stats(tpl, qv_r, vals_r, vals_d):
    """y_own / y_don / y_hyb(t) for t in {1, 2} under both schedules -> dict of flags."""
    lv = levels(tpl); y_own, y_don = vals_r[-1], vals_d[-1]; out = {"m": len(tpl["eqt"]), "d_star": max(lv), "own_ne_don": not tol_eq(y_don, y_own)}
    for t in (1, 2):
        for sch, fixed in (("step", {s: vals_d[s] for s in range(min(t, len(vals_d)))}), ("level", {s: vals_d[s] for s in range(len(lv)) if lv[s] <= t})):
            ch = chain(tpl, qv_r, fixed); y_h = ch[-1] if ch else None
            kept = y_h is not None and math.isfinite(y_h) and not tol_eq(y_h, y_own) and not tol_eq(y_h, y_don)
            out["%s%d" % (sch, t)] = {"kept": kept, "strict": kept and render_ans(y_h) is not None, "eq_don": tol_eq(y_h, y_don), "eq_own": tol_eq(y_h, y_own), "y": y_h}
    out["hyb12_differ"] = out["step1"]["kept"] and out["step2"]["kept"] and not tol_eq(out["step1"]["y"], out["step2"]["y"])
    out["step_level_agree1"] = tol_eq(out["step1"]["y"], out["level1"]["y"]) if out["step1"]["y"] is not None and out["level1"]["y"] is not None else False
    return out


def summarise(rows):
    """kept / strict fractions overall and by m (pairs with y_own != y_don only)."""
    rows = [r for r in rows if r["own_ne_don"]]; res = {"n_pairs": len(rows)}
    if not rows:
        return res
    for key in ("step1", "step2", "level1", "level2"):
        res[key] = {"kept": float(np.mean([r[key]["kept"] for r in rows])), "strict": float(np.mean([r[key]["strict"] for r in rows])),
                    "eq_don": float(np.mean([r[key]["eq_don"] for r in rows])), "eq_own": float(np.mean([r[key]["eq_own"] for r in rows]))}
    res["hyb12_differ"] = float(np.mean([r["hyb12_differ"] for r in rows])); res["step_level_agree1"] = float(np.mean([r["step_level_agree1"] for r in rows]))
    res["strict_both_t1_t2_and_differ"] = float(np.mean([r["step1"]["strict"] and r["step2"]["strict"] and r["hyb12_differ"] for r in rows]))
    by = {}
    for r in rows:
        by.setdefault(min(r["m"], 5), []).append(r)
    res["by_m"] = {("m%d" % k if k < 5 else "m5+"): {"n": len(v), "step1_strict": float(np.mean([x["step1"]["strict"] for x in v])),
                   "step2_strict": float(np.mean([x["step2"]["strict"] for x in v])), "level1_strict": float(np.mean([x["level1"]["strict"] for x in v]))}
                   for k, v in sorted(by.items())}
    return res


def main():
    out = {"script": "theory/ansbottle/pairs_calc.py", "draws_per_recipient": D_DRAWS}
    tr, _ = G.load_aug("train"); te, _ = G.load_aug("test", filter_bad=False); va, _ = G.load_aug("valid", filter_bad=False)
    # (1) K'0 chance -------------------------------------------------------------------------------------------------------------
    ytr = Counter(norm_number(r["ans"]) for r in tr); yte = [norm_number(r["ans"]) for r in te]; cte = Counter(yte); Ntr = sum(ytr.values())
    mode_tr, n_mode_tr = ytr.most_common(1)[0]; mode_te, n_mode_te = cte.most_common(1)[0]
    n1 = sum(1 for v in ytr.values() if v == 1); lam = n1 / Ntr                                   # Good-Turing unseen mass
    tok_counts = Counter(); n_tok = 0
    for y, c in ytr.items():
        for tk in ENC.encode(" " + y) + [50256]:
            tok_counts[tk] += c; n_tok += c
    V = len(tok_counts) + 1

    def p_bg(y):                                                             # token-unigram backoff over answer strings (+ EOS), add-one
        return math.exp(sum(math.log((tok_counts[tk] + 1) / (n_tok + V)) for tk in ENC.encode(" " + y) + [50256]))
    nll = [-math.log((1 - lam) * ytr[y] / Ntr + lam * p_bg(y)) for y in yte]
    lab_tok = [len(ENC.encode("### " + r["ans"])) + 1 for r in te]                                # labelled tokens: "###", answer tokens, EOS
    H_tr = -sum(c / Ntr * math.log(c / Ntr) for c in ytr.values())
    out["kprime0_chance"] = {"test_max_answer_freq": n_mode_te / len(te), "test_mode": mode_te, "train_mode": mode_tr, "train_mode_freq_train": n_mode_tr / Ntr,
                             "train_mode_freq_test": cte[mode_tr] / len(te), "train_answer_entropy_nats": H_tr, "test_xent_under_train_marginal_nats_per_item": float(np.mean(nll)),
                             "mean_labelled_tokens_per_item": float(np.mean(lab_tok)), "chance_answer_ce_per_token": float(np.sum(nll) / np.sum(lab_tok)),
                             "note": "an item-independent decoder: accuracy <= test_max_answer_freq; its best answer CE ~ the test cross-entropy under the train marginal "
                                     "(Good-Turing mixture with a token-unigram backoff); per token = summed item NLL / summed labelled tokens ('###' costs ~0)"}
    # (2) answer token counts ------------------------------------------------------------------------------------------------------
    na = [len(ENC.encode(" " + r["ans"])) for r in te]
    out["answer_tokens_test"] = {"share_1_token": float(np.mean([n == 1 for n in na])), "share_2_tokens": float(np.mean([n == 2 for n in na])), "mean": float(np.mean(na))}
    # (5) literal counts ---------------------------------------------------------------------------------------------------------
    tpl_te = [G.parse_recomputable(r)[0] for r in te]
    rec_te = [(r, t) for r, t in zip(te, tpl_te) if t is not None]
    qnums = [len(G.parse_recomputable(r)[0]["qvals"]) if t is not None else None for r, t in zip(te, tpl_te)]
    out["literals_test"] = {"n_recomputable": len(rec_te), "share_recomputable": len(rec_te) / len(te),
                            "mean_used_literals_recomputable": float(np.mean([len(t["used_q"]) for _, t in rec_te])),
                            "mean_distinct_question_numbers_recomputable": float(np.mean([q for q in qnums if q is not None])),
                            "mean_m_recomputable": float(np.mean([len(t["eqt"]) for _, t in rec_te]))}
    print("part 1/2/5 done", round(time.time() - T0, 1), flush=True)
    # (3) TEMPLATE-SWAP pairs --------------------------------------------------------------------------------------------------------
    rng = np.random.default_rng(20710); rows_te = []; per_rec_te = []
    for r, t in rec_te:
        qv = own_qv(t); vr = chain(t, qv); kept_any = 0
        for _ in range(D_DRAWS):
            d = redraw(r, t, rng)
            if d is None:
                continue
            hs = hybrid_stats(t, qv, vr, d[1]); rows_te.append(hs); kept_any += int(hs["own_ne_don"] and hs["step1"]["strict"])
        per_rec_te.append(kept_any)
    out["template_swap_test"] = summarise(rows_te); out["template_swap_test"]["share_recipients_with_a_strict_t1_donor"] = float(np.mean([k > 0 for k in per_rec_te]))
    print("part 3 test done", round(time.time() - T0, 1), flush=True)
    rng = np.random.default_rng(20711); idx = rng.permutation(len(tr))[:3000]; rows_tr = []; n_tpl = 0
    for i in idx:
        r = tr[i]; t, _ = G.parse_recomputable(r)
        if t is None:
            continue
        n_tpl += 1; base = redraw(r, t, rng)                                   # the recipient is itself a fresh redraw (recall impossible, Theorem X)
        if base is None:
            continue
        qv = base[0]; vr = base[1]
        for _ in range(D_DRAWS - 1):
            d = redraw(r, t, rng)
            if d is not None:
                rows_tr.append(hybrid_stats(t, qv, vr, d[1]))
    out["template_swap_train_redraw"] = summarise(rows_tr); out["template_swap_train_redraw"]["n_templates_sampled"] = int(len(idx))
    out["template_swap_train_redraw"]["n_recomputable_templates"] = n_tpl
    print("part 3 train done", round(time.time() - T0, 1), flush=True)
    # expected readable pairs (one pair per recipient: the first strict-kept, clean-correct donor) under assumed per-item accuracies
    k_te = out["template_swap_test"]["step1"]["strict"]
    try:
        ea = json.load(open(os.path.join(ROOT, "boxes", "runs_fsq", "gsm_lat_aug_s1", "eval_items.json")))
        okA = {e["i"]: bool(e["ok"]) for e in ea}; recA = sum(1 for j, t in enumerate(tpl_te) if t is not None and okA.get(j))
    except Exception:                                                         # noqa: BLE001
        recA = None
    exp = {"A_correct_recomputable_test_recipients": recA}
    for p_r in (0.08, 0.12, 0.147):
        for p_d in (0.06, 0.10, 0.175):
            exp["test_pool_pr%.3f_pd%.3f" % (p_r, p_d)] = len(rec_te) * p_r * (1 - (1 - p_d * k_te) ** D_DRAWS)
    k_tr = out["template_swap_train_redraw"]["step1"]["strict"]; n_t = n_tpl
    for p in (0.06, 0.10, 0.175):                                             # TRAIN-REDRAW: >= 1 correct recipient draw and >= 1 correct, kept donor among the rest
        p_pair = sum(math.comb(D_DRAWS, j) * p ** j * (1 - p) ** (D_DRAWS - j) * (1 - (1 - k_tr) ** (j - 1)) for j in range(2, D_DRAWS + 1))
        exp["train_redraw_%d_templates_p%.3f" % (n_t, p)] = n_t * p_pair
    out["expected_readable_pairs"] = exp
    # (4) LITERAL-ONLY-SWAP pairs ----------------------------------------------------------------------------------------------------
    rng = np.random.default_rng(20712); pool = []
    for i in rng.permutation(len(tr))[:8000]:
        t, _ = G.parse_recomputable(tr[i])
        if t is not None:
            pool.append((tr[i], t, op_sig(t)))
    by_k = {}
    for r, t, s in pool:
        by_k.setdefault(len(t["used_q"]), []).append((r, t, s))
    lit = {"n_recipients": 0, "found": 0, "tries": [], "kept": 0, "strict": 0, "eq_own": 0, "own_ne_don": 0}
    for r, t, s in pool[:1500]:
        base = redraw(r, t, rng)
        if base is None:
            continue
        lit["n_recipients"] += 1; qv_r = base[0]; vr = base[1]; vals_lit = [qv_r[i] for i in t["used_q"]]; cand = by_k.get(len(t["used_q"]), [])
        for tries in range(1, 301):
            rd, td, sd = cand[int(rng.integers(0, len(cand)))]
            if sd == s:
                continue
            newq = {i: v for i, v in zip(td["used_q"], vals_lit)}
            if not ok_values(td, newq):
                continue
            res, vd = G.render(rd, td, newq)
            if res[0] is None:
                continue
            lit["found"] += 1; lit["tries"].append(tries)
            ch = chain(t, qv_r, {0: vd[0]}); y_h = ch[-1] if ch else None; y_own, y_don = vr[-1], vd[-1]
            if not tol_eq(y_own, y_don):
                lit["own_ne_don"] += 1
                kept = y_h is not None and math.isfinite(y_h) and not tol_eq(y_h, y_own) and not tol_eq(y_h, y_don)
                lit["kept"] += int(kept); lit["strict"] += int(kept and render_ans(y_h) is not None); lit["eq_own"] += int(tol_eq(y_h, y_own))
            break
    n_ = max(1, lit["own_ne_don"])
    out["literal_only_swap"] = {"n_recipients": lit["n_recipients"], "share_found_within_300": lit["found"] / max(1, lit["n_recipients"]),
                                "median_tries": float(np.median(lit["tries"])) if lit["tries"] else None, "share_own_ne_don": lit["own_ne_don"] / max(1, lit["found"]),
                                "t1_kept": lit["kept"] / n_, "t1_strict": lit["strict"] / n_, "t1_hyb_eq_own": lit["eq_own"] / n_,
                                "note": "donor = another recomputable train template (different operator signature, same number of used literals) carrying the "
                                        "recipient's used literal values in slot order; hybrid = the recipient chain with v_1 := the donor's v_1 (A-SLOT)"}
    out["sec"] = round(time.time() - T0, 1)
    with open(OUT, "w", newline="\n") as f_:
        json.dump(out, f_, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("template_swap_test", "template_swap_train_redraw")}, indent=1))
    print(json.dumps({"test": out["template_swap_test"], "train": out["template_swap_train_redraw"]}, indent=1))


if __name__ == "__main__":
    main()
