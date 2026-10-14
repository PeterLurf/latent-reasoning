"""STEP-TARGET-FACTOR facts (CPU, no torch): token-share split of the aux target " <<e_t>>" + EOT on GSM8K-Aug, operand grounding
(gsm8k_data.parse_recomputable: q = question literal, r = previous result, k = constant / number word), pointer-vocabulary sizes, operator
vocabulary, value token lengths, and the equation dependency-DAG depth d* (the serial need) per item. Output: step_stats.json (this folder)."""
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import psutil

try:
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass
sys.path.insert(0, "D:/Latent")
import boxes.gsm8k_data as G  # noqa: E402
import tiktoken  # noqa: E402

ENC = tiktoken.get_encoding("gpt2")
OUT = os.path.dirname(os.path.abspath(__file__))
OPCH = set("+-*/()%")


def char_cats(e, tpl_eq):
    """per-character category of ' <<' + e + '>>': fmt / op / eq / sp / val / opd_q / opd_r / opd_k / opd_u (ungrounded: item not parsed) / oth."""
    s = " <<" + e + ">>"; cat = ["fmt"] * len(s); off = 3
    lhs, _, rhs = e.partition("=")
    for i, ch in enumerate(e):
        j = off + i
        if i < len(lhs):
            cat[j] = "op" if ch in OPCH else ("sp" if ch == " " else "opd_u" if (ch.isdigit() or ch in ".,") else "oth")
        elif i == len(lhs):
            cat[j] = "eq"
        else:
            cat[j] = "sp" if ch == " " else ("val" if (ch.isdigit() or ch in ".,") else "oth")
    if tpl_eq is not None:
        for (a, b, tok, kind, ref) in tpl_eq["ops"]:
            for i in range(a, b):
                cat[off + i] = "opd_" + kind
    return s, cat


def tok_cats(s, cat):
    ids = ENC.encode(s); out = []; pos = 0
    for t in ids:
        w = ENC.decode_single_token_bytes(t).decode("utf-8", errors="replace"); n = len(w)
        cs = [c for c in cat[pos:pos + n] if c != "sp"] or ["sp"]; pos += n
        u = sorted(set(cs)); out.append(u[0] if len(u) == 1 else "mixed:" + "+".join(u))
    assert pos == len(s), (s, pos)
    return ids, out


def dag_depth(tpl):
    d = []
    for t in tpl["eqt"]:
        rs = [o[4] for o in t["ops"] if o[3] == "r"]; d.append(1 + max((d[j] for j in rs), default=0))
    return d


def run(split, n_max, seed):
    R, _ = G.load_aug(split, filter_bad=(split == "train"))
    if n_max and len(R) > n_max:
        R = [R[i] for i in np.random.default_rng(seed).permutation(len(R))[:n_max]]
    tokc = Counter(); tokc_first = Counter(); n_items = len(R); n_parsed = 0; reasons = Counter(); m_hist = Counter(); dstar = Counter(); dstar_by_m = {}
    opd_kind = Counter(); n_ops_eq = Counter(); opv = Counter(); nq = Counter(); nq_used = Counter(); val_len = Counter(); opd_len = Counter()
    eq_has_r = Counter(); tok_len = []; first_opd_kind = Counter(); levels_hist = Counter(); maxwidth = Counter()
    for r in R:
        tpl, why = G.parse_recomputable(r); reasons[why] += 1; m = len(r["eqs"]); m_hist[m] += 1
        if tpl is not None:
            n_parsed += 1; d = dag_depth(tpl); dstar[max(d)] += 1; dstar_by_m.setdefault(m, Counter())[max(d)] += 1
            nq[len(tpl["qvals"])] += 1; nq_used[len(tpl["used_q"])] += 1
            lv = Counter(d); maxwidth[max(lv.values())] += 1
            for t_, e in zip(tpl["eqt"], r["eqs"]):
                kinds = [o[3] for o in t_["ops"]]; opd_kind.update(kinds); n_ops_eq[len(kinds)] += 1; eq_has_r["R" if "r" in kinds else "Q"] += 1
                if kinds:
                    first_opd_kind[kinds[0]] += 1
                for o in t_["ops"]:
                    opd_len[len(ENC.encode(o[2]))] += 1
                val_len[len(ENC.encode(t_["rtok"]))] += 1
        for t, e in enumerate(r["eqs"]):
            s, cat = char_cats(e, tpl["eqt"][t] if tpl is not None else None); ids, tc = tok_cats(s, cat); tc = tc + ["eot"]
            tokc.update(tc); tok_len.append(len(tc))
            first_nonfmt = next((c for c in tc if c != "fmt"), None); tokc_first[first_nonfmt] += 1
            lhs = e.partition("=")[0]
            opv.update(ch for ch in lhs if ch in "+-*/%"); opv["()"] += int("(" in lhs)
    tot = sum(tokc.values())
    share = {k: round(v / tot, 4) for k, v in sorted(tokc.items(), key=lambda kv: -kv[1])}
    grp = {"format (' <<', '>>', EOT, '=')": 0, "operator": 0, "operand_q (question literal)": 0, "operand_r (previous result)": 0, "operand_k (constant/word)": 0,
           "operand_unparsed_item": 0, "value (RHS)": 0, "mixed/other": 0}
    for k, v in tokc.items():
        key = ("format (' <<', '>>', EOT, '=')" if k in ("fmt", "eot", "eq") else "operator" if k == "op" else "operand_q (question literal)" if k == "opd_q" else
               "operand_r (previous result)" if k == "opd_r" else "operand_k (constant/word)" if k == "opd_k" else "operand_unparsed_item" if k == "opd_u" else
               "value (RHS)" if k == "val" else "mixed/other")
        grp[key] += v
    grp = {k: round(v / tot, 4) for k, v in grp.items()}
    np_ = max(1, n_parsed); ok = sum(opd_kind.values())
    return {"split": split, "n_items": n_items, "n_parsed_recomputable": n_parsed, "parse_reasons": dict(reasons), "m_hist": dict(sorted(m_hist.items())),
            "mean_m": float(np.mean([len(r["eqs"]) for r in R])), "mean_target_tokens_per_step_incl_eot": float(np.mean(tok_len)), "token_share_raw": share,
            "token_share_grouped": grp, "first_content_token_category": dict(tokc_first),
            "parsed_only": {"operand_kind_share": {k: round(v / ok, 4) for k, v in opd_kind.items()}, "first_operand_kind": dict(first_opd_kind),
                            "operands_per_equation": dict(sorted(n_ops_eq.items())), "eq_with_R_operand_share": round(eq_has_r["R"] / max(1, sum(eq_has_r.values())), 4),
                            "distinct_question_numbers": dict(sorted(nq.items())), "mean_distinct_question_numbers": float(sum(k * v for k, v in nq.items()) / np_),
                            "used_question_literals": dict(sorted(nq_used.items())), "operand_tokens_per_literal": dict(sorted(opd_len.items())),
                            "value_tokens": dict(sorted(val_len.items())), "dag_depth_dstar": dict(sorted(dstar.items())),
                            "mean_dstar": float(sum(k * v for k, v in dstar.items()) / np_), "max_level_width": dict(sorted(maxwidth.items())),
                            "dstar_by_m": {str(m): dict(sorted(c.items())) for m, c in sorted(dstar_by_m.items())}},
            "operator_vocab": dict(opv)}


if __name__ == "__main__":
    res = {"train_sample": run("train", 20000, 7), "test": run("test", 0, 0)}
    json.dump(res, open(os.path.join(OUT, "step_stats.json"), "w", newline="\n"), indent=1)
    for k, v in res.items():
        print(k, json.dumps({kk: v[kk] for kk in ("n_items", "n_parsed_recomputable", "mean_m", "mean_target_tokens_per_step_incl_eot", "token_share_grouped")}))
        print("   ", json.dumps(v["parsed_only"])[:1500])
        print("   ops", v["operator_vocab"], "first", v["first_content_token_category"])
