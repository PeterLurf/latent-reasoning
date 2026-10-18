"""PROOFS_LANG item 5: independent GSM8K-Aug / test structure + DECOMP sizing (shares no code with gsm_stats.py /
gsm_mbin.py). Per item: m = number of <<...>> equations; per equation an ast parse of the LHS (commas removed; 'x' read
as '*' as gsm_mbin does, and separately WITHOUT that rewrite), binary-op count, exact re-evaluation vs the RHS.
DECOMP-parse(item) = every equation parses into + - * / BinOps over numeric literals (unary minus allowed), >= 1 op.
Provenance by value for every leaf of every parsed equation: earlier step result (internal sub-steps included) /
question literal (unique / non-unique) / both (ambiguous) / neither (CONST).
    python -B gsm_check.py <out.json> [--train_cap 60000]
"""
import argparse
import ast
import collections
import json
import re
import statistics

NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def load(p, cap=None):
    out = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if cap and len(out) >= cap:
                break
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "||" not in line or "####" not in line:
                out.append(None); continue
            q, rest = line.split("||", 1); cot, ans = rest.rsplit("####", 1)
            out.append((q, re.findall(r"<<([^>]*)>>", cot), ans.strip()))
    return out


def tree(expr, xrewrite=True):
    s = expr.replace(",", "")
    if xrewrite:
        s = s.replace("x", "*")
    try:
        t = ast.parse(s, mode="eval").body
    except Exception:  # noqa: BLE001
        return None
    steps = []  # (op, a, b, value) post-order

    def ev(nd):
        if isinstance(nd, ast.Constant) and isinstance(nd.value, (int, float)):
            return float(nd.value), ("leaf", float(nd.value))
        if isinstance(nd, ast.UnaryOp) and isinstance(nd.op, ast.USub) and isinstance(nd.operand, ast.Constant) and isinstance(nd.operand.value, (int, float)):
            return -float(nd.operand.value), ("leaf", -float(nd.operand.value))
        if isinstance(nd, ast.BinOp) and isinstance(nd.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a = ev(nd.left); b = ev(nd.right)
            if a is None or b is None:
                return None
            op = type(nd.op).__name__
            try:
                v = {"Add": a[0] + b[0], "Sub": a[0] - b[0], "Mult": a[0] * b[0], "Div": a[0] / b[0]}[op]
            except ZeroDivisionError:
                return None
            steps.append((op, a[1], b[1], v))
            return v, ("step", len(steps) - 1)
        return None
    r = ev(t)
    if r is None:
        return None
    return r[0], steps


def lits(q):
    return [float(s.replace(",", "")) for s in NUM.findall(q)]


def analyse(data, name, tok=None):
    rec = {"items": len(data), "malformed": sum(d is None for d in data)}
    d = [x for x in data if x is not None]
    M = collections.Counter(len(e) for _, e, _ in d); rec["m_dist"] = dict(sorted(M.items()))
    single = 0; vals = []; parsed = 0; parsed_nox = 0; mbin = collections.Counter(); m_of_parsed = []; exact_bad = 0; zero_op_eq = 0
    prov = collections.Counter(); maxlit = collections.Counter(); nlits = []; mbin_gt12 = 0; eq_total = 0
    for q, eqs, a in d:
        L = lits(q); nlits.append(len(L)); maxlit[min(len(L), 20)] += 1
        lit_count = collections.Counter(L)
        ok = bool(eqs); okx = bool(eqs); tot = 0; all1 = True; earlier = []
        trees = []
        for e in eqs:
            eq_total += 1
            if "=" not in e:
                ok = okx = False; all1 = False; trees.append(None); continue
            lhs, rhs = e.rsplit("=", 1)
            t = tree(lhs); tn = tree(lhs, xrewrite=False)
            if tn is None or len(tn[1]) == 0:
                okx = False
            if t is None:
                ok = False; all1 = False; trees.append(None); continue
            nops = len(t[1])
            if nops == 0:
                zero_op_eq += 1; ok = False
            if nops != 1:
                all1 = False
            tot += nops; trees.append(t)
            try:
                rv = float(rhs.replace(",", "").strip()); vals.append(rv)
                if abs(t[0] - rv) > 1e-6:
                    exact_bad += 1
            except Exception:  # noqa: BLE001
                pass
        single += all1
        if ok:
            parsed += 1; mbin[tot] += 1; m_of_parsed.append(len(eqs)); mbin_gt12 += tot > 12
            for t in trees:
                for op, A, B, v in t[1]:
                    for kind, val in (A, B):
                        if kind != "leaf":
                            prov["internal_step"] += 1; continue
                        er = any(abs(val - u) < 1e-9 for u in earlier); nl = lit_count.get(val, 0)
                        if er and nl:
                            prov["ambiguous_both"] += 1
                        elif er:
                            prov["earlier_result"] += 1
                        elif nl == 1:
                            prov["literal_unique"] += 1
                        elif nl > 1:
                            prov["literal_nonunique"] += 1
                        else:
                            prov["const_or_unmatched"] += 1
                    earlier.append(v)
        parsed_nox += okx
    n = len(d)
    rec["single_op_only_share"] = round(single / n, 4)
    rec["decomp_parse_share"] = round(parsed / n, 4); rec["decomp_parse_share_no_x_rewrite"] = round(parsed_nox / n, 4)
    rec["n_parsed"] = parsed; rec["zero_op_equations"] = zero_op_eq; rec["equations"] = eq_total
    rec["exact_eval_mismatch_equations"] = exact_bad
    cum = 0; cs = {}
    for k in sorted(mbin):
        cum += mbin[k]; cs[k] = round(cum / parsed, 4)
    rec["mbin_dist"] = dict(sorted(mbin.items())); rec["mbin_cum"] = cs
    rec["mbin_mean_parsed"] = round(sum(k * v for k, v in mbin.items()) / parsed, 4); rec["mbin_sum_parsed"] = sum(k * v for k, v in mbin.items())
    rec["m_mean_parsed"] = round(statistics.mean(m_of_parsed), 4); rec["m_mean_all"] = round(sum(k * v for k, v in M.items()) / n, 4)
    rec["mbin_gt12_share_parsed"] = round(mbin_gt12 / parsed, 5)
    ptot = sum(prov.values()); rec["provenance_share"] = {k: round(v / ptot, 4) for k, v in sorted(prov.items())}
    dig = sorted(len(str(int(abs(v)))) for v in vals)
    rec["values"] = {"n": len(vals), "nonint": round(sum(v != int(v) for v in vals) / len(vals), 4), "neg": round(sum(v < 0 for v in vals) / len(vals), 4),
                     "digits_p50": dig[len(dig) // 2], "digits_p99": dig[int(0.99 * len(dig))], "digits_max": dig[-1], "ge_1e7": sum(abs(v) >= 1e7 for v in vals),
                     "not_exact_2dp": sum(abs(round(v, 2) - v) > 1e-9 for v in vals)}
    rec["literals_per_q"] = {"mean": round(statistics.mean(nlits), 3), "max": max(nlits), "gt14": sum(x > 14 for x in nlits)}
    if tok is not None:
        T = sorted(len(tok(q)["input_ids"]) for q, _, _ in d[:3000])
        rec["q_tokens"] = {"n": len(T), "p50": T[len(T) // 2], "p90": T[int(0.9 * len(T))], "max": T[-1]}
    print("==", name, json.dumps(rec)[:3000], flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--train_cap", type=int, default=60000)
    a = ap.parse_args()
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception:  # noqa: BLE001
        tok = None
    out = {}
    out["test"] = analyse(load("D:/Latent/bench/data/gsm8k_aug/test.txt"), "test", tok)
    out["train60k"] = analyse(load("D:/Latent/bench/data/gsm8k_aug/train.txt", a.train_cap), "train60k")
    json.dump(out, open(a.out, "w", newline="\n"), indent=1)
    out["train_all"] = analyse(load("D:/Latent/bench/data/gsm8k_aug/train.txt"), "train_all")
    json.dump(out, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
