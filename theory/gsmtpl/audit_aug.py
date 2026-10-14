"""GSM-TEMPLATE draft, measurement 2 (CPU, read-only; Audit's asks 2026-10-13 22:4x): GSM8K-Aug equation audit.
  (1) every <<expr=val>> re-evaluated by a whitelisted-AST arithmetic evaluator (numbers, + - * / ** % //, unary +/-, parentheses; NEVER eval);
  (2) operand grounding (question literal / previous result / number word / common constant / ungrounded);
  (3) computation skeletons (numbers -> q_i / r_j / k_v / u) and their overlap Aug-train vs GSM8K test / train (structure-memorisation proxy);
  (4) the 'recomputable' share for number resampling; (5) a capitalised-name heuristic; (6) exact question overlap; (7) length residuals.
Writes aug_audit.json next to this script.   python -I audit_aug.py"""
import ast
import collections
import ctypes
import json
import math
import operator
import os
import re

try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
except Exception:
    pass
import tiktoken  # noqa: E402

ENC = tiktoken.get_encoding("gpt2")
HERE = os.path.dirname(os.path.abspath(__file__)); BD = "D:/Latent/bench/data/"
EQ = re.compile(r"<<([^>]*)>>"); NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])|(?<![\w.])\.\d+")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Mod: operator.mod,
       ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow}
WORDS = {"half": [0.5, 2], "twice": [2], "double": [2], "triple": [3], "thrice": [3], "dozen": [12], "dozens": [12], "quarter": [0.25, 4],
         "third": [3], "one": [1], "two": [2], "three": [3], "four": [4], "five": [5], "six": [6], "seven": [7], "eight": [8], "nine": [9], "ten": [10],
         "eleven": [11], "twelve": [12], "fifteen": [15], "twenty": [20], "thirty": [30], "forty": [40], "fifty": [50], "hundred": [100],
         "thousand": [1000], "percent": [100], "%": [100], "week": [7], "weeks": [7], "hour": [60], "hours": [60], "minutes": [60], "year": [12, 52, 365],
         "years": [12, 52, 365], "month": [30, 4], "months": [12, 30], "day": [24], "days": [24, 7], "cents": [100], "pair": [2], "pairs": [2]}
CONST = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 24, 30, 52, 60, 100, 1000, 365, 0.5, 0.25, 0.1, 0.01}


class Bad(Exception):
    pass


def safe_eval(s):
    """whitelisted arithmetic only; raises Bad on anything else (names, calls, attributes, huge powers)."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand); return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > 12 or abs(a) > 1e6):
                raise Bad("pow")
            try:
                return OPS[type(n.op)](a, b)
            except (ZeroDivisionError, OverflowError, ValueError):
                raise Bad("arith")
        raise Bad("node %s" % type(n).__name__)
    if len(s) > 200 or not re.fullmatch(r"[\d\s.+\-*/%()]+", s):
        raise Bad("chars")
    try:
        return ev(ast.parse(s, mode="eval"))
    except SyntaxError:
        raise Bad("syntax")


def fnum(t):
    return float(t.replace(",", ""))


def decimals(t):
    t = t.replace(",", ""); return len(t.split(".")[1]) if "." in t else 0


def check_eq(e):
    """'expr=val' -> (status, lhs value, rhs value, expr string). status: exact / round / wrong / unparsed."""
    e = e.replace(",", "").replace("$", "").strip()
    if e.count("=") != 1:
        return "unparsed", None, None, e
    lhs, rhs = (x.strip() for x in e.split("="))
    try:
        r = safe_eval(rhs) if rhs else None
        l_ = safe_eval(lhs)
    except Bad:
        return "unparsed", None, None, lhs
    if r is None:
        return "unparsed", None, None, lhs
    tol = 1e-9 * max(1.0, abs(r))
    if abs(l_ - r) <= tol:
        return "exact", l_, r, lhs
    d = decimals(rhs) if re.fullmatch(r"-?[\d.]+", rhs) else 6
    if abs(round(l_, d) - r) <= 1e-9 * max(1.0, abs(r)) or abs(l_ - r) <= 0.5 * 10 ** (-d) + 1e-9 or (d == 0 and abs(l_ - r) < 1.0 and abs(r) >= 1 and abs(l_ - r) / abs(r) < 0.02):
        return "round", l_, r, lhs
    return "wrong", l_, r, lhs


def q_numbers(q):
    out = []
    for m_ in NUM.finditer(q):
        try:
            out.append((fnum(m_.group(0)), m_.group(0)))
        except ValueError:
            pass
    return out


def near(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def analyse(q, eqs, ans):
    """-> dict for one item."""
    qn = q_numbers(q); qvals = []
    for v, _ in qn:
        if not any(near(v, w) for w in qvals):
            qvals.append(v)
    ql = q.lower(); wvals = set(v for w, vs in WORDS.items() if re.search(r"\b" + re.escape(w) + r"\b" if w != "%" else "%", ql) for v in vs)
    st, rhs_vals, sk, sh, ungrounded, cats = [], [], [], [], 0, collections.Counter()
    for e in eqs:
        s, l_, r, lhs = check_eq(e); st.append(s)
        if s == "unparsed":
            sk.append("?"); sh.append("?"); continue
        toks = []
        def rep(m_):
            nonlocal ungrounded
            v = fnum(m_.group(0))
            for j in range(len(rhs_vals) - 1, -1, -1):
                if near(v, rhs_vals[j]):
                    cats["R"] += 1; return "r%d" % (len(rhs_vals) - j)
            for i, w in enumerate(qvals):
                if near(v, w):
                    cats["Q"] += 1; return "q%d" % i
            if v in wvals:
                cats["W"] += 1; return "k%g" % v
            if v in CONST:
                cats["K"] += 1; return "k%g" % v
            ungrounded += 1; cats["U"] += 1; return "u"
        skel = re.sub(r"\d+(?:\.\d+)?|\.\d+", rep, lhs.replace(" ", ""))
        sk.append(skel); sh.append(re.sub(r"[qrku][\d.]*", "x", skel)); rhs_vals.append(r)
    try:
        a = fnum(ans.replace("$", "").strip().split()[0]) if ans.strip() else None
    except (ValueError, IndexError):
        a = None
    q_unique = all(sum(1 for _, t in qn if near(fnum(t), v)) == 1 for v in qvals)
    return {"st": st, "skel": ";".join(sk), "shape": ";".join(sh), "ungrounded": ungrounded, "cats": cats,
            "ans_last": (a is not None and rhs_vals and near(a, rhs_vals[-1])), "ans_any": (a is not None and any(near(a, r) for r in rhs_vals)),
            "ans_parsed": a is not None, "int_all": all(abs(r - round(r)) < 1e-9 for r in rhs_vals) if rhs_vals else True, "q_unique": q_unique}


def load_aug(name):
    Q, E, A = [], [], []
    for l in open(BD + "gsm8k_aug/" + name + ".txt", encoding="utf-8"):
        l = l.rstrip("\n")
        if "||" not in l or "####" not in l:
            continue
        qq, rest = l.split("||", 1); eqs, an = rest.rsplit("####", 1); Q.append(qq.strip()); E.append(EQ.findall(eqs)); A.append(an.strip())
    return Q, E, A


def load_openai(name):
    D = [json.loads(l) for l in open(BD + name + ".jsonl", encoding="utf-8") if l.strip()]
    return [x["question"].strip() for x in D], [EQ.findall(x["answer"]) for x in D], [x["answer"].split("####")[-1].strip() for x in D]


STOP = set("""The A An In On At If He She It They We You I His Her Their Our My Your This That These Those There Then When What How Each Every After Before
If Mr Mrs Ms Dr Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August September October November
December Christmas Halloween Easter Thanksgiving Valentine New York City Street Avenue Store School Market One Two Three Four Five Six Seven Eight Nine Ten
Also But And So For With From To Of By As Since While During Last Next First Second Third All Some Most Both Half Today Tomorrow Yesterday""".split())
CAP = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b")


def summarize(items, Q, name):
    n = len(items); neq = sum(len(x["st"]) for x in items); stc = collections.Counter(s for x in items for s in x["st"])
    cats = collections.Counter()
    for x in items:
        cats.update(x["cats"])
    tot = sum(cats.values())
    r = {"n_items": n, "n_equations": neq, "eq_status": {k: [v, round(v / neq, 5)] for k, v in stc.items()},
         "items_all_exact": round(sum(all(s == "exact" for s in x["st"]) for x in items) / n, 4),
         "items_exact_or_round": round(sum(all(s in ("exact", "round") for s in x["st"]) for x in items) / n, 4),
         "items_any_wrong": round(sum(any(s == "wrong" for s in x["st"]) for x in items) / n, 4),
         "items_any_unparsed": round(sum(any(s == "unparsed" for s in x["st"]) for x in items) / n, 4),
         "items_no_equation": round(sum(len(x["st"]) == 0 for x in items) / n, 4),
         "answer_eq_last_rhs": round(sum(bool(x["ans_last"]) for x in items) / n, 4), "answer_eq_any_rhs": round(sum(bool(x["ans_any"]) for x in items) / n, 4),
         "operand_categories": {k: [v, round(v / tot, 4)] for k, v in cats.items()},
         "items_with_ungrounded_operand": round(sum(x["ungrounded"] > 0 for x in items) / n, 4),
         "items_all_int_intermediates": round(sum(x["int_all"] for x in items) / n, 4)}
    rec = [len(x["st"]) > 0 and all(s == "exact" for s in x["st"]) and x["ungrounded"] == 0 and x["ans_last"] and x["q_unique"] for x in items]
    r["recomputable"] = round(sum(rec) / n, 4)
    r["recomputable_def"] = ("all equations parse and are exact; every operand is a question literal, a previous result, a number word in the question or a "
                             "common constant; answer == last equation's value; each question literal value occurs once in the question (unambiguous substitution)")
    capc = collections.Counter(); has = 0
    for qq in Q:
        c = [w for w in CAP.findall(qq) if w not in STOP]
        has += bool(c); capc.update(set(c))
    r["name_heuristic"] = {"items_with_capitalised_candidate": round(has / n, 4), "distinct_candidates": len(capc), "top30": capc.most_common(30)}
    return r, rec


def main():
    res = {}
    Qa, Ea, Aa = load_aug("train"); Ia = [analyse(q, e, a) for q, e, a in zip(Qa, Ea, Aa)]
    res["aug_train"], rec_a = summarize(Ia, Qa, "aug_train")
    sets = {}
    for nm, (Q, E, A) in (("gsm8k_train", load_openai("gsm8k_train")), ("gsm8k_test", load_openai("gsm8k_test")), ("aug_valid", load_aug("valid"))):
        I = [analyse(q, e, a) for q, e, a in zip(Q, E, A)]; res[nm], _ = summarize(I, Q, nm); sets[nm] = (Q, E, A, I)
    # skeleton overlap (structure proxy)
    skA = collections.Counter(x["skel"] for x in Ia); shA = collections.Counter(x["shape"] for x in Ia)
    Itr = sets["gsm8k_train"][3]; Ite = sets["gsm8k_test"][3]
    skT = collections.Counter(x["skel"] for x in Itr)
    ov = {}
    for nm, I in (("gsm8k_test", Ite), ("gsm8k_train", Itr), ("aug_valid", sets["aug_valid"][3])):
        ov[nm] = {"skel_seen_in_aug_train": round(sum(x["skel"] in skA for x in I) / len(I), 4),
                  "shape_seen_in_aug_train": round(sum(x["shape"] in shA for x in I) / len(I), 4),
                  "skel_seen_in_gsm8k_train": round(sum(x["skel"] in skT for x in I) / len(I), 4)}
    cum = 0; tot = len(Ia); cover = {}
    for k_, (s, c) in enumerate(skA.most_common(), 1):
        cum += c
        for f in (0.5, 0.9):
            if f not in cover and cum >= f * tot:
                cover[f] = k_
    res["skeletons"] = {"aug_train_distinct_skel": len(skA), "aug_train_distinct_shape": len(shA), "gsm8k_train_distinct_skel": len(skT),
                        "aug_train_skel_count_covering_50pct": cover.get(0.5), "aug_train_skel_count_covering_90pct": cover.get(0.9),
                        "overlap": ov, "top10_aug_skel": skA.most_common(10),
                        "skel_def": "each equation's LHS with numbers -> q<i> (i-th distinct question number), r<j> (result of the j-th previous equation), "
                                    "k<v> (number word / common constant), u (ungrounded); equations joined by ';'. shape = skel with all symbols -> x"}
    # exact question overlap
    sA = set(Qa)
    res["question_overlap"] = {nm: {"exact_in_aug_train": sum(q in sA for q in sets[nm][0])} for nm in ("gsm8k_train", "gsm8k_test", "aug_valid")}
    # draws per distinct item
    res["draws"] = {f"{S // 1000}k_steps_bs64": {"draws": S * 64, "epochs_aug_train": round(S * 64 / len(Qa), 2), "epochs_gsm8k_train_7473": round(S * 64 / 7473, 1),
                                               "epochs_p8_na_train_11985": round(S * 64 / 11985, 1)} for S in (12000, 24000, 48000)}
    # length residuals: test think_L vs Aug-train mean think_L at the same equation count m
    La = [1 + len(t) for t in ENC.encode_ordinary_batch(Qa, num_threads=3)]; ma = [len(e) for e in Ea]
    byM = collections.defaultdict(list)
    for L, m in zip(La, ma):
        byM[m].append(L)
    mu = {m: sum(v) / len(v) for m, v in byM.items()}
    xs = [m for m in ma if 1 <= m <= 8]; ys = [L for L, m in zip(La, ma) if 1 <= m <= 8]; mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    Qt, Et = sets["gsm8k_test"][0], sets["gsm8k_test"][1]; Lt = [1 + len(t) for t in ENC.encode_ordinary_batch(Qt, num_threads=3)]
    resid = [L - mu.get(len(e), my) for L, e in zip(Lt, Et)]; rs = sorted(resid); n = len(rs)
    t1, t2 = rs[n // 3], rs[2 * n // 3]
    terc = [[r for r in resid if r < t1], [r for r in resid if t1 <= r < t2], [r for r in resid if r >= t2]]
    res["length_residual_test_vs_aug"] = {"aug_tokens_per_equation_ols_m1to8": round(slope, 3), "aug_mean_L_by_m": {m: round(mu[m], 2) for m in sorted(mu)},
                                          "test_resid_mean": round(sum(resid) / n, 2), "test_resid_median": rs[n // 2], "tercile_cuts": [round(t1, 2), round(t2, 2)],
                                          "tercile_mean_resid": [round(sum(t) / len(t), 2) for t in terc], "tercile_n": [len(t) for t in terc],
                                          "tercile_mean_resid_in_equations": [round(sum(t) / len(t) / slope, 2) for t in terc],
                                          "frac_test_resid_gt_0": round(sum(r > 0 for r in resid) / n, 4)}
    Lv = [1 + len(t) for t in ENC.encode_ordinary_batch(sets["aug_valid"][0], num_threads=3)]
    rv = [L - mu.get(len(e), my) for L, e in zip(Lv, sets["aug_valid"][1])]
    res["length_residual_test_vs_aug"]["aug_valid_resid_mean"] = round(sum(rv) / len(rv), 2)
    # recomputable x length: is the recomputable subset representative?
    Lr = [L for L, r in zip(La, rec_a) if r]; mr = [m for m, r in zip(ma, rec_a) if r]
    res["aug_train"]["recomputable_subset"] = {"n": len(Lr), "mean_think_L": round(sum(Lr) / max(1, len(Lr)), 2), "mean_m": round(sum(mr) / max(1, len(mr)), 3),
                                               "all_mean_think_L": round(sum(La) / len(La), 2), "all_mean_m": round(sum(ma) / len(ma), 3)}
    # examples of wrong equations (first 8) for the record
    ex = []
    for q, e, x in zip(Qa, Ea, Ia):
        if "wrong" in x["st"]:
            ex.append({"eqs": e, "st": x["st"]})
        if len(ex) >= 8:
            break
    res["aug_train"]["wrong_examples"] = ex
    json.dump(res, open(os.path.join(HERE, "aug_audit.json"), "w", newline="\n"), indent=1, default=lambda o: dict(o) if isinstance(o, collections.Counter) else str(o))
    print(json.dumps(res, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
