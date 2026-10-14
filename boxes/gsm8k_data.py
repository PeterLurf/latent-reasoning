"""THEORY §200 GSM8K-AUG PILOT (58947b26; design docs/THEORY_GSM8K_TEMPLATE_DRAFT_2026-10-13.md §3): the data layer.
* load_aug(split): bench/data/gsm8k_aug/<split>.txt lines "question||<<e_1>> .. <<e_m>> #### answer" -> raw dicts {q, eqs (inner 'expr=val' strings), ans}.
  Train is filtered (filter_bad_eq): any item with a wrong or unparsable equation is dropped (theory/gsmtpl/audit_aug.py's whitelisted-AST evaluator,
  copied here - that module lowers the process priority on import). Test = GSM8K's 1,319 (gsm8k_aug/test.txt, the same questions), valid = valid.txt.
* encode(raw) -> ProsMem's item schema (boxes/prosqa168s2): q / pre ([question] / [question + "\\n"] ids), a1 ("### " + answer ids), depth = m,
  step_ids (" <<e_t>>" ids, the aux target of pass t), cot (the TEXT-COT lines "<<e_t>>\\n").
* R_nu relabel_names (Theorem N): the item's person names (whitelist: s200_names.json, the top 250 per gender in Aug train by pronoun co-occurrence)
  mapped injectively into the same-gender pool, word-boundary consistent (possessives included); equations / answer untouched (A0 asserted).
* R_x resample (Theorem X): for a RECOMPUTABLE item (every equation exact; every operand grounded as exactly one of: a question literal, a previous
  result, a number word / constant; answer == the last result; each question literal value unique), the used question literals are redrawn with the
  same integer-digit count, decimals and integrality, distinct from every other number in the question and from the item's constants; equations are
  recomputed in order and a draw is rejected on any error, a sign flip from the original, or a result not exactly representable at the original
  result's decimals (so integrality is kept and every rendered equation evaluates exactly). Up to 50 tries, else the original. Rendering keeps every
  number token's original style (commas, decimals, a leading '.'), so the identity draw reproduces the item exactly (checked at parse; failures are
  not resampled and counted).
Test / valid items are never transformed (relabel_names / resample assert raw["_split"] == "train")."""
import ast
import json
import operator
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUG = os.path.join(ROOT, "bench", "data", "gsm8k_aug")
EQ = re.compile(r"<<([^>]*)>>")
NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])|(?<![\w.])\.\d+")      # audit_aug.NUM (question literals)
NUMTOK = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")                                   # number tokens inside an equation side
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Mod: operator.mod,
       ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow}
WORDS = {"half": [0.5, 2], "twice": [2], "double": [2], "triple": [3], "thrice": [3], "dozen": [12], "dozens": [12], "quarter": [0.25, 4],
         "third": [3], "one": [1], "two": [2], "three": [3], "four": [4], "five": [5], "six": [6], "seven": [7], "eight": [8], "nine": [9], "ten": [10],
         "eleven": [11], "twelve": [12], "fifteen": [15], "twenty": [20], "thirty": [30], "forty": [40], "fifty": [50], "hundred": [100],
         "thousand": [1000], "percent": [100], "%": [100], "week": [7], "weeks": [7], "hour": [60], "hours": [60], "minutes": [60], "year": [12, 52, 365],
         "years": [12, 52, 365], "month": [30, 4], "months": [12, 30], "day": [24], "days": [24, 7], "cents": [100], "pair": [2], "pairs": [2]}
CONST = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 24, 30, 52, 60, 100, 1000, 365, 0.5, 0.25, 0.1, 0.01}                 # audit_aug.WORDS / CONST
TRIES = 50


class Bad(Exception):
    pass


def safe_eval(s):
    """audit_aug.safe_eval: whitelisted arithmetic only (numbers, + - * / % // **, unary signs, parentheses); raises Bad on anything else."""
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


def decimals(t):
    t = t.replace(",", ""); return len(t.split(".")[1]) if "." in t else 0


def fnum(t):
    return float(t.replace(",", ""))


def check_eq(e):
    """audit_aug.check_eq: 'expr=val' -> status exact / round / wrong / unparsed."""
    e = e.replace(",", "").replace("$", "").strip()
    if e.count("=") != 1:
        return "unparsed"
    lhs, rhs = (x.strip() for x in e.split("="))
    try:
        r = safe_eval(rhs) if rhs else None
        l_ = safe_eval(lhs)
    except Bad:
        return "unparsed"
    if r is None:
        return "unparsed"
    if abs(l_ - r) <= 1e-9 * max(1.0, abs(r)):
        return "exact"
    d = decimals(rhs) if re.fullmatch(r"-?[\d.]+", rhs) else 6
    if abs(round(l_, d) - r) <= 1e-9 * max(1.0, abs(r)) or abs(l_ - r) <= 0.5 * 10 ** (-d) + 1e-9 or (d == 0 and abs(l_ - r) < 1.0 and abs(r) >= 1 and abs(l_ - r) / abs(r) < 0.02):
        return "round"
    return "wrong"


def load_aug(split, filter_bad=None):
    """-> (raw items, filter counts). filter_bad (default: train only) drops items with any 'wrong' / 'unparsed' equation."""
    filter_bad = (split == "train") if filter_bad is None else filter_bad; out = []; cnt = {"lines": 0, "kept": 0, "wrong": 0, "unparsed": 0, "malformed": 0}
    for l in open(os.path.join(AUG, split + ".txt"), encoding="utf-8"):
        l = l.rstrip("\n")
        if not l.strip():
            continue
        cnt["lines"] += 1
        if "||" not in l or "####" not in l:
            cnt["malformed"] += 1; continue
        q, rest = l.split("||", 1); eqs, an = rest.rsplit("####", 1); eqs = EQ.findall(eqs); st = [check_eq(e) for e in eqs]
        if filter_bad and ("wrong" in st or "unparsed" in st):
            cnt["wrong" if "wrong" in st else "unparsed"] += 1; continue
        out.append({"q": q.strip(), "eqs": eqs, "ans": an.strip(), "_split": split, "_line": l}); cnt["kept"] += 1
    return out, cnt


def render_line(raw):
    """the inverse of load_aug's parse for the data round-trip gate: question||<<e_1>> .. <<e_m>> #### answer."""
    return "%s||%s #### %s" % (raw["q"], " ".join("<<%s>>" % e for e in raw["eqs"]), raw["ans"])


def encode(raw, ENC, EOS):
    """raw -> ProsMem's item schema (+ m, ans, eqs, cot)."""
    q = raw["q"]; eqs = raw["eqs"]
    return {"q": ENC.encode(q), "pre": ENC.encode(q + "\n"), "a1": ENC.encode("### " + raw["ans"]), "depth": len(eqs), "m": len(eqs), "ans": raw["ans"],
            "eqs": list(eqs), "step_ids": [ENC.encode(" <<%s>>" % e) for e in eqs], "cot": [t for e in eqs for t in ENC.encode("<<%s>>\n" % e)]}


# ---------------------------------------------------------------- R_nu: names
CAPW = re.compile(r"\b([A-Z][a-z]{2,})\b")


def load_names(path=None):
    d = json.load(open(path or os.path.join(AUG, "s200_names.json"), encoding="utf-8")); return {"male": list(d["male"]), "female": list(d["female"])}


def relabel_names(raw, pools, rng):
    """R_nu on a TRAIN raw item -> (new raw, n_names). Every whitelist name in the question gets a distinct same-gender pool name not already a
    capitalised word of the question; one regex pass (word boundary: possessives keep their 's)."""
    assert raw.get("_split") == "train", "§200: only training items are transformed"
    gender = pools.get("_gender") or {**{n: "male" for n in pools["male"]}, **{n: "female" for n in pools["female"]}}; pools["_gender"] = gender
    caps = set(CAPW.findall(raw["q"])); names = sorted(w for w in caps if w in gender)
    if not names:
        return raw, 0
    mp = {}
    for gd in ("male", "female"):
        src = [w for w in names if gender[w] == gd]
        if src:
            avail = [w for w in pools[gd] if w not in caps]; pick = rng.choice(len(avail), len(src), replace=False); mp.update({w: avail[i] for w, i in zip(src, pick)})
    pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(mp, key=len, reverse=True))) + r")\b")
    y = dict(raw); y["q"] = pat.sub(lambda m_: mp[m_.group(1)], raw["q"]); return y, len(mp)


# ---------------------------------------------------------------- R_x: numbers
def fmt(v, like):
    """v rendered in the style of the number token `like` (its decimals, thousands commas, a leading '.')."""
    d = decimals(like); s = ("{:,.%df}" if "," in like else "{:.%df}") % d
    s = s.format(v)
    if like.startswith(".") and s.startswith("0."):
        s = s[1:]
    return s


def _near(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def parse_recomputable(raw):
    """-> a resampling template, or (None, reason). Stricter than audit_aug's 'recomputable' by one rule: an operand matching more than one of
    {a previous result, a question literal} or a question literal that is also a number-word value of the question is ambiguous -> not resampled."""
    q = raw["q"]; qn = [(m_.start(), m_.end(), m_.group(0), fnum(m_.group(0))) for m_ in NUM.finditer(q)]
    qvals = []
    for _, _, _, v in qn:
        if not any(_near(v, w) for w in qvals):
            qvals.append(v)
    if not all(sum(1 for _, _, _, v in qn if _near(v, w)) == 1 for w in qvals):
        return None, "q_not_unique"
    ql = q.lower(); wvals = set(v for w, vs in WORDS.items() if re.search(r"\b" + re.escape(w) + r"\b" if w != "%" else "%", ql) for v in vs)
    if not raw["eqs"]:
        return None, "no_equation"
    eqt, rvals, used_q = [], [], set()
    for e in raw["eqs"]:
        if check_eq(e) != "exact" or e.count("=") != 1:
            return None, "not_exact"
        lhs, rhs = e.split("=")
        if not re.fullmatch(r"\s*[\d,]*\.?\d+\s*", rhs) or "$" in e:
            return None, "rhs_or_dollar"
        ops = []
        for m_ in NUMTOK.finditer(lhs):
            v = fnum(m_.group(0)); rj = [j for j in range(len(rvals) - 1, -1, -1) if _near(v, rvals[j])]; qi = [i for i, w in enumerate(qvals) if _near(v, w)]
            if rj and qi:
                return None, "ambiguous_r_q"
            if qi and (v in wvals):
                return None, "ambiguous_q_word"
            if rj:
                ops.append((m_.start(), m_.end(), m_.group(0), "r", rj[0]))
            elif qi:
                ops.append((m_.start(), m_.end(), m_.group(0), "q", qi[0])); used_q.add(qi[0])
            elif v in wvals or v in CONST:
                ops.append((m_.start(), m_.end(), m_.group(0), "k", v))
            else:
                return None, "ungrounded"
        r_tok = rhs.strip(); eqt.append({"lhs": lhs, "rhs": rhs, "rtok": r_tok, "ops": ops, "r0": fnum(r_tok)}); rvals.append(fnum(r_tok))
    am = list(NUMTOK.finditer(raw["ans"]))
    if len(am) != 1 or not _near(fnum(am[0].group(0)), rvals[-1]):
        return None, "answer_not_last"
    if not used_q:
        return None, "no_q_literal"
    tpl = {"qn": qn, "qvals": qvals, "used_q": sorted(used_q), "eqt": eqt, "ans_tok": am[0], "consts": sorted({o[4] for t in eqt for o in t["ops"] if o[3] == "k"} | wvals)}
    if render(raw, tpl, {i: qvals[i] for i in tpl["used_q"]})[0] != (raw["q"], raw["eqs"], raw["ans"]):
        return None, "identity_render"
    return tpl, "ok"


def _fmt_exact(v, tok):
    """fmt(v, tok) when it renders v exactly (else Bad: the draw is rejected, e.g. a decimal result in an integer-styled operand slot)."""
    s = fmt(v, tok)
    if abs(fnum(s) - v) > 1e-9 * max(1.0, abs(v)):
        raise Bad("format_loss")
    return s


def render(raw, tpl, newq):
    """the item with question literals newq {q index: value} -> ((q, eqs, ans), values) or ((None, reason), None) when a constraint fails. Every
    rendered number must equal its value exactly (_fmt_exact), so every rendered equation evaluates exactly."""
    try:
        return _render(raw, tpl, newq)
    except Bad as e_:
        return (None, str(e_)), None


def _render(raw, tpl, newq):
    vals = []; eqs = []
    for t in tpl["eqt"]:
        ex, out, last = [], [], 0
        for s, e_, tok, kind, ref in t["ops"]:
            v = newq[ref] if kind == "q" else (vals[ref] if kind == "r" else ref)
            out.append(t["lhs"][last:s]); out.append(_fmt_exact(v, tok)); last = e_
            ex.append((s, e_, repr(float(v))))
        out.append(t["lhs"][last:])
        expr = "".join(t["lhs"][a:b] if k is None else k for a, b, k in _segments(t["lhs"], ex))
        try:
            v_new = safe_eval(expr.replace(",", ""))
        except Bad:
            return (None, "eval_error"), None
        d = decimals(t["rtok"])
        if abs(v_new - round(v_new, d)) > 1e-9 * max(1.0, abs(v_new)):
            return (None, "not_representable"), None
        if (v_new < 0) != (t["r0"] < 0):
            return (None, "sign"), None
        v_new = round(v_new, d); vals.append(v_new); eqs.append("".join(out) + "=" + t["rhs"].replace(t["rtok"], _fmt_exact(v_new, t["rtok"]), 1))
    q = raw["q"]; parts, last = [], 0
    for s, e_, tok, v in tpl["qn"]:
        i = next(i for i, w in enumerate(tpl["qvals"]) if _near(v, w))
        parts.append(q[last:s]); parts.append(_fmt_exact(newq[i], tok) if i in newq else tok); last = e_
    parts.append(q[last:]); at = tpl["ans_tok"]; ans = raw["ans"][:at.start()] + _fmt_exact(vals[-1], at.group(0)) + raw["ans"][at.end():]
    return ("".join(parts), eqs, ans), vals


def _segments(s, ex):
    """[(start, end, replacement or None)] covering s, the number spans in ex replaced (by full-precision values for evaluation)."""
    out, last = [], 0
    for a, b, rep in ex:
        out.append((last, a, None)); out.append((a, b, rep)); last = b
    out.append((last, len(s), None)); return out


def draw_literal(tok, rng):
    """a fresh value with tok's integer-digit count and decimals (a nonzero last decimal; a nonzero leading digit when the integer part has > 1 digit)."""
    t = tok.replace(",", ""); ip, _, fp = t.partition("."); d = len(fp)
    if ip in ("", "0"):                                                      # ".5" / "0.25": integer part stays 0
        iv = 0
    elif len(ip) == 1:
        iv = int(rng.integers(1, 10))
    else:
        iv = int(rng.integers(10 ** (len(ip) - 1), 10 ** len(ip)))
    if not d:
        return float(iv)
    fv = int(rng.integers(0, 10 ** d)); fv = fv - fv % 10 + int(rng.integers(1, 10)) if fv % 10 == 0 else fv
    return round(iv + fv / 10 ** d, d)


def resample(raw, tpl, rng, test_q=None):
    """R_x on a TRAIN raw item with its template -> (new raw, accepted). New literals are distinct from each other, from every other number in the
    question and from the item's constants; up to TRIES draws, else the original. test_q: the set of test / valid questions (asserted never produced)."""
    assert raw.get("_split") == "train", "§200: only training items are transformed"
    other = [v for _, _, _, v in tpl["qn"]]
    for _ in range(TRIES):
        newq = {}
        for i in tpl["used_q"]:
            tok = next(tk for _, _, tk, v in tpl["qn"] if _near(v, tpl["qvals"][i])); newq[i] = draw_literal(tok, rng)
        nv = list(newq.values())
        if len(set(nv)) != len(nv) or any(_near(v, w) for v in nv for w in other if not any(_near(w, tpl["qvals"][i]) for i in newq)) \
                or any(_near(v, c) for v in nv for c in tpl["consts"]):
            continue
        (res, _vals) = render(raw, tpl, newq)
        if res[0] is None:
            continue
        y = dict(raw); y["q"], y["eqs"], y["ans"] = res
        if test_q is not None:
            assert y["q"] not in test_q, "§200: a resampled training question equals a test / valid question"
        return y, True
    return raw, False
