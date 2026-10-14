"""GSM-TEMPLATE draft, measurement 3: a small TemplateGSM (math-ai/TemplateGSM, config templategsm-7473-1k) sample via the HF datasets-server rows API
(JSON only, saved under dl_tgsm/; untrusted data: solution_code is NEVER executed; it is parsed with ast and evaluated by a whitelisted evaluator).
Per sampled row: template_id -> GSM8K train[template_id] (the prototype), token lengths, the number of computed assignments vs the prototype's equation count,
'intermediate leak' (a computed non-final value printed in the problem text), result vs the code's value.   python -I tgsm_sample.py [n_templates]"""
import ast
import ctypes
import json
import operator
import os
import random
import re
import sys
import time
import urllib.request

try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
except Exception:
    pass
import tiktoken  # noqa: E402

ENC = tiktoken.get_encoding("gpt2"); HERE = os.path.dirname(os.path.abspath(__file__)); DL = os.path.join(HERE, "dl_tgsm")
API = "https://datasets-server.huggingface.co/rows?dataset=math-ai/TemplateGSM&config=templategsm-7473-1k&split=train&offset=%d&length=%d"
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Mod: operator.mod,
       ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow}
EQ = re.compile(r"<<([^>]*)>>"); NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")


LEAKS = []


class Bad(Exception):
    pass


def code_values(src):
    """straight-line 'name = arithmetic over numbers / earlier names' only -> [(name, value, is_literal)]; anything else -> Bad."""
    env, out = {}, []
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id in env:
            return env[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand); return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 12:
                raise Bad("pow")
            try:
                return OPS[type(n.op)](a, b)
            except (ZeroDivisionError, OverflowError):
                raise Bad("arith")
        raise Bad(type(n).__name__)
    for st in ast.parse(src).body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            v = ev(st.value); env[st.targets[0].id] = v; out.append((st.targets[0].id, v, not isinstance(st.value, ast.BinOp)))
        else:
            raise Bad("stmt " + type(st).__name__)
    return out


def fetch(off, n):
    p = os.path.join(DL, "rows_%d_%d.json" % (off, n))
    if not os.path.exists(p):
        with urllib.request.urlopen(API % (off, n), timeout=40) as r:
            open(p, "wb").write(r.read())
        time.sleep(0.5)
    return json.load(open(p, encoding="utf-8"))


def main():
    nt = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    os.makedirs(DL, exist_ok=True); rng = random.Random(19300)
    gtr = [json.loads(l) for l in open("D:/Latent/bench/data/gsm8k_train.jsonl", encoding="utf-8") if l.strip()]
    offs = sorted(rng.sample(range(7473), nt))
    rows = []
    for j in offs:
        try:
            d = fetch(j * 1000 + rng.randrange(990), 3)
        except Exception as e:  # noqa: BLE001
            print("fetch failed", j, e); continue
        rows += [r["row"] for r in d.get("rows", [])]
    recs = []
    for r in rows:
        t = int(r["template_id"]); src = gtr[t] if 0 <= t < len(gtr) else None
        rec = {"template_id": t, "problem_id": r.get("problem_id"), "L_problem": 1 + len(ENC.encode_ordinary(r["problem"])),
               "L_proto": 1 + len(ENC.encode_ordinary(src["question"])) if src else None, "m_proto": len(EQ.findall(src["answer"])) if src else None}
        try:
            cv = code_values(r["solution_code"]); comp = [(n, v) for n, v, lit in cv if not lit]
            rec["n_computed"] = len(comp); fin = cv[-1][1] if cv else None
            pn = [float(x.replace(",", "")) for x in NUM.findall(r["problem"])]
            inter = [v for n, v in comp if n != "result"][:-1] if comp else []
            lits = [v for n, v, lit in cv if lit]
            lk = [v for v in inter if abs(v) >= 3 and not any(abs(v - w) <= 1e-9 for w in lits) and any(abs(v - w) <= 1e-6 * max(1, abs(v)) for w in pn)]
            rec["intermediate_leak"] = bool(lk); rec["leaked"] = lk[:3]
            if lk and len(LEAKS) < 6:
                LEAKS.append({"t": t, "problem": r["problem"][:400], "leaked": lk[:3], "proto": src["question"][:300] if src else None})
            try:
                rec["result_matches_code"] = abs(float(r["result"]) - fin) <= 1e-6 * max(1, abs(fin))
            except (TypeError, ValueError):
                rec["result_matches_code"] = None
            rec["result_integer"] = fin is not None and abs(fin - round(fin)) < 1e-9
        except (Bad, SyntaxError) as e:
            rec["n_computed"] = None; rec["code_unparsed"] = str(e)[:40]
        recs.append(rec)
    ok = [x for x in recs if x.get("n_computed") is not None]
    def mean(xs):
        xs = [x for x in xs if x is not None]; return round(sum(xs) / len(xs), 3) if xs else None
    s = {"n_rows": len(recs), "n_templates": len({x["template_id"] for x in recs}), "code_parsed": len(ok),
         "mean_L_problem": mean([x["L_problem"] for x in recs]), "mean_L_proto": mean([x["L_proto"] for x in recs]),
         "mean_L_ratio": mean([x["L_problem"] / x["L_proto"] for x in recs if x["L_proto"]]),
         "mean_n_computed": mean([x["n_computed"] for x in ok]), "mean_m_proto_parsed_rows": mean([x["m_proto"] for x in ok]),
         "n_computed_def": "assignments whose RHS is an arithmetic BinOp (aliases / literals excluded)", "frac_n_computed_eq_m_proto": mean([float(x["n_computed"] == x["m_proto"]) for x in ok]),
         "frac_n_computed_lt_m_proto": mean([float(x["n_computed"] < x["m_proto"]) for x in ok]),
         "leak_def": "a computed non-final value (|v| >= 3, not equal to any code literal) printed as a number in the problem text", "frac_intermediate_leak": mean([float(x["intermediate_leak"]) for x in ok]),
         "frac_result_matches_code": mean([float(x["result_matches_code"]) for x in ok if x.get("result_matches_code") is not None]),
         "frac_result_integer": mean([float(x["result_integer"]) for x in ok])}
    s["leak_examples"] = LEAKS
    json.dump({"summary": s, "rows": recs}, open(os.path.join(HERE, "tgsm_sample.json"), "w", newline="\n"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
