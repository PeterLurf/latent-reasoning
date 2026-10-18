"""LANG-VB draft $0: DECOMP sizing. m_bin = total binary operators over an item's equations (ast parse of each LHS);
share of items whose every equation parses into + - * / binary ops on numeric literals / earlier results, and whose
equation values re-evaluate exactly (|eval - stated| < 1e-6 after the stated rounding)."""
import ast, collections, re, sys
def ops(expr):
    try:
        t = ast.parse(expr.replace(",", "").replace("x", "*"), mode="eval")
    except Exception:
        return None
    n = 0
    for nd in ast.walk(t):
        if isinstance(nd, ast.BinOp):
            if not isinstance(nd.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)): return None
            n += 1
        elif isinstance(nd, (ast.Expression, ast.Constant, ast.UnaryOp, ast.USub)) or isinstance(nd, ast.operator) or isinstance(nd, ast.unaryop):
            continue
        else:
            return None
    try:
        v = eval(compile(t, "e", "eval"))
    except Exception:
        return None
    return n, v
for name, p, cap in (("test", "D:/Latent/bench/data/gsm8k_aug/test.txt", None), ("train", "D:/Latent/bench/data/gsm8k_aug/train.txt", 60000)):
    M = collections.Counter(); ok = 0; n = 0; exact = 0
    for i, line in enumerate(open(p, encoding="utf-8", errors="replace")):
        if cap and i >= cap: break
        if "||" not in line or "####" not in line: continue
        n += 1; eqs = re.findall(r"<<([^>]*)>>", line.split("||", 1)[1]); tot = 0; good = True; ex = True
        for e in eqs:
            if "=" not in e: good = False; break
            lhs, rhs = e.rsplit("=", 1); r = ops(lhs)
            if r is None or r[0] == 0: good = False; break
            tot += r[0]
            try:
                ex = ex and abs(r[1] - float(rhs.replace(",", ""))) < 1e-6
            except Exception:
                ex = False
        if good and eqs:
            ok += 1; M[tot] += 1; exact += ex
    cum = 0; tot_ok = sum(M.values()); s = []
    for k in sorted(M):
        cum += M[k]; s.append("%d:%d(%.3f)" % (k, M[k], cum / tot_ok))
    print(name, "items", n, "DECOMP-parse ok %.3f, exact-eval %.3f of ok" % (ok / n, exact / max(ok, 1)), "m_bin dist (cum):", " ".join(s))
