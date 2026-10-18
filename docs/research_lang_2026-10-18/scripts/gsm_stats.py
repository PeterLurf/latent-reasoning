"""LANG-VB draft $0: GSM8K-Aug / test structure stats for the slot design: m (equations), single-op share per item, values'
digit lengths, integrality, question length in GPT-2 tokens (if a tokenizer is available), number literals per question."""
import collections, json, re, sys, statistics
def load(p, cap=None):
    out = []
    for i, line in enumerate(open(p, encoding="utf-8", errors="replace")):
        if cap and i >= cap: break
        if "||" not in line or "####" not in line: continue
        q, rest = line.rstrip("\n").split("||", 1); cot, ans = rest.split("####", 1)
        out.append((q, re.findall(r"<<([^>]*)>>", cot), ans.strip()))
    return out
try:
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
except Exception as e:
    tok = None; print("no tokenizer:", type(e).__name__)
for name, p, cap in (("test", "D:/Latent/bench/data/gsm8k_aug/test.txt", None), ("train", "D:/Latent/bench/data/gsm8k_aug/train.txt", 60000)):
    d = load(p, cap); M = collections.Counter(len(e) for _, e, _ in d)
    allsingle = 0; vals = []; nonint = 0; neg = 0; lits = []; qtok = []; mxdig = []
    for q, eqs, a in d:
        sig = [re.sub(r"[^+\-*/%()]", "", e.split("=")[0]) for e in eqs]
        allsingle += all(len(s) == 1 for s in sig)
        for e in eqs:
            v = e.split("=")[-1].strip().replace(",", "")
            try:
                f = float(v); vals.append(f); nonint += (f != int(f)); neg += f < 0; mxdig.append(len(str(int(abs(f)))))
            except Exception: pass
        lits.append(len(re.findall(r"\d[\d,]*\.?\d*", q)))
        if tok is not None and len(qtok) < 3000: qtok.append(len(tok(q)["input_ids"]))
    n = len(d); print("==", name, "items", n, "m dist", sorted(M.items()))
    print("  all-equations-single-op share %.3f | values: n %d, non-integer %.4f, negative %.4f, digits p50 %d p99 %d max %d" % (allsingle / n, len(vals), nonint / len(vals), neg / len(vals), statistics.median(mxdig), sorted(mxdig)[int(0.99 * len(mxdig))], max(mxdig)))
    print("  numeric literals per question: mean %.2f p90 %d max %d" % (statistics.mean(lits), sorted(lits)[int(0.9 * n)], max(lits)))
    if qtok: print("  question GPT-2 tokens: p50 %d p90 %d max %d (first %d items)" % (statistics.median(qtok), sorted(qtok)[int(0.9 * len(qtok))], max(qtok), len(qtok)))
