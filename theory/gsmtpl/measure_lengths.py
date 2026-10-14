import ast
"""GSM-TEMPLATE draft, measurement 1 (CPU, read-only): GPT-2 (tiktoken 'gpt2') token lengths of the think input [EOT] + question
for ProsQA p8 (na train / na test / ctl) and GSM8K (openai train / test) and GSM8K-Aug (train / valid / test); equation counts; aux-target and
text-CoT lengths; alpha^L at init (alpha = sigmoid(5) = 0.99331, THEORY_SEEDSTAB Part I 4.2). Writes lengths.json next to this script.
    python -I measure_lengths.py"""
import ctypes
import json
import math
import os
import re
import sys

try:                                                                   # idle priority (memory: CPU work must not starve the GPU queues)
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
except Exception:
    pass
import tiktoken  # noqa: E402

ENC = tiktoken.get_encoding("gpt2"); NT = 3
ALPHA = 1 / (1 + math.exp(-5.0))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lengths.json")
P8 = "D:/Latent/prosqa/p8data/"; BD = "D:/Latent/bench/data/"
EQ = re.compile(r"<<([^>]*)>>")


def q(xs, ps=(0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    s = sorted(xs); n = len(s)
    r = {"n": n, "mean": round(sum(s) / n, 2), "min": s[0], "max": s[-1]}
    for p in ps:
        r["p%02d" % round(100 * p)] = s[min(n - 1, int(p * (n - 1) + 0.5))]
    r["alpha_L_at_mean"] = round(ALPHA ** r["mean"], 4); r["mean_alpha_L"] = round(sum(ALPHA ** L for L in s) / n, 4)
    return r


def toklens(texts):
    return [len(t) for t in ENC.encode_ordinary_batch(texts, num_threads=NT)]


def prosqa():
    out = {}
    for name in ("p8_na_train", "p8_na_test", "p8_ctl"):
        D = json.load(open(P8 + name + ".json"))
        Lq = [1 + n for n in toklens([x["question"] for x in D])]          # think input = [EOT] + question
        steps = [x["steps"] if isinstance(x["steps"], list) else ast.literal_eval(x["steps"]) for x in D]
        Ls = toklens([" " + s for st in steps for s in st])
        cot = [a + b for a, b in zip(toklens([x["question"] + "\n" for x in D]),
                                     toklens(["".join(s + "\n" for s in st) + "### " + x["answer"] for x, st in zip(D, steps)]))]
        dep = [len(st) for st in steps]
        r = {"think_L": q(Lq), "aux_target_tokens": q(Ls), "textcot_seq_tokens": q(cot), "by_depth": {}}
        for d in sorted(set(dep)):
            r["by_depth"][d] = q([L for L, e in zip(Lq, dep) if e == d], ps=(0.5,))
        if name == "p8_ctl":
            ext = [x.get("ext") for x in D]; r["by_ext"] = {}
            for e in sorted(set(map(str, ext))):
                r["by_ext"][e] = q([L for L, x in zip(Lq, ext) if str(x) == e], ps=(0.05, 0.5, 0.95))
            r["ctl_depth_hist"] = {d: dep.count(d) for d in sorted(set(dep))}
        out[name] = r
    return out


def gsm_openai():
    out = {}
    for name in ("gsm8k_train", "gsm8k_test"):
        D = [json.loads(l) for l in open(BD + name + ".jsonl", encoding="utf-8") if l.strip()]
        Lq = [1 + n for n in toklens([x["question"] for x in D])]
        m = [len(EQ.findall(x["answer"])) for x in D]
        cot = toklens([x["question"] + "\n" + x["answer"] for x in D])
        r = {"think_L": q(Lq), "textcot_seq_tokens_full_nl": q(cot), "n_eq_hist": {k: m.count(k) for k in sorted(set(m))}, "by_n_eq": {}}
        for k in sorted(set(m)):
            r["by_n_eq"][k] = q([L for L, e in zip(Lq, m) if e == k], ps=(0.5,))
        out[name] = r
    return out


def gsm_aug():
    out = {}
    for name in ("train", "valid", "test"):
        Q, E, A, bad = [], [], [], 0
        for l in open(BD + "gsm8k_aug/" + name + ".txt", encoding="utf-8"):
            l = l.rstrip("\n")
            if not l.strip():
                continue
            if "||" not in l or "####" not in l:
                bad += 1; continue
            qq, rest = l.split("||", 1); eqs, ans = rest.rsplit("####", 1)
            Q.append(qq.strip()); E.append(EQ.findall(eqs)); A.append(ans.strip())
        Lq = [1 + n for n in toklens(Q)]; m = [len(e) for e in E]
        aux = toklens([" <<" + e + ">>" for es in E for e in es])
        cot = toklens([qq + "\n" + " ".join("<<" + e + ">>" for e in es) + " #### " + a for qq, es, a in zip(Q, E, A)])
        r = {"n_lines_parsed": len(Q), "n_lines_unparsed": bad, "think_L": q(Lq), "aux_target_tokens": q(aux), "textcot_seq_tokens_eq_format": q(cot),
             "n_eq_hist": {k: m.count(k) for k in sorted(set(m))}, "by_n_eq": {}, "frac_m_gt_6": round(sum(k > 6 for k in m) / len(m), 4),
             "n_unique_questions": len(set(Q))}
        for k in sorted(set(m)):
            r["by_n_eq"][k] = q([L for L, e in zip(Lq, m) if e == k], ps=(0.5,))
        if name == "train":
            out["_train_Lq"] = Lq
        out["aug_" + name] = r
    return out


def main():
    res = {"tokenizer": "tiktoken gpt2 (= boxes/prosqa168.ENC)", "think_input": "[EOT] + question tokens (prosqa168s2.ProsMem.think_inputs)",
           "alpha_init": ALPHA}
    res.update(prosqa()); res.update(gsm_openai()); ag = gsm_aug(); trL = sorted(ag.pop("_train_Lq")); res.update(ag)
    # overlap: share of each eval set's think_L above the training distribution's quantiles
    import bisect
    def frac_above(xs, thr):
        return round(sum(x > thr for x in xs) / len(xs), 4)
    D = [json.loads(l) for l in open(BD + "gsm8k_test.jsonl", encoding="utf-8") if l.strip()]
    teL = [1 + n for n in toklens([x["question"] for x in D])]
    p = lambda s, f: s[min(len(s) - 1, int(f * (len(s) - 1) + 0.5))]  # noqa: E731
    res["gsm8k_test_vs_aug_train"] = {"aug_train_p95": p(trL, 0.95), "aug_train_p99": p(trL, 0.99), "aug_train_max": trL[-1],
                                      "test_frac_above_aug_p95": frac_above(teL, p(trL, 0.95)), "test_frac_above_aug_p99": frac_above(teL, p(trL, 0.99)),
                                      "test_frac_above_aug_max": frac_above(teL, trL[-1])}
    P = {n: [1 + k for k in toklens([x["question"] for x in json.load(open(P8 + n + ".json"))])] for n in ("p8_na_train", "p8_na_test", "p8_ctl")}
    tr = sorted(P["p8_na_train"])
    res["prosqa_eval_vs_train"] = {n: {"frac_above_train_p95": frac_above(P[n], p(tr, 0.95)), "frac_above_train_max": frac_above(P[n], tr[-1])}
                                   for n in ("p8_na_test", "p8_ctl")}
    res["prosqa_over_gsm8k_test_mean_think_L"] = round(res["p8_na_train"]["think_L"]["mean"] / res["gsm8k_test"]["think_L"]["mean"], 3)
    json.dump(res, open(OUT, "w", newline="\n"), indent=1)
    print(json.dumps({k: (v["think_L"] if isinstance(v, dict) and "think_L" in v else v) for k, v in res.items()}, indent=1)[:6000])


if __name__ == "__main__":
    main()
