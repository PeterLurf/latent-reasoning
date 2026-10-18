"""PROOFS_TS checks P1 + P2 (CPU fp32, 2 threads, Idle priority): the TF-GREEDY labeller vs real KV-cache greedy generation, and the base's
REROOT / FULL keep and purity on FRESH items (rng 23187; unused in the repo at check time; these syms must join the TEXT-SELF exclusion sets).
TF side: TS_draft's own label_reroot / label_full (imported read-only from docs/research_textself_2026-10-18/scripts/ts_label_check.py).
GEN side (independent code): per step, KV-cache greedy generation through core.GPT2KV.forward (the cot_answers pattern: batches of equal prompt
length, no padding), stopped at the first ' =' / EOT; a step is form-valid iff it equals render(tree with the leftmost-ready handle -> leaf v)
for some v in 0..10 (brute force over v; no _render_pos / candidate()) AND its terminator is ' =' iff that tree still has an operator.
FULL-GEN: whole-chain generation (budget = chain length + 50), split at ' =' / EOT, each segment validated the same way.
Gold (N.apply) is used only to score purity, after labelling. Outputs pc_reroot.json (per-n summaries + per-item records), incrementally.
python -B pc_reroot.py"""
import ctypes
import json
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"; sys.dont_write_bytecode = True
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)   # IDLE_PRIORITY_CLASS
except Exception:
    pass
sys.path.insert(0, "D:/Latent/docs/research_textself_2026-10-18/scripts"); sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ts_label_check as T  # noqa: E402  (read-only import; its main() is never called)

N = T.N; EOT = T.EOT; EQ = T.EQ; NUMS = T.NUMS
torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "pc_reroot.json")
SEED = 23187


def reduce_copy(cur, v):
    path = N._leftmost_ready(cur); t2 = N._copy(cur); tgt = t2
    for p in path:
        tgt = getattr(tgt, p)
    tgt.op = tgt.l = tgt.r = None; tgt.v = v; return t2


def valid_step(cur, s, term):
    """-> (v, new tree) if s + term is the form-valid step from cur, else None (brute force over the value)."""
    for v in range(11):
        t2 = reduce_copy(cur, v)
        if N.ids(N.render(t2)) == s:
            return (v, t2) if term == (EQ if t2.op is not None else EOT) else None
    return None


@torch.no_grad()
def gen_batch(g, prompts, budget, stop_at_eq):
    """greedy KV-cache generation for equal-length prompts (core.GPT2KV.forward, as cot_answers). -> per row (tokens incl. the terminator, min top-2 margin)."""
    x = torch.tensor(prompts); h, past = g(g.embed(x), None); B = len(prompts); out = [[] for _ in range(B)]; done = [False] * B; mar = [float("inf")] * B
    for _ in range(budget):
        lg = g.logits(h[:, -1]); tv, ti = lg.topk(2, -1); nt = ti[:, 0]
        for r in range(B):
            if not done[r]:
                tok = int(nt[r]); out[r].append(tok); mar[r] = min(mar[r], float(tv[r, 0] - tv[r, 1]))
                if tok == EOT or (stop_at_eq and tok == EQ):
                    done[r] = True
        if all(done):
            break
        h, past = g(g.embed(nt[:, None]), past)
    return out, mar


def nops(root):
    return sum(1 for s in N.render(root) if s.strip() in N.PREC)


def reroot_gen(g, roots, bs=32):
    S = len(roots); cur = [N._copy(r) for r in roots]; n0 = [nops(r) for r in roots]; ok = [True] * S; vals = [[] for _ in range(S)]; fail = [None] * S
    mar = [float("inf")] * S
    for k in range(1, max(n0) + 1):
        act = [i for i in range(S) if ok[i] and k <= n0[i]]; by = {}
        for i in act:
            by.setdefault(len(N.render(cur[i])), []).append(i)
        for L, idx in by.items():
            for b0 in range(0, len(idx), bs):
                ii = idx[b0:b0 + bs]; pr = [[EOT] + N.ids(N.render(cur[i])) + [EQ] for i in ii]
                outs, ms = gen_batch(g, pr, L + 1, True)
                for r, i in enumerate(ii):
                    o = outs[r]; mar[i] = min(mar[i], ms[r])
                    if not o or o[-1] not in (EQ, EOT):
                        ok[i] = False; fail[i] = k; continue
                    vs = valid_step(cur[i], o[:-1], o[-1])
                    if vs is None:
                        ok[i] = False; fail[i] = k
                    else:
                        vals[i].append(vs[0]); cur[i] = vs[1]
    return [{"keep": ok[i], "values": vals[i] if ok[i] else None, "fail_step": fail[i], "min_margin": mar[i]} for i in range(S)]


def full_gen(g, roots, bs=32):
    S = len(roots); by = {}; res = [None] * S
    for i, r in enumerate(roots):
        by.setdefault(len(N.render(r)), []).append(i)
    for L, idx in by.items():
        for b0 in range(0, len(idx), bs):
            ii = idx[b0:b0 + bs]; pr = [[EOT] + N.ids(N.render(roots[i])) + [EQ] for i in ii]
            budget = max(T.chain_ids(roots[i], [0] * nops(roots[i])).__len__() for i in ii) + 50   # shape-only chain length + 50 (= len(cot_ids) + 50)
            outs, ms = gen_batch(g, pr, budget, False)
            for r, i in enumerate(ii):
                o = outs[r]; cur = N._copy(roots[i]); vals = []; pos = 0; ok = True; fail = None; k = 0
                while cur.op is not None:
                    k += 1; j = pos
                    while j < len(o) and o[j] not in (EQ, EOT):
                        j += 1
                    if j >= len(o):
                        ok = False; fail = k; break
                    vs = valid_step(cur, o[pos:j], o[j])
                    if vs is None:
                        ok = False; fail = k; break
                    vals.append(vs[0]); cur = vs[1]; pos = j + 1
                # lenient answer (cot_answers' rule): text after the last ' =' before EOT, if one number
                e = o.index(EOT) if EOT in o else None; ans = -1
                if e is not None:
                    body = o[:e]; last = max([q for q, t in enumerate(body) if t == EQ], default=-1); tail = body[last + 1:]
                    if len(tail) == 1 and tail[0] in NUMS:
                        ans = T.NUM2V[tail[0]]
                res[i] = {"keep": ok, "values": vals if ok else None, "fail_step": fail, "min_margin": ms[r], "lenient_answer": ans, "ntok": len(o)}
    return res


def main():
    t0 = time.time()
    excl = {tuple(it["syms"]) for it in N.test_set(1000, n_max=24)}; r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        excl.add(tuple(N.render(N.generate(r_sp, 8)[0])))
    r_d = np.random.default_rng(23180)                                       # the draft's precondition items (n 9..20, 200 per n): excluded too
    for n in range(9, 21):
        c = 0
        while c < 200:
            r_, _ = N.generate(r_d, n)
            if tuple(N.render(r_)) in excl:
                continue
            excl.add(tuple(N.render(r_))); c += 1
    plan = [(9, 200, ("reroot",)), (10, 200, ("reroot", "full")), (11, 200, ("reroot", "full")), (12, 200, ("reroot", "full")), (16, 200, ("reroot",)), (20, 100, ("reroot",))]
    rng = np.random.default_rng(SEED); items = {}; hits = {}
    for n in range(9, 21):                                                  # one stream, n 9..20 in order, 200 per n (used: the plan's counts)
        items[n] = []; hits[n] = 0
        while len(items[n]) < 200:
            r_, _ = N.generate(rng, n)
            if tuple(N.render(r_)) in excl:
                hits[n] += 1; continue
            items[n].append(r_)
    g = T.load("s170_cot_s1")
    res = {"seed": SEED, "excl": "test_set(1000, 24) + SERIAL(17300, 2200) + the draft's rng-23180 items", "excl_hits": hits, "by_n": {}, "items": {}}
    print("loaded %.0f s" % (time.time() - t0), flush=True)
    for n, k, protos in plan:
        roots = items[n][:k]; gv = [T.gold_values(r) for r in roots]; rn = {}
        for proto in protos:
            t1 = time.time(); tf = (T.label_reroot if proto == "reroot" else T.label_full)(g, roots, 25); t_tf = time.time() - t1
            t1 = time.time(); ge = (reroot_gen if proto == "reroot" else full_gen)(g, roots); t_ge = time.time() - t1
            keep_tf = [l_["keep"] for l_ in tf]; keep_ge = [l_["keep"] for l_ in ge]
            agree_keep = sum(a == b for a, b in zip(keep_tf, keep_ge))
            agree_full = sum((a["keep"] == b["keep"]) and (a["values"] == b["values"]) for a, b in zip(tf, ge))
            kept = [i for i in range(k) if keep_tf[i]]; kw = [i for i in kept if tf[i]["values"] != gv[i]]
            fs = [tf[i]["fail_step"] for i in range(k) if not keep_tf[i]]
            mg = np.array([x["min_margin"] for x in ge])
            row = {"items": k, "keep_tf": len(kept) / k, "n_kept_tf": len(kept), "kept_wrong_tf": len(kw), "pi_tf": (1 - len(kw) / len(kept)) if kept else None,
                   "keep_gen": float(np.mean(keep_ge)), "agree_keep": agree_keep, "agree_keep_and_values": agree_full,
                   "fail_step_hist_tf": {str(s): fs.count(s) for s in sorted(set(fs))}, "sec_tf": t_tf, "sec_gen": t_ge,
                   "gen_min_margin_q": [float(np.quantile(mg, q)) for q in (0.0, 0.01, 0.1, 0.5)], "gen_items_margin_lt_1e-3": int((mg < 1e-3).sum()),
                   "disagreements": [{"i": i, "tf": [tf[i]["keep"], tf[i]["fail_step"], tf[i]["values"]], "gen": [ge[i]["keep"], ge[i]["fail_step"], ge[i]["values"]],
                                      "gen_min_margin": ge[i]["min_margin"]} for i in range(k) if (tf[i]["keep"] != ge[i]["keep"]) or (tf[i]["values"] != ge[i]["values"])]}
            if proto == "full":
                row["lenient_acc_gen"] = float(np.mean([ge[i]["lenient_answer"] == gv[i][-1] for i in range(k)]))
            rn[proto] = row
            res["items"]["%d_%s" % (n, proto)] = [[int(tf[i]["keep"]), tf[i]["fail_step"], int(tf[i]["values"] == gv[i]) if tf[i]["keep"] else None,
                                                  int(ge[i]["keep"]), ge[i]["fail_step"]] for i in range(k)]
            print(n, proto, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items() if kk != "disagreements"}, "disagree", len(row["disagreements"]), flush=True)
            res["by_n"][str(n)] = rn
            json.dump(res, open(OUT, "w", newline="\n"), indent=1)
    res["sec"] = time.time() - t0; json.dump(res, open(OUT, "w", newline="\n"), indent=1)
    open(os.path.join(HERE, "pc_reroot.done"), "w", newline="\n").write("done %.0f s\n" % res["sec"])


if __name__ == "__main__":
    main()
