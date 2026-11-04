"""QB0 (07_REDTEAM HIGH-1; pre-stated as Note 263-4): is a QUESTION-BLIND 'longest read-chain' policy (LP) correct on the training law and on the eval families?
$0 CPU, generator output only, one process (BLAS / torch threads capped at 3). No model is loaded; torch is imported only because the repo's generator modules
(inject_ow / inject_aw / of_parts) import it.

Populations (exact generator calls; see the pre-statement for the full list):
  T-bun   bun_s1's training law replayed EXACTLY (args boxes/runs_fsq/bun_s1/args.json: seed 1, bs 64, d_max 4, distract_mix s256,chain,givop, distract_p 1.0, perm_p 0,
          mul_s_p 0): EXCL = build_sets(0)['excl'] | inject_aw.reg_excl (len asserted == args excl_n), rng_d = default_rng(seed + SEED_D), stream =
          OW.Stream(OW.SEED_TRAIN, EXCL, FRAME_TRAIN, topo, dag, mul_s_p 0), per item k of step s the trainer's coins / draws (25196 coin, distract_mix 25205).
          Trainer steps 1..313, the first 20,000 items in (step, k) order. gb_s1 (same seed, --cue generic) saw exactly these items.
  T-of    of_s1's law (args boxes/runs_fsq/of_s1/args.json: + perm_p 0.5 perm-then-insert, ask_p 0.25 -> 48 chain + 16 ask items per step): chain items steps 1..417,
          first 20,000; ask items (own stream Stream(26210), perm coin 26211 / perm 26212, class ask_render.ask_of): the shortest prefix with >= 8,000 items (~steps 1..500).
          Every population is the shortest (step, k) prefix with >= its count AND >= 1,000 items at each d 1-4 (asserted within the replayed steps 1..1000).
  E       REG TEST2 (build_sets(0)['test2'], sha asserted == args test2_sha) d 1-8; TEST-PERM d 2-4 (perm_render.test_perm_sets); LATE-PREMISE d 2-4
          (of_parts.late_order + perm_render.render); INSERT-INDIST (insert_render.insert_eval_sets, 7 conds); insert_render.held_out_sets (shas asserted ==
          scratch/s256_ho_shas.json) and held_out2_sets (== scratch/s257_ho_shas.json) on TEST2 d 4.
No new seeds: every training draw replays the trainer's own seeds (seed, family, step, k).

Policy RT (BINDING; 07_REDTEAM HIGH-1's walk, Theory 11-04; never reads the question): start = the first given sentence in text order; at a node the candidates are
the entities whose op sentence reads it; none -> stop and predict the node; one -> move; several -> the non-sink (a candidate with a reader) if exactly one,
else uniform over the non-sinks (several) or over all candidates (all sinks), drawn from default_rng((27150, crc32(cell), item, 0)). rt = realised, rt_exp =
the analytic expectation over the tie draws (the rule uses rt_ent_exp); RT-START = start == S_0; RT-STOP = stop == S_d. RT-S (descriptive): RT with start =
the first given that has a reader (draws (..., 1)).
Policy LP (secondary; never reads the question): parse every sentence except the final question into a dataflow graph (node per entity; a given sentence = a root; an op
sentence defines B from A (+ C), edges A -> B, C -> B; mirrored frames resolved as OW.solve does: the reading whose inputs are known and whose B is not).
L(n) = 0 for roots, 1 + max L(inputs) else. L* = max L. End = the node with L = L* whose defining sentence is LATEST in the text (tie rule a); start = among the
roots reachable backwards along L-decreasing edges from End, the one whose sentence is EARLIEST (tie rule b). Prediction = the value of End.
  LP (value)  P(value(End) == answer);  LP-ENT  P(End == the asked entity);  LP-STOP  L* - d (share == 0 / > 0);  LP-START  P(start == S_0);  tie  P(#ends > 1).
  Ask items: P(value(End) == v_X); Q1 = X's own defining sentence applied to its inputs' values (Q1 == v_X; Q1-ONEHOP = every input of X is a root).
Parse check: the parser's value of the asked entity == it['answer'] AND == OW.solve(text) (ask items: == v_X, and S_d == answer); mismatches excluded and counted.
    python theory/read_qb0.py            -> theory/results/read_qb0.json
    python theory/read_qb0.py --smoke    -> <= 20 items; prints 'smoke ok' + parse-agreement counts only (no accuracies)"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "3")
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
import boxes.ow_words as OW  # noqa: E402
import boxes.nested_words as NW  # noqa: E402
import boxes.insert_render as IR  # noqa: E402
import boxes.perm_render as PR  # noqa: E402
import boxes.ask_render as AR  # noqa: E402
import boxes.inject_ow as IO  # noqa: E402
import boxes.inject_aw as AW  # noqa: E402
import boxes.of_parts as OP  # noqa: E402
try:
    import torch
    torch.set_num_threads(3)
except Exception:                                                           # noqa: BLE001
    pass

OUT = "theory/results/read_qb0.json"
SEED_RT = 27150                                                             # RT / RT-S tie draws: default_rng((27150, crc32(cell), item, 0 RT / 1 RT-S)); unused in boxes/ theory/ scratch/*.py (grep 11-04)
N_CHAIN, N_ASK, MIN_PER_D = 20000, 8000, 1000                       # + the prefix extends until every d 1-4 cell has >= 1,000 items
BASE_OPS = ("add_c", "sub_c", "mul_c", "add_v", "sub_v", "mul_v")           # add_s / mul_s share add_v / mul_v's frames (same text -> same equation)
GB = {"gb_s1": "boxes/runs_fsq/gb_s1", "gb_s2": "boxes/runs_fsq/gb_s2"}
RULE_FAMS = ["TEST2-d1-4", "INSERT-INDIST", "H-OP", "H-CHAINSRC", "H-GIVPOS", "H-EXTEND", "H-LIVE", "K3", "H-CHAINSRC-OP", "H-CHAINSRC-LIVE"]   # pre-stated (Note 263-4)


# ---------------------------------------------------------------- parser / policy
_PATS = {}


def pats(u):
    if u not in _PATS:
        L = []
        for op in BASE_OPS:
            for f in NW.FRAMES[op]:
                rx = re.escape(f).replace(re.escape("{u}"), re.escape(u))
                for k, g in (("B", r"(?P<B>\w+)"), ("A", r"(?P<A>\w+)"), ("C", r"(?P<C>\w+)"), ("c", r"(?P<c>\d+)")):
                    rx = rx.replace(re.escape("{%s}" % k), g, 1)
                L.append((op, re.compile("^" + rx + "$")))
        _PATS[u] = (re.compile(r"^(\w+) has (\d+) " + re.escape(u) + r"\.$"), L)
    return _PATS[u]


class ParseError(Exception):
    pass


def split(text):
    s = [x.strip() for x in re.split(r"(?<=[.?])\s+", text) if x.strip()]
    if not s[-1].startswith("How many"):
        raise ParseError("no final question")
    return s[:-1], s[-1]


def parse(sents, u):
    """-> val {name: value}, defn {name: sentence index}, inputs {name: tuple}, opof {name: (op, c or None)}."""
    gv, P = pats(u); val, defn, inputs, opof, rules = {}, {}, {}, {}, []
    for i, s in enumerate(sents):
        m = gv.match(s)
        if m:
            if m.group(1) in defn:
                raise ParseError("entity defined twice")
            val[m.group(1)] = int(m.group(2)); defn[m.group(1)] = i; inputs[m.group(1)] = (); continue
        hits = [(op, mm.groupdict()) for op, p in P for mm in [p.match(s)] if mm]
        if not hits:
            raise ParseError("unparsed sentence: " + s)
        rules.append((i, hits))
    pending = list(rules); changed = True
    while pending and changed:
        changed = False
        for r in list(pending):
            i, hits = r
            for op, gd in hits:
                a = val.get(gd["A"]); cst = gd.get("c"); b = int(cst) if cst is not None else val.get(gd.get("C"))
                if a is not None and b is not None and gd["B"] not in val:
                    B = gd["B"]; val[B] = OW.apply(op, a, b); defn[B] = i; inputs[B] = (gd["A"],) + ((gd["C"],) if cst is None else ())
                    opof[B] = (op, None if cst is None else int(cst)); pending.remove(r); changed = True; break
    if pending:
        raise ParseError("unresolved op sentences")
    return val, defn, inputs, opof


def lp(val, defn, inputs):
    L = {}

    def depth(n):
        if n not in L:
            L[n] = 0 if not inputs[n] else 1 + max(depth(x) for x in inputs[n])
        return L[n]
    for n in val:
        depth(n)
    Ls = max(L.values()); ends = [n for n in val if L[n] == Ls]; end = max(ends, key=lambda n: defn[n])   # tie (a): latest defining sentence
    roots, seen, st = set(), set(), [end]
    while st:
        n = st.pop()
        if n in seen:
            continue
        seen.add(n)
        if not inputs[n]:
            roots.add(n)
        else:
            st += [x for x in inputs[n] if L[x] == L[n] - 1]
    start = min(roots, key=lambda n: defn[n])                               # tie (b): earliest given
    return {"end": end, "value": val[end], "L": Ls, "n_ends": len(ends), "start": start}


def rt_policy(val, defn, inputs, target, tval, sd, rng, skip_sink_start=False):
    """RT (question-blind forward walk): start = the first given sentence in text order (RT-S: the first given that has a reader); at a node, candidates = the
    entities whose op sentence reads it (ordered by sentence); none -> stop; one -> move; several -> the non-sink if exactly one, else uniform (seeded) over the
    non-sinks if several, over all candidates if all are sinks. -> realised stop (value / entity) + the analytic expectation over the tie draws."""
    readers = {n: [] for n in val}
    for b_, ins in inputs.items():
        for x in set(ins):
            readers[x].append(b_)
    for n in readers:
        readers[n].sort(key=lambda b_: defn[b_])
    givens = sorted((n for n in val if not inputs[n]), key=lambda n: defn[n]); start = givens[0]
    if skip_sink_start:
        gs = [g for g in givens if readers[g]]; start = gs[0] if gs else givens[0]

    def choices(n):
        c = readers[n]
        if len(c) <= 1:
            return c
        ns = [x for x in c if readers[x]]
        return ns if ns else c                                              # exactly one non-sink -> [it]; several -> them; none -> all (sinks)
    n = start
    while True:
        ch = choices(n)
        if not ch:
            break
        n = ch[0] if len(ch) == 1 else ch[int(rng.integers(len(ch)))]
    memo = {}

    def ex(m):
        if m not in memo:
            ch = choices(m)
            memo[m] = (float(m == target), float(val[m] == tval)) if not ch else tuple(float(x) for x in np.mean([ex(y) for y in ch], axis=0))
        return memo[m]
    e_ent, e_val = ex(start)
    return {"ok": int(val[n] == tval), "ent": int(n == target), "exp": e_val, "ent_exp": e_ent, "start": start, "stop_sd": int(n == sd)}


def policies(val, defn, inputs, target, tval, s0, sd, d, key):
    r = lp(val, defn, inputs); c = zlib.crc32(key[0].encode())
    o = {"ok": int(r["value"] == tval), "ent": int(r["end"] == target), "stop": r["L"] - d, "start_s0": int(r["start"] == s0), "tie": int(r["n_ends"] > 1)}
    for pre, sk in (("rt", False), ("rts", True)):
        w = rt_policy(val, defn, inputs, target, tval, sd, np.random.default_rng((SEED_RT, c, int(key[1]), int(sk))), sk)
        o.update({pre + "_ok": w["ok"], pre + "_ent": w["ent"], pre + "_exp": w["exp"], pre + "_ent_exp": w["ent_exp"], pre + "_start_s0": int(w["start"] == s0),
                  pre + "_stop_sd": w["stop_sd"]})
    return o


def parsed(text, u, asked, answer, spine_vals=None):
    """-> (val, defn, inputs, opof) or None (parse mismatch): the parser's asked value == answer == OW.solve(text) (+ every spine value)."""
    try:
        sents, q = split(text); P_ = parse(sents, u)
    except ParseError:
        return None
    if P_[0].get(asked) != answer or OW.solve(text) != answer:
        return None
    if spine_vals is not None and any(P_[0].get(n) != v for n, v in spine_vals):
        return None
    return P_


def eval_item(it, d=None, key=("", 0)):
    sp = it["names"]["spine"]; ans = int(it["answer"])
    P_ = parsed(it["text"], it["unit"], sp[-1], ans, list(zip(sp, [int(v) for v in it["values"]])))
    if P_ is None:
        return None
    return policies(P_[0], P_[1], P_[2], sp[-1], ans, sp[0], sp[-1], it["d"] if d is None else d, key)


def ask_item(r, key=("", 0)):
    cls, X, v, t2, fb = AR.ask_of(r); sp = r["names"]["spine"]
    P_ = parsed(t2, r["unit"], sp[-1], int(r["answer"]))
    if P_ is None or P_[0].get(X) != v:
        return None, cls
    val, defn, inputs, opof = P_; ins = inputs[X]
    if ins:
        op, c = opof[X]; q1 = OW.apply(op, val[ins[0]], c if c is not None else val[ins[1]])
    else:
        q1 = val[X]
    o = policies(val, defn, inputs, X, v, sp[0], sp[-1], r["d"], key)
    o.update(q1=int(q1 == v), q1_onehop=int(all(not inputs[x] for x in ins)), cls=cls, d=r["d"]); return o, cls


# ---------------------------------------------------------------- stats
def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 6), round(c + h, 6)]


def summ(recs, n_mis, extra=None):
    recs = [r for r in recs if r is not None]; n = len(recs); k = sum(r["ok"] for r in recs); ke = sum(r["ent"] for r in recs)
    o = {"n": n, "parse_mismatch": n_mis, "lp": k / n if n else None, "lp_ci95": wilson(k, n), "lp_ent": ke / n if n else None, "lp_ent_ci95": wilson(ke, n),
         "lp_stop_eq_d": float(np.mean([r["stop"] == 0 for r in recs])) if n else None, "lp_stop_gt_d": float(np.mean([r["stop"] > 0 for r in recs])) if n else None,
         "lp_stop_mean": float(np.mean([r["stop"] for r in recs])) if n else None, "lp_start_s0": float(np.mean([r["start_s0"] for r in recs])) if n else None,
         "tie_share": float(np.mean([r["tie"] for r in recs])) if n else None}
    for pre in ("rt", "rts"):                                               # realised (seeded) with Wilson CIs; *_exp = the analytic expectation over the tie draws
        k_, ke_ = sum(r[pre + "_ok"] for r in recs), sum(r[pre + "_ent"] for r in recs)
        o.update({pre: k_ / n if n else None, pre + "_ci95": wilson(k_, n), pre + "_ent": ke_ / n if n else None, pre + "_ent_ci95": wilson(ke_, n)})
        for f in ("_exp", "_ent_exp", "_start_s0", "_stop_sd"):
            o[pre + f] = float(np.mean([r[pre + f] for r in recs])) if n else None
    if extra:
        o.update(extra)
    return o


def run_set(items, d=None, cell=""):
    recs = [eval_item(it, d, (cell, i)) for i, it in enumerate(items)]; return [r for r in recs if r is not None], sum(r is None for r in recs)


def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x)); out = r.copy()
    for v in np.unique(x):
        m = x == v; out[m] = r[m].mean()
    return out


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- populations
def load_args(run):
    return json.load(open(os.path.join("boxes/runs_fsq", run, "args.json"), encoding="utf-8"))


def build(a):
    t0 = time.time(); SETS = IO.build_sets(mul_s_p=a["mul_s_p"]); EXCL = SETS["excl"] | AW.reg_excl(argparse.Namespace(**a), SETS["excl"])
    assert len(EXCL) == a["excl_n"], ("EXCL size != the trainer's", len(EXCL), a["excl_n"])
    assert SETS["shas"]["test2_sha"] == a["test2_sha"], "TEST2 sha != the run's"
    return SETS, EXCL, round(time.time() - t0)


def replay(a, EXCL, n_chain, n_ask):
    """the trainer's draw_items / ask_items (boxes/inject_aw.py main) for steps 1.. until n_chain chain and n_ask ask items; -> chain [(step, k, item)], ask [...], steps."""
    seed, bs = a["seed"], a["bs"]; NASK = int(round(a.get("ask_p", 0.0) * bs)); perm_p = a.get("perm_p", 0.0); dp = a["distract_p"]
    MIXT = tuple(a["distract_mix"].split(",")) if a.get("distract_mix") else (); nmax = a.get("distract_nmax", 2)
    assert a["src"] == "ow" and not a.get("train_frames") and not a.get("resume"), "replay: OW law, default frames, from step 1"
    rng_d = np.random.default_rng(seed + IO.SEED_D)
    stream = OW.Stream(OW.SEED_TRAIN, EXCL, frames=OW.FRAME_TRAIN, mul_s_p3=a["mul_s_p3"], mul_s_p=a["mul_s_p"], **IO.OWKW)
    ask_stream = OW.Stream(AW.SEED_ASK_STREAM, EXCL, frames=OW.FRAME_TRAIN, mul_s_p3=a["mul_s_p3"], mul_s_p=a["mul_s_p"], **IO.OWKW) if NASK else None
    chain, ask, step = [], [], 0
    while len(chain) < n_chain or (NASK and len(ask) < n_ask):
        step += 1; d = int(rng_d.integers(1, a["d_max"] + 1))
        for k, r in enumerate([stream.draw(d) for _ in range(bs - NASK)]):
            if perm_p and d >= 2 and np.random.default_rng((seed, PR.SEED_PERM_TRAIN + 1, step, k)).random() < perm_p:
                r = PR.train_perm(r, seed, step, k)
            if dp and np.random.default_rng((seed, IR.SEED_DISTRACT_COIN, step, k)).random() < dp:
                r2, n = IR.distract_mix(r, seed, step, k, MIXT, nmax=nmax) if MIXT else IR.distract(r, seed, step, k, nmax=nmax)
                if n:
                    r = r2
            chain.append((step, k, r))
        for k in range(NASK):
            r = ask_stream.draw(d)
            if d >= 2 and np.random.default_rng((seed, AW.SEED_ASK_COIN, step, k)).random() < a["ask_perm_p"]:
                pit, _, inv = PR.permuted(r, np.random.default_rng((seed, AW.SEED_ASK_PERM, step, k))); PR.check(r, pit); assert inv >= 1; pit["perm"] = 1; r = pit
            ask.append((step, k, r))
    return chain[:n_chain], ask[:n_ask], step


def prefix(items, n):
    """the shortest (step, k)-ordered prefix with >= n items and >= MIN_PER_D items at each d 1-4 (asserted reachable within the replayed steps 1..1000)."""
    cnt = {}
    for i, (_, _, it) in enumerate(items):
        cnt[it["d"]] = cnt.get(it["d"], 0) + 1
        if i + 1 >= n and all(cnt.get(d, 0) >= MIN_PER_D for d in (1, 2, 3, 4)):
            return items[:i + 1]
    raise AssertionError(("prefix: per-d minimum not reached", cnt))


def replay_gate(run, chain, ask):
    """the replayed steps 1..1000 vs the run's first log.jsonl bin: distract_n / distract_types counts (+ ask_n)."""
    for l in open(os.path.join("boxes/runs_fsq", run, "log.jsonl"), encoding="utf-8"):
        j = json.loads(l)
        if "distract_n" in j:
            break
    dn, dt = {}, {}
    for _, _, it in chain:
        n = it.get("distract_n", 0); dn[str(n)] = dn.get(str(n), 0) + 1
        for t in it.get("distract_types", ()):
            dt[t] = dt.get(t, 0) + 1
    g = {"log_step": j["step"], "distract_n": dn, "log_distract_n": j["distract_n"], "distract_types": dict(sorted(dt.items())), "log_distract_types": j["distract_types"],
         "ask_n": len(ask), "log_ask_n": j.get("ask_n")}
    g["match"] = bool(j["step"] == 1000 and dn == j["distract_n"] and dict(sorted(dt.items())) == j["distract_types"] and (j.get("ask_n") is None or j["ask_n"] == len(ask)))
    return g


def chain_pop(chain, pop):
    recs = []; mis = 0; by_d = {}
    for i, (_, _, it) in enumerate(chain):
        r = eval_item(it, None, (pop, i)); recs.append(r); mis += r is None; by_d.setdefault(it["d"], []).append(r)
    out = summ(recs, mis); out["by_d"] = {str(d): summ(v, sum(x is None for x in v)) for d, v in sorted(by_d.items())}
    v24 = [x for d, v in by_d.items() if d >= 2 for x in v]; out["by_d"]["2-4"] = summ(v24, sum(x is None for x in v24))   # the binding d 2-4 pool
    out["insert_share"] = float(np.mean([bool(it.get("distract_n")) for _, _, it in chain])); out["perm_share"] = float(np.mean([bool(it.get("perm")) for _, _, it in chain]))
    return out


def ask_pop(ask):
    recs, mis, by = [], 0, {}
    for i, (_, _, it) in enumerate(ask):
        r, cls = ask_item(it, ("T-of-ask", i)); mis += r is None
        if r is not None:
            recs.append(r); by.setdefault(cls, []).append(r); by.setdefault("d%d" % it["d"], []).append(r)
    out = summ(recs, mis)
    q = sum(r["q1"] for r in recs); out.update(q1=q / len(recs), q1_ci95=wilson(q, len(recs)), q1_onehop=float(np.mean([r["q1_onehop"] for r in recs])))
    out["by_class"] = {c: dict(summ(v, 0), q1=float(np.mean([r["q1"] for r in v])), share=len(v) / len(recs)) for c, v in sorted(by.items())}
    return out


def eval_sets(SETS):
    t2 = SETS["test2"]; E = {}
    for d in range(1, 9):
        E["TEST2-d%d" % d] = {"items": t2[d]}
    P_, _ = PR.test_perm_sets({d: t2[d] for d in (2, 3, 4)}, (2, 3, 4))
    for d in (2, 3, 4):
        E["TEST-PERM-d%d" % d] = {"items": P_[d]}
        E["LATE-PREMISE-d%d" % d] = {"items": [OP._render(it, OP.late_order(it), "late") for it in t2[d]]}
    ins, _ = IR.insert_eval_sets(t2[4])
    for c in IR.CONDS:
        E["INSERT-INDIST:" + c] = {"items": ins[c], "d": 4}
    HO, _ = IR.held_out_sets(t2[4]); assert IR.set_shas(HO) == json.load(open("scratch/s256_ho_shas.json", encoding="utf-8"))["held_out"], "held_out shas"
    HO2, _ = IR.held_out2_sets(t2[4]); assert IR.set_shas(HO2) == json.load(open("scratch/s257_ho_shas.json", encoding="utf-8"))["held_out2"], "held_out2 shas"
    for H in (HO, HO2):
        for fam, dd in H.items():
            for pos, its in dd.items():
                E["%s:%s" % (fam, pos)] = {"items": its, "d": 4}
    return E


def pool(parts, res):
    recs = [r for p in parts for r in res[p][0]]; return summ(recs, sum(res[p][1] for p in parts), {"parts": parts})


def families(E, res):
    keys = list(E); F = {}
    F["TEST2-d1-4"] = pool(["TEST2-d%d" % d for d in (1, 2, 3, 4)], res)
    for d in range(1, 9):
        F["TEST2-d%d" % d] = pool(["TEST2-d%d" % d], res)
    for d in (2, 3, 4):
        F["TEST-PERM-d%d" % d] = pool(["TEST-PERM-d%d" % d], res); F["LATE-PREMISE-d%d" % d] = pool(["LATE-PREMISE-d%d" % d], res)
    F["INSERT-INDIST"] = pool([k for k in keys if k.startswith("INSERT-INDIST:")], res)
    tops = {"H-OP": "H-OP-", "H-CHAINSRC": "H-CHAINSRC:", "H-GIVPOS": "H-GIVPOS:", "H-EXTEND": "H-EXTEND:", "H-LIVE": "H-LIVE:", "K3": "K3:",
            "H-CHAINSRC-OP": "H-CHAINSRC-OP-", "H-CHAINSRC-LIVE": "H-CHAINSRC-LIVE:"}
    for top, pre in tops.items():
        F[top] = pool([k for k in keys if k.startswith(pre)], res)
    for k in keys:                                                          # every (family, position) cell
        if ":" in k:
            F["cell " + k] = pool([k], res)
    return F


def gb_fams():
    out = {}
    for s, p in GB.items():
        A = json.load(open(os.path.join(p, "eval_acc.json"), encoding="utf-8")); H = json.load(open(os.path.join(p, "eval_insert_ho.json"), encoding="utf-8"))
        g = {"TEST2-d1-4": float(np.mean([A["by_d"][str(d)]["acc"] for d in (1, 2, 3, 4)])), "INSERT-INDIST": float(np.mean([H["indist"][c]["acc8"] for c in IR.CONDS]))}
        for f in ("H-OP", "H-CHAINSRC", "H-GIVPOS", "H-EXTEND", "H-LIVE", "K3", "H-CHAINSRC-OP", "H-CHAINSRC-LIVE"):
            g[f] = float(H["fam"][f]["acc8"])
        for d in range(5, 9):
            g["TEST2-d%d" % d] = float(A["by_d"][str(d)]["acc"])
        out[s] = g
    out["mean"] = {f: (out["gb_s1"][f] + out["gb_s2"][f]) / 2 for f in out["gb_s1"]}
    return out


POLICY_KEY = {"RT": "rt_ent_exp", "RT-S": "rts_ent_exp", "LP": "lp_ent"}      # the rule's per-family policy value (entity level; RT / RT-S analytic over ties)


def agreement(F, G, pol):
    key = POLICY_KEY[pol]; pv = {f: F[f][key] for f in RULE_FAMS}; lo = [f for f in RULE_FAMS if pv[f] <= 0.10]; hi = [f for f in RULE_FAMS if pv[f] >= 0.95]
    mid = [f for f in RULE_FAMS if f not in lo and f not in hi]; fails = []; per = {}
    for s_ in ("gb_s1", "gb_s2"):
        c1 = [f for f in lo if G[s_][f] > 0.45]; c2 = [f for f in hi if G[s_][f] < 0.85]; per[s_] = {"clause_i_fail": c1, "clause_ii_fail": c2}
        fails += [(s_, "i")] * bool(c1) + [(s_, "ii")] * bool(c2)
    vac = [c for c, fs in (("i", lo), ("ii", hi)) if not fs]
    fit = {s_: [f for f in mid if abs(G[s_][f] - pv[f]) <= 0.10] for s_ in ("gb_s1", "gb_s2")}   # descriptive: mid-band families gb matches within 0.10
    lab = ("QB0-UNINFORMATIVE" if vac else "QB0-SUPPORTED" if not fails else "QB0-PARTIAL" if len(fails) == 1 else "QB0-REFUTED")
    return {"policy": pol, "measure": key, "families": RULE_FAMS, "n_families": len(RULE_FAMS), "policy_value": pv, "low": lo, "high": hi, "mid": mid,
            "vacuous_clauses": vac, "mid_fit_within_0.10": fit, "per_seed": per, "clause_seed_failures": [list(x) for x in fails], "label": lab,
            "spearman_descriptive": {s_: spearman([pv[f] for f in RULE_FAMS], [G[s_][f] for f in RULE_FAMS]) for s_ in ("gb_s1", "gb_s2", "mean")}}


def smoke():
    a = load_args("bun_s1"); ao = load_args("of_s1"); assert ao["excl_n"] == a["excl_n"] and ao["mul_s_p"] == a["mul_s_p"]
    SETS, EXCL, sec = build(a)
    cb, _, _ = replay(a, EXCL, 4, 0); co, ak, _ = replay(ao, EXCL, 4, 4); E = eval_sets(SETS)
    picks = [("TEST2-d4", 0), ("TEST-PERM-d3", 0), ("LATE-PREMISE-d3", 0), ("H-EXTEND:after_Sd", 0), ("H-LIVE:after", 0), ("H-CHAINSRC-LIVE:3", 0),
             ("H-CHAINSRC-OP-mul_c:3", 0), ("INSERT-INDIST:after_k2", 0)]
    agree = {"T-bun": [0, 0], "T-of": [0, 0], "T-of-ask": [0, 0], "E": [0, 0]}
    for i, (_, _, it) in enumerate(cb):
        agree["T-bun"][0] += eval_item(it, None, ("T-bun", i)) is not None; agree["T-bun"][1] += 1
    for i, (_, _, it) in enumerate(co):
        agree["T-of"][0] += eval_item(it, None, ("T-of", i)) is not None; agree["T-of"][1] += 1
    for i, (_, _, it) in enumerate(ak):
        agree["T-of-ask"][0] += ask_item(it, ("T-of-ask", i))[0] is not None; agree["T-of-ask"][1] += 1
    for k, i in picks:
        agree["E"][0] += eval_item(E[k]["items"][i], E[k].get("d"), (k, i)) is not None; agree["E"][1] += 1
    tot = sum(v[1] for v in agree.values()); assert tot <= 20
    print("smoke ok")
    print("parse agreement (parser == it['answer'] == OW.solve, spine values equal):", {k: "%d/%d" % tuple(v) for k, v in agree.items()}, "total %d items" % tot)


def binding_block(out):
    """Note 263-4 (A) labels + section 5 ask pressure, in place on out."""
    lab_ = lambda p: "QB-SUFFICIENT" if p >= 0.99 else ("QB-PARTIAL" if p >= 0.90 else "QB-INSUFFICIENT")  # noqa: E731
    PV = {"LP": "lp", "RT": "rt_exp", "RT-S": "rts_exp"}                   # BINDING (Note 263-4 A): value level; RT / RT-S = the analytic expectation over ties
    labels = {p: {k: {c: {"value": (out[k] if c == "pooled" else out[k]["by_d"][c])[f], "label": lab_((out[k] if c == "pooled" else out[k]["by_d"][c])[f])}
                      for c in ("1", "2", "3", "4", "2-4", "pooled")} for k in ("T-bun", "T-of")} for p, f in PV.items()}
    suff = {p: all(labels[p]["T-bun"][c]["label"] == "QB-SUFFICIENT" for c in ("2", "3", "4")) for p in PV}   # QB-SUFFICIENT at each of d 2, 3, 4
    out["binding"] = {"labels": labels, "t_bun_d2_4_sufficient": suff, "QB0_headline": "YES" if any(suff.values()) else "NO",
                      "def": "any of LP / RT / RT-S QB-SUFFICIENT (>= 0.99, value level) at each of d 2, 3, 4 on T-bun chain items"}
    AB = out["T-of-ask"]["by_class"]                                        # Note 263-4 §5: one-hop question pressure = 1 - max policy (value level) per d
    pr = {str(d): 1.0 - max(AB["d%d" % d][f] for f in PV.values()) for d in (1, 2, 3, 4)}
    out["T-of-ask"]["pressure"] = {"by_d": pr, "ge_0.5_at_d2_4": all(pr[c] >= 0.5 for c in ("2", "3", "4"))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args()
    if A_.smoke:
        return smoke()
    t0 = time.time(); a = load_args("bun_s1"); ao = load_args("of_s1"); assert ao["excl_n"] == a["excl_n"] and ao["mul_s_p"] == a["mul_s_p"]
    SETS, EXCL, sec = build(a)
    gate = {}; pops = {}
    for run, aa in (("bun_s1", a), ("of_s1", ao)):                         # exactness gate: steps 1..1000 == the run's first log bin (insert n / type counts)
        NASK = int(round(aa.get("ask_p", 0.0) * aa["bs"])); ch, ak_, _ = replay(aa, EXCL, 1000 * (aa["bs"] - NASK), 1000 * NASK); g = replay_gate(run, ch, ak_)
        assert g["match"], ("replay != the trainer's first 1k bin", run, g); gate[run] = g; pops[run] = (ch, ak_)
    cb = prefix(pops["bun_s1"][0], N_CHAIN); co = prefix(pops["of_s1"][0], N_CHAIN); ak = prefix(pops["of_s1"][1], N_ASK)
    out = {"def": __doc__, "build_sec": sec, "replay_gate": gate,
           "T-bun": dict(chain_pop(cb, "T-bun"), steps=[1, max(s for s, _, _ in cb)], run="bun_s1"),
           "T-of": dict(chain_pop(co, "T-of"), steps=[1, max(s for s, _, _ in co)], run="of_s1"),
           "T-of-ask": dict(ask_pop(ak), steps=[1, max(s for s, _, _ in ak)], run="of_s1")}
    binding_block(out)
    E = eval_sets(SETS); res = {k: run_set(v["items"], v.get("d"), k) for k, v in E.items()}
    F = families(E, res); out["E"] = F; G = gb_fams(); out["gb"] = G
    out["descriptive_gb_agreement"] = {p: agreement(F, G, p) for p in ("RT", "LP", "RT-S")}   # DESCRIPTIVE ONLY (Note 263-4 B): gb numbers already published
    out["parse_mismatch_total"] = out["T-bun"]["parse_mismatch"] + out["T-of"]["parse_mismatch"] + out["T-of-ask"]["parse_mismatch"] + sum(v[1] for v in res.values())
    out["sec"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False); f.write("\n")
    print("wrote", OUT, "QB0 headline", out["binding"]["QB0_headline"], out["binding"]["t_bun_d2_4_sufficient"])


if __name__ == "__main__":
    main()
