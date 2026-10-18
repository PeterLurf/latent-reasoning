"""§218 LANG-VB A2 labeller and self pool (LANG_v1.md §2.4, §3.4, §4.1 `labeller (A2)`, §7 G-L3 / G-L15 / G-L20, §8): SOUND-CLOSURE (parses the
ORIGINAL item's text), the two labeller chains L-ROLL / chunked L-RR on a frozen LangSLVA (boxes/langvb.py), GATE-1-L, the resumable pool builder,
the trainer's loader (load_pool / self_item) and the rng_s replay. Gold (generator fields) is read only by gate1_stats (GATE-1-L) and the gates.

API
  parse_ids(ids) -> label input {ids, P, slot_idx, concepts, ord}: P = {"R", "options": [o1, o2], "stmts": [[subj, obj], ...]} parsed from
      tiktoken-gpt2-decoded text_ids ([EOT] + " " + text); slot_idx = sorted first-mention last-token positions, concepts = names in that order,
      ord {name: slot ordinal}. The parse is re-rendered chunkwise and asserted == ids (so the parse is exact, never generator internals).
  render_parsed(P) -> (ids, {concept name: slot position}); concepts_in_order(P) -> concept names in first-mention (= slot) order.
  sound_closure(P, marks, kmax=16) -> {keep, stop_s, label, tie, Mp}: §2.4 SOUND-CLOSURE on marks M_1..M_K (iterables of concept NAMES);
      Mp = [M'_0 = set(), M'_1, ..] (stops at the first option pass, or at min(K, kmax)).
  l_roll(m, inputs, dev, ac, K=16, bs=16) / l_rr(m, inputs, dev, ac, kmax=16, bs=16) -> per input {keep, stop_s, label, tie, raw (M_t as slot-ordinal
      sets, t = 1..), Mp_add (M'_t \\ M'_{t-1} as sorted ordinals, t = 1..stop), frontier (sizes), n_rollouts}. run_chain(chain, ...) dispatches.
  make_ac(dev, bf16=True) -> the trainer's ac(): bf16 autocast on a cuda card with capability >= 8, else disabled (fp32; always fp32 on CPU).
  gate1_l(m, dev, ac, items_by_D=None, chains=CHAINS, bs=16) -> the §4.1 labeller dict {gate1 {chain: {D: {...}}, choice, items}, kept_vs_drawn, ...};
      items_by_D = prosqa_sym.gate_set() when None (2,000 per D 9-12, seed 22400). choose(gate1) applies the §2.4 selection rule.
  make_record(inp, res) -> a pool record (keys exactly REC_KEYS); build_pool_part(m, dev, ac, out_dir, D, chain, ...) labels Stream(22300) band D
      with the test + GATE exclusion into out_dir/part_D<D>.jsonl (resumable: state file + byte offset); assemble_pool(out_dir, ...) writes
      out_dir/pool.jsonl (interleaved D 9, 10, 11, 12 by record; D never stored), pool_sigs.txt (side file, pool order) and pool_meta.json.
  load_pool(path, meta_path, verify=True, compact=True) -> (records, meta): sha256 == meta, key set == REC_KEYS, n == meta; verify re-renders
      stmts_parsed == text_ids / slot_idx and re-checks marks_sound's soundness, stop and label.  self_item(rec, kmax) -> a langvb.lvb_loss item.
  replay_self_demand(seed, steps=12000, p_self=0.5, bs=64, self_kextra=4) -> {self_steps, records}; g_l20_check(meta, seed=1, ...) -> dict.
  CLI: python -B -m boxes.langvb_label gate1 --model A1/model.pt --out eval_gate1.json --device cuda:0
       python -B -m boxes.langvb_label pool --model A1/model.pt --chain L-ROLL --out_dir POOL --D 9,10 --device cuda:0   (resumable)
       python -B -m boxes.langvb_label assemble --out_dir POOL --chain L-ROLL

AMBIGUITY (each resolved; prototypes cited where they decide it):
  A1 SOUND-CLOSURE's tie: both options entering M' at the same first pass s is impossible on a SYM v1 item (o_P not in Reach(R), §3.3b, so
     M' (subset of Reach(R) by S-L') never holds o_P), but the spec is silent; such an item is NOT kept (tie = True). If the options enter at
     different passes the first one is the label (the spec's "first such s").
  A2 marks feeding SOUND-CLOSURE: L-ROLL's M_t = the slots whose argmax R_P(V_t) = IN at pass t (not monotone; SOUND-CLOSURE does not need it).
     L-RR's M_t is CUMULATIVE (M_t = M_{t-1} u F_t): a mark made at step t stays a mark, so a concept marked before its parent was soundly marked
     can still enter M' later (the "late chain" that §3.4's scope admits).
  A3 L-RR's first step (the spec defines steps from a frontier, not the start): step 1 runs the ORIGINAL item (= the re-render whose frontier is R's
     own direct statements at their own slots) at K = 2; M_1 = its pass-1 marks (the model's own, not parsed: for a competent labeller they are R's
     direct concepts), F_2 = (pass-2 marks \\ pass-1 marks) \\ M_1. Step t >= 2 re-renders with F_t; F_{t+1} = U_chunks (pass-2 \\ pass-1) \\ M_t.
     Rollouts per chain = 1 + sum_{t=2}^{s-1} ceil(|F_t| / 4) (the prototype v1_sym.py counts sum over all BFS layers of ceil(w / 4) as its cost
     estimate: one more chunk group, its last layer).
  A4 L-RR re-render: R's direct statements are found in the parsed text (subject == R). Frontier nodes are chunked in original slot order
     (chunk k = F[4k : 4k + 4]); chunk member i takes R's i-th original statement slot (text order); surplus original slots (chunk smaller than r)
     are deleted; members beyond r are inserted one by one at a uniform sentence boundary of the current statement list (0..len, i.e. also
     before the first statement and just before the question), rng = random.Random("<RR_SEED>:<sha1 of the item's ids>:<t>:<k>") (deterministic,
     unregistered seed RR_SEED 21850; no reading depends on which boundary). The question is unchanged. Slots of the re-render are recomputed
     (first-mention last token, the same rule as prosqa_sym.render); a concept whose only mention was an R statement is absent from it. Marks map
     back by concept name (names are unique per item).
  A5 L-RR stopping: at the first KEEP pass; at t = kmax (16); when F_{t+1} is empty the chain makes no more rollouts but SOUND-CLOSURE runs on
     M_t held constant up to kmax (late soundness can still reach an option).
  A6 L-ROLL runs K' = 16 passes for every item (batched); SOUND-CLOSURE is read up to 16. "frontier" for L-ROLL = |M_t \\ U_{s<t} M_s| per pass
     (first-time raw marks), for L-RR = |F_t| (|M_1| at t = 1); frontier_hist pools these over steps and items.
  A7 GATE-1-L fields (§4.1 names; definitions not given in the spec): yield = kept / n; pi_label = share of kept with label == gold o_R;
     omit_rate = mean over kept items of |R_s \\ M'_s| / |R_s| (gold REACH at the stop pass; R_s = R_D since s >= D); false_in = raw marks (c, t)
     with dist(R, c) > t over all raw marks the chain produced (L-ROLL: passes 1..16; L-RR: M_t for the steps run), false_in_unreach the same
     with c not in Reach(R) at all; stop_lag = mean (s - D) over kept (+ hist). kept_vs_drawn per D: T, concepts (|S|), cross (C_RP + C_PR),
     decoys (both worlds), frontier = the item's widest gold BFS concept layer (the prototype v1_sym.py `widths`); summaries for drawn and kept.
     The top-level kept_vs_drawn is the chosen chain's; both chains are under kept_vs_drawn_by_chain.
  A8 Selection (§2.4 + N9, Addendum 1 item 7): pooled kept yield over D 11-12 (sum kept / sum n); L-RR iff yield(L-RR) - yield(L-ROLL) >= 0.05,
     else L-ROLL. GATE-1-L pass = pi_label 1.000 at every D under the chosen chain AND no wrong label in either chain (LABEL-BUG = a code fault)
     AND yield >= 0.2 at every D 9-12 under the chosen chain.
  A9 Pool record: marks_sound = M' by pass in DELTA form: element t-1 (t = 1..stop_s) = the sorted slot ordinals (indices into slot_idx) first
     entering M' at pass t (M'_t = the union of elements 0..t-1); stop_s = len(marks_sound); label = the option's name; stmts_parsed = parse_ids's
     P. Records stop at s (self targets use M'_s only, §2.4).
  A10 Pool counts: "draws" = items handed to the labeller (accepted Stream draws); signature collisions (redraws) are logged, not counted. A band
     stops when its kept count reaches 106,000 (the draws after the 106,000th kept item in its chunk are not counted) or at 530,000 draws.
     Interleave: record 4k + (D - 9) = band D's k-th kept record; if a band is short the cycle skips it after it runs out (POOL-SHORT is then
     set and A2 is not launched anyway). Per-record signatures go to pool_sigs.txt (pool order, no D) for G-L20's exclusion check.
  A11 load_pool does not refuse a POOL-SHORT pool (it prints a warning): G-L20 (g_l20_check) is the launch gate. compact=True keeps text_ids /
     slot_idx as array('H') (GPT-2 ids < 65,536) and interns the names, to keep 424k records in RAM; self_item returns plain lists.
  A12 bf16: the labeller computes R_P outside autocast on V_t (cast to fp32 if bf16), as langvb.lvb_loss does; argmax ties go to OUT (torch argmax).
"""
import array
import collections
import hashlib
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch

import boxes.prosqa_sym as PS

ENC = PS.ENC; EOT = PS.EOT; OUT, IN = 0, 1
KMAX_LABEL = 16                    # §2.4: s <= 16, L-ROLL K' 16
RR_CHUNK, RR_K, RR_SEED = 4, 2, 21850
CHAINS = ("L-ROLL", "L-RR")
POOL_D = (9, 10, 11, 12); KEPT_PER_D = 106000; DRAW_CAP_PER_D = 530000
SEL_MARGIN = 0.05; YIELD_MIN = 0.2
REC_KEYS = {"text_ids", "slot_idx", "stmts_parsed", "marks_sound", "stop_s", "label"}
GOLD_KEYS = {"D", "dist_R", "dist_P", "o_R", "o_P", "sig", "names", "edges", "levels", "tag", "gold"}

_STM = re.compile(r" (?:Every ([a-z]+)|([A-Z][a-z]*)) is a ([a-z]+)\.")
_QST = re.compile(r" Is ([A-Z][a-z]*) a ([a-z]+) or ([a-z]+)\?")


# ---------------------------------------------------------------------- parsing (text only)
def _person(w):
    return w[:1].isupper()


def render_parsed(P):
    """P {R, options, stmts} -> (ids [EOT] + enc(" " + text), {concept name: last-token index of its first mention}); the chunk encoding of
    prosqa_sym.render (asserted == the full tiktoken encoding)."""
    ids = [EOT]; slot = {}; parts_all = []
    for a, b in P["stmts"]:
        parts = ((" " + a, a), (" is", None), (" a", None), (" " + b, b), (".", None)) if _person(a) else \
            ((" Every", None), (" " + a, a), (" is", None), (" a", None), (" " + b, b), (".", None))
        parts_all.append(parts)
    o1, o2 = P["options"]
    parts_all.append(((" Is", None), (" " + P["R"], None), (" a", None), (" " + o1, o1), (" or", None), (" " + o2, o2), ("?", None)))
    for parts in parts_all:
        for s, c in parts:
            ids += PS._tk(s)
            if c is not None and c not in slot and not _person(c):
                slot[c] = len(ids) - 1
    return ids, slot


def _text(P):
    return " ".join(("%s is a %s." if _person(a) else "Every %s is a %s.") % (a, b) for a, b in P["stmts"]) + \
        " Is %s a %s or %s?" % (P["R"], P["options"][0], P["options"][1])


def concepts_in_order(P):
    """concept names in first-mention order (= slot position order): statement subjects / objects in text order, then the options."""
    seen = {}; out = []
    for a, b in P["stmts"]:
        for w in (a, b):
            if not _person(w) and w not in seen:
                seen[w] = 1; out.append(w)
    for w in P["options"]:
        if w not in seen:
            seen[w] = 1; out.append(w)
    return out


def _input_of(P, ids, check_full=True):
    ids2, slot = render_parsed(P)
    if ids is not None and list(ids2) != list(ids):
        raise AssertionError("re-rendered parse != text_ids")
    if check_full and ids2[1:] != ENC.encode(" " + _text(P)):
        raise AssertionError("chunked ids != full encoding")
    cs = concepts_in_order(P); si = [slot[c] for c in cs]
    assert si == sorted(si) and len(set(si)) == len(si), "slot order != first-mention order"
    if len(ids2) > 1024:
        raise AssertionError("T > 1024")
    return {"ids": ids2, "P": P, "slot_idx": si, "concepts": cs, "ord": {c: j for j, c in enumerate(cs)}}


def parse_ids(ids):
    """§2.4 / PROOFS §3: the ORIGINAL item's statements and question parsed from its GPT-2 ids (decoded with tiktoken gpt2)."""
    ids = list(ids)
    if ids[0] != EOT:
        raise AssertionError("ids must start with EOT")
    s = ENC.decode(ids[1:]); pos = 0; st = []
    while True:
        mt = _STM.match(s, pos)
        if mt is None:
            break
        st.append([mt.group(1) or mt.group(2), mt.group(3)]); pos = mt.end()
    q = _QST.fullmatch(s, pos)
    if q is None:
        raise AssertionError("question not parsed at offset %d" % pos)
    P = {"R": q.group(1), "options": [q.group(2), q.group(3)], "stmts": st}
    return _input_of(P, ids)


# ---------------------------------------------------------------------- SOUND-CLOSURE (§2.4, Lemma S-L' §3.4)
def _closure_tables(P):
    R = P["R"]; direct = set(); par = collections.defaultdict(set)
    for a, b in P["stmts"]:
        if a == R:
            direct.add(b)
        elif not _person(a):
            par[b].add(a)
    return direct, par


def sound_closure(P, marks, kmax=KMAX_LABEL, tables=None):
    """M'_0 = {}; M'_t = M'_{t-1} u {c in M_t : "<R> is a c." or some a in M'_{t-1} with "Every a is a c."}, t = 1..min(len(marks), kmax).
    KEEP iff an option is in M'_s for some s <= kmax (first such s = stop_s, label = that option); both options at the first s -> not kept
    (AMBIGUITY A1). marks: list over t = 1.. of iterables of concept names (persons / unknown names never enter: they have no statement)."""
    direct, par = tables if tables is not None else _closure_tables(P)
    o1, o2 = P["options"]; Mp = [set()]
    for t in range(1, min(len(marks), kmax) + 1):
        prev = Mp[-1]; new = set(prev)
        for c in marks[t - 1]:
            if c not in prev and (c in direct or (par.get(c) and not par[c].isdisjoint(prev))):
                new.add(c)
        Mp.append(new)
        h1, h2 = o1 in new, o2 in new
        if h1 and h2:
            return {"keep": False, "stop_s": None, "label": None, "tie": True, "Mp": Mp}
        if h1 or h2:
            return {"keep": True, "stop_s": t, "label": o1 if h1 else o2, "tie": False, "Mp": Mp}
    return {"keep": False, "stop_s": None, "label": None, "tie": False, "Mp": Mp}


def _mp_add(inp, Mp):
    o = inp["ord"]; return [sorted(o[c] for c in Mp[t] - Mp[t - 1]) for t in range(1, len(Mp))]


# ---------------------------------------------------------------------- the labeller chains
def make_ac(dev, bf16=True):
    """the trainer's ac(): bf16 autocast on cuda (capability >= 8) unless bf16 False; fp32 on CPU."""
    amp = torch.bfloat16 if (bf16 and str(dev).startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8) else None
    f = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    f.dtype = str(amp) if amp is not None else "fp32"
    return f


def _batch(inps, dev):
    T = max(len(u["ids"]) for u in inps); S = max(len(u["slot_idx"]) for u in inps); B = len(inps)
    x = torch.full((B, T), EOT, dtype=torch.long); pad = torch.zeros(B, T, dtype=torch.bool)
    si = torch.zeros(B, S, dtype=torch.long); sp = torch.zeros(B, S, dtype=torch.bool)
    for b, u in enumerate(inps):
        n = len(u["ids"]); x[b, :n] = torch.tensor(u["ids"]); pad[b, :n] = True; k = len(u["slot_idx"])
        si[b, :k] = torch.tensor(u["slot_idx"]); sp[b, :k] = True
    return x.to(dev), pad.to(dev), si.to(dev), sp.to(dev)


@torch.no_grad()
def _marks(m, inps, dev, ac, K, bs):
    """frozen rollout to K passes -> per input: [set of slot ordinals with argmax R_P(V_t) = IN for t = 1..K]."""
    out = [None] * len(inps); order = sorted(range(len(inps)), key=lambda i: -len(inps[i]["ids"]))
    for c0 in range(0, len(order), bs):
        part = order[c0:c0 + bs]; x, pad, si, sp = _batch([inps[i] for i in part], dev)
        with ac():
            r = m.rollout_ls(x, pad, si, sp, K)
        per = []
        for t in range(1, K + 1):
            V = r["V"][t]; V = V.float() if V.dtype in (torch.bfloat16, torch.float16) else V
            per.append(((m.rec_logits(V).argmax(-1) == IN) & sp).cpu().numpy())
        del r
        for b, i in enumerate(part):
            out[i] = [set(np.nonzero(per[t][b])[0].tolist()) for t in range(K)]
    return out


def l_roll(m, inps, dev, ac, K=KMAX_LABEL, bs=16):
    """L-ROLL (§2.4): the frozen model's own rollout to K' 16; M_t = {c : argmax R_P(V_t)[c] = IN}; SOUND-CLOSURE on the original parse."""
    mk = _marks(m, inps, dev, ac, K, bs); res = []
    for u, raw in zip(inps, mk):
        cs = u["concepts"]; sc = sound_closure(u["P"], [{cs[j] for j in s_} for s_ in raw])
        seen = set(); fr = []
        for s_ in raw:
            fr.append(len(s_ - seen)); seen |= s_
        res.append({"keep": sc["keep"], "stop_s": sc["stop_s"], "label": sc["label"], "tie": sc["tie"], "raw": raw, "Mp_add": _mp_add(u, sc["Mp"]),
                    "frontier": fr, "n_rollouts": 1})
    return res


def rerender(P, chunk, rng):
    """AMBIGUITY A4: R's direct statements replaced by "R is a f." for f in chunk (<= 4): at R's original statement slots in text order, surplus
    slots deleted, extra members at uniform sentence boundaries."""
    R = P["R"]; new = []; k = 0
    for a, b in P["stmts"]:
        if a == R:
            if k < len(chunk):
                new.append([R, chunk[k]]); k += 1
        else:
            new.append([a, b])
    for f in chunk[k:]:
        new.insert(rng.randint(0, len(new)), [R, f])
    return {"R": R, "options": list(P["options"]), "stmts": new}


def _ids_key(ids):
    return hashlib.sha1(json.dumps(list(ids)).encode()).hexdigest()[:16]


def l_rr(m, inps, dev, ac, kmax=KMAX_LABEL, bs=16, chunk=RR_CHUNK, seed=RR_SEED):
    """chunked L-RR (§2.4 v1; AMBIGUITY A3-A5). Per step t the active items' frontier chunks are re-rendered and batched (K = 2)."""
    n = len(inps); keys = [_ids_key(u["ids"]) for u in inps]; tabs = [_closure_tables(u["P"]) for u in inps]
    first = _marks(m, inps, dev, ac, RR_K, bs)                              # step 1: the original item, K = 2 (AMBIGUITY A3)
    M = [set(p1) for p1, _ in first]; F = [(p2 - p1) - set(p1) for p1, p2 in first]     # M_1, F_2
    raw = [[set(M[i])] for i in range(n)]; fr = [[len(M[i])] for i in range(n)]; nroll = [1] * n; res = [None] * n
    t = 1
    while True:
        for i in range(n):                                                  # SOUND-CLOSURE on M_1..M_t (M held constant once F is empty)
            if res[i] is None:
                cs = inps[i]["concepts"]; sc = sound_closure(inps[i]["P"], [{cs[j] for j in s_} for s_ in raw[i]], kmax, tabs[i])
                if sc["keep"] or sc["tie"] or t == kmax:
                    res[i] = {"keep": sc["keep"], "stop_s": sc["stop_s"], "label": sc["label"], "tie": sc["tie"], "raw": raw[i],
                              "Mp_add": _mp_add(inps[i], sc["Mp"]), "frontier": fr[i], "n_rollouts": nroll[i]}
        act = [i for i in range(n) if res[i] is None]
        if t == kmax or not act:
            break
        if t >= 2:                                                          # F_{t+1} = U_chunks (pass-2 \ pass-1 marks of the F_t re-render) \ M_t
            jobs = []
            for i in act:
                cs = inps[i]["concepts"]; Fn = [cs[j] for j in sorted(F[i])]; F[i] = set()
                for k in range(0, len(Fn), chunk):
                    rng = random.Random("%d:%s:%d:%d" % (seed, keys[i], t, k // chunk))
                    jobs.append((i, _input_of(rerender(inps[i]["P"], Fn[k:k + chunk], rng), None, check_full=True)))
            if jobs:
                mk = _marks(m, [u for _, u in jobs], dev, ac, RR_K, bs)
                for (i, u), (p1, p2) in zip(jobs, mk):
                    nroll[i] += 1; o = inps[i]["ord"]; F[i] |= {o[u["concepts"][j]] for j in (p2 - p1)}
            for i in act:
                F[i] -= M[i]
        t += 1
        for i in act:                                                       # M_t = M_{t-1} u F_t (cumulative; AMBIGUITY A2)
            M[i] = M[i] | F[i]; raw[i].append(set(M[i])); fr[i].append(len(F[i]))
    return res


def run_chain(chain, m, inps, dev, ac, bs=16):
    if chain == "L-ROLL":
        return l_roll(m, inps, dev, ac, KMAX_LABEL, bs)
    if chain == "L-RR":
        return l_rr(m, inps, dev, ac, KMAX_LABEL, bs)
    raise ValueError(chain)


# ---------------------------------------------------------------------- GATE-1-L (§2.4, §4.1, G-L15)
def _hexmask(s_, S):
    v = 0
    for j in s_:
        v |= 1 << j
    return format(v, "x")


def _gold(it, inp):
    nid = {nm: i for i, nm in enumerate(it["names"])}; dR = it["dist_R"]
    d = [dR.get(nid[c], -1) for c in inp["concepts"]]
    lay = collections.Counter(x for x in d if x > 0)
    return {"D": it["D"], "o_R": it["names"][it["o_R"]], "dist": d, "T": it["T"], "concepts": len(inp["concepts"]),
            "cross": int(sum(it["n_cross"])), "decoys": int(sum(it["n_decoys"])), "frontier": max(lay.values()) if lay else 0}


def _summ(v):
    if not v:
        return {"n": 0}
    v = sorted(v); n = len(v)
    return {"n": n, "mean": round(sum(v) / n, 3), "p05": v[int(0.05 * (n - 1))], "p50": v[n // 2], "p95": v[int(0.95 * (n - 1))], "min": v[0], "max": v[-1],
            "hist": dict(sorted(collections.Counter(v).items())) if len(set(v)) <= 60 else None}


def gate1_stats(golds, res):
    """per-D GATE-1-L numbers for one chain (AMBIGUITY A7)."""
    n = len(res); kept = [k for k, r in enumerate(res) if r["keep"]]; nk = len(kept)
    wrong = [k for k in kept if res[k]["label"] != golds[k]["o_R"]]
    om, fin, fin_u, nraw, lag, fh = [], 0, 0, 0, [], collections.Counter()
    for k, (g, r) in enumerate(zip(golds, res)):
        d = g["dist"]
        for t, s_ in enumerate(r["raw"], 1):
            for j in s_:
                nraw += 1; fin += not (0 < d[j] <= t); fin_u += d[j] < 0
        fh.update(r["frontier"])
        if r["keep"]:
            s = r["stop_s"]; Rs = {j for j, x in enumerate(d) if 0 < x <= s}; Ms = {j for a in r["Mp_add"] for j in a}
            om.append(len(Rs - Ms) / max(1, len(Rs))); lag.append(s - g["D"])
    return {"n": n, "kept": nk, "yield": nk / n if n else None, "pi_label": (nk - len(wrong)) / nk if nk else None, "n_wrong_label": len(wrong),
            "n_tie": sum(r["tie"] for r in res), "omit_rate": float(np.mean(om)) if om else None, "false_in": fin / nraw if nraw else None,
            "false_in_unreach": fin_u / nraw if nraw else None, "n_raw_marks": nraw, "stop_lag": float(np.mean(lag)) if lag else None,
            "stop_lag_hist": dict(sorted(collections.Counter(lag).items())), "frontier_hist": dict(sorted(fh.items())),
            "rollouts_mean": float(np.mean([r["n_rollouts"] for r in res])) if res else None}


def kept_vs_drawn(golds, res):
    kept = [g for g, r in zip(golds, res) if r["keep"]]
    return {f: {"drawn": _summ([g[f] for g in golds]), "kept": _summ([g[f] for g in kept])} for f in ("T", "concepts", "cross", "decoys", "frontier")}


def choose(g1, chains=CHAINS):
    """§2.4 selection rule (AMBIGUITY A8): pooled kept yield over D 11-12; L-RR only if it beats L-ROLL by >= 0.05."""
    y = {}
    for c in chains:
        n = sum(g1[c][D]["n"] for D in (11, 12) if D in g1[c]); k = sum(g1[c][D]["kept"] for D in (11, 12) if D in g1[c])
        y[c] = k / n if n else None
    if "L-RR" not in y or "L-ROLL" not in y:
        return chains[0], y
    return ("L-RR" if (y["L-RR"] is not None and y["L-ROLL"] is not None and y["L-RR"] - y["L-ROLL"] >= SEL_MARGIN) else "L-ROLL"), y


def gate1_l(m, dev, ac, items_by_D=None, chains=CHAINS, bs=16, log=print):
    """G-L15 GATE-1-L: both chains on the GATE items (gold read only here), the §4.1 labeller fields + the pass rule."""
    t0 = time.time(); m.eval()
    items_by_D = items_by_D if items_by_D is not None else PS.gate_set()
    g1 = {c: {} for c in chains}; store = {c: [] for c in chains}; kvd = {c: {} for c in chains}; sec = {}
    for D in sorted(items_by_D):
        its = items_by_D[D]; inps = [parse_ids(it["ids"]) for it in its]; golds = [_gold(it, u) for it, u in zip(its, inps)]
        for c in chains:
            t1 = time.time(); res = run_chain(c, m, inps, dev, ac, bs); sec["%s_D%d" % (c, D)] = round(time.time() - t1, 1)
            g1[c][D] = gate1_stats(golds, res); kvd[c][D] = kept_vs_drawn(golds, res)
            for k, (g, u, r) in enumerate(zip(golds, inps, res)):
                S = len(u["concepts"])
                store[c].append({"D": D, "i": k, "keep": r["keep"], "stop_s": r["stop_s"], "label": r["label"], "gold_o_R": g["o_R"], "tie": r["tie"],
                                 "marks_hex": [_hexmask(s_, S) for s_ in r["raw"]], "marks_sound": r["Mp_add"], "frontier": r["frontier"],
                                 "n_rollouts": r["n_rollouts"], "gold_dist": g["dist"], "T": g["T"]})
            log("GATE1 %s D %d: kept %d / %d, pi %s, %.0f s" % (c, D, g1[c][D]["kept"], g1[c][D]["n"], g1[c][D]["pi_label"], sec["%s_D%d" % (c, D)]))
    choice, y1112 = choose(g1, chains)
    bug = any(g1[c][D]["n_wrong_label"] for c in chains for D in g1[c])
    ok_pi = all(g1[choice][D]["pi_label"] in (None, 1.0) for D in g1[choice]) and not bug
    ok_y = all(D in g1[choice] and (g1[choice][D]["yield"] or 0.0) >= YIELD_MIN for D in POOL_D)
    out = {"gate1": {**{c: {str(D): v for D, v in g1[c].items()} for c in chains}, "choice": choice, "items": store},
           "kept_vs_drawn": {str(D): v for D, v in kvd[choice].items()}, "kept_vs_drawn_by_chain": {c: {str(D): v for D, v in kvd[c].items()} for c in chains},
           "yield_D11_12": y1112, "label_bug": bug, "pass_pi_label": ok_pi, "pass_yield": ok_y, "pass": bool(ok_pi and ok_y),
           "rule": "choice: pooled kept yield D 11-12, L-RR iff it beats L-ROLL by >= %.2f; pass: pi_label 1.000 (no wrong label in any chain) and yield >= %.1f at every D 9-12 (chosen chain)" % (SEL_MARGIN, YIELD_MIN),
           "dtype": getattr(ac, "dtype", None), "c_anneal": float(m.c_anneal), "n_per_D": {str(D): len(v) for D, v in items_by_D.items()}, "sec": sec,
           "sec_total": round(time.time() - t0, 1)}
    return out


# ---------------------------------------------------------------------- pool records, builder, loader
def make_record(inp, res):
    """a kept labeller result -> the pool record (keys exactly REC_KEYS; AMBIGUITY A9)."""
    assert res["keep"] and len(res["Mp_add"]) == res["stop_s"]
    rec = {"text_ids": list(inp["ids"]), "slot_idx": list(inp["slot_idx"]), "stmts_parsed": inp["P"], "marks_sound": res["Mp_add"],
           "stop_s": int(res["stop_s"]), "label": res["label"]}
    assert set(rec) == REC_KEYS
    return rec


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="\n") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _part_paths(out_dir, D):
    b = os.path.join(out_dir, "part_D%d" % D); return b + ".jsonl", b + ".sigs", b + ".state.json"


def build_pool_part(m, dev, ac, out_dir, D, chain, kept_target=KEPT_PER_D, draw_cap=DRAW_CAP_PER_D, chunk=1024, bs=16, excl=None,
                    seed=PS.SEED_POOL, max_draws=None, log=print, tag=None):
    """§2.4 / §8: label band D of Stream(seed) (test + GATE signatures excluded) with `chain` until kept_target kept or draw_cap draws
    (AMBIGUITY A10). Resumable: part_D<D>.state.json holds the stream rng state, counts and the part files' byte lengths (a crash mid-chunk is
    truncated back on resume). max_draws stops early (smokes / a planned stop); the band is `done` only at kept_target or draw_cap.
    tag = provenance {model, model_sha256, card} (checked equal on resume; assemble_pool checks model / dtype / chain agree across bands)."""
    os.makedirs(out_dir, exist_ok=True); m.eval(); pj, ps_, pst = _part_paths(out_dir, D)
    excl = excl if excl is not None else PS.exclusion_sigs(); stream = PS.Stream(seed, excl)
    st = {"D": D, "chain": chain, "seed": seed, "drawn": 0, "kept": 0, "collisions": [], "bytes": 0, "bytes_sigs": 0, "sec": 0.0, "chunks": [],
          "done": False, "kept_target": kept_target, "draw_cap": draw_cap, "dtype": getattr(ac, "dtype", None), "c_anneal": float(m.c_anneal),
          "tag": tag or {}}
    if os.path.exists(pst):
        st = json.load(open(pst)); assert st["chain"] == chain and st["seed"] == seed and st["D"] == D, "resume: chain / seed / D differ"
        assert st.get("dtype") == getattr(ac, "dtype", None) and (tag is None or st.get("tag", {}).get("model_sha256") == tag.get("model_sha256")), \
            "resume: model / dtype differ"
        rs = st["rng_state"]; stream.rngs[D] = random.Random(); stream.rngs[D].setstate((rs[0], tuple(rs[1]), rs[2])); stream.n[D] = st["stream_n"]
        for p_, nb in ((pj, st["bytes"]), (ps_, st["bytes_sigs"])):
            with open(p_, "ab") as f:
                f.truncate(nb)
        log("RESUME D %d: drawn %d kept %d" % (D, st["drawn"], st["kept"]))
    else:
        for p_ in (pj, ps_):
            open(p_, "wb").close()
    stop = st["drawn"] + max_draws if max_draws else None
    while not st["done"] and (stop is None or st["drawn"] < stop):
        n = min(chunk, draw_cap - st["drawn"]) if stop is None else min(chunk, draw_cap - st["drawn"], stop - st["drawn"])
        t1 = time.time(); nc0 = len(stream.collisions); its = [stream.draw(D) for _ in range(n)]; tg = time.time() - t1
        inps = [parse_ids(it["ids"]) for it in its]; res = run_chain(chain, m, inps, dev, ac, bs); used = 0; recs = []; sigs = []
        for it, u, r in zip(its, inps, res):
            used += 1
            if r["keep"]:
                recs.append(make_record(u, r)); sigs.append(it["sig"])
                if st["kept"] + len(recs) >= kept_target:
                    break
        with open(pj, "ab") as f:
            for r_ in recs:
                f.write((json.dumps(r_, separators=(",", ":")) + "\n").encode())
        with open(ps_, "ab") as f:
            for s_ in sigs:
                f.write((s_ + "\n").encode())
        dt = time.time() - t1
        st["drawn"] += used; st["kept"] += len(recs); st["bytes"] = os.path.getsize(pj); st["bytes_sigs"] = os.path.getsize(ps_); st["sec"] += dt
        st["collisions"] += stream.collisions[nc0:]
        st["chunks"].append({"n": used, "kept": len(recs), "sec": round(dt, 2), "sec_gen": round(tg, 2), "sec_per_drawn": round(dt / max(1, used), 4)})
        st["done"] = st["kept"] >= kept_target or st["drawn"] >= draw_cap
        rs = stream.rngs[D].getstate(); st["rng_state"] = [rs[0], list(rs[1]), rs[2]]; st["stream_n"] = stream.n[D]
        _jdump(st, pst)
        log("POOL D %d: drawn %d kept %d (yield %.3f), %.3f s / drawn" % (D, st["drawn"], st["kept"], st["kept"] / max(1, st["drawn"]), dt / max(1, used)))
    return st


def assemble_pool(out_dir, Ds=POOL_D, chain=None, kept_target=KEPT_PER_D, draw_cap=DRAW_CAP_PER_D, extra=None, require_done=True):
    """§2.4: pool.jsonl in the fixed interleaved order (D 9, 10, 11, 12 cycling by record; AMBIGUITY A10), pool_sigs.txt, pool_meta.json."""
    sts = {}; lines = {}; sg = {}
    for D in Ds:
        pj, ps_, pst = _part_paths(out_dir, D); st = json.load(open(pst)); sts[D] = st
        assert not require_done or st["done"], "band D %d not done" % D
        assert chain is None or st["chain"] == chain, "chain differs in D %d" % D
        with open(pj, "rb") as f:
            lines[D] = f.read(st["bytes"]).splitlines(keepends=True)
        with open(ps_, "rb") as f:
            sg[D] = f.read(st["bytes_sigs"]).splitlines(keepends=True)
        assert len(lines[D]) == st["kept"] == len(sg[D]), "D %d: part lines != kept" % D
    for k_ in ("chain", "seed", "dtype", "c_anneal", "kept_target", "draw_cap"):
        assert len({json.dumps(sts[D].get(k_)) for D in Ds}) == 1, "bands disagree on %s" % k_
    assert len({sts[D].get("tag", {}).get("model_sha256") for D in Ds}) == 1, "bands labelled by different models"
    pool = os.path.join(out_dir, "pool.jsonl"); sigp = os.path.join(out_dir, "pool_sigs.txt"); cur = {D: 0 for D in Ds}; n = 0
    with open(pool + ".tmp", "wb") as f, open(sigp + ".tmp", "wb") as g:
        while any(cur[D] < len(lines[D]) for D in Ds):
            for D in Ds:
                if cur[D] < len(lines[D]):
                    ln = lines[D][cur[D]]; assert ln.endswith(b"\n") and not ln.endswith(b"\r\n")
                    assert set(json.loads(ln)) == REC_KEYS
                    f.write(ln); g.write(sg[D][cur[D]]); cur[D] += 1; n += 1
    os.replace(pool + ".tmp", pool); os.replace(sigp + ".tmp", sigp)
    short = [D for D in Ds if sts[D]["kept"] < kept_target]
    try:
        import subprocess
        head = subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip() or None
    except Exception:
        head = None
    meta = {"sha256": file_sha256(pool), "n_records": n, "rec_keys": sorted(REC_KEYS),
            "counts": {str(D): {"drawn": sts[D]["drawn"], "kept": sts[D]["kept"], "yield": sts[D]["kept"] / max(1, sts[D]["drawn"]),
                                "collisions": len(sts[D]["collisions"]), "sec": round(sts[D]["sec"], 1),
                                "sec_per_drawn": sts[D]["sec"] / max(1, sts[D]["drawn"])} for D in Ds},
            "POOL_SHORT": bool(short), "pool_short_D": short, "kept_target_per_D": kept_target, "draw_cap_per_D": draw_cap,
            "chain": sts[Ds[0]]["chain"], "seeds": {"pool_stream": sts[Ds[0]]["seed"], "rr_seed": RR_SEED,
                                                     "exclusion": "test 22200 (1,000 per D 3-16) + GATE 22400 (2,000 per D 9-12) role-symmetric signatures"},
            "order": "interleaved D %s cycling by record (an exhausted band is skipped); D not stored" % ",".join(map(str, Ds)),
            "builder_commit": "PLACEHOLDER (fill with the registration commit of boxes/langvb_label.py)", "git_head_at_assemble": head,
            "collisions": {str(D): sts[D]["collisions"] for D in Ds}, "dtype": sts[Ds[0]].get("dtype"), "c_anneal": sts[Ds[0]].get("c_anneal"),
            "sigs_file": "pool_sigs.txt", "sigs_sha256": file_sha256(sigp), "kmax_label": KMAX_LABEL,
            "tags": {str(D): sts[D].get("tag", {}) for D in Ds}, **(extra or {})}
    _jdump(meta, os.path.join(out_dir, "pool_meta.json"))
    return meta


def check_record(rec):
    """the loader's per-record verification: stmts_parsed re-renders to text_ids / slot_idx; marks_sound is a sound chain (S-L' rule on the
    record's own statements) whose first option pass is stop_s and whose option is the label."""
    P = rec["stmts_parsed"]; ids, slot = render_parsed(P)
    assert ids == list(rec["text_ids"]), "stmts_parsed != text_ids"
    cs = concepts_in_order(P); assert [slot[c] for c in cs] == list(rec["slot_idx"]), "slot_idx != first-mention positions"
    direct, par = _closure_tables(P); s = rec["stop_s"]; ms = rec["marks_sound"]; assert 1 <= s <= KMAX_LABEL and len(ms) == s
    prev = set(); opts = set(P["options"])
    for t, add in enumerate(ms, 1):
        names = {cs[j] for j in add}; assert len(names) == len(add) and not (names & prev)
        for c in names:
            assert c in direct or not par.get(c, set()).isdisjoint(prev), "unsound mark %s at pass %d" % (c, t)
        prev = prev | names; hit = prev & opts
        assert (len(hit) == 1) if t == s else not hit, "option pass != stop_s"
    assert rec["label"] in prev and rec["label"] in opts
    return True


def load_pool(path, meta_path, verify=True, compact=True):
    """the trainer's self-pool loader: sha256 == pool_meta.json's; key set == REC_KEYS (no gold field); n == meta; verify -> check_record per
    record; compact -> text_ids / slot_idx as array('H'), names interned (AMBIGUITY A11)."""
    meta = json.load(open(meta_path)); sha = file_sha256(path); assert sha == meta["sha256"], ("pool sha mismatch", sha, meta["sha256"])
    recs = []
    with open(path, "rb") as f:
        for ln in f:
            assert ln.endswith(b"\n") and not ln.endswith(b"\r\n"), "pool line ending"
            r = json.loads(ln); assert set(r) == REC_KEYS, ("pool record keys", sorted(r)); assert not (set(r) & GOLD_KEYS)
            if verify:
                check_record(r)
            if compact:
                r["text_ids"] = array.array("H", r["text_ids"]); r["slot_idx"] = array.array("H", r["slot_idx"])
                P = r["stmts_parsed"]; P["R"] = sys.intern(P["R"]); P["options"] = [sys.intern(w) for w in P["options"]]
                P["stmts"] = [(sys.intern(a), sys.intern(b)) for a, b in P["stmts"]]; r["label"] = sys.intern(r["label"])
            recs.append(r)
    assert len(recs) == meta["n_records"], ("pool n_records", len(recs), meta["n_records"])
    if meta.get("POOL_SHORT"):
        print("WARNING: POOL-SHORT pool (%s): A2 is not launched on it (G-L20)" % meta.get("pool_short_D"), flush=True)
    return recs, meta


def self_item(rec, kmax=KMAX_LABEL):
    """a pool record -> a langvb.lvb_loss / lvb_batch / reach_targets item: ids, slot_idx (sorted), concepts (slot order), marks (per slot the
    first M' pass or None), stop_s, o_R = the label, o_P = the other displayed option (parsed question), D None (lvb_loss reads it.get("stop_s",
    it["D"]))."""
    P = rec["stmts_parsed"]; cs = concepts_in_order(P); si = list(rec["slot_idx"]); s = int(rec["stop_s"])
    assert len(cs) == len(si) and si == sorted(si) and 1 <= s <= kmax and len(rec["marks_sound"]) == s
    marks = [None] * len(cs)
    for t, add in enumerate(rec["marks_sound"], 1):
        for j in add:
            assert marks[j] is None; marks[j] = t
    o1, o2 = P["options"]; lab = rec["label"]; assert lab in (o1, o2) and marks[cs.index(lab)] == s
    return {"ids": list(rec["text_ids"]), "slot_idx": si, "concepts": cs, "marks": marks, "stop_s": s, "o_R": lab, "o_P": o2 if lab == o1 else o1, "D": None}


# ---------------------------------------------------------------------- G-L20: the trainer's rng_s replay
def replay_self_demand(seed, steps=12000, p_self=0.5, bs=64, self_kextra=4):
    """langvb.main draw_step's rng_s order: rng_s = default_rng(seed + 21800); every step one rng_s.random() < p_self coin (the plain batch is
    drawn first with rng / rng_k, which rng_s never sees); on a self step one rng_s.integers(0, self_kextra + 1) per record (bs records)."""
    r = np.random.default_rng(seed + 21800); n = 0
    for _ in range(steps):
        if bool(r.random() < p_self):
            n += 1
            for _ in range(bs):
                r.integers(0, self_kextra + 1)
    return {"seed": seed, "self_steps": n, "records": n * bs, "per_D": n * bs / len(POOL_D)}


def g_l20_check(meta, seed=1, steps=12000, p_self=0.5, bs=64, self_kextra=4, kept_target=KEPT_PER_D):
    """§7 G-L20 (the pool_meta half): kept per D >= 106,000, total >= 1.1 x the seed's replayed demand, not POOL-SHORT, n_records = sum kept."""
    dem = replay_self_demand(seed, steps, p_self, bs, self_kextra); kept = {D: meta["counts"][D]["kept"] for D in meta["counts"]}
    tot = sum(kept.values())
    out = {"demand": dem, "kept": kept, "total": tot, "kept_ok": all(v >= kept_target for v in kept.values()), "margin_ok": tot >= 1.1 * dem["records"],
           "not_short": not meta.get("POOL_SHORT"), "n_records_ok": meta["n_records"] == tot}
    out["pass"] = all(out[k] for k in ("kept_ok", "margin_ok", "not_short", "n_records_ok"))
    return out


# ---------------------------------------------------------------------- CLI
def _load_model(path, dev):
    from boxes import s5
    import boxes.langvb as LV
    m = LV.make_model(s5.load_gpt2(dev)).to(dev)
    if path:
        sd = torch.load(path, map_location=dev, weights_only=True); m.load_state_dict(sd, strict=True); del sd
    m.eval(); return m


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gate1", "pool", "assemble"]); ap.add_argument("--model", default="")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--fp32", action="store_true"); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default=""); ap.add_argument("--out_dir", default=""); ap.add_argument("--chain", default="L-ROLL", choices=list(CHAINS))
    ap.add_argument("--D", default="9,10,11,12"); ap.add_argument("--chunk", type=int, default=1024); ap.add_argument("--max_draws", type=int, default=0)
    ap.add_argument("--kept_target", type=int, default=KEPT_PER_D); ap.add_argument("--draw_cap", type=int, default=DRAW_CAP_PER_D)
    a = ap.parse_args(); Ds = [int(v) for v in a.D.split(",")]
    if a.cmd == "assemble":
        print(json.dumps({k: v for k, v in assemble_pool(a.out_dir, tuple(Ds), a.chain, a.kept_target, a.draw_cap).items() if k != "collisions"}, indent=1)); return
    m = _load_model(a.model, a.device); ac = make_ac(a.device, not a.fp32)
    card = torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu"
    if a.cmd == "gate1":
        out = gate1_l(m, a.device, ac, bs=a.bs); out.update({"model": a.model, "model_sha256": file_sha256(a.model) if a.model else None, "card": card}); _jdump(out, a.out)
        print("GATE1 choice", out["gate1"]["choice"], "pass", out["pass"], flush=True); return
    excl = PS.exclusion_sigs(); tag = {"model": a.model, "model_sha256": file_sha256(a.model) if a.model else None, "card": card}
    for D in Ds:
        build_pool_part(m, a.device, ac, a.out_dir, D, a.chain, a.kept_target, a.draw_cap, a.chunk, a.bs, excl, max_draws=a.max_draws or None, tag=tag)


if __name__ == "__main__":
    main()
