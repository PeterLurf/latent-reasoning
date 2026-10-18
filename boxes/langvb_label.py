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
  make_ac(dev, bf16=True) -> the trainer's ac(): bf16 autocast on a cuda card with capability >= 8, else disabled; ac.dtype = "bf16" | "fp32".
  gate1_l(m, dev, ac, items_by_D=None, chains=CHAINS, bs=16, chunk=1024, tag=None, require_bf16=True) -> eval_gate1.json (score_s218 LAYOUTS):
      meta {card, dtype "bf16", bs, chunk, model_sha256, ...}, choice, chains {L-ROLL|L-RR: {"9".."12": {n, kept, yield, omit_rate, false_in,
      stop_lag, frontier_hist, n_wrong_label, n_tie, n_unsound, ...}}}, items {chain: {D: [{keep, label, gold, stop, ...}]}} (label and gold =
      option NAME strings), kept_vs_drawn {D: {T, concepts, cross, decoys, frontier}}, label_bug (wrong label, tie or unsound kept item in either
      chain), pass. items_by_D = prosqa_sym.gate_set() when None (2,000 per D 9-12, seed 22400); inputs run through the chain in chunks of `chunk`
      (as the pool builder does). choose(chains) applies the §2.4 selection rule.
  make_record(inp, res) -> a pool record (keys exactly REC_KEYS).
  build_pool_part(m, dev, ac, out_dir, D, chain, gate1_path, tag, ...) labels Stream(22300) band D (test + GATE exclusion) into
      out_dir/part_D<D>.jsonl: chain / dtype / model_sha256 asserted == eval_gate1.json's choice / meta (and GATE-1-L passed, no LABEL-BUG);
      bs / chunk / card / kept_target / draw_cap / tag / gate1 sha asserted equal on resume; a band-claim lock file; resumable (state + byte offset).
  assemble_pool(out_dir, gate1_path, ...) -> pool.jsonl (interleaved D 9, 10, 11, 12 by record; D never stored), pool_sigs.txt and pool_meta.json
      (score_s218 LAYOUTS: kept_by_D, draws_by_D, total_kept, demand_replay, rng_replay_equal, key_set, sig_hits {test, gate, probe}, gold_key_hits,
      n_tie, label_bug, card_by_band, bs, chunk, builder_commit {head, dirty, ...}, n_dup_sigs, ...).
  probe_stream_sigs(excl, scale=1.0) -> signatures of the Stream(22500) items langvb_eval's probe / FD parts draw (for sig_hits.probe).
  load_pool(path, meta_path, verify=True, compact=True) -> (records, meta): sha256 == meta, key set == REC_KEYS, n == meta; verify re-renders
      stmts_parsed == text_ids / slot_idx and re-checks marks_sound's soundness, stop and label.  self_item(rec, kmax) -> a langvb.lvb_loss item.
  labeller_part(m, dev, pool_path, meta_path, gate1_path, cursor, eval_bs, n_audit=5000, n_memo=2000) -> eval_lvb.json's "labeller" part
      {gate1 (copy: choice, yields, label_bug), kept_vs_drawn (copy), label_audit {n, n_wrong, pi_label, n_unsound, items}, memo {D: {acc_pool,
      acc_fresh, n_pool, n_fresh, items}}, meta {dtype "fp32", eval_bs, card, cursor}}.
  replay_self_demand(seed, steps=12000, p_self=0.5, bs=64, self_kextra=4) -> {self_steps, records}; g_l20_check(meta, seed=1, ...) -> dict.
  git_info() -> {head, dirty, dirty_files, files}; state_sha256(m) -> sha256 of a state dict (model identity for smokes without a model.pt file).
  CLI (gate1 / pool refuse to run without bf16; --chain has no default):
       python -B -m boxes.langvb_label gate1 --model A1/model.pt --out RUNS/lvb_gate1_s1/eval_gate1.json --device cuda:0
       python -B -m boxes.langvb_label pool --model A1/model.pt --gate1 RUNS/lvb_gate1_s1/eval_gate1.json --chain L-ROLL --out_dir RUNS/lvb_pool_s1 --D 9,10
       python -B -m boxes.langvb_label assemble --gate1 RUNS/lvb_gate1_s1/eval_gate1.json --chain L-ROLL --out_dir RUNS/lvb_pool_s1
       python -B -m boxes.langvb_label labeller --model A2/model.pt --gate1 ... --pool RUNS/lvb_pool_s1/pool.jsonl --cursor N --eval_bs 16 --out F

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
  (A1-A12 accepted by Theory, Note 218-3. A13-A20 follow Audit's G-L13 part 2a review.)
  A13 A tie (both options in M' at the same first pass) is a parse / closure fault (o_P not in Reach(R)): counted (n_tie) and part of label_bug in
     GATE-1-L and in the pool. A kept GATE item whose M'_t is not inside the gold R_t, or whose s < D, is counted (n_unsound) and is a label_bug
     too; the gate1 CLI writes the file first, then exits non-zero on label_bug. In the pool every kept record passes check_record (else raise).
  A14 Pool <-> GATE-1-L: the builder and assemble read eval_gate1.json and assert chain == choice, dtype == meta.dtype, model_sha256 ==
     meta.model_sha256, label_bug false and (require_gate_pass) pass true; the gate1 file's sha256 is stored per band and asserted on resume.
  A15 Batch composition (bf16): _marks sorts by length within one run_chain call, so the chunk (draws per call) and bs fix the bf16 batches;
     both are stored per band, asserted equal on resume and across bands at assemble; GATE-1-L uses the same chunking (meta.chunk). The card is
     part of the band's tag: a resume on another card raises (a band never mixes cards); bands may sit on different cards (card_by_band).
  A16 model_sha256 = sha256 of the model.pt file (CLI); a smoke without a file uses "state:" + state_sha256(m).
  A17 Band-claim lock: part_D<D>.lock is created with O_EXCL (pid, host, time) and removed on exit; an existing lock raises (remove it by hand only
     if its process is dead).
  A18 Duplicate signatures inside the pool: signatures include D and the full name-free structure, so two pool records can share one only if
     the generator drew two structurally identical items in one band (never observed); assemble counts them (n_dup_sigs) and does not drop any.
  A19 sig_hits: test / gate = pool records whose signature is in prosqa_sym's test / GATE set (0 by the stream exclusion); probe = in the
     Stream(22500) sets langvb_eval draws (probe: 1,000 held-out + 200,000 pool, D U{3..8} from np rng 22500; FD: 2,000 hold D U{6..8} rng 22565
     and 200,000 training draws D U{6..8} rng 22560 in batches of 64). VACUOUS by construction: those are D 3-8 and the signature includes D, the
     pool is D 9-12; computed anyway. gold_key_hits = pool lines holding a GOLD_KEYS key ('"<key>":').
  A20 labeller_part: LABEL-AUDIT's gold = the unique option reachable from R by an independent BFS over the statements parsed afresh from text_ids
     (never the label or the record's marks); no / two reachable options -> gold None (counted wrong). MEMO's D for a pool record = the BFS
     distance of its label from R; acc = langvb_eval.score_b's rule (sum log p("### gold.") > sum log p("### other.")) at K' = D + 2, fp32
     (no autocast, tf32 off), pool records = the first n_memo consumed records of that D, fresh = the first n_memo GATE items of that D.
"""
import array
import collections
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
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
    f.dtype = "bf16" if amp is not None else "fp32"
    return f


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIT_FILES = ("boxes/langvb_label.py", "boxes/prosqa_sym.py", "boxes/langvb.py", "boxes/slva.py", "boxes/vbank.py", "boxes/deltamem.py", "boxes/s5.py")


def git_info(files=GIT_FILES):
    """builder_commit: git rev-parse HEAD + a dirty flag from `git status --porcelain` on the module files."""
    def run(*a):
        try:
            return subprocess.run(["git", "-C", ROOT_DIR] + list(a), capture_output=True, text=True, timeout=60).stdout
        except Exception:
            return None
    head = run("rev-parse", "HEAD"); st = run("status", "--porcelain", "--", *files)
    return {"head": head.strip() if head else None, "dirty": None if st is None else bool(st.strip()),
            "dirty_files": None if st is None else [ln[3:] for ln in st.splitlines() if ln.strip()], "files": list(files)}


def state_sha256(m):
    """sha256 over a model's state dict (sorted names + raw tensor bytes): the model identity for smokes that have no model.pt file (A16)."""
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


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
            "hist": _shist(collections.Counter(v)) if len(set(v)) <= 60 else None}


def _shist(cnt):
    """a Counter -> {str(key): count} sorted by key (JSON-exact: string keys)."""
    return {str(k): v for k, v in sorted(cnt.items())}


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
            "stop_lag_hist": _shist(collections.Counter(lag)), "frontier_hist": _shist(fh),
            "rollouts_mean": float(np.mean([r["n_rollouts"] for r in res])) if res else None}


def kept_vs_drawn(golds, res):
    kept = [g for g, r in zip(golds, res) if r["keep"]]
    return {f: {"drawn": _summ([g[f] for g in golds]), "kept": _summ([g[f] for g in kept])} for f in ("T", "concepts", "cross", "decoys", "frontier")}


def choose(chains_out, chains=CHAINS):
    """§2.4 selection rule (AMBIGUITY A8): pooled kept yield over D 11-12; L-RR only if it beats L-ROLL by >= 0.05. chains_out {chain: {"D": stats}}."""
    y = {}
    for c in chains:
        n = sum(chains_out[c][str(D)]["n"] for D in (11, 12) if str(D) in chains_out[c])
        k = sum(chains_out[c][str(D)]["kept"] for D in (11, 12) if str(D) in chains_out[c]); y[c] = k / n if n else None
    if "L-RR" not in y or "L-ROLL" not in y:
        return chains[0], y
    return ("L-RR" if (y["L-RR"] is not None and y["L-ROLL"] is not None and y["L-RR"] - y["L-ROLL"] >= SEL_MARGIN) else "L-ROLL"), y


def run_chain_chunked(chain, m, inps, dev, ac, bs=16, chunk=1024):
    """run_chain over consecutive chunks of `chunk` inputs (the pool builder's batching unit; AMBIGUITY A15)."""
    out = []
    for c0 in range(0, len(inps), chunk):
        out += run_chain(chain, m, inps[c0:c0 + chunk], dev, ac, bs)
    return out


def unsound_kept(res, dist, D):
    """a kept result vs gold: M'_t inside R_t for every t (each pass-t addition has 0 < dist <= t) and s >= D -> list of problems (A13)."""
    bad = [] if res["stop_s"] >= D else ["s %d < D %d" % (res["stop_s"], D)]
    for t, add in enumerate(res["Mp_add"], 1):
        if any(not (0 < dist[j] <= t) for j in add):
            bad.append("M'_%d not in R_%d" % (t, t)); break
    return bad


def gate1_l(m, dev, ac, items_by_D=None, chains=CHAINS, bs=16, chunk=1024, tag=None, require_bf16=True, log=print):
    """G-L15 GATE-1-L: both chains on the GATE items (gold read only here) -> the eval_gate1.json dict (score_s218 LAYOUTS; module docstring)."""
    dtype = getattr(ac, "dtype", None)
    if require_bf16:
        assert dtype == "bf16", "GATE-1-L runs under bf16 autocast (§2.4); got %s" % dtype
    t0 = time.time(); m.eval(); tag = tag or {}
    items_by_D = items_by_D if items_by_D is not None else PS.gate_set()
    chains_out = {c: {} for c in chains}; items = {c: {} for c in chains}; kvd = {c: {} for c in chains}; sec = {}; unsound_ex = []
    for D in sorted(items_by_D):
        its = items_by_D[D]; inps = [parse_ids(it["ids"]) for it in its]; golds = [_gold(it, u) for it, u in zip(its, inps)]
        for c in chains:
            t1 = time.time(); res = run_chain_chunked(c, m, inps, dev, ac, bs, chunk); sec["%s_D%d" % (c, D)] = round(time.time() - t1, 1)
            st = gate1_stats(golds, res); nu = 0; lst = []
            for k, (g, u, r) in enumerate(zip(golds, inps, res)):
                bad = unsound_kept(r, g["dist"], D) if r["keep"] else []
                if bad:
                    nu += 1
                    if len(unsound_ex) < 20:
                        unsound_ex.append({"chain": c, "D": D, "i": k, "problems": bad})
                S = len(u["concepts"])
                lst.append({"keep": bool(r["keep"]), "label": r["label"], "gold": g["o_R"], "stop": r["stop_s"], "i": k, "tie": bool(r["tie"]),
                            "unsound": bad, "marks_hex": [_hexmask(s_, S) for s_ in r["raw"]], "marks_sound": r["Mp_add"], "frontier": r["frontier"],
                            "n_rollouts": r["n_rollouts"], "gold_dist": g["dist"], "T": g["T"]})
            st["n_unsound"] = nu; chains_out[c][str(D)] = st; items[c][str(D)] = lst; kvd[c][str(D)] = kept_vs_drawn(golds, res)
            log("GATE1 %s D %d: kept %d / %d, pi %s, ties %d, unsound %d, %.0f s" % (c, D, st["kept"], st["n"], st["pi_label"], st["n_tie"], nu,
                                                                                  sec["%s_D%d" % (c, D)]))
    choice, y1112 = choose(chains_out, chains)
    n_bug = {k: sum(chains_out[c][D][k] for c in chains for D in chains_out[c]) for k in ("n_wrong_label", "n_tie", "n_unsound")}
    bug = any(n_bug.values())
    ok_pi = all(chains_out[choice][D]["pi_label"] in (None, 1.0) for D in chains_out[choice]) and not bug
    ok_y = all(str(D) in chains_out[choice] and (chains_out[choice][str(D)]["yield"] or 0.0) >= YIELD_MIN for D in POOL_D)
    meta = {"card": tag.get("card"), "dtype": dtype, "bs": bs, "chunk": chunk, "model_sha256": tag.get("model_sha256"), "model": tag.get("model"),
            "c_anneal": float(m.c_anneal), "seed": PS.SEED_GATE, "n_per_D": {str(D): len(v) for D, v in items_by_D.items()}, "chains_run": list(chains),
            "builder_commit": git_info()}
    return {"meta": meta, "choice": choice, "chains": chains_out, "items": items, "kept_vs_drawn": kvd[choice], "kept_vs_drawn_by_chain": kvd,
            "label_bug": bool(bug), "label_bug_counts": n_bug, "unsound_examples": unsound_ex, "yield_D11_12": y1112, "pass_pi_label": bool(ok_pi),
            "pass_yield": bool(ok_y), "pass": bool(ok_pi and ok_y),
            "rule": "choice: pooled kept yield D 11-12, L-RR iff it beats L-ROLL by >= %.2f; pass: no LABEL-BUG (wrong label, tie or unsound kept item in "
                    "either chain), pi_label 1.000 and yield >= %.1f at every D 9-12 (chosen chain)" % (SEL_MARGIN, YIELD_MIN),
            "sec": sec, "sec_total": round(time.time() - t0, 1)}


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


def gate1_link(gate1_path, chain, dtype, model_sha256, require_pass=True):
    """A14: the pool is built only with GATE-1-L's chosen chain, dtype and model (and only after GATE-1-L passed without LABEL-BUG)."""
    g = json.load(open(gate1_path)); gm = g.get("meta") or {}
    assert chain is not None and g.get("choice") == chain, "chain %s != eval_gate1.json choice %s" % (chain, g.get("choice"))
    assert gm.get("dtype") == dtype, "dtype %s != eval_gate1.json meta.dtype %s" % (dtype, gm.get("dtype"))
    assert gm.get("model_sha256") and gm.get("model_sha256") == model_sha256, "model_sha256 %s != eval_gate1.json %s" % (model_sha256, gm.get("model_sha256"))
    assert g.get("label_bug") is False, "eval_gate1.json reads LABEL-BUG: no pool (A2 voided and re-gated)"
    assert not require_pass or g.get("pass") is True, "GATE-1-L did not pass: no pool (§2.9)"
    return {"path": os.path.abspath(gate1_path), "sha256": file_sha256(gate1_path), "choice": g["choice"], "pass": g.get("pass"),
            "yield_D11_12": g.get("yield_D11_12"), "meta": {k: gm.get(k) for k in ("card", "dtype", "bs", "chunk", "model_sha256")}}


def _lock(path):
    """A17: claim a band (O_EXCL); raises if the lock exists."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("band already claimed (lock %s: %s); remove it by hand only if that process is dead" % (path, open(path).read()))
    os.write(fd, json.dumps({"pid": os.getpid(), "host": platform.node(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}).encode()); os.close(fd)


BAND_FIXED = ("D", "chain", "seed", "dtype", "c_anneal", "bs", "chunk", "kept_target", "draw_cap")


def build_pool_part(m, dev, ac, out_dir, D, chain, gate1_path, tag, kept_target=KEPT_PER_D, draw_cap=DRAW_CAP_PER_D, chunk=1024, bs=16, excl=None,
                    seed=PS.SEED_POOL, max_draws=None, require_bf16=True, require_gate_pass=True, log=print):
    """§2.4 / §8: label band D of Stream(seed) (test + GATE signatures excluded) with `chain` until kept_target kept or draw_cap draws
    (AMBIGUITY A10). Resumable: part_D<D>.state.json holds the stream rng state, counts and the part files' byte lengths (a crash mid-chunk is
    truncated back on resume). max_draws stops early (smokes / a planned stop); the band is `done` only at kept_target or draw_cap.
    tag = {model, model_sha256, card} (required). A14: chain / dtype / model_sha256 == eval_gate1.json; A15: BAND_FIXED, tag and the gate1 file's
    sha asserted equal on resume; A17: band-claim lock; A13: ties counted, every kept record passes check_record."""
    dtype = getattr(ac, "dtype", None)
    if require_bf16:
        assert dtype == "bf16", "the pool is labelled under bf16 autocast (§2.4); got %s" % dtype
    assert tag and tag.get("model_sha256") and tag.get("card"), "tag {model_sha256, card} required"
    g1 = gate1_link(gate1_path, chain, dtype, tag["model_sha256"], require_gate_pass)
    os.makedirs(out_dir, exist_ok=True); m.eval(); pj, ps_, pst = _part_paths(out_dir, D); lk = pst[:-len(".state.json")] + ".lock"
    _lock(lk)
    try:
        return _build_band(m, dev, ac, D, chain, g1, tag, kept_target, draw_cap, chunk, bs, excl, seed, max_draws, log, pj, ps_, pst, dtype)
    finally:
        os.remove(lk)


def _build_band(m, dev, ac, D, chain, g1, tag, kept_target, draw_cap, chunk, bs, excl, seed, max_draws, log, pj, ps_, pst, dtype):
    excl = excl if excl is not None else PS.exclusion_sigs(); stream = PS.Stream(seed, excl)
    new = {"D": D, "chain": chain, "seed": seed, "dtype": dtype, "c_anneal": float(m.c_anneal), "bs": bs, "chunk": chunk, "kept_target": kept_target,
           "draw_cap": draw_cap}
    if os.path.exists(pst):
        st = json.load(open(pst))
        for k_ in BAND_FIXED:
            assert st.get(k_) == new[k_], "resume: %s %s != %s (a band is labelled with one fixed setting)" % (k_, st.get(k_), new[k_])
        assert st["tag"] == tag, "resume: tag %s != %s (model / card must not change inside a band)" % (st["tag"], tag)
        assert st["gate1"]["sha256"] == g1["sha256"], "resume: eval_gate1.json changed"
        rs = st["rng_state"]; stream.rngs[D] = random.Random(); stream.rngs[D].setstate((rs[0], tuple(rs[1]), rs[2])); stream.n[D] = st["stream_n"]
        for p_, nb in ((pj, st["bytes"]), (ps_, st["bytes_sigs"])):
            with open(p_, "ab") as f:
                f.truncate(nb)
        st["resumes"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "drawn": st["drawn"], "builder_commit": git_info()})
        log("RESUME D %d: drawn %d kept %d" % (D, st["drawn"], st["kept"]))
    else:
        st = {**new, "drawn": 0, "kept": 0, "n_tie": 0, "collisions": [], "bytes": 0, "bytes_sigs": 0, "sec": 0.0, "chunks": [], "done": False,
              "tag": tag, "card": tag["card"], "gate1": g1, "builder_commit": git_info(), "resumes": []}
        for p_ in (pj, ps_):
            open(p_, "wb").close()
    stop = st["drawn"] + max_draws if max_draws else None
    while not st["done"] and (stop is None or st["drawn"] < stop):
        n = min(chunk, draw_cap - st["drawn"]) if stop is None else min(chunk, draw_cap - st["drawn"], stop - st["drawn"])
        t1 = time.time(); nc0 = len(stream.collisions); its = [stream.draw(D) for _ in range(n)]; tg = time.time() - t1
        inps = [parse_ids(it["ids"]) for it in its]; res = run_chain(chain, m, inps, dev, ac, bs); used = 0; recs = []; sigs = []; ties = 0
        for it, u, r in zip(its, inps, res):
            used += 1; ties += bool(r["tie"])
            if r["keep"]:
                rec = make_record(u, r); check_record(rec); recs.append(rec); sigs.append(it["sig"])
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
        st["collisions"] += stream.collisions[nc0:]; st["n_tie"] += ties
        st["chunks"].append({"n": used, "kept": len(recs), "n_tie": ties, "sec": round(dt, 2), "sec_gen": round(tg, 2),
                             "sec_per_drawn": round(dt / max(1, used), 4)})
        st["done"] = st["kept"] >= kept_target or st["drawn"] >= draw_cap
        rs = stream.rngs[D].getstate(); st["rng_state"] = [rs[0], list(rs[1]), rs[2]]; st["stream_n"] = stream.n[D]
        _jdump(st, pst)
        log("POOL D %d: drawn %d kept %d (yield %.3f), %.3f s / drawn" % (D, st["drawn"], st["kept"], st["kept"] / max(1, st["drawn"]), dt / max(1, used)))
    return st


def probe_stream_sigs(excl=None, scale=1.0):
    """A19: signatures of the Stream(22500) items langvb_eval draws (part_probe: 1,000 held-out + 200,000 pool, D U{3..8} from np rng 22500 in one
    call; part_fd: 2,000 hold, D U{6..8} from rng 22565 in one call, and 200,000 training draws D U{6..8} from rng 22560 in batches of 64). Each
    consumer starts at its band's first draw, so the union is the first max_consumer n(D) draws of each band. scale shrinks the counts (smokes)."""
    excl = excl if excl is not None else PS.exclusion_sigs(); need = collections.Counter()
    n_pr, n_fh, n_ft = int(201000 * scale), int(2000 * scale), int(200000 * scale)
    for cnt in (collections.Counter(int(D) for D in np.random.default_rng(22500).integers(3, 9, size=n_pr)),
                collections.Counter(int(D) for D in np.random.default_rng(22565).integers(6, 9, size=n_fh))):
        for D, v in cnt.items():
            need[D] = max(need[D], v)
    r = np.random.default_rng(22560); ft = collections.Counter(); done = 0
    while done < n_ft:
        n = min(64, n_ft - done); done += n; ft.update(int(D) for D in r.integers(6, 9, size=n))
    for D, v in ft.items():
        need[D] = max(need[D], v)
    s = PS.Stream(22500, excl); return {s.draw(D)["sig"] for D in sorted(need) for _ in range(need[D])}, _shist(need)


def assemble_pool(out_dir, gate1_path, Ds=POOL_D, chain=None, kept_target=KEPT_PER_D, draw_cap=DRAW_CAP_PER_D, extra=None, require_done=True,
                  require_gate_pass=True, sig_sets=None, probe_scale=1.0, demand_seed=1):
    """§2.4: pool.jsonl in the fixed interleaved order (D 9, 10, 11, 12 cycling by record; AMBIGUITY A10), pool_sigs.txt, pool_meta.json (A13-A19).
    sig_sets {test, gate, probe: set} (None = computed: prosqa_sym's fixed sets and probe_stream_sigs(scale=probe_scale))."""
    sts = {}; lines = {}; sg = {}
    for D in Ds:
        pj, ps_, pst = _part_paths(out_dir, D); st = json.load(open(pst)); sts[D] = st
        assert not os.path.exists(pst[:-len(".state.json")] + ".lock"), "band D %d is claimed (lock present): still building?" % D
        assert not require_done or st["done"], "band D %d not done" % D
        assert chain is None or st["chain"] == chain, "chain differs in D %d" % D
        with open(pj, "rb") as f:
            lines[D] = f.read(st["bytes"]).splitlines(keepends=True)
        with open(ps_, "rb") as f:
            sg[D] = f.read(st["bytes_sigs"]).splitlines(keepends=True)
        assert len(lines[D]) == st["kept"] == len(sg[D]), "D %d: part lines != kept" % D
    for k_ in BAND_FIXED[1:]:
        assert len({json.dumps(sts[D].get(k_)) for D in Ds}) == 1, "bands disagree on %s" % k_
    assert len({sts[D]["tag"]["model_sha256"] for D in Ds}) == 1, "bands labelled by different models"
    assert len({sts[D]["gate1"]["sha256"] for D in Ds}) == 1, "bands linked to different eval_gate1.json files"
    s0 = sts[Ds[0]]; g1 = gate1_link(gate1_path, s0["chain"], s0["dtype"], s0["tag"]["model_sha256"], require_gate_pass)
    assert g1["sha256"] == s0["gate1"]["sha256"], "eval_gate1.json changed since the bands were built"
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
    keys = set(); gold_hits = 0; pats = [('"%s":' % k).encode() for k in sorted(GOLD_KEYS)]
    with open(pool, "rb") as f:
        for ln in f:
            keys |= set(json.loads(ln)); gold_hits += any(p_ in ln for p_ in pats)
    psigs = open(sigp).read().split(); n_dup = len(psigs) - len(set(psigs))
    if sig_sets is None:
        sig_sets = {"test": PS.fixed_sigs(PS.SEED_TEST, 1000, range(3, 17)), "gate": PS.fixed_sigs(PS.SEED_GATE, 2000, range(9, 13))}
        sig_sets["probe"], probe_n = probe_stream_sigs(sig_sets["test"] | sig_sets["gate"], probe_scale)
    else:
        probe_n = None
    sig_hits = {k: sum(s_ in v for s_ in psigs) for k, v in sig_sets.items()}
    dem = replay_self_demand(demand_seed); v1p = os.path.join(ROOT_DIR, "docs", "research_lang_2026-10-18", "v1_checks", "v1_pool.json")
    ref = json.load(open(v1p)) if os.path.exists(v1p) else {"seed1_self_steps": 5992, "seed1_demand": 383488}
    rng_eq = demand_seed == 1 and dem["self_steps"] == ref["seed1_self_steps"] and dem["records"] == ref["seed1_demand"]
    n_tie = {str(D): sts[D]["n_tie"] for D in Ds}; bc_now = git_info()
    bc = {"head": bc_now["head"], "dirty": bc_now["dirty"], "dirty_files": bc_now["dirty_files"], "files": bc_now["files"],
          "by_band": {str(D): sts[D]["builder_commit"] for D in Ds}, "resumes": {str(D): [r_["builder_commit"] for r_ in sts[D]["resumes"]] for D in Ds}}
    heads = {sts[D]["builder_commit"]["head"] for D in Ds} | {r_["builder_commit"]["head"] for D in Ds for r_ in sts[D]["resumes"]} | {bc_now["head"]}
    bc["consistent"] = len(heads) == 1 and not any(sts[D]["builder_commit"]["dirty"] for D in Ds)
    meta = {"sha256": file_sha256(pool), "n_records": n, "rec_keys": sorted(REC_KEYS),
            "kept_by_D": {str(D): sts[D]["kept"] for D in Ds}, "draws_by_D": {str(D): sts[D]["drawn"] for D in Ds}, "total_kept": sum(sts[D]["kept"] for D in Ds),
            "demand_replay": dem["records"], "demand_self_steps": dem["self_steps"], "demand_seed": demand_seed, "rng_replay_equal": bool(rng_eq),
            "key_set": sorted(keys), "sig_hits": sig_hits, "sig_hits_note": "probe hits are vacuous by construction (probe / FD streams are D 3-8, the "
            "signature includes D, the pool is D 9-12); computed anyway (A19)", "probe_draws_by_D": probe_n, "probe_scale": probe_scale,
            "gold_key_hits": gold_hits, "n_dup_sigs": n_dup, "n_tie": sum(n_tie.values()), "n_tie_by_D": n_tie,
            "label_bug": sum(n_tie.values()) > 0, "card_by_band": {str(D): sts[D]["card"] for D in Ds}, "bs": s0["bs"], "chunk": s0["chunk"],
            "gate1": g1, "model_sha256": s0["tag"]["model_sha256"], "model": s0["tag"].get("model"),
            "counts": {str(D): {"drawn": sts[D]["drawn"], "kept": sts[D]["kept"], "yield": sts[D]["kept"] / max(1, sts[D]["drawn"]),
                                "collisions": len(sts[D]["collisions"]), "sec": round(sts[D]["sec"], 1),
                                "sec_per_drawn": sts[D]["sec"] / max(1, sts[D]["drawn"])} for D in Ds},
            "POOL_SHORT": bool(short), "pool_short_D": short, "kept_target_per_D": kept_target, "draw_cap_per_D": draw_cap,
            "chain": sts[Ds[0]]["chain"], "seeds": {"pool_stream": sts[Ds[0]]["seed"], "rr_seed": RR_SEED,
                                                     "exclusion": "test 22200 (1,000 per D 3-16) + GATE 22400 (2,000 per D 9-12) role-symmetric signatures"},
            "order": "interleaved D %s cycling by record (an exhausted band is skipped); D not stored" % ",".join(map(str, Ds)),
            "builder_commit": bc,
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
    """§7 G-L20 (the pool_meta half): kept per D >= 106,000 (every D 9-12 present), total >= 1.1 x the seed's replayed demand, not POOL-SHORT,
    n_records = sum kept, the replay equal to v1_pool's count, key set exact, sig_hits 0 (test / gate / probe), gold_key_hits 0, no LABEL-BUG."""
    dem = replay_self_demand(seed, steps, p_self, bs, self_kextra)
    kept = meta.get("kept_by_D") or {D: v["kept"] for D, v in meta["counts"].items()}; tot = sum(kept.values())
    out = {"demand": dem, "kept": kept, "total": tot, "kept_ok": all(kept.get(str(D), 0) >= kept_target for D in POOL_D),
           "margin_ok": tot >= 1.1 * dem["records"], "not_short": not meta.get("POOL_SHORT"), "n_records_ok": meta["n_records"] == tot,
           "replay_ok": meta.get("rng_replay_equal") is True and meta.get("demand_replay") == dem["records"],
           "keys_ok": sorted(meta.get("key_set") or []) == sorted(REC_KEYS), "sig_ok": isinstance(meta.get("sig_hits"), dict)
           and set(meta["sig_hits"]) >= {"test", "gate", "probe"} and sum(meta["sig_hits"].values()) == 0,
           "gold_ok": meta.get("gold_key_hits") == 0, "no_label_bug": meta.get("label_bug") is False}
    out["pass"] = all(out[k] for k in ("kept_ok", "margin_ok", "not_short", "n_records_ok", "replay_ok", "keys_ok", "sig_ok", "gold_ok", "no_label_bug"))
    return out


# ---------------------------------------------------------------------- the A2 eval file's "labeller" part
def _bfs_names(P):
    """independent of SOUND-CLOSURE: BFS distances from R over every parsed statement (persons included as nodes)."""
    succ = collections.defaultdict(list)
    for a, b in P["stmts"]:
        succ[a].append(b)
    d = {P["R"]: 0}; q = collections.deque([P["R"]])
    while q:
        u = q.popleft()
        for v in succ[u]:
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


def labeller_part(m, dev, pool_path, meta_path, gate1_path, cursor, eval_bs, n_audit=5000, n_memo=2000, gate_items=None, card=None, log=print):
    """eval_lvb.json's labeller part (A2; score_s218 LAYOUTS 'eval_lvb.json labeller (A2)', AMBIGUITY A20), fp32 on the eval card:
    gate1 = a copy of eval_gate1.json's choice (+ yields, label_bug), kept_vs_drawn = its copy, label_audit on the first n_audit CONSUMED records
    (pool order; cursor = the run's final pool_cursor), memo {D: acc_pool vs acc_fresh at K' = D + 2} (MEMO-GAP-L, descriptive)."""
    import boxes.langvb as LV
    import boxes.langvb_eval as LE
    t0 = time.time(); m.eval(); g1 = json.load(open(gate1_path)); recs, meta = load_pool(pool_path, meta_path, verify=True, compact=False)
    assert 0 < cursor <= len(recs), "cursor %s outside the pool (%d records)" % (cursor, len(recs))
    cons = recs[:cursor]; aud = []; n_wrong = n_uns = n_none = 0; Dof = []
    for j, r in enumerate(cons):
        u = parse_ids(r["text_ids"]); d = _bfs_names(u["P"]); reach = [o for o in u["P"]["options"] if o in d]
        gold = reach[0] if len(reach) == 1 else None; Dl = d.get(r["label"]); Dof.append(Dl)
        if j >= n_audit:
            continue
        cs = u["concepts"]; sound = r["stop_s"] >= (Dl or 10 ** 6) and all(cs[k] in d and d[cs[k]] <= t for t, add in enumerate(r["marks_sound"], 1) for k in add)
        ok = gold is not None and r["label"] == gold; n_wrong += not ok; n_uns += not sound; n_none += gold is None
        aud.append({"j": j, "label": r["label"], "gold": gold, "D": Dl, "stop_s": r["stop_s"], "ok": bool(ok), "sound": bool(sound)})
    na = len(aud)
    label_audit = {"n": na, "n_wrong": n_wrong, "pi_label": (na - n_wrong) / na if na else None, "n_unsound": n_uns, "n_no_gold": n_none,
                   "rule": "gold = the unique option reachable from R by BFS over the statements parsed from text_ids (A20)", "items": aud}
    tf = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    gate_items = gate_items if gate_items is not None else PS.gate_set(n_memo, range(9, 13))

    def acc(its, K):
        ok = []
        for i in range(0, len(its), eval_bs):
            out, _ = LE.score_b(m, LV.lvb_batch(its[i:i + eval_bs], dev), K); sR, sP = out[K]; ok += (sR > sP).long().tolist()
        return ok
    memo = {}
    try:
        for D in POOL_D:
            pj = [j for j, Dl in enumerate(Dof) if Dl == D][:n_memo]; ip = acc([self_item(cons[j]) for j in pj], D + 2) if pj else []
            fr = gate_items.get(D, [])[:n_memo]; ifr = acc([LV.lvb_item(it) for it in fr], D + 2) if fr else []
            memo[str(D)] = {"acc_pool": float(np.mean(ip)) if ip else None, "acc_fresh": float(np.mean(ifr)) if ifr else None, "n_pool": len(ip),
                            "n_fresh": len(ifr), "K": D + 2, "items": {"pool": [{"j": j, "ok": o} for j, o in zip(pj, ip)], "fresh": ifr}}
            log("MEMO D %d: pool %s (n %d), fresh %s (n %d)" % (D, memo[str(D)]["acc_pool"], len(ip), memo[str(D)]["acc_fresh"], len(ifr)))
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf
    ys = {c: {D: v.get("yield") for D, v in (g1.get("chains") or {}).get(c, {}).items()} for c in CHAINS}
    return {"gate1": {"choice": g1.get("choice"), "yield_D11_12": g1.get("yield_D11_12"), "yields": ys, "label_bug": g1.get("label_bug"),
                      "pass": g1.get("pass"), "sha256": file_sha256(gate1_path), "meta": g1.get("meta")},
            "kept_vs_drawn": g1.get("kept_vs_drawn"), "label_audit": label_audit, "memo": memo,
            "meta": {"card": card, "dtype": "fp32", "eval_bs": {"labeller": eval_bs}, "cursor": cursor, "pool_sha256": meta["sha256"],
                     "n_audit_req": n_audit, "n_memo_req": n_memo, "n_consumed_by_D": {str(D): sum(Dl == D for Dl in Dof) for D in POOL_D}},
            "sec": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------- CLI
def _load_model(path, dev):
    from boxes import s5
    import boxes.langvb as LV
    m = LV.make_model(s5.load_gpt2(dev)).to(dev)
    if path:
        sd = torch.load(path, map_location=dev, weights_only=True); m.load_state_dict(sd, strict=True); del sd
    m.eval(); return m


def main():
    """gate1 / pool refuse to run without bf16 (§2.4); --chain has no default (pool / assemble assert it == eval_gate1.json's choice)."""
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gate1", "pool", "assemble", "labeller"]); ap.add_argument("--model", default="")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--out", default=""); ap.add_argument("--out_dir", default=""); ap.add_argument("--chain", default=None, choices=list(CHAINS))
    ap.add_argument("--gate1", default=""); ap.add_argument("--pool", default=""); ap.add_argument("--cursor", type=int, default=0)
    ap.add_argument("--eval_bs", type=int, default=16); ap.add_argument("--D", default="9,10,11,12"); ap.add_argument("--max_draws", type=int, default=0)
    ap.add_argument("--kept_target", type=int, default=KEPT_PER_D); ap.add_argument("--draw_cap", type=int, default=DRAW_CAP_PER_D)
    a = ap.parse_args(); Ds = [int(v) for v in a.D.split(",")]
    if a.cmd in ("pool", "assemble"):
        assert a.chain and a.gate1, "--chain and --gate1 are required (the pool uses GATE-1-L's choice)"
    if a.cmd == "assemble":
        mt = assemble_pool(a.out_dir, a.gate1, tuple(Ds), a.chain, a.kept_target, a.draw_cap)
        print(json.dumps({k: v for k, v in mt.items() if k not in ("collisions", "gate1", "builder_commit")}, indent=1)); return
    assert a.model, "--model required"
    m = _load_model(a.model, a.device); card = torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu"
    tag = {"model": a.model, "model_sha256": file_sha256(a.model), "card": card}
    if a.cmd == "labeller":
        assert a.gate1 and a.pool and a.cursor and a.out, "--gate1 --pool --cursor --out required"
        out = labeller_part(m, a.device, a.pool, os.path.join(os.path.dirname(a.pool), "pool_meta.json"), a.gate1, a.cursor, a.eval_bs, card=card)
        out["meta"]["model_sha256"] = tag["model_sha256"]; _jdump(out, a.out); print("LABEL_AUDIT", out["label_audit"]["n"], out["label_audit"]["n_wrong"]); return
    ac = make_ac(a.device, True)
    assert ac.dtype == "bf16", "bf16 autocast unavailable on %s (needs a cuda card with compute capability >= 8): §2.4 requires bf16" % a.device
    if a.cmd == "gate1":
        out = gate1_l(m, a.device, ac, bs=a.bs, chunk=a.chunk, tag=tag); _jdump(out, a.out)
        print("GATE1 choice", out["choice"], "pass", out["pass"], "label_bug", out["label_bug"], flush=True)
        if out["label_bug"]:
            raise SystemExit("LABEL-BUG in GATE-1-L (%s): a code fault; A2 voided and re-gated" % out["label_bug_counts"])
        return
    excl = PS.exclusion_sigs()
    for D in Ds:
        build_pool_part(m, a.device, ac, a.out_dir, D, a.chain, a.gate1, tag, a.kept_target, a.draw_cap, a.chunk, a.bs, excl, max_draws=a.max_draws or None)


if __name__ == "__main__":
    main()
