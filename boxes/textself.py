"""§219 TEXT-SELF (TS_v1.md at 146a25d0; §2.2-§2.4, §2.6 T2): the text labeller, the CV-T checks, compose_T, the self-pool record format, the
exclusion sets and an INDEPENDENT KV-cache generation (G-T16). Shared by scratch/s219_build_pool.py (B1), scratch/s219_gate.py (B2), the trainer's
loader (boxes/textself_ft.py, which uses only compose_T / load_pool / self_item / n_of / file_sha256) and boxes/text_eval.py (E1 / E2).
  candidate (shape only): from the current tree, the next CV-T step is c_k(v) = render(tree with its leftmost-ready handle -> the leaf v); its token ids
    split at the value slot: (prefix, suffix, has_operator_after). No N.apply / N.value / N.cot_* anywhere in the labeller path.
  REROOT (TF-GREEDY, the registered labeller): per step a fresh prompt [EOT] + e_{k-1} + [' ='] (+ the candidate's prefix, teacher-forced); the greedy
    argmax must equal every fixed token of c_k and be a value token at the value slot, then (second forward on the cached prefix) the suffix and the
    terminator (' =' iff c_k has an operator, else EOT). Kept = all n steps pass (KEEP-T and CV-T, TS_v1 §2.3); label = v_n.
  FULL (TF-GREEDY, descriptive LABELLER-FULL): one sequence [EOT] e_0 ' =' e_1 ' =' ... e_n EOT (the model's own whole chain).
  Independent generation (G-T16): per step KV-cache greedy decoding through core.GPT2KV.forward (nested170.cot_answers' loop, equal-length prompt
    batches, no padding), stopped at the first ' =' / EOT, validity by check_step_T (the CV-T decomposition), never by candidate(); FULL likewise.
  record (pool): keys exactly {syms, n, chain_label, values}; values = the labeller's v_1..v_n, chain_label = v_n. No gold field.
  compose_T(syms, values) = c_1(v_1) ' =' ... c_n(v_n) EOT from the shape and the values only.
Ambiguities resolved here (TS_v1 silent; nearest precedent followed):
  A1 parse_tree is a verbatim copy of vbank_diag._parse_tree (gold-free shunting-yard), so the text trainer does not import the vbank stack;
     scratch/gate_s219.py asserts both parsers agree on every gate / perturbation input.
  A2 TF-GREEDY reads the LM head only at the positions it checks (prefix slots, the value slot, suffix slots, terminator), not at every padded position as
     the draft's ts_label_check did: identical in exact arithmetic, smaller memory at label batch 128; G-T16 measures the fp32 agreement.
  A3 check_step_T's SPAN-T accepts every alignment p with s[:p] == e[:p] and s[p+1:] == e[p+d:], d = |e| - |s| + 1 >= 3 (a value can equal a neighbour, so
     a single longest-common-prefix split would mis-flag valid steps); ORDER-T = the handle's own alignment is among them; VALUE-SLOT reads the token at
     the handle slot when FRAME-T holds, else at any SPAN alignment.
  A4 The G-T18 smoke items (rng 23195) are 1,000 draws of generate(rng, 24) in order, redrawn on a hit against test_set(1000, 30) + SERIAL
     (gt18_items); all 1,000 join every exclusion set.
  A5 Exclusion of the TEXT-SELF precondition streams regenerates rng 23180 with ts_label_check.main's loop (no within-stream dedup) and rng 23187 with
     pc_reroot.main's loop (its own 23180 regeneration added to its exclusion set item by item); both full 200-per-n streams at n 9..20 (V7).
  A6 Deleted latent pools are not regenerated (Addendum 1 item 4); every latent / s219 gate_items.json and pool file on disk is read (smoke dirs never)."""
import glob
import hashlib
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import boxes.nested170 as N

EOT = N.EOT
EQ = N.TID[" ="]
NUM_IDS = [int(x) for x in N.NUM_IDS]
NUMS = frozenset(NUM_IDS)
NUM2V = {t: v for v, t in enumerate(NUM_IDS)}
ID2SYM = {N.TID[w]: w for w in N.VOCAB}
REC_KEYS = {"syms", "n", "chain_label", "values"}
GOLD_KEYS = {"root", "t", "y", "gold", "cot", "cot_ids", "steps", "E", "E_self"}
R = "boxes/runs_fsq/"
BASE_RUN = R + "s170_cot_s1"
BASE_MODEL = BASE_RUN + "/model.pt"
BASE_SHA256 = "36f4659b99f08a65202b303f4af74ef20ac65f1e60d2a25f3e84aafbfb2e3392"           # TS_v1 V6
NESTED170_SHA256 = "1961bd3687a903f3e44968b16aed854d78c253e2d674e367731b9a7eae088b9e"      # G-T7 pin (v1_argkeys.json src_sha256)
CTX = 1025                                                                  # P + C <= 1025 (GPT-2's 1,024 position rows; TS_v1 §2.8)
LABEL_BS = 128                                                              # TS_v1 §2.2 (fixed)
LATENT_N50 = {"r0": 10.306, "r1": 15.460, "r2": 20.539}                     # Addendum 2 item 4 (score_s219 owns the readings)
ROUNDS = {   # TS_v1 §2.2 / §2.4 items 1, 9 / §2.5 / §2.7 (one lineage, s1)
    1: {"pool_seed": 23200, "perm_seed": 23201, "per_n": {9: 97_000, 10: 97_000, 11: 97_000, 12: 97_000}, "pool_dir": R + "s219_pool_t1_s1", "pool_file": "pool_t1.jsonl",
        "gate_dir": R + "s219_gate_t1_s1", "windows": {"W1": (23260, (9, 10, 11, 12), "GATE_T1")}, "labeller": BASE_RUN, "fallback": None,
        "audit_seed": 23270, "kept_seed": 23280, "cell": R + "s219_t1_s1", "init_from": BASE_MODEL, "self_micro": 2},
    2: {"pool_seed": 23300, "perm_seed": 23301, "per_n": {9: 32_500, 10: 32_500, 11: 32_500, 12: 32_500, 13: 65_000, 14: 65_000, 15: 65_000, 16: 65_000},
        "pool_dir": R + "s219_pool_t2_s1", "pool_file": "pool_t2.jsonl", "gate_dir": R + "s219_gate_t2_s1",
        "windows": {"W2": (23360, (13, 14, 15, 16), "GATE_T2"), "W1": (23361, (9, 10, 11, 12), "GATE_T1p")}, "labeller": R + "s219_t1_s1", "fallback": BASE_RUN,
        "audit_seed": 23370, "kept_seed": 23380, "cell": R + "s219_t2_s1", "init_from": R + "s219_t1_s1/model.pt", "self_micro": 4},
}
GATE_PER_N = 2000
EQUIV_PER_N = 250


# ------------------------------------------------------------------------------------------------ parsing / shape (gold-free)
def parse_tree(syms):
    """A1: verbatim copy of vbank_diag._parse_tree. decoded symbols -> a nested170 Node tree (shunting-yard; precedence + left associativity +
    brackets), or None if the symbols are not one valid expression over values 0..P-1. Redundant brackets parse."""
    out, st = [], []
    prev = "op"                                                             # what the previous token allows next: "op" -> an operand or "(" may follow

    def red():
        if len(out) < 2:
            raise ValueError
        b, a = out.pop(), out.pop(); out.append(N.Node(op=st.pop(), l=a, r=b))
    try:
        for s_ in syms:
            w = s_.strip()
            if w.isdigit():
                if prev != "op" or not 0 <= int(w) < N.P:
                    return None
                out.append(N.Node(v=int(w))); prev = "val"
            elif w in N.PREC:
                if prev != "val":
                    return None
                while st and st[-1] in N.PREC and N.PREC[st[-1]] >= N.PREC[w]:
                    red()
                st.append(w); prev = "op"
            elif w == "(":
                if prev != "op":
                    return None
                st.append(w)
            elif w == ")":
                if prev != "val":
                    return None
                while st and st[-1] != "(":
                    red()
                if not st:
                    return None
                st.pop()
            else:
                return None
        if prev != "val":
            return None
        while st:
            if st[-1] == "(":
                return None
            red()
    except (ValueError, IndexError):
        return None
    return out[0] if len(out) == 1 else None


def n_of(syms):
    return sum(1 for s in syms if s.strip() in N.PREC)


def height(nd):
    return 0 if nd.op is None else 1 + max(height(nd.l), height(nd.r))


def reduce_shape(tree, v):
    """tree with its leftmost-ready handle replaced by the leaf v (in place; a SHAPE operation, no apply). -> the new leaf node."""
    path = N._leftmost_ready(tree); tgt = tree
    for p in path:
        tgt = getattr(tgt, p)
    tgt.op = tgt.l = tgt.r = None; tgt.v = int(v); return tgt


def candidate(tree):
    """(prefix ids, suffix ids, has_op_after) of the next CV-T step from `tree` (copied), value slot open (ts_label_check.candidate)."""
    t2 = N._copy(tree); leaf = reduce_shape(t2, 0); syms, pos = N._render_pos(t2, leaf); ids = N.ids(syms)
    return ids[:pos], ids[pos + 1:], t2.op is not None


def compose_T(syms, values, slots=False):
    """the self target: c_1(v_1) ' =' c_2(v_2) ' =' ... c_n(v_n) EOT from the shape of syms and the labeller's values only (TS_v1 §2.3).
    slots=True -> (ids, value-slot indices into ids)."""
    tree = parse_tree(syms); assert tree is not None, "compose_T: syms do not parse"
    assert len(values) == n_of(syms), ("compose_T: len(values) != n", len(values), n_of(syms))
    out, vpos = [], []
    for v in values:
        v = int(v); assert 0 <= v < N.P
        pre, suf, more = candidate(tree); vpos.append(len(out) + len(pre)); out += pre + [NUM_IDS[v]] + suf + [EQ if more else EOT]; reduce_shape(tree, v)
    assert tree.op is None
    return (out, vpos) if slots else out


def c_struct(syms):
    """the shape-only chain length (= len(compose_T(syms, any values)) = len(N.cot_ids(root)) on every item; never calls cot_ids)."""
    return len(compose_T(syms, [0] * n_of(syms)))


def _render_own(nd, parent=None, right=False):
    """render() with the owning node of every symbol (a node owns its operator and its own brackets); an independent copy of render's bracket rule."""
    if nd.op is None:
        return [(" %d" % nd.v, nd)]
    s = _render_own(nd.l, nd.op, False) + [(" " + nd.op, nd)] + _render_own(nd.r, nd.op, True)
    if parent is not None and (N.PREC[nd.op] < N.PREC[parent] or (right and N.PREC[nd.op] == N.PREC[parent])):
        s = [(" (", nd)] + s + [(" )", nd)]
    return s


def handle_span(tree):
    """(lo, hi): the symbol indices of render(tree) owned by the leftmost-ready handle (its operands, operator and own brackets); contiguous."""
    own = _render_own(tree); path = N._leftmost_ready(tree); h = tree
    for p in path:
        h = getattr(h, p)
    idx = [k for k, (_, o) in enumerate(own) if o is h or o is h.l or o is h.r]
    assert idx == list(range(idx[0], idx[-1] + 1)), "handle span not contiguous"
    return idx[0], idx[-1]


CV_FLAGS = ("PARSE", "TERM", "FRAME", "SPAN", "VALUE", "ORDER", "CANON")


def check_step_T(cur, s, term):
    """TS_v1 §2.3 per-step checks on an EMITTED step s (token ids, terminator excluded) with terminator `term`, from the parsed previous tree `cur`.
    -> (flags {PARSE, TERM, FRAME, SPAN, VALUE, ORDER, CANON}, the value v or None, the next tree or None); kept step = every flag true. Shape only:
    the value is never checked (VALUE-SLOT accepts any value token)."""
    e = N.ids(N.render(cur)); lo, hi = handle_span(cur); f = {}
    sy = [ID2SYM.get(int(t)) for t in s]
    tree = parse_tree(sy) if (s and all(x is not None and x not in (" =", " .") for x in sy)) else None
    f["PARSE"] = tree is not None
    f["TERM"] = bool(f["PARSE"] and term == (EQ if tree.op is not None else EOT))
    Lc = len(e) - (hi - lo + 1) + 1
    f["FRAME"] = len(s) == Lc and list(s[:lo]) == e[:lo] and list(s[lo + 1:]) == e[hi + 1:]
    d = len(e) - len(s) + 1; s_ = list(s)
    spans = [p for p in range(len(s_)) if s_[:p] == e[:p] and s_[p + 1:] == e[p + d:]] if (d >= 3 and s_) else []
    f["SPAN"] = bool(spans)
    f["VALUE"] = (int(s_[lo]) in NUMS) if f["FRAME"] else any(int(s_[p]) in NUMS for p in spans)
    f["ORDER"] = lo in spans
    f["CANON"] = bool(f["PARSE"] and N.ids(N.render(tree)) == s_)
    ok = all(f.values())
    return f, (NUM2V[int(s_[lo])] if ok else None), (tree if ok else None)


def cand_equal(cur, s, term):
    """the candidate-equality test: v if s == prefix + [value v] + suffix of candidate(cur) and term == the candidate's terminator, else None."""
    pre, suf, more = candidate(cur); s = [int(t) for t in s]
    if len(s) != len(pre) + 1 + len(suf) or s[:len(pre)] != pre or s[len(pre) + 1:] != suf or s[len(pre)] not in NUMS or term != (EQ if more else EOT):
        return None
    return NUM2V[s[len(pre)]]


def chain_form(syms, gen):
    """gold-free form check of a generated chain `gen` (tokens after the prompt, EOT included if emitted): -> (form_valid, values or None,
    first_bad_step (1-based) or None, first_bad_idx (index into gen of the first token off every CV-T continuation) or None). The candidate test per
    step; a chain that is CV-T up to the end of gen without its final EOT is bad at idx len(gen) (the first missing token)."""
    tree = parse_tree(syms); pos = 0; vals = []; k = 0; gen = [int(t) for t in gen]
    while tree.op is not None:
        k += 1; pre, suf, more = candidate(tree); want = pre + [None] + suf + [EQ if more else EOT]
        for j, w in enumerate(want):
            q = pos + j
            if q >= len(gen):
                return False, None, k, len(gen)
            if (w is None and gen[q] not in NUMS) or (w is not None and gen[q] != w):
                return False, None, k, q
        v = NUM2V[gen[pos + len(pre)]]; vals.append(v); reduce_shape(tree, v); pos += len(want)
    if pos != len(gen):                                                     # tokens after the final EOT never exist in a stopped generation
        return False, None, k + 1, pos
    return True, vals, None, None


def cot_answer(gen):
    """nested170.cot_answers' answer rule: the text after the last ' =' before EOT when it is a single number, else -1 (no EOT -> -1)."""
    row = [int(t) for t in gen]
    if EOT not in row:
        return -1
    body = row[:row.index(EOT)]; last = max([k for k, tk in enumerate(body) if tk == EQ], default=-1); tail = body[last + 1:]
    return NUM2V[tail[0]] if (len(tail) == 1 and tail[0] in NUMS) else -1


# ------------------------------------------------------------------------------------------------ model I/O
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_model(path, dev):
    """a text model.pt -> (GPT2KV in eval mode, strict load asserted, sha256). The local memory cap applies (s5.load_gpt2)."""
    from boxes import s5
    g = s5.load_gpt2(dev); sd = torch.load(path, map_location=dev, weights_only=True); mk = g.load_state_dict(sd, strict=True); del sd
    assert not mk.missing_keys and not mk.unexpected_keys, mk
    sha = file_sha256(path)
    if os.path.normpath(path).replace("\\", "/").endswith("s170_cot_s1/model.pt"):
        assert sha == BASE_SHA256, ("§219 V6: the base model.pt sha256", sha)              # the round-1 labeller / T2 fallback / base eval identity
    g.eval(); return g, sha


# ------------------------------------------------------------------------------------------------ TF-GREEDY labellers (batched, right-padded)
@torch.no_grad()
def _fwd(g, ids, pos, past, kvalid):
    """batched GPT-2 forward with per-row positions and key validity (ts_label_check.fwd, device-aware): ids / pos (B, T) long; past per layer (k, v)
    (B, H, P, hd) or None; kvalid (B, P + T) bool (False = a pad key nobody may attend to; a query always sees itself). -> hidden after ln_f, new past."""
    B, T = ids.shape; P = 0 if past is None else past[0][0].shape[2]; D = g.D; H = g.H; hd = D // H; dev = ids.device
    h = g.w("wte.weight")[ids] + g.w("wpe.weight")[pos]
    causal = torch.ones(T, P + T, dtype=torch.bool, device=dev).tril(P)
    selfm = torch.zeros(T, P + T, dtype=torch.bool, device=dev); selfm[torch.arange(T, device=dev), P + torch.arange(T, device=dev)] = True
    mask = (causal[None] & (kvalid[:, None, :] | selfm[None]))[:, None]
    new = []
    for i in range(g.L):
        pre = f"h.{i}."
        a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
        q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
        q, k, v = (t.view(B, T, H, hd).transpose(1, 2) for t in (q, k, v))
        if past is not None:
            k = torch.cat([past[i][0], k], 2); v = torch.cat([past[i][1], v], 2)
        new.append((k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
        a = F.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
        h = h + F.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh") @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")
    return F.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias")), new


@torch.no_grad()
def _run_chunks(g, chunks, lens0, past, kvalid, reads):
    """feed one chunk per row (right-padded) after `past`; lens0 = tokens already in each row. reads[r] = chunk-local positions whose argmax (and top-2
    margin) are returned (A2). -> (argmax lists, margin lists, past, kvalid)."""
    dev = g.w("wte.weight").device; B = len(chunks); T = max(len(c) for c in chunks)
    ids = torch.full((B, T), EOT, dtype=torch.long); pos = torch.zeros((B, T), dtype=torch.long); kv_new = torch.zeros((B, T), dtype=torch.bool)
    for r, c in enumerate(chunks):
        ids[r, :len(c)] = torch.tensor(c, dtype=torch.long); pos[r, :len(c)] = torch.arange(lens0[r], lens0[r] + len(c)); kv_new[r, :len(c)] = True
        pos[r, len(c):] = min(1023, lens0[r] + len(c))
    assert int(pos.max()) <= 1023, "TF-GREEDY: a position past GPT-2's 1,024 rows"
    ids, pos, kv_new = ids.to(dev), pos.to(dev), kv_new.to(dev)
    kvalid = kv_new if kvalid is None else torch.cat([kvalid, kv_new], 1)
    h, past = _fwd(g, ids, pos, past, kvalid)
    rr = [r for r in range(B) for _ in reads[r]]; cc = [j for r in range(B) for j in reads[r]]; am = [[] for _ in range(B)]; mg = [[] for _ in range(B)]
    if rr:
        lg = h[torch.tensor(rr, device=dev), torch.tensor(cc, device=dev)] @ g.w("wte.weight").T
        a_ = lg.argmax(-1).cpu().tolist(); tv = lg.topk(2, -1).values; m_ = (tv[:, 0] - tv[:, 1]).cpu().tolist(); q = 0
        for r in range(B):
            for _ in reads[r]:
                am[r].append(int(a_[q])); mg[r].append(float(m_[q])); q += 1
    return am, mg, past, kvalid


@torch.no_grad()
def label_reroot(g, syms_list, bs=LABEL_BS):
    """REROOT TF-GREEDY (ts_label_check.label_reroot, batched over items at the same step index, sorted by length). -> per input {n, keep, values,
    label, fail_step, fail (region prefix / value / suffix / term, slot, want, got), min_margin}. Values are always the model's argmax."""
    S = len(syms_list); trees = [parse_tree(s) for s in syms_list]; assert all(t is not None for t in trees), "an input does not parse"
    n0 = [n_of(s) for s in syms_list]; ok = [True] * S; vals = [[] for _ in range(S)]; fail = [None] * S; finfo = [None] * S; mm = [math.inf] * S
    for k in range(1, max(n0) + 1):
        act = [i for i in range(S) if ok[i] and k <= n0[i]]; act.sort(key=lambda i: len(N.render(trees[i])))
        for b0 in range(0, len(act), bs):
            part = act[b0:b0 + bs]; ca, pm = [], []
            for i in part:
                pre, suf, more = candidate(trees[i]); pr = [EOT] + N.ids(N.render(trees[i])) + [EQ]; ca.append(pr + pre); pm.append((len(pr), pre, suf, more))
            am, mg, past, kvalid = _run_chunks(g, ca, [0] * len(part), None, None, [list(range(P_ - 1, len(c))) for (P_, _, _, _), c in zip(pm, ca)])
            vb = {}
            for r, i in enumerate(part):
                P_, pre, suf, more = pm[r]; a_ = am[r]; mm[i] = min(mm[i], min(mg[r]))
                bad = next((j for j in range(len(pre)) if a_[j] != pre[j]), None)
                if bad is not None:
                    ok[i] = False; fail[i] = k; finfo[i] = {"region": "prefix", "slot": bad, "want": pre[bad], "got": a_[bad]}
                elif a_[len(pre)] not in NUMS:
                    ok[i] = False; fail[i] = k; finfo[i] = {"region": "value", "slot": len(pre), "want": None, "got": a_[len(pre)]}
                else:
                    vb[r] = NUM2V[a_[len(pre)]]
            cb = [([NUM_IDS[vb[r]]] + pm[r][2]) if r in vb else [EOT] for r in range(len(part))]
            am2, mg2, _, _ = _run_chunks(g, cb, [len(c) for c in ca], past, kvalid, [list(range(len(c))) if r in vb else [] for r, c in enumerate(cb)])
            del past, kvalid
            for r, i in enumerate(part):
                if r not in vb:
                    continue
                P_, pre, suf, more = pm[r]; tgt = suf + [EQ if more else EOT]; mm[i] = min(mm[i], min(mg2[r]))
                bad = next((j for j in range(len(tgt)) if am2[r][j] != tgt[j]), None)
                if bad is not None:
                    ok[i] = False; fail[i] = k
                    finfo[i] = {"region": "suffix" if bad < len(suf) else "term", "slot": len(pre) + 1 + bad, "want": tgt[bad], "got": am2[r][bad]}
                else:
                    vals[i].append(vb[r]); reduce_shape(trees[i], vb[r])
    return [{"n": n0[i], "keep": bool(ok[i]), "values": vals[i] if ok[i] else None, "label": vals[i][-1] if ok[i] else None, "fail_step": fail[i],
             "fail": finfo[i], "min_margin": mm[i]} for i in range(S)]


@torch.no_grad()
def label_full(g, syms_list, bs=LABEL_BS):
    """FULL TF-GREEDY (ts_label_check.label_full; descriptive LABELLER-FULL): one sequence per item, chunk k = [v_{k-1}] + suffix_{k-1} + term +
    prefix_k. -> per input {n, keep, values, label, fail_step, min_margin}."""
    out = []
    order = sorted(range(len(syms_list)), key=lambda i: len(syms_list[i])); res = [None] * len(syms_list)
    for b0 in range(0, len(order), bs):
        idx = order[b0:b0 + bs]; B = len(idx); trees = [parse_tree(syms_list[i]) for i in idx]; n0 = [n_of(syms_list[i]) for i in idx]
        ok = [True] * B; vals = [[] for _ in range(B)]; fail = [None] * B; lens = [0] * B; past = None; kvalid = None; pend = [None] * B; mm = [math.inf] * B
        for k in range(1, max(n0) + 2):
            chunks, checks, reads = [], [], []
            for r in range(B):
                if k == 1:
                    pre, suf, more = candidate(trees[r]); pr = [EOT] + N.ids(N.render(trees[r])) + [EQ]; c = pr + pre
                    chk = [(len(pr) - 1 + j, pre[j]) for j in range(len(pre))] + [(len(c) - 1, None)]; pend[r] = (suf, more)
                elif k <= n0[r] + 1 and ok[r]:
                    suf, more = pend[r]; v = vals[r][-1]; mid = suf + ([EQ] if more else [EOT])
                    if more:
                        pre, suf2, more2 = candidate(trees[r]); pend[r] = (suf2, more2)
                    else:
                        pre = []
                    c = [NUM_IDS[v]] + mid + pre; chk = [(j, c[j + 1]) for j in range(len(c) - 1)] + ([(len(c) - 1, None)] if more else [])
                else:
                    c = [EOT]; chk = []
                chunks.append(c); checks.append(chk); reads.append([j for j, _ in chk])
            assert all(lens[r] + len(chunks[r]) <= 1024 for r in range(B)), "FULL: past GPT-2's 1,024 rows"
            am, mg, past, kvalid = _run_chunks(g, chunks, lens, past, kvalid, reads)
            for r in range(B):
                if ok[r] and (k == 1 or k <= n0[r] + 1):
                    for q, (j, tok) in enumerate(checks[r]):
                        mm[r] = min(mm[r], mg[r][q])
                        if tok is None:
                            v_ = am[r][q]
                            if v_ not in NUMS:
                                ok[r] = False; fail[r] = k
                            else:
                                vals[r].append(NUM2V[v_]); reduce_shape(trees[r], NUM2V[v_])
                            break
                        if am[r][q] != tok:
                            ok[r] = False; fail[r] = k; break
                lens[r] += len(chunks[r])
        del past, kvalid
        for r, i in enumerate(idx):
            res[i] = {"n": n0[r], "keep": bool(ok[r]), "values": vals[r] if ok[r] else None, "label": vals[r][-1] if ok[r] else None, "fail_step": fail[r],
                      "min_margin": mm[r]}
    out.extend(res)
    return out


# ------------------------------------------------------------------------------------------------ KV-cache greedy generation (independent code)
@torch.no_grad()
def greedy_batch(g, prompts, budget, stop_at_eq=False, margins=False):
    """KV-cache greedy generation for EQUAL-LENGTH prompts (nested170.cot_answers' loop: core.GPT2KV.forward, argmax of the fp32 logits at the last
    row). Stops when every row has emitted EOT (or ' =' with stop_at_eq), after `budget` tokens, and BREAKS BEFORE FEEDING ROW 1,024 (PROOFS P3).
    -> (per row the tokens up to and incl. its terminator, else every produced token; per row min top-2 margin (inf unless margins); the largest row fed)."""
    dev = g.w("wte.weight").device; P0 = len(prompts[0]); assert all(len(p) == P0 for p in prompts) and P0 <= 1024 and budget >= 1
    x = torch.tensor(prompts, device=dev, dtype=torch.long); h, past = g(g.embed(x), None); B = len(prompts)
    out = [[] for _ in range(B)]; done = [False] * B; mar = [math.inf] * B; fed = P0 - 1
    for j in range(budget):
        lg = g.logits(h[:, -1]).float(); nt = lg.argmax(-1); ntl = nt.cpu().tolist()
        if margins:
            tv = lg.topk(2, -1).values; mv = (tv[:, 0] - tv[:, 1]).cpu().tolist()
        for r in range(B):
            if not done[r]:
                out[r].append(int(ntl[r]))
                if margins:
                    mar[r] = min(mar[r], float(mv[r]))
                if ntl[r] == EOT or (stop_at_eq and ntl[r] == EQ):
                    done[r] = True
        if all(done) or j == budget - 1:
            break
        if P0 + j > 1023:                                                   # token j would sit at row P0 + j = 1,024: never fed
            break
        h, past = g(g.embed(nt[:, None]), past); fed = P0 + j
    del h, past
    return out, mar, fed


def _by_len(keys):
    by = {}
    for i, k in enumerate(keys):
        by.setdefault(k, []).append(i)
    return by


@torch.no_grad()
def gen_reroot(g, syms_list, bs=32):
    """G-T16's INDEPENDENT REROOT: per step a fresh prompt [EOT] + e_{k-1} + [' ='], KV-cache greedy decoding to the first ' =' / EOT (budget |e| + 1),
    equal-length prompt batches; validity by check_step_T. -> per input {n, keep, values, fail_step, fail_flags, steps (emitted ids incl. terminator),
    min_margin}."""
    S = len(syms_list); cur = [parse_tree(s) for s in syms_list]; n0 = [n_of(s) for s in syms_list]; ok = [True] * S; vals = [[] for _ in range(S)]
    fail = [None] * S; ff = [None] * S; steps = [[] for _ in range(S)]; mar = [math.inf] * S
    for k in range(1, max(n0) + 1):
        act = [i for i in range(S) if ok[i] and k <= n0[i]]
        for L, idx in _by_len([len(N.render(cur[i])) for i in act]).items():
            ii = [act[q] for q in idx]
            for b0 in range(0, len(ii), bs):
                sub = ii[b0:b0 + bs]; outs, ms, _ = greedy_batch(g, [[EOT] + N.ids(N.render(cur[i])) + [EQ] for i in sub], L + 1, True, True)
                for r, i in enumerate(sub):
                    o = outs[r]; mar[i] = min(mar[i], ms[r]); steps[i].append(o)
                    if not o or o[-1] not in (EQ, EOT):
                        ok[i] = False; fail[i] = k; ff[i] = {"no_terminator": True}; continue
                    f, v, t2 = check_step_T(cur[i], o[:-1], o[-1])
                    if t2 is None:
                        ok[i] = False; fail[i] = k; ff[i] = f
                    else:
                        vals[i].append(v); cur[i] = t2
    return [{"n": n0[i], "keep": bool(ok[i]), "values": vals[i] if ok[i] else None, "fail_step": fail[i], "fail_flags": ff[i], "steps": steps[i],
             "min_margin": mar[i]} for i in range(S)]


@torch.no_grad()
def gen_full(g, syms_list, bs=32):
    """G-T16's FULL generation (descriptive): whole-chain greedy generation (budget C_struct + 50, capped at 1025 - P), split at ' =' / EOT, each
    segment validated by check_step_T. -> per input {n, keep, values, fail_step, answer, ntok, min_margin}."""
    S = len(syms_list); res = [None] * S
    for L, idx in _by_len([len(s) for s in syms_list]).items():
        for b0 in range(0, len(idx), bs):
            ii = idx[b0:b0 + bs]; P0 = L + 2; budget = min(max(c_struct(syms_list[i]) for i in ii) + 50, CTX - P0)
            outs, ms, _ = greedy_batch(g, [[EOT] + N.ids(syms_list[i]) + [EQ] for i in ii], budget, False, True)
            for r, i in enumerate(ii):
                o = outs[r]; cur = parse_tree(syms_list[i]); vals = []; pos = 0; okk = True; fail = None; k = 0
                while cur.op is not None:
                    k += 1; j = pos
                    while j < len(o) and o[j] not in (EQ, EOT):
                        j += 1
                    if j >= len(o):
                        okk = False; fail = k; break
                    f, v, t2 = check_step_T(cur, o[pos:j], o[j])
                    if t2 is None:
                        okk = False; fail = k; break
                    vals.append(v); cur = t2; pos = j + 1
                if okk and pos != len(o):
                    okk = False; fail = k + 1
                res[i] = {"n": n_of(syms_list[i]), "keep": okk, "values": vals if okk else None, "fail_step": fail, "answer": cot_answer(o), "ntok": len(o),
                          "min_margin": ms[r]}
    return res


# ------------------------------------------------------------------------------------------------ pool records
def record(syms, values):
    """the pool writer (B1): keys exactly REC_KEYS."""
    v = [int(x) for x in values]; r = {"syms": list(syms), "n": n_of(syms), "chain_label": v[-1], "values": v}; assert set(r) == REC_KEYS; return r


def load_pool(path, meta_path=None):
    """the trainer's self-pool loader: sha256 == pool_meta.json's; every record's key set == REC_KEYS (no gold field); n recomputed from syms;
    len(values) == n; chain_label == values[-1]; every value in 0..P-1. -> (records, meta)."""
    meta_path = meta_path or os.path.join(os.path.dirname(path), "pool_meta.json")
    meta = json.load(open(meta_path)); sha = file_sha256(path); assert sha == meta["sha256"], ("pool sha mismatch", sha, meta["sha256"])
    recs = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln); assert set(r) == REC_KEYS, ("pool record keys", sorted(r)); assert not (set(r) & GOLD_KEYS)
            assert n_of(r["syms"]) == r["n"] == len(r["values"]), "pool record n != #operators(syms) / len(values)"
            assert r["chain_label"] == r["values"][-1] and all(isinstance(v, int) and 0 <= v < N.P for v in r["values"]), "pool record values"
            recs.append(r)
    return recs, meta


def self_item(rec):
    """a pool record -> a trainer item: syms, n and the labeller's values (SD-LABEL-MAP-T). No root: any gold path is a KeyError."""
    it = {"syms": rec["syms"], "n": rec["n"]}
    it["values"] = rec["values"]  # SD-LABEL-MAP-T
    return it


# ------------------------------------------------------------------------------------------------ gold (gate / audit / eval processes only)
def gold_values(syms):
    """GOLD (never in a labeller or trainer path): the value written at each CoT step (N.reductions on the parsed tree; uses N.apply)."""
    return [int(v) for v in N.reductions(parse_tree(syms))]


def patch_gold():
    """G-T0 / TS_v1 §2.4 item 3: the builder's gold patch (N.apply, N.value, N.eval_tokens, N.cot_steps, N.cot_ids, N.reductions raise)."""
    def boom(*a, **k):
        raise RuntimeError("G-T0: a gold function was called in a gold-free stage")
    for nm in ("apply", "value", "eval_tokens", "cot_steps", "cot_ids", "reductions"):
        setattr(N, nm, boom)


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


# ------------------------------------------------------------------------------------------------ exclusion (TS_v1 §2.4 item 2; Addendum 1 item 4)
def _serial():
    r_sp = np.random.default_rng(17300)
    return [tuple(N.render(N.generate(r_sp, 8)[0])) for _ in range(2200)]


def fresh_213_37(n_lo=9, n_hi=24, n_items=1000):
    """part_chainlabel_fresh's item draw (rng 21337), n 9..24 sequentially, excluding test_set(1000) + SERIAL (s216_build_pool.fresh_213_37)."""
    excl = {tuple(it["syms"]) for it in N.test_set(1000)} | set(_serial()); rng = np.random.default_rng(21337); out = set()
    for n in range(n_lo, n_hi + 1):
        got = 0
        while got < n_items:
            s_ = tuple(N.render(N.generate(rng, n)[0]))
            if s_ in excl:
                continue
            out.add(s_); got += 1
    return out


def precond_streams():
    """A5 / V7: the TEXT-SELF precondition items (read by the base): {23180: {n: [syms]}, 23187: {n: [syms]}}, the full 200-per-n streams at n 9..20,
    each regenerated with its script's own loop."""
    ex24 = {tuple(it["syms"]) for it in N.test_set(1000, n_max=24)}; ser = _serial(); excl = ex24 | set(ser)
    rng = np.random.default_rng(23180); s80 = {}
    for n in range(9, 21):                                                  # ts_label_check.main: one stream, n 9..20 in order, 200 per n, redrawn on a hit
        s80[n] = []
        while len(s80[n]) < 200:
            r_, _ = N.generate(rng, n)
            if tuple(N.render(r_)) in excl:
                continue
            s80[n].append(N.render(r_))
    excl2 = set(excl); r_d = np.random.default_rng(23180)                   # pc_reroot.main: the 23180 items added to its exclusion one by one
    for n in range(9, 21):
        c = 0
        while c < 200:
            r_, _ = N.generate(r_d, n)
            if tuple(N.render(r_)) in excl2:
                continue
            excl2.add(tuple(N.render(r_))); c += 1
    rng = np.random.default_rng(23187); s87 = {}
    for n in range(9, 21):
        s87[n] = []
        while len(s87[n]) < 200:
            r_, _ = N.generate(rng, n)
            if tuple(N.render(r_)) in excl2:
                continue
            s87[n].append(N.render(r_))
    return {23180: s80, 23187: s87}


def gt18_items(n_draw=1000):
    """A4: the G-T18 smoke items (fresh n-24 shapes, rng 23195, read by the base on the 3070): 1,000 draws, redrawn on a hit vs test_set(1000, 30) + SERIAL."""
    excl = {tuple(it["syms"]) for it in N.test_set(1000, n_max=30)} | set(_serial()); rng = np.random.default_rng(23195); out = []
    while len(out) < n_draw:
        s_ = N.render(N.generate(rng, 24)[0])
        if tuple(s_) in excl:
            continue
        out.append(s_)
    return out


def _gate_syms(path):
    g = json.load(open(path))
    its = [it for w in g["windows"].values() for it in w["items"]] if "windows" in g else g["items"]
    return {tuple(it["syms"]) for it in its}


def _pool_syms(path):
    out = set()
    with open(path, encoding="utf-8") as f:
        for ln in f:
            out.add(tuple(json.loads(ln)["syms"]))
    return out


def stream_sha(st):
    return hashlib.sha256(json.dumps({str(k): {str(n): v for n, v in d.items()} for k, d in sorted(st.items())}, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def exclusion_sets(own_pool=None, own_gate=None, with_own_gate=True):
    """{name: set of syms tuples} per TS_v1 §2.4 item 2 (+ T11): test_set(1000, n_max 30) (incl. the sealed n 25..30), SERIAL, the 213-37 fresh items,
    every latent gate_items.json and latent pool file on disk (A6), the precondition streams 23180 / 23187 (A5), the G-T18 smoke items 23195 (A4), and
    every s219 gate_items.json / pool file already built (own_pool never; own_gate only if with_own_gate). Smoke dirs never count. Also -> meta {stream
    sha, files read}."""
    S = {"test_n30": {tuple(it["syms"]) for it in N.test_set(1000, n_max=30)}, "serial": set(_serial()), "s213_37_n9_24": fresh_213_37()}
    pst = precond_streams(); S["precond_23180"] = {tuple(s) for d in pst[23180].values() for s in d}; S["precond_23187"] = {tuple(s) for d in pst[23187].values() for s in d}
    S["gt18_23195"] = {tuple(s) for s in gt18_items()}
    files = []
    nrm = lambda p: os.path.normpath(p) if p else None  # noqa: E731
    for gp in sorted(x for x in glob.glob(R + "s21[56]_gate*/gate_items.json") + glob.glob(R + "s219_gate*/gate_items.json") if "_smoke" not in x):
        if nrm(gp) == nrm(own_gate) and not with_own_gate:
            continue
        S["gates:" + gp.replace("\\", "/")] = _gate_syms(gp); files.append(gp.replace("\\", "/"))
    for pp in sorted(x for x in glob.glob(R + "s21[56]_pool*/pool_r*.jsonl") + glob.glob(R + "s219_pool*/pool_t*.jsonl") if "_smoke" not in x):
        if nrm(pp) == nrm(own_pool):
            continue
        S["pool:" + pp.replace("\\", "/")] = _pool_syms(pp); files.append(pp.replace("\\", "/"))
    meta = {"precondition_stream_sha": stream_sha(pst), "precondition_counts": {str(k): {str(n): len(v) for n, v in d.items()} for k, d in pst.items()},
            "files": files, "numpy": np.__version__}
    return S, meta
