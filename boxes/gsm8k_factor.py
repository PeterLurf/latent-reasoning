"""THEORY §204 STEP-TARGET-FACTOR (f6f8d51d; contract docs/THEORY_STEPFACTOR_DRAFT_2026-10-14.md §8, frozen at 3ec408c6): the factorised step target for
gsm8k_stepaux's stepaux arm, in a NEW module so prosqa168s2.ProsMem (shared with §168s2 / §191) is untouched (§8.1's zero-risk alternative).
  FactorMem(ProsMem): the same DeltaMem construction (identical global-RNG use) with add_norm passed through (§8.1 RMS plumbing: the §185 path, no
      deltamem change), nopass (aux prompt [EOT, " Step", " :"], asserted pass-token-free), aux_logits / aux_greedy on aux_prompt(t), and - only with
      factor=True - the heads W_op (768 -> 17) and W_p1 / W_p2 (768 -> 768, identity init), initialised from a dedicated generator (seed + 30000) so the
      global RNG stream is unchanged. The heads read only the aux decoder's h_last (the final hidden at the prompt's last position); answers() never
      calls them (L6).
  factor_targets(raw, tpl, op_vocab) (pure data, §8.3): per step t, the value ids ENC(" " + rhs) + [EOT] (None unless the rhs is one number token
      string), the operator class (signature = the LHS's "+-*/%()" characters; index in op_vocab, else OTHER = len(op_vocab)), and pointers j = 0, 1
      (recomputable items only, from parse_recomputable's ops in LHS order): (target first token, candidate first tokens C_t); "coll" = C_t's sources
      include two different values sharing a first BPE token (pointer accuracy is over first-token classes; note 204-3 adds a collision-free field).
      Value fields compare by STRING (decoded greedy value .strip() == the gold RHS; note 204-3); value_exact_tokid keeps token-id equality.
  factor_loss(m, S, its, t, dev, mode): per-item L_aux,t = the mean of the present terms (value: token-mean LM CE teacher-forced after the prompt;
      op: CE over 17; pointers: CE over C_t with logits <W_pj h_last, wte[c]> (wte detached)); mode 'value' = the value term only.
  aux_suite(...): the §8.4 eval fields (value exact by pass / grounding, op / pointer accuracy, the string arm's in-string proxies) from M_t, M_1, M_0,
      and COUNTER-SWAP's synthetic states.
    python -m boxes.gsm8k_factor --build_op_vocab   -> bench/data/gsm8k_aug/s204_op_vocab.json (frozen top-16 signatures of the filtered train set)"""
import hashlib
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
from boxes.deltas5 import fwd_read  # noqa: E402
from boxes.prosqa168s2 import COLON_ID, ENC, EOT, STEP_ID, ProsMem, pad, pass_tok  # noqa: E402

OPCH = "+-*/%()"; RHS_RE = re.compile(r"[\d,]*\.?\d+"); OPV_PATH = os.path.join(G.AUG, "s204_op_vocab.json"); N_OP = 16
PASS_TOKS = {pass_tok(t) for t in range(0, 40)}


def signature(e):
    return "".join(ch for ch in e.split("=")[0] if ch in OPCH)


def load_op_vocab(path=None):
    p = path or OPV_PATH; raw = open(p, "rb").read(); return json.loads(raw)["vocab"], hashlib.sha256(raw).hexdigest()


def build_op_vocab():
    """the frozen top-16 operator signatures over every step of the FILTERED train set (count desc, ties by first occurrence)."""
    R, fc = G.load_aug("train"); cnt = Counter(); first = {}
    for i, r in enumerate(R):
        for e in r["eqs"]:
            s = signature(e); cnt[s] += 1; first.setdefault(s, (i, len(first)))
    order = sorted(cnt, key=lambda s: (-cnt[s], first[s][1])); voc = order[:N_OP]; tot = sum(cnt.values())
    out = {"vocab": voc, "counts": {s: cnt[s] for s in voc}, "coverage": sum(cnt[s] for s in voc) / tot, "n_steps": tot, "n_signatures": len(cnt),
           "tie_order": "count desc, then first occurrence in train order", "train_filter": fc,
           "def": "§204 §8.2 --op_vocab: signature = the LHS's '+-*/%()' characters in order; OTHER = index 16"}
    json.dump(out, open(OPV_PATH, "w", newline="\n"), indent=1); return out


def factor_targets(raw, tpl, op_vocab):
    """§8.3, per step: {"v": value ids or None, "op": class, "p": [(target, cands) or None] * 2, "g": "R" | "Q" | "unparsed", "rhs": str}."""
    out = []; qlit = set()
    if tpl is not None:
        for _, _, tok, _ in tpl["qn"]:
            qlit.add(ENC.encode(tok)[0]); qlit.add(ENC.encode(tok.replace(",", ""))[0])
    for t, e in enumerate(raw["eqs"]):
        lhs, _, rhs = e.partition("="); rhs = rhs.strip(); sig = signature(e)
        st = {"v": (ENC.encode(" " + rhs) + [EOT]) if RHS_RE.fullmatch(rhs) else None, "op": op_vocab.index(sig) if sig in op_vocab else len(op_vocab),
              "p": [None, None], "rhs": rhs, "g": "unparsed", "coll": None}
        if tpl is not None:
            eqt = tpl["eqt"][t]; prev = {ENC.encode(tpl["eqt"][j]["rtok"])[0] for j in range(t)}; st["g"] = "R" if any(o[3] == "r" for o in eqt["ops"]) else "Q"
            vals = {}                                                       # note 204-3: C_t's first-token collision (two different values, one first token)
            for _, _, tok, val in tpl["qn"]:
                for tk in {ENC.encode(tok)[0], ENC.encode(tok.replace(",", ""))[0]}:
                    vals.setdefault(tk, set()).add(round(val, 6))
            for j in range(t):
                vals.setdefault(ENC.encode(tpl["eqt"][j]["rtok"])[0], set()).add(round(tpl["eqt"][j]["r0"], 6))
            st["coll"] = any(len(v_) > 1 for v_ in vals.values())
            for j in range(2):
                if j < len(eqt["ops"]):
                    tg = ENC.encode(eqt["ops"][j][2])[0]; st["p"][j] = (tg, sorted(qlit | prev | {tg}))
        out.append(st)
    return out


class FactorMem(ProsMem):
    """ProsMem with --add_norm passed to DeltaMem, --aux_nopass and (factor=True) the factor heads (§8.1 / §8.2)."""
    def __init__(self, g, add_norm=None, nopass=False, factor=False, seed=1, n_op=N_OP + 1):
        torch.nn.Module.__init__(self); self.g = g; self.mem = DM.DeltaMem(g, 6, "top", add_norm=add_norm or None)   # = ProsMem's call when add_norm is None
        self.nopass = bool(nopass); self.factor = bool(factor)
        if factor:
            dev = g.w("wte.weight").device; gen = torch.Generator(device="cpu").manual_seed(int(seed) + 30000); D = g.w("wte.weight").shape[1]; bd = 1 / np.sqrt(D)
            self.w_op = torch.nn.Linear(D, n_op).to(dev); self.w_p1 = torch.nn.Linear(D, D).to(dev); self.w_p2 = torch.nn.Linear(D, D).to(dev)
            with torch.no_grad():                                           # no global-RNG draw (the default Linear init would consume it)
                self.w_op.weight.copy_((torch.rand(n_op, D, generator=gen) * 2 - 1) * bd); self.w_op.bias.copy_((torch.rand(n_op, generator=gen) * 2 - 1) * bd)
                for w_ in (self.w_p1, self.w_p2):
                    w_.weight.copy_(torch.eye(D)); w_.bias.zero_()

    def aux_prompt(self, t):
        if self.nopass:
            pre = [EOT, STEP_ID, COLON_ID]; assert not any(tk in PASS_TOKS for tk in pre), "§204: a pass token in the NOPASS aux prompt"
            return pre
        return [EOT, STEP_ID, pass_tok(t), COLON_ID]

    def aux_logits(self, S, its, t, dev):
        """prosqa168s2.ProsMem.aux_logits with the prompt from aux_prompt(t) (identical when nopass is off)."""
        pre = self.aux_prompt(t); seqs, tg = [], []
        for it in its:
            assert t <= it["depth"], "§168s2: no aux target past the last hop"
            tgt = it["step_ids"][t - 1] + [EOT]; seqs.append(pre + tgt[:-1]); tg.append(tgt)
        x_, _ = pad(seqs); y_, _ = pad(tg, -100); x = torch.tensor(x_, device=dev); y = torch.tensor(y_, device=dev)
        assert x.shape[1] == len(pre) + y.shape[1] - 1 and bool((x[:, :len(pre)] == torch.tensor(pre, device=dev)).all())
        h = fwd_read(self.g, self.g.embed(x), self.mem, S)
        return self.g.logits(h[:, len(pre) - 1:]).float(), y

    @torch.no_grad()
    def aux_greedy(self, S, t, L):
        pre = self.aux_prompt(t); x = torch.tensor([pre] * S.shape[0], device=S.device)
        for _ in range(L):
            nt = self.g.logits(fwd_read(self.g, self.g.embed(x), self.mem, S)[:, -1]).float().argmax(-1); x = torch.cat([x, nt[:, None]], 1)
        return x[:, len(pre):].cpu().numpy()

    def op_logits(self, h_last):
        return self.w_op(h_last)

    def ptr_logits(self, h_last, cands, j):
        """pointer j (0 / 1): <W_pj h_last, wte[c]> over each item's candidate first tokens (wte detached) -> list of 1-d tensors (None = masked)."""
        wte = self.g.w("wte.weight").detach(); q_ = (self.w_p1, self.w_p2)[j](h_last)
        return [wte[torch.tensor(c_, device=q_.device)] @ q_[i] if c_ is not None else None for i, c_ in enumerate(cands)]


def factor_loss(m, S, its, t, dev, mode="factor"):
    """per-item L_aux,t (list aligned with its; None = no term present) and the per-term CE sums / counts for logging. One forward of
    prompt + value[:-1] (right-padded); h_last = the hidden at the prompt's last position (causal: identical with or without the value tokens)."""
    pre = m.aux_prompt(t); fts = [it["ft"][t - 1] for it in its]
    seqs = [pre + (f_["v"][:-1] if f_["v"] is not None else []) for f_ in fts]; x_, _ = pad(seqs); x = torch.tensor(x_, device=dev)
    h = fwd_read(m.g, m.g.embed(x), m.mem, S); P = len(pre) - 1; terms = [[] for _ in its]; log = {}
    vi = [i for i, f_ in enumerate(fts) if f_["v"] is not None]
    if vi:
        lg = m.g.logits(h[vi, P:]).float(); y_, _ = pad([fts[i]["v"] for i in vi], -100); y = torch.tensor(y_, device=dev)
        ce = Fn.cross_entropy(lg[:, :y.shape[1]].transpose(1, 2), y, ignore_index=-100, reduction="none").sum(1) / (y != -100).sum(1)
        for r_, i in enumerate(vi):
            terms[i].append(ce[r_])
        log["val"] = (float(ce.detach().sum()), len(vi))
    if mode == "factor":
        hl = h[:, P].float(); ceo = Fn.cross_entropy(m.op_logits(hl).float(), torch.tensor([f_["op"] for f_ in fts], device=dev), reduction="none")
        for i in range(len(its)):
            terms[i].append(ceo[i])
        log["op"] = (float(ceo.detach().sum()), len(its))
        for j in range(2):
            lpj = m.ptr_logits(hl, [f_["p"][j][1] if f_["p"][j] is not None else None for f_ in fts], j); s_, n_ = 0.0, 0
            for i, f_ in enumerate(fts):
                if f_["p"][j] is not None:
                    tg, cs = f_["p"][j]; c_ = Fn.cross_entropy(lpj[i].float()[None], torch.tensor([cs.index(tg)], device=dev)); terms[i].append(c_); s_ += float(c_.detach()); n_ += 1
            log["ptr%d" % (j + 1)] = (s_, n_)
    return [torch.stack(tt).mean() if tt else None for tt in terms], log


# ---------------------------------------------------------------- eval (§8.4)
@torch.no_grad()
def greedy_value(m, S, t, L=12):
    """greedy after the aux prompt until EOT (<= L tokens) -> list of token lists."""
    gen = m.aux_greedy(S, t, L); out = []
    for row in gen:
        r_ = []
        for v in row:
            if int(v) == EOT:
                break
            r_.append(int(v))
        out.append(r_)
    return out


@torch.no_grad()
def head_preds(m, S, its, t, dev):
    """(op argmax, [pointer argmax token or None] x 2) reading S with the prompt (factor arm)."""
    pre = m.aux_prompt(t); x = torch.tensor([pre] * len(its), device=dev); hl = fwd_read(m.g, m.g.embed(x), m.mem, S)[:, -1].float(); fts = [it["ft"][t - 1] for it in its]
    out = [None, None]
    for j in range(2):
        lpj = m.ptr_logits(hl, [f_["p"][j][1] if f_["p"][j] is not None else None for f_ in fts], j)
        out[j] = [(fts[i]["p"][j][1][int(lpj[i].argmax())] if lpj[i] is not None else None) for i in range(len(its))]
    return m.op_logits(hl).argmax(-1).tolist(), out


STR_RE = re.compile(r"<<([^=<>]*)=([^<>]*)>>")


def aux_suite(m, D, think_states_fn, K, dev, arm, op_vocab, tmax=8, bs=64):
    """§8.4: for passes t <= min(m, tmax, K), from M_t / M_1 (pass-token arms) / M_0 and COUNTER-SWAP's (t S_add(M_1), S_del(M_1)), (S_add(M_t),
    S_del(M_1)), (S_add(M_1), S_del(M_t)): value exact (factor / value arms: greedy value; string arm: value-in-string), op accuracy (factor: W_op;
    string: greedy signature), pointer-1/2 accuracy (factor: heads; string: first operand's first token), by pass and by grounding (R / Q / unparsed).
    think_states_fn(items, k) -> M_k (b, ...). Works on a plain ProsMem too (the string comparator's eval-only re-run: same fields, same items).
    Returns {source: {field: {t: acc}}} plus counts."""
    HA = m.mem.h_add; HD_ = m.mem.h_del; R = {}

    def add(src, fld, t, g_, v):
        R.setdefault(src, {}).setdefault(fld, {}).setdefault(str(t), []).append(v)
        if g_ is not None:
            R[src].setdefault(fld + "_by_grounding", {}).setdefault(g_, {}).setdefault(str(t), []).append(v)
    for i0 in range(0, len(D), bs):
        ch = D[i0:i0 + bs]; Ms = {0: None}; T_ = min(K, tmax, max(it["m"] for it in ch))
        for k in range(1, T_ + 1):
            Ms[k] = think_states_fn(ch, k)
        M0 = torch.zeros_like(Ms[1])
        for t in range(1, T_ + 1):
            ix = [j for j, it in enumerate(ch) if t <= it["m"]]
            if not ix:
                continue
            its = [ch[j] for j in ix]; Mt, M1 = Ms[t][ix], Ms[1][ix]
            srcs = {"Mt": Mt, "M0": M0[ix]}
            if not getattr(m, "nopass", False):                             # M_1 with P_t is undefined under NOPASS
                srcs["M1"] = M1
            synth = M1.clone(); synth[:, :HA] = t * M1[:, :HA]; sw1 = M1.clone(); sw1[:, :HA] = Mt[:, :HA]; sw2 = Mt.clone(); sw2[:, :HA] = M1[:, :HA]
            srcs.update({"cs_counter": synth, "cs_addMt_delM1": sw1, "cs_addM1_delMt": sw2})
            for src, S in srcs.items():
                fts = [it["ft"][t - 1] for it in its]
                if arm in ("value", "factor"):
                    gv = greedy_value(m, S, t)
                    for f_, g_ in zip(fts, gv):                             # note 204-3: STRING equality (cross-arm); token-id equality kept as _tokid
                        if f_["v"] is not None:
                            add(src, "value_exact", t, f_["g"], ENC.decode(g_).strip() == f_["rhs"]); add(src, "value_exact_tokid", t, f_["g"], g_ == f_["v"][:-1])
                    if arm == "factor":
                        op_, (p1, p2) = head_preds(m, S, its, t, dev)
                        for r_, f_ in enumerate(fts):
                            add(src, "op_acc", t, None, op_[r_] == f_["op"])
                            for j, pj in ((0, p1), (1, p2)):
                                if f_["p"][j] is not None:
                                    add(src, "ptr%d_acc" % (j + 1), t, f_["g"], pj[r_] == f_["p"][j][0])
                                    if not f_["coll"]:                      # note 204-3: collision-free items only
                                        add(src, "ptr%d_acc_nocollide" % (j + 1), t, f_["g"], pj[r_] == f_["p"][j][0])
                else:                                                       # string arm: the in-string proxies of its greedy " <<lhs=rhs>>"
                    gen = m.aux_greedy(S, t, max(len(it["step_ids"][t - 1]) + 1 for it in its))
                    for r_, (it, f_) in enumerate(zip(its, fts)):
                        txt = ENC.decode([int(v) for v in gen[r_] if int(v) != EOT]); mm = STR_RE.search(txt)
                        add(src, "value_in_string", t, f_["g"], bool(mm) and mm.group(2).strip() == f_["rhs"])
                        add(src, "op_in_string", t, None, bool(mm) and signature(mm.group(1) + "=") == signature(it["eqs"][t - 1]))
                        if f_["p"][0] is not None:
                            num = re.match(r"[\d,.]+", mm.group(1).lstrip("( ")) if mm else None
                            ok_ = bool(num) and ENC.encode(num.group(0))[0] == f_["p"][0][0]; add(src, "ptr1_in_string", t, f_["g"], ok_)
                            if not f_["coll"]:
                                add(src, "ptr1_in_string_nocollide", t, f_["g"], ok_)
    out = {}
    for src, d_ in R.items():
        o_ = out.setdefault(src, {})
        for fld, v in d_.items():
            if fld.endswith("_by_grounding"):
                o_[fld] = {g_: {t: float(np.mean(x)) for t, x in sorted(vv.items(), key=lambda kv: int(kv[0]))} for g_, vv in v.items()}
            else:
                o_[fld] = {t: float(np.mean(x)) for t, x in sorted(v.items(), key=lambda kv: int(kv[0]))}; o_["n_" + fld] = {t: len(x) for t, x in v.items()}
    return out


if __name__ == "__main__":
    if "--build_op_vocab" in sys.argv:
        o = build_op_vocab(); print(json.dumps({k: v for k, v in o.items() if k != "counts"}), o["counts"])
