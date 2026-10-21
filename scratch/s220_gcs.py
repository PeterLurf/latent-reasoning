"""§220 GSM-CACHE-SWAP, stage 0 (GPT-2): the eval battery -> <out_dir>/eval_gcs.json (contract = theory/score_s220.py's docstring, 5f8b7361).
Registered text: docs/research_scaleprobe_2026-10-20/GCS_v1.md at 3a88762e (v1 + Addendum 1); THEORY §220. A NEW scratch file: it imports
boxes/codi_port.py and boxes/codi_g1.py read-only and edits nothing under boxes/ or theory/. Stage 1 (CODI-1B) is NOT built here.

Models (--model): codi20 = boxes/runs_fsq/gsm_codi_20ep_s1/trainable.pt through codi_port.build -> merged_fp32 (210-4's path);
                  zen    = zen-E/CODI-gpt2 prosqa/ckpt_ext/hf/codi/pytorch_model.bin, loaded as boxes/codi_g1.py does (sha256-checked) -> merged_fp32;
                  tiny   = a random-init 4-layer GPT-2 (smokes only).
Path (GCS_v1 §1; GPT-2 has no BOS, so "[BOS] q <bot>" is "q <bot>", as CODI's GPT-2 recipe and codi_port.codi_loss):
  q <bot> -> z_1 = prj(h_last); z_j fed as inputs_embeds, z_{j+1} = prj(h_last); K inputs fed (z_{K+1} discarded); <eot> embedding; greedy argmax over
  logits[:, :V-1] (<eot> never generated), stop at EOS (50256), max 256 new tokens; parser = CODI test.py's: commas removed, the LAST number anywhere
  (regex -?\\d+\\.?\\d*), float equality with the gold (no number -> None, = wrong / neither). bs 1, fp32, eager attention, TF32 off.
  bs-1 path: explicit position_ids 0..T-1, no attention_mask. Official-semantics path (G-R, "virtual bs 128"): each item left-padded with [PAD] to the
  max (question + <bot>) length of its dataset-order block of 128, attention_mask on the question pass only, no position_ids anywhere (HF default =
  pad-inclusive arange), so pads are attended in the latent and answer phases, as test.py.
Arms (GCS_v1 §4): INTACT (K 6, captures z_1..z_6 inputs and the latent-row final hidden states), OFFICIAL (G-R), K-SWEEP {0,1,2,3,4,5,6,8,10},
  MEAN-Z, NOISE (torch rng 22022), QDROP (+ K 0), QMASK (+ K 0), LOGIT-LENS; per exact-length pair: CACHE-SWAP, STATE-SWAP, LEAK-SEG, SELF-SWAP
  (cache / state), variants (k_only, v_only, last_row, layers_lo, layers_hi), PREFIX-SWAP k 1/3/5, SCIT bridge (cache / state); per donor: donor-own
  (cache / state) bitwise; GRAPH-MATCHED Δ = 0 (cache, state, SCIT incl. oracle-cf) and the Δ != 0 graph-matched row under its own key; CODI20 only:
  210-4's original 500 pairs (STATE-SWAP only, descriptive).
Cache arms: the recipient's intact cache (an exact clone) gets the donor's rows written at the stated slots/layers/kinds; EVERY other row of every
  layer's keys and values is asserted torch.equal to the recipient's intact rows, and the written rows torch.equal to the donor's source rows
  (asserts.cache_rows_ok; failures are recorded, the run continues, and the scorer censors to UNREAD).
Pairs: exact question-token-length donors (np rng 22020, recipients in index order, donor uniform among INTACT-correct items with a different gold,
  the same question token length and use count < 3); graph-matched (np rng 22021) from the <<expr=c>> annotations (reviewer's signature function,
  review_scripts/oc_cache.py), dropping pairs whose oracle-cf equals either gold.

UNSPECIFIED CHOICES (registration silent or ambiguous; the conservative option taken, for Audit's G-D review):
  U1  GPT-2 has no BOS: QMASK masks every question token (positions 0..Lq-1); <bot>, latents, <eot> and generated tokens stay visible; positions are
      unchanged. QDROP = "<bot> z_1..z_6 <eot>" with the item's own captured z, positions 0.. (fresh forward). QDROP-K0 = "<bot> <eot>" (one constant
      output, broadcast to every item); QMASK-K0 = K 0 with the question masked (per item).
  U2  NOISE "per-position norm matched" = 210-4's definition: each (item, position) Gaussian vector rescaled to that item's own z_j L2 norm
      (torch.Generator seed 22022, one (N, 6, D) draw in item order).
  U3  MEAN-Z = per-position mean of the INTACT z_j over the evaluated items (all 1,319 in the real run).
  U4  LOGIT-LENS: z_j through the tied LM head directly (z is prj's LayerNorm output; no ln_f); latent-row residual = the final-layer hidden state at
      latent position j (HF hidden_states[-1], ln_f applied) through the LM head; "top number token" = argmax over vocabulary ids whose decoded text
      (stripped) is a plain number; stored with the overall top-1 id.
  U5  SCIT bridge: one teacher-forced forward of [<eot>] + tokens("The answer is: N") + [EOS] on a clone of the arm's cache; log-softmax over
      logits[:, :V-1] (as decoding); stored: summed log-prob (all tokens), token count, and the tail sum/count (the tokens after "The answer is:").
      N strings: golds as written in the test file (commas removed); oracle-cf formatted as an integer when integral, else %.10g. Random gold: a fresh
      np rng 22023 per pair set, an item drawn uniformly among items whose gold differs from recipient, donor and oracle-cf golds.
  U6  Graph-matched oracle-cf: primary = substitute the donor's step-1 result into the recipient's step 1 and re-execute the recipient's later steps
      (step 1 is always a leaf); the step (n-1) substitution is stored too (ocf_klast). Both programs must be "executable": every annotated step
      re-evaluates to its stated result and the last step equals the gold; signatures need >= 2 steps (a 1-step oracle-cf is the donor gold, dropped
      by rule). ocf values rounded to 6 decimals. The reuse cap 3 (Addendum 1.2) also applies to the graph-matched draw (one counter over both rows);
      Δ = 0 candidates are preferred; a recipient with only Δ != 0 candidates goes to the separate descriptive row graph_matched_dne0, run with
      STATE-SWAP and an uncorrected reverse-distance CACHE-SWAP (latent rows to the recipient's latent slots; GPT-2 positions cannot be corrected).
  U7  LEAK-SEG at Δ = 0: the donor's rows at positions Lq-6..Lq-1 (its last 6 question tokens, before <bot>) are written into the SAME positions of the
      recipient's cache (GCS_v1 §4 wording); the recipient's latent rows are kept. (REVIEW_SP §8 iii's "written into the same destination slots"
      could be read as the latent slots; GCS_v1 wins.)
  U8  Test file: prosqa/ckpt_ext/hf/data/gsm8k_test.json, sha256 pinned below (reviewer: identical in order and golds to HF openai/gsm8k main test,
      parquet sha256 prefix ee7b8da9e381df27; the parquet is not present locally, so meta.test_sha256 is the json's sha256).
  U9  Decoded text drops ids >= 50257 (pad / <bot>) before tokenizer.decode(skip_special_tokens=True), equal to the slow tokenizer's own handling.
  U10 Token sha = sha256 of ",".join(generated ids incl. the stopping EOS). G-R "agree" = share of items whose parsed predictions are equal (None ==
      None counts as agreement); per-item preds of both runs are stored.
  U11 210-4 pairs (CODI20 only) are read from boxes/runs_fsq/gsm_codi_20ep_s1/eval_latent_use.json items (recipient, donor; pred_swap_all stored
      beside ours). zen-E has no 210-4 pairs, so pairs_210_4 is absent for it.
  U12 Graph-matched and 210-4 pairs run no SELF-SWAP (identity is gated on the primary pairs).
Smokes: --tiny (random 4-layer GPT-2), --synthetic N (template arithmetic questions, NOT GSM8K), --limit N, --max_new; any of these requires
--out_dir under scratch/_tmp. --smoke_all_correct (tiny/synthetic only) treats every item as INTACT-correct so the pair arms are exercised.
    python scratch/s220_gcs.py --model codi20 --device cuda:0                      # -> boxes/runs_fsq/gsm_codi_20ep_s1/eval_gcs.json
    python scratch/s220_gcs.py --model zen --device cuda:0                         # -> scratch/s220/zen_codi_gpt2/eval_gcs.json
    python scratch/s220_gcs.py --model codi20 --device cuda:0 --gate_m --out_dir scratch/_tmp/s220/gm_codi20   # G-M only (synthetic tokens)
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("THEORY_THREADS", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

TEST = os.path.join(ROOT, "prosqa", "ckpt_ext", "hf", "data", "gsm8k_test.json")
TEST_SHA = "3d497ddbd728358309e25046b47a383adf582428cd3b5574767a55376b291d8d"
RUN20 = os.path.join(ROOT, "boxes", "runs_fsq", "gsm_codi_20ep_s1")
OUT_DEFAULT = {"codi20": RUN20, "zen": os.path.join(ROOT, "scratch", "s220", "zen_codi_gpt2")}
KS = (0, 1, 2, 3, 4, 5, 6, 8, 10)
K6 = 6
NLEAK = 6
MAX_REUSE = 3
SEED_PAIRS, SEED_GM, SEED_NOISE, SEED_RAND = 22020, 22021, 22022, 22023
ANS_PREFIX = "The answer is: "
NUM_RE = re.compile(r"-?\d+\.?\d*")


# ------------------------------------------------------------------ data, parser, signatures (no model)
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def gold_str(a):
    return str(a).replace(",", "").split("####")[-1].strip()


def load_test():
    d = json.load(open(TEST, encoding="utf-8"))
    return [{"i": i, "q": q.strip().replace("  ", " "), "gold": float(gold_str(a)), "gold_s": gold_str(a), "cot": c or ""}
            for i, (q, c, a) in enumerate(zip(d["question"], d["cot"], d["answer"]))]


def synthetic(n, seed=0):
    """template arithmetic word problems with <<a op b = c>> annotations (NOT GSM8K; smokes only)."""
    rng = np.random.default_rng(seed); names = ["Ann", "Bob", "Cal", "Dee"]; out = []
    for i in range(n):
        t = i % 3; nm = names[i % 4]; a, b, c = (int(x) for x in rng.integers(2, 60, 3))
        if i % 5 == 0:
            a = int(rng.integers(1000, 99999))                                         # multi-token numbers: unequal lengths (exercises the Δ != 0 rows)
        if t == 0:
            q, cot, g = f"{nm} has {a} apples and buys {b} more. How many apples does {nm} have?", f"<<{a}+{b}={a + b}>>", a + b
        elif t == 1:
            s = a + b; q = f"{nm} has {a} pens, buys {b} more and gives away {c}. How many pens are left?"; cot = f"<<{a}+{b}={s}>> <<{s}-{c}={s - c}>>"; g = s - c
        else:
            s = a * b; q = f"{nm} buys {a} boxes of {b} nails and then {c} more nails. How many nails in total?"; cot = f"<<{a}*{b}={s}>> <<{s}+{c}={s + c}>>"; g = s + c
        out.append({"i": i, "q": q, "gold": float(g), "gold_s": str(g), "cot": cot})
    return out


def parse_official(txt):
    """CODI test.py extract_answer_number: commas removed, the last number anywhere; none -> None (test.py: inf, scored wrong)."""
    nums = NUM_RE.findall(txt.replace(",", ""))
    return float(nums[-1]) if nums else None


def program(cot):
    """reviewer's graph signature (review_scripts/oc_cache.py sig) plus the executable program it describes."""
    steps = re.findall(r"<<([^=<>]+)=([^<>]+)>>", cot.replace(",", "")); res, sig, prog = [], [], []
    for expr, val in steps:
        s, p = [], []
        for t in re.findall(r"\d+\.?\d*|[-+*/()]", expr):
            if re.fullmatch(r"\d+\.?\d*", t):
                v = float(t)
                if v in res:
                    k = res.index(v); s.append("r%d" % k); p.append(("r", k))
                else:
                    s.append("q"); p.append(("q", v))
            else:
                s.append(t); p.append(("op", t))
        sig.append("".join(s)); prog.append(p)
        try:
            res.append(float(val))
        except ValueError:
            res.append(None)
    return tuple(sig), prog, res


def execute(prog, override=None):
    vals = []
    for k, p in enumerate(prog):
        if override and k in override:
            vals.append(float(override[k])); continue
        try:
            expr = "".join(repr(vals[x[1]]) if x[0] == "r" else (repr(x[1]) if x[0] == "q" else x[1]) for x in p)
            vals.append(float(eval(expr, {"__builtins__": {}}, {})))      # tokens are digits / + - * / ( ) only (regex above)
        except Exception:
            return None
    return vals


def close(a, b):
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))


def graph_info(it):
    sig, prog, res = program(it["cot"]); vals = execute(prog) if prog else None
    ok = bool(prog) and vals is not None and all(r is None or close(v, r) for v, r in zip(vals, res)) and close(vals[-1], it["gold"])
    return {"sig": sig, "prog": prog, "vals": vals, "exec": ok}


def oracle_cf(gi, gj, k):
    """the recipient's program with its step-k result replaced by the donor's step-k result, later steps re-executed; rounded to 6 decimals."""
    v = execute(gi["prog"], {k: gj["vals"][k]})
    if v is None or not np.isfinite(v[-1]):
        return None
    return round(v[-1], 6)


def fmt_num(v):
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else "%.10g" % v


def draw_exact(correct, gold, L, seed=SEED_PAIRS, cap=MAX_REUSE):
    rng = np.random.default_rng(seed); uses = Counter(); pairs, nod = [], []
    for i in correct:
        pool = [j for j in correct if j != i and gold[j] != gold[i] and L[j] == L[i] and uses[j] < cap]
        if not pool:
            nod.append(i); continue
        j = pool[int(rng.integers(0, len(pool)))]; uses[j] += 1; pairs.append((i, j))
    return pairs, nod, uses


def draw_graph(correct, gold, L, GI, seed=SEED_GM, cap=MAX_REUSE):
    rng = np.random.default_rng(seed); uses = Counter(); d0, dne0 = [], []; cnt = Counter()
    for i in correct:
        gi = GI[i]
        if not gi["exec"] or len(gi["sig"]) < 2:
            cnt["no_exec_or_1step"] += 1; continue
        cands = []
        for j in correct:
            if j == i or gold[j] == gold[i] or uses[j] >= cap or not GI[j]["exec"] or GI[j]["sig"] != gi["sig"]:
                continue
            o = oracle_cf(gi, GI[j], 0)
            if o is None or o == gold[i] or o == gold[j]:
                continue
            cands.append((j, o))
        c0 = [c for c in cands if L[c[0]] == L[i]]
        pick = c0 or cands
        if not pick:
            cnt["no_candidate"] += 1; continue
        j, o = pick[int(rng.integers(0, len(pick)))]; uses[j] += 1
        rec = {"r": i, "d": j, "delta": L[i] - L[j], "sig": "|".join(gi["sig"]), "ocf": o, "ocf_klast": oracle_cf(gi, GI[j], len(gi["sig"]) - 2)}
        (d0 if c0 else dne0).append(rec)
    return d0, dne0, dict(cnt), uses


# ------------------------------------------------------------------ models
def load_model(name, dev):
    import boxes.codi_port as CP
    tok = CP.codi_tok(); info = {"model": name}
    if name == "tiny":
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(0); cfg = GPT2Config(n_layer=4, n_embd=64, n_head=2, vocab_size=50260, n_positions=1024); cfg._attn_implementation = "eager"
        m = GPT2LMHeadModel(cfg); prj = nn.Sequential(nn.Dropout(0.0), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64)); prj.add_module("ln", nn.LayerNorm(64))
        info.update({"model_sha256": "random-init seed 0", "path": None})
        return m.to(dev).eval(), prj.to(dev).eval(), tok, info
    m, prj = CP.build(dev)
    if name == "codi20":
        p = os.path.join(RUN20, "trainable.pt"); TP = dict(CP.trainable(m, prj)); sd = torch.load(p, map_location=dev, weights_only=False)
        assert set(sd) == set(TP), "trainable.pt keys != codi_port.trainable"
        for n, p_ in TP.items():
            p_.data.copy_(sd[n])
        info.update({"path": os.path.relpath(p, ROOT).replace(os.sep, "/"), "model_sha256": sha256_file(p), "params_loaded": len(TP)})
    elif name == "zen":
        import boxes.codi_g1 as G1
        sha = sha256_file(G1.REL); assert sha == G1.SHA, ("zen-E checkpoint sha256 mismatch", sha)
        sd = torch.load(G1.REL, map_location="cpu", weights_only=False); pre = "codi.base_model.model."; mine = dict(m.named_parameters()); n_set = 0
        for k, v in sd.items():                                                        # = boxes/codi_g1.py main()'s loader
            if k.startswith("prj."):
                continue
            k2 = k[len(pre):].replace(".base_layer.", ".base.").replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
            if k2 in mine:
                with torch.no_grad():
                    mine[k2].copy_(v.to(mine[k2].dtype)); n_set += 1
        prj.load_state_dict({k[4:]: v.float() for k, v in sd.items() if k.startswith("prj.")})
        info.update({"path": os.path.relpath(G1.REL, ROOT).replace(os.sep, "/"), "model_sha256": sha, "params_loaded": n_set, "hf": "zen-E/CODI-gpt2 (sha pinned in boxes/codi_g1.py)"})
    else:
        raise ValueError(name)
    m.eval(); prj.eval(); mm, pj = CP.merged_fp32(m, prj, dev); del m
    return mm, pj, tok, info


def clone_cache(cache, cfg):
    from transformers import DynamicCache
    c = DynamicCache(config=cfg)
    for l_, L in enumerate(cache.layers):
        c.update(L.keys.clone(), L.values.clone(), l_)
    return c


def gen_sha(g):
    return hashlib.sha256(",".join(map(str, g)).encode()).hexdigest()


class Runner:
    def __init__(self, m, prj, tok, dev, max_new):
        self.m, self.prj, self.tok, self.dev, self.max_new = m, prj, tok, dev, max_new
        self.V = m.config.vocab_size; self.BOT, self.EOT = self.V - 2, self.V - 1; self.EOS = tok.eos_token_id; self.PAD = tok.pad_token_id
        self.V0 = self.V - 3; self.wte = m.transformer.wte; self.nl = m.config.n_layer; self.cfg = m.config
        assert self.PAD == self.V0 and self.EOS == self.V0 - 1, (self.PAD, self.EOS, self.V)
        self.n_checks = 0; self.fails = []

    def t(self, x):
        return torch.tensor(x, device=self.dev)

    def ids(self, q):
        return self.tok(q)["input_ids"]

    @torch.no_grad()
    def prefix(self, q_ids, K=K6, override=None, npad=None):
        """q <bot> + K latents. override {j: (D,) tensor} replaces latent input j (1-based). npad not None -> official semantics."""
        seq = list(q_ids) + [self.BOT]
        if npad is None:
            P = len(seq); o = self.m(input_ids=self.t([seq]), position_ids=torch.arange(P, device=self.dev)[None], use_cache=True, output_hidden_states=True)
        else:
            seq = [self.PAD] * npad + seq; P = len(seq); am = self.t([[0] * npad + [1] * (P - npad)])
            o = self.m(input_ids=self.t([seq]), attention_mask=am, use_cache=True, output_hidden_states=True)
        cache = o.past_key_values; z = self.prj(o.hidden_states[-1][:, -1:, :]); zs, hs = [], []
        for j in range(1, K + 1):
            if override is not None and j in override:
                z = override[j].to(self.dev, z.dtype).view(1, 1, -1)
            zs.append(z[0, 0])
            kw = {} if npad is not None else {"position_ids": self.t([[P + j - 1]])}
            o = self.m(inputs_embeds=z, past_key_values=cache, use_cache=True, output_hidden_states=True, **kw)
            cache = o.past_key_values; h = o.hidden_states[-1][:, -1:, :]; hs.append(h[0, 0]); z = self.prj(h)
        assert cache.get_seq_length() == P + K
        return {"cache": cache, "P": P, "K": K, "next": P + K, "Lq": len(q_ids), "z": zs, "h": hs, "official": npad is not None}

    def copy(self, pre):
        return {**pre, "cache": clone_cache(pre["cache"], self.cfg)}

    @torch.no_grad()
    def decode(self, pre, qmask=False):
        cache = pre["cache"]; pos = pre["next"]; x = self.wte(self.t([[self.EOT]])); gen = []
        for _ in range(self.max_new):
            kw = {}
            if not pre["official"]:
                kw["position_ids"] = self.t([[pos]])
            if qmask:
                am = torch.ones(1, pos + 1, dtype=torch.long, device=self.dev); am[0, :pre["Lq"]] = 0; kw["attention_mask"] = am
            o = self.m(inputs_embeds=x, past_key_values=cache, use_cache=True, **kw); cache = o.past_key_values
            nxt = int(o.logits[0, -1, :self.V - 1].argmax()); gen.append(nxt)
            if nxt == self.EOS:
                break
            x = self.wte(self.t([[nxt]])); pos += 1
        txt = self.tok.decode([g for g in gen if g < self.V0], skip_special_tokens=True)
        return {"pred": parse_official(txt), "sha": gen_sha(gen), "ntok": len(gen), "txt": txt[:200]}

    @torch.no_grad()
    def write_rows(self, dst, src, dst_slots, src_slots, layers, kinds, tag, _test_corrupt=None):
        """write src's rows (src_slots) into dst's cache at dst_slots for the given layers / kinds ('k', 'v'); assert every other row is unchanged and
        the written rows equal the source rows (bitwise)."""
        c, s = dst["cache"], src["cache"]; before = [(L.keys.clone(), L.values.clone()) for L in c.layers]
        di, si = self.t(list(dst_slots)), self.t(list(src_slots))
        for l_ in layers:
            if "k" in kinds:
                c.layers[l_].keys[:, :, di] = s.layers[l_].keys[:, :, si]
            if "v" in kinds:
                c.layers[l_].values[:, :, di] = s.layers[l_].values[:, :, si]
        if _test_corrupt is not None:                                                  # gate_s220 only: proves the assertion fires
            _test_corrupt(c)
        ok = len(c.layers) == len(before)
        for l_, L in enumerate(c.layers):
            for kind, cur, old, srcT in (("k", L.keys, before[l_][0], s.layers[l_].keys), ("v", L.values, before[l_][1], s.layers[l_].values)):
                T = cur.shape[2]; w = list(dst_slots) if (l_ in layers and kind in kinds) else []
                keep = self.t([t for t in range(T) if t not in set(w)])
                ok &= cur.shape == old.shape and torch.equal(cur[:, :, keep], old[:, :, keep])
                if w:
                    ok &= torch.equal(cur[:, :, di], srcT[:, :, si])
        self.n_checks += 1
        if not ok:
            self.fails.append(tag); print("CACHE-ROW ASSERT FAILED", tag, flush=True)
        return ok

    def latent_slots(self, pre):
        return list(range(pre["P"], pre["P"] + pre["K"]))

    def leak_slots(self, pre):
        assert pre["Lq"] >= NLEAK
        return list(range(pre["Lq"] - NLEAK, pre["Lq"]))

    @torch.no_grad()
    def scit(self, pre, cands):
        """teacher-forced log-prob of [<eot>] + "The answer is: N" + [EOS] per candidate, on a clone of pre's cache."""
        out = {}; npre = len(self.tok.encode(ANS_PREFIX.strip()))
        for name, nstr in cands.items():
            if nstr is None:
                continue
            toks = self.tok.encode(ANS_PREFIX + nstr) + [self.EOS]; c = self.copy(pre); L = len(toks)
            x = self.wte(self.t([[self.EOT] + toks[:-1]])); pos = torch.arange(pre["next"], pre["next"] + L, device=self.dev)[None]
            o = self.m(inputs_embeds=x, position_ids=pos, past_key_values=c["cache"], use_cache=True)
            lp = torch.log_softmax(o.logits[0, :, :self.V - 1].float(), -1).gather(1, self.t(toks)[:, None])[:, 0]
            out[name] = {"n": nstr, "sum": float(lp.sum()), "ntok": L, "tail_sum": float(lp[npre:].sum()), "tail_ntok": L - npre}
        return out


# ------------------------------------------------------------------ the battery
def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return "error: %s" % e


def acc_of(rows, items):
    return float(np.mean([r["pred"] is not None and r["pred"] == it["gold"] for r, it in zip(rows, items)])) if rows else None


def cols(rows):
    return {"pred": [r["pred"] for r in rows], "sha": [r["sha"] for r in rows], "ntok": [r["ntok"] for r in rows]}


def dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="\n") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def gate_m(R, items, out_dir, dev):
    """G-M: memory at the largest shape first (longest question, K 10, 256 forced decode steps) on SYNTHETIC token ids of that length."""
    Lmax = max(len(R.ids(it["q"])) for it in items); synth = [R.ids(" the")[0]] * Lmax
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time(); pre = R.prefix(synth, K=10); eos, save = R.EOS, R.max_new
    R.EOS = -1; R.max_new = 256; r = R.decode(pre); R.EOS, R.max_new = eos, save                      # EOS disabled: all 256 steps forced
    pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else None
    res = {"gate": "G-M", "longest_question_tokens": Lmax, "K": 10, "decode_steps": r["ntok"], "peak_gib": pk, "sec": round(time.time() - t0, 1),
           "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "input": "synthetic token ids (' the' x Lmax), no test item"}
    dump(os.path.join(out_dir, "gate_m.json"), res); print("G-M", res, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["codi20", "zen", "tiny"]); ap.add_argument("--device", required=True)
    ap.add_argument("--out_dir", default=None); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=256); ap.add_argument("--smoke_all_correct", action="store_true")
    ap.add_argument("--gate_m", action="store_true"); ap.add_argument("--load_only", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("S220_THREADS", "2")))
    ap.add_argument("--official_shuffle", type=int, default=None, help="Note 220-3: ONLY the official-semantics arm, item order shuffled with this rng seed before forming the 128-blocks -> scratch/s220/[zen_codi_gpt2/]official_shuffle_<seed>.json; no registered arm runs")
    a = ap.parse_args(); dev = a.device; torch.set_num_threads(a.threads)
    smoke = bool(a.limit or a.synthetic or a.model == "tiny" or a.max_new != 256 or a.smoke_all_correct)
    out_dir = os.path.abspath(a.out_dir or OUT_DEFAULT.get(a.model, ""))
    tmp_root = os.path.abspath(os.path.join(ROOT, "scratch", "_tmp"))
    if smoke or a.gate_m or a.load_only:
        assert a.out_dir and out_dir.startswith(tmp_root + os.sep), "smokes / gates write under scratch/_tmp only"
    if a.smoke_all_correct:
        assert a.model == "tiny" or a.synthetic, "--smoke_all_correct is for tiny / synthetic smokes only"
    os.makedirs(out_dir, exist_ok=True); fo = os.path.join(out_dir, "eval_gcs.json"); fp = os.path.join(out_dir, "eval_gcs.partial.json")
    assert a.official_shuffle is not None or not os.path.exists(fo), "refusing to overwrite %s" % fo   # --official_shuffle writes only its own file (own guard)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.use_deterministic_algorithms(True, warn_only=True)
    t_start = time.time()
    test_sha = sha256_file(TEST)
    if a.synthetic:
        items = synthetic(a.synthetic)
    else:
        assert test_sha == TEST_SHA, ("test file sha256 mismatch", test_sha)
        items = load_test()
    items = items[:a.limit] if a.limit else items; N = len(items)
    m, prj, tok, info = load_model(a.model, dev)
    import transformers
    meta = {**info, "stage": 0, "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "device": dev, "dtype": "fp32 (LoRA merged)" if a.model != "tiny" else "fp32",
            "bs": 1, "test_sha256": test_sha if not a.synthetic else "synthetic", "test_file": os.path.relpath(TEST, ROOT).replace(os.sep, "/"), "transformers": transformers.__version__,
            "torch": torch.__version__, "attn_implementation": m.config._attn_implementation, "tf32": False, "max_new": a.max_new, "n_items": N,
            "commit": git("rev-parse", "HEAD"), "script_sha256": sha256_file(os.path.abspath(__file__)), "script_git_status": git("status", "--porcelain", "--", "scratch/s220_gcs.py"),
            "codi_port_sha256": sha256_file(os.path.join(ROOT, "boxes", "codi_port.py")), "codi_port_git": git("log", "-1", "--format=%H", "--", "boxes/codi_port.py"),
            "registered": "docs/research_scaleprobe_2026-10-20/GCS_v1.md at 3a88762e", "smoke": smoke, "limit": a.limit or None, "synthetic": a.synthetic or None,
            "smoke_all_correct": a.smoke_all_correct, "rng": {"pairs": SEED_PAIRS, "graph": SEED_GM, "noise": SEED_NOISE, "scit_random": SEED_RAND}}
    if a.load_only:
        dump(os.path.join(out_dir, "load_only.json"), meta); print("LOAD_ONLY", json.dumps(meta), flush=True); return
    R = Runner(m, prj, tok, dev, a.max_new)
    if a.gate_m:
        gate_m(R, items, out_dir, dev); return
    ids = [R.ids(it["q"]) for it in items]; L = [len(x) for x in ids]; gold = [it["gold"] for it in items]
    if a.official_shuffle is not None:                                      # Note 220-3 (+ Audit addendum): official-vs-official self-consistency
        fo2 = os.path.join(out_dir if smoke else os.path.join(ROOT, "scratch", "s220", *(() if a.model == "codi20" else ("zen_codi_gpt2",))),
                           "official_shuffle_%d.json" % a.official_shuffle)
        assert not os.path.exists(fo2), "refusing to overwrite %s" % fo2
        order = np.random.default_rng(a.official_shuffle).permutation(N); preds = {}; npads = {}
        for b in range(0, N, 128):
            blk = [int(k) for k in order[b:b + 128]]; mx = max(L[k] for k in blk) + 1
            for k in blk:
                r = R.decode(R.prefix(ids[k], npad=mx - (L[k] + 1))); preds[str(items[k]["i"])] = r["pred"]; npads[str(items[k]["i"])] = mx - (L[k] + 1)
        os.makedirs(os.path.dirname(fo2), exist_ok=True)
        dump(fo2, {"meta": {**meta, "official_shuffle_seed": a.official_shuffle, "note": "Note 220-3: official-semantics arm only, order shuffled before the 128-blocks"},
                   "pred": preds, "npad": npads, "acc": float(np.mean([preds[str(it["i"])] is not None and preds[str(it["i"])] == it["gold"] for it in items]))})   # = acc_of
        print("OFFICIAL_SHUFFLE", a.official_shuffle, fo2, flush=True); return
    res = {"meta": meta, "golds": {str(it["i"]): it["gold"] for it in items}}

    def log(*x):
        print("[%6.0fs]" % (time.time() - t_start), *x, flush=True)

    # 1. INTACT (K 6): captures z inputs and latent-row hiddens for MEAN-Z / NOISE / STATE-SWAP / LOGIT-LENS
    intact, Z, H = [], [], []
    for k, x in enumerate(ids):
        pre = R.prefix(x); Z.append(torch.stack(pre["z"])); H.append(torch.stack(pre["h"])); intact.append(R.decode(pre))
    Z = torch.stack(Z); H = torch.stack(H); acc = {"intact": acc_of(intact, items)}; log("INTACT acc", acc["intact"])
    res["intact_sha"] = {str(it["i"]): r["sha"] for it, r in zip(items, intact)}
    per_item = {"intact": {**cols(intact), "txt": [r["txt"] for r in intact]}}
    # 2. OFFICIAL semantics (virtual bs 128)
    official = []
    for b in range(0, N, 128):
        mx = max(L[b:b + 128]) + 1
        for k in range(b, min(N, b + 128)):
            official.append(R.decode(R.prefix(ids[k], npad=mx - (L[k] + 1))))
    acc["official"] = acc_of(official, items); per_item["official"] = {**cols(official), "txt": [r["txt"] for r in official],
                                                                     "npad": [max(L[(k // 128) * 128:(k // 128) * 128 + 128]) - L[k] for k in range(N)]}
    agree = float(np.mean([x["pred"] == y["pred"] for x, y in zip(intact, official)]))
    res["gate"] = {"agree": agree, "acc_bs1": acc["intact"], "acc_official": acc["official"], "n": N,
                   "agree_def": "share of items with equal parsed predictions (None == None agrees), bs-1 INTACT vs official-semantics virtual bs 128"}
    if a.model == "codi20":
        res["gate"]["ref_210_4"] = 0.379
    log("OFFICIAL acc", acc["official"], "agree", agree)
    res.update({"acc": acc, "per_item": per_item}); dump(fp, res)
    # 3. K-SWEEP
    ks = {}
    for K in KS:
        rows = intact if K == K6 else [R.decode(R.prefix(x, K=K)) for x in ids]
        ks[str(K)] = {"acc": acc_of(rows, items), **cols(rows)}; acc["k%d" % K] = ks[str(K)]["acc"]
    res["k_sweep"] = ks; log("K-SWEEP", {k: v["acc"] for k, v in ks.items()})
    # 4. MEAN-Z, NOISE, QDROP, QMASK, LOGIT-LENS
    mz = Z.mean(0); rows = [R.decode(R.prefix(x, override={j: mz[j - 1] for j in range(1, K6 + 1)})) for x in ids]
    acc["mean_z"] = acc_of(rows, items); per_item["mean_z"] = cols(rows)
    g = torch.Generator().manual_seed(SEED_NOISE); nz = torch.randn(tuple(Z.shape), generator=g, dtype=torch.float32).to(Z.device)
    nz = nz / nz.norm(dim=-1, keepdim=True) * Z.norm(dim=-1, keepdim=True)
    rows = [R.decode(R.prefix(x, override={j: nz[k, j - 1] for j in range(1, K6 + 1)})) for k, x in enumerate(ids)]
    acc["noise"] = acc_of(rows, items); per_item["noise"] = cols(rows)
    rows = [R.decode(R.prefix([], override={j: Z[k, j - 1] for j in range(1, K6 + 1)})) for k in range(N)]
    r0 = R.decode(R.prefix([], K=0)); acc["qdrop"] = acc_of(rows, items); acc["qdrop_k0"] = acc_of([r0] * N, items)
    res["qdrop"] = {"acc": acc["qdrop"], **cols(rows), "k0": {"acc": acc["qdrop_k0"], "pred": r0["pred"], "sha": r0["sha"], "txt": r0["txt"]},
                    "def": "fresh '<bot> z_1..z_6 <eot>' with the item's own INTACT z, positions 0..; k0 = '<bot> <eot>' (one constant output)"}
    rows = [R.decode(R.prefix(x), qmask=True) for x in ids]; rows0 = [R.decode(R.prefix(x, K=0), qmask=True) for x in ids]
    acc["qmask"] = acc_of(rows, items); acc["qmask_k0"] = acc_of(rows0, items)
    res["qmask"] = {"acc": acc["qmask"], **cols(rows), "k0": {"acc": acc["qmask_k0"], **cols(rows0)},
                    "def": "answer phase (<eot> + generated queries) cannot attend to question tokens 0..Lq-1 (all layers; additive key mask, positions unchanged)"}
    log("MEAN-Z", acc["mean_z"], "NOISE", acc["noise"], "QDROP", acc["qdrop"], "QDROP-K0", acc["qdrop_k0"], "QMASK", acc["qmask"], "QMASK-K0", acc["qmask_k0"])
    num_ids, num_val = [], []
    for t in range(R.V0):
        s = tok.decode([t]).strip()
        if re.fullmatch(r"\d+(\.\d+)?", s):
            num_ids.append(t); num_val.append(float(s))
    W = m.lm_head.weight; nid = torch.tensor(num_ids, device=W.device); LLk = {"z_num": [], "h_num": [], "z_top": [], "h_top": []}
    with torch.no_grad():
        for b in range(0, N, 32):                                                      # chunks: (32, 6, V) logits at a time
            for nm, X in (("z", Z[b:b + 32]), ("h", H[b:b + 32])):
                lg = X @ W.T
                LLk[nm + "_num"] += [[num_val[int(i)] for i in row] for row in lg[:, :, nid].argmax(-1).tolist()]; LLk[nm + "_top"] += lg.argmax(-1).tolist()
            del lg
    res["logit_lens"] = {"def": "U4: z_j via the tied LM head (no ln_f); h_j = final-layer (ln_f) hidden at latent position j; top number token among %d numeric ids" % len(num_ids), **LLk}
    dump(fp, res)
    # 5. pairs
    correct = [k for k in range(N) if a.smoke_all_correct or (intact[k]["pred"] is not None and intact[k]["pred"] == gold[k])]
    pairs, nod, uses = draw_exact(correct, gold, L)
    GI = [graph_info(it) for it in items]; gm0, gmne, gcnt, guses = draw_graph(correct, gold, L, GI)
    res["pairs_meta"] = {"pool_size": len(correct), "n_pairs": len(pairs), "n_no_donor": len(nod), "no_donor": nod, "max_reuse": max(uses.values()) if uses else 0,
                         "reuse_hist": dict(Counter(uses.values())), "rule": "np rng 22020; recipients ascending; donor uniform among INTACT-correct, gold differs, same question token length, uses < 3"}
    log("PAIRS", res["pairs_meta"]["n_pairs"], "pool", len(correct), "no donor", len(nod), "graph d0", len(gm0), "dne0", len(gmne))
    allsl = list(range(R.nl)); lo, hi = allsl[:R.nl // 2], allsl[R.nl // 2:]
    VAR = {"k_only": ("lat", allsl, "k"), "v_only": ("lat", allsl, "v"), "last_row": ("last", allsl, "kv"), "layers_lo": ("lat", lo, "kv"), "layers_hi": ("lat", hi, "kv")}
    gate_c = []; prs = []; var = {k: {"pred": [], "sha": []} for k in VAR}; pfx = {str(k): {"pred": [], "sha": []} for k in (1, 3, 5)}
    scit = {"cands": [], "cache": [], "state": []}; rng_rand = np.random.default_rng(SEED_RAND); donor_own = {}

    def rand_cand(rng, excl):
        pool = [k for k in range(N) if all(gold[k] != e for e in excl if e is not None)]
        return items[pool[int(rng.integers(0, len(pool)))]]["gold_s"] if pool else None

    def cache_arm(base, src, dst_sl, src_sl, layers, kinds, tag, cands=None):
        c = R.copy(base); R.write_rows(c, src, dst_sl, src_sl, layers, kinds, tag)
        sc = R.scit(c, cands) if cands else None
        return R.decode(c), sc

    for q, (r, d) in enumerate(pairs):
        pr = R.prefix(ids[r]); pdn = R.prefix(ids[d]); ps = R.prefix(ids[r])            # recipient, donor, a fresh recipient run (SELF source)
        cands = {"recipient": items[r]["gold_s"], "donor": items[d]["gold_s"], "random": rand_cand(rng_rand, (gold[r], gold[d]))}
        o_c, sc_c = cache_arm(pr, pdn, R.latent_slots(pr), R.latent_slots(pdn), allsl, "kv", ("cache", r, d), cands)
        o_l, _ = cache_arm(pr, pdn, R.leak_slots(pr), R.leak_slots(pdn), allsl, "kv", ("leak", r, d))
        o_sc, _ = cache_arm(pr, ps, R.latent_slots(pr), R.latent_slots(ps), allsl, "kv", ("self_cache", r, r))
        st = R.prefix(ids[r], override={j: Z[d, j - 1] for j in range(1, K6 + 1)}); sc_s = R.scit(st, cands); o_s = R.decode(st)
        o_ss = R.decode(R.prefix(ids[r], override={j: Z[r, j - 1] for j in range(1, K6 + 1)}))
        for nm, (sl, layers, kinds) in VAR.items():
            dsl = R.latent_slots(pr)[-1:] if sl == "last" else R.latent_slots(pr); ssl = R.latent_slots(pdn)[-1:] if sl == "last" else R.latent_slots(pdn)
            o_v, _ = cache_arm(pr, pdn, dsl, ssl, layers, kinds, (nm, r, d)); var[nm]["pred"].append(o_v["pred"]); var[nm]["sha"].append(o_v["sha"])
        for kp in (1, 3, 5):
            o_p = R.decode(R.prefix(ids[r], override={j: Z[d, j - 1] for j in range(1, kp + 1)})); pfx[str(kp)]["pred"].append(o_p["pred"]); pfx[str(kp)]["sha"].append(o_p["sha"])
        if d not in donor_own:                                                         # G-I donor-own: the donor with its own rows / own z, bitwise
            b_ = R.prefix(ids[d]); s_ = R.prefix(ids[d]); oc_, _ = cache_arm(b_, s_, R.latent_slots(b_), R.latent_slots(s_), allsl, "kv", ("donor_own", d, d))
            os_ = R.decode(R.prefix(ids[d], override={j: Z[d, j - 1] for j in range(1, K6 + 1)}))
            donor_own[d] = {"cache": oc_["sha"], "state": os_["sha"], "ok": oc_["sha"] == intact[d]["sha"] and os_["sha"] == intact[d]["sha"]}
        prs.append({"r": items[r]["i"], "d": items[d]["i"], "Lq": L[r], "out": {"cache": o_c["pred"], "state": o_s["pred"], "leak": o_l["pred"]},
                    "sha": {"self_cache": o_sc["sha"], "self_state": o_ss["sha"], "cache": o_c["sha"], "state": o_s["sha"], "leak": o_l["sha"]},
                    "txt": {"cache": o_c["txt"], "state": o_s["txt"], "leak": o_l["txt"]}, "ntok": {"cache": o_c["ntok"], "state": o_s["ntok"], "leak": o_l["ntok"]}})
        scit["cands"].append(cands); scit["cache"].append(sc_c); scit["state"].append(sc_s)
        if q < 20:
            gate_c.append({"r": items[r]["i"], "d": items[d]["i"], "Lq_r": pr["Lq"], "Lq_d": pdn["Lq"], "P_r": pr["P"], "T_before_eot": pr["next"],
                           "latent_slots": R.latent_slots(pr), "leak_slots": R.leak_slots(pr), "n_layers": R.nl,
                           "rows_ok_so_far": not R.fails})
        if q % 50 == 49:
            log("pairs", q + 1, "/", len(pairs)); res["pairs"] = prs; dump(fp, res)
    res["pairs"] = prs
    res["variants"] = {"def": "CACHE-SWAP variants on the primary pairs (same order): k_only / v_only = keys / values only (6 latent rows, all layers); last_row = the "
                              "6th latent row only (k+v, all layers); layers_lo / layers_hi = layers 0..%d / %d..%d (6 rows, k+v)" % (R.nl // 2 - 1, R.nl // 2, R.nl - 1), **var}
    res["prefix"] = {"def": "STATE-SWAP of z_1..z_k only (later z computed by the recipient), primary pairs, same order", **pfx}
    res["scit"] = {"def": "U5; candidates per primary pair (same order); summed log-prob of '<eot>'-conditioned 'The answer is: N'+EOS under CACHE-SWAP and STATE-SWAP", **scit}
    res["donor_own"] = {str(items[k]["i"]): v for k, v in donor_own.items()}
    log("PRIMARY done; donor-own ok", all(v["ok"] for v in donor_own.values()))
    dump(fp, res)

    # 6. graph-matched (Δ = 0) and its Δ != 0 row
    def gm_rows(recs, rng):
        out = []
        for rec in recs:
            r, d = rec["r"], rec["d"]; pr = R.prefix(ids[r]); pdn = R.prefix(ids[d])
            cands = {"recipient": items[r]["gold_s"], "donor": items[d]["gold_s"], "oracle_cf": fmt_num(rec["ocf"]), "random": rand_cand(rng, (gold[r], gold[d], rec["ocf"]))}
            o_c, sc_c = cache_arm(pr, pdn, R.latent_slots(pr), R.latent_slots(pdn), allsl, "kv", ("gm_cache", r, d), cands)
            st = R.prefix(ids[r], override={j: Z[d, j - 1] for j in range(1, K6 + 1)}); sc_s = R.scit(st, cands); o_s = R.decode(st)
            out.append({**rec, "r": items[r]["i"], "d": items[d]["i"], "out": {"cache": o_c["pred"], "state": o_s["pred"]}, "sha": {"cache": o_c["sha"], "state": o_s["sha"]},
                        "scit": {"cands": cands, "cache": sc_c, "state": sc_s}})
        return out
    res["graph_matched"] = {"def": "Δ = 0 graph-matched pairs (np rng 22021; U6); int-follow = out == ocf", "counts": gcnt, "max_reuse": max(guses.values()) if guses else 0,
                            "pairs": gm_rows(gm0, np.random.default_rng(SEED_RAND))}
    res["graph_matched_dne0"] = {"def": "graph-matched pairs with Δ != 0 (separate descriptive row, never pooled; CACHE-SWAP is reverse-distance and position-UNCORRECTED)",
                                 "pairs": gm_rows(gmne, np.random.default_rng(SEED_RAND))}
    log("GRAPH-MATCHED", len(gm0), len(gmne)); dump(fp, res)
    # 7. 210-4's original 500 pairs (CODI20 only; STATE-SWAP only)
    if a.model == "codi20" and not a.synthetic:
        src = json.load(open(os.path.join(RUN20, "eval_latent_use.json")))["items"]; idx = {it["i"]: k for k, it in enumerate(items)}; p4 = []
        for it in src:
            if it["recipient"] in idx and it["donor"] in idx:
                r, d = idx[it["recipient"]], idx[it["donor"]]; o = R.decode(R.prefix(ids[r], override={j: Z[d, j - 1] for j in range(1, K6 + 1)}))
                p4.append({"r": it["recipient"], "d": it["donor"], "delta": L[r] - L[d], "out": {"state": o["pred"]}, "sha": o["sha"], "pred_210_4": it["pred_swap_all"],
                           "intact_ok_bs1": intact[r]["pred"] == gold[r]})
        res["pairs_210_4"] = {"source": "boxes/runs_fsq/gsm_codi_20ep_s1/eval_latent_use.json items (rng 21002; bs 16, codi_probe parser)", "pairs": p4}
        log("210-4 pairs", len(p4))
    res["asserts"] = {"cache_rows_ok": not R.fails and R.n_checks > 0, "n_checks": R.n_checks, "fails": [list(map(str, f)) for f in R.fails[:50]], "n_fails": len(R.fails),
                      "donor_own_ok": all(v["ok"] for v in donor_own.values()), "self_ok": all(p["sha"]["self_cache"] == res["intact_sha"][str(p["r"])] and
                                                                                         p["sha"]["self_state"] == res["intact_sha"][str(p["r"])] for p in prs),
                      "gate_c": gate_c}
    meta["sec"] = round(time.time() - t_start)
    dump(fo, res); log("DONE", fo, "cache_rows_ok", res["asserts"]["cache_rows_ok"], "self_ok", res["asserts"]["self_ok"], "donor_own_ok", res["asserts"]["donor_own_ok"])
    if os.path.exists(fp):
        os.replace(fp, fp + ".done")                                                   # keep the last partial (never deleted)


if __name__ == "__main__":
    main()
