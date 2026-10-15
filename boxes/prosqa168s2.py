"""THEORY §168 step 2 STEP-AUX-ProsQA (note 168-7, 410d7cbc; scorer theory/score_s168s2.py). p8 NA (prosqa/p8data/p8_na_*.json, depths 3..14), GPT-2 small, 12k steps.
Three arms (--arm):
  stepaux    : §170's DeltaMem-R (6 + 6 heads after block 6, top writes, gamma 2k -> 6k, W_o fade 500, g_add calibrated at step 0), fast_add. K think passes,
               each over [EOT, question] (right-padded, padded writes masked). After pass t <= min(depth, K), §175's read-only aux decoder
               [EOT, " Step", " t", " :"] + steps[t-1] (teacher-forced, + EOT; it sees NO question token) reading M_t. The answer: the main decoder over
               [EOT, question, "\\n", "### " + answer, EOS] reading M_K (CE on the answer + EOS). loss = answer CE + lambda * aux (lambda = aux_step to f0 of
               --aux_fade, then linear to 0 at f1).
  currdirect : 168-4's curriculum DIRECT (Coconut's stages with zero thoughts): stage k drops the first k steps of the CoT text, stage length
               steps / (STAGES + 2) with the last (no-CoT) stage taking the remaining 2 lengths; format question\\n + steps[k:] (each + "\\n") + "### " + answer + EOS.
  textcot    : GPT-2 on question\\n + steps (each + "\\n") + "### " + answer + EOS (Coconut's CoT format); eval = greedy CoT then candidate scoring (eval_p8 cot).
Candidate accuracy (all arms): log p("### " + correct answer) > log p(the negative's) after the arm's prefix (+ memory for stepaux); chance 0.5.
eval.json: acc, acc_by_depth, n_by_depth, acc_ctl (p8_ctl.json); stepaux adds s168s2 {acc_by_depth_kprime, aux_exact_by_hop, delta_cos (+ at init),
aux_done, path_indep, step_recover, normcap (note 185-1: K' curve with S_add capped at C from train items)}. --resume <ckpt> continues a run.
§191 PROSQA-RELABEL (--relabel): every training item drawn is relabelled afresh (entity / concept names, premise order, option order; relabel()); test items never are.
--test500_every N logs test500_acc on a fixed 500-item test subset; with --relabel the eval adds train_fit_relabel (1,000 relabelled train items).
    python -m boxes.prosqa168s2 --arm stepaux --think_k 10 --aux_step 1.0 --aux_fade 12000,12000 --out boxes/runs_fsq/s168s2_stepaux_s1"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import ast  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes.deltas5 import fwd_read, write_grad_norm  # noqa: E402
from boxes.fsq120 import driver_version, env_info, lr_at  # noqa: E402
from boxes.prosqa168 import ENC, EOS, candidate_margins  # noqa: E402

EOT = EOS; ANS = ENC.encode("###")[0]; NL = ENC.encode("\n")
STEP_ID, COLON_ID = ENC.encode(" Step")[0], ENC.encode(" :")[0]
KPRIMES = (0, 1, 2, 4, 6, 8, 10, 12, 14, 16)
HIST_168 = {3: 815, 4: 1628, 5: 1711, 6: 1777, 7: 1716, 8: 1675, 9: 1756, 10: 907}   # p8_na_train len(steps) histogram (note 168-12)
AUX_W = (0.583, 0.583, 0.583, 0.671, 0.869, 1.156, 1.619, 2.42, 4.193, 13.214)      # note 168-12 STEP-AUX-REWEIGHT (bb1b10c6): w_t = E_bar / E_t, E_t = sum_{d>=t} c_d / d
_E = [sum(c / d for d, c in HIST_168.items() if d >= t) for t in range(1, 11)]
assert all(abs(w - sum(_E) / 10 / e) < 1e-3 for w, e in zip(AUX_W, _E)) and abs(sum(e * w for e, w in zip(_E, AUX_W)) - sum(_E)) < 1e-2 * sum(_E) / 10, "168-12: the weight table"


ENT17 = ("Alex", "Bob", "Carol", "Davis", "Eva", "Fae", "Gabriel", "Jack", "Max", "Oliver", "Polly", "Rex", "Sally", "Sam", "Stella", "Tom", "Wren")   # p8's entity names
CON38 = ("bompus", "boompus", "brimpus", "chorpus", "dumpus", "felpus", "fompus", "gerpus", "gorpus", "grimpus", "gwompus", "hilpus", "impus", "jelpus", "jompus",
         "kerpus", "lempus", "lorpus", "numpus", "quimpus", "rempus", "rompus", "rorpus", "scrompus", "shumpus", "sterpus", "storpus", "terpus", "timpus", "tumpus",
         "vumpus", "worpus", "wumpus", "yerpus", "yimpus", "yumpus", "zhorpus", "zumpus")                                   # p8's concept names (train = test = ctl vocabulary)
Q_RE = re.compile(r"(.*\.) Is (\w+) a (\w+) or (\w+)\?")


def pass_tok(t):
    tk = ENC.encode(" %d" % t); assert len(tk) == 1; return tk[0]


def _lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def encode(x):
    """one raw p8 item -> dict: q (question ids), pre (question + "\\n" ids), a1 / a2 ("### " + correct / negative answer ids), depth, steps (texts), step_ids
    (" " + step ids), node (hop t's node id = the object of steps[t-1], an index into idx_to_symbol)."""
    sym = _lit(x["idx_to_symbol"]); steps = _lit(x["steps"]); tgt, neg = sym[int(x["target"])], sym[int(x["neg_target"])]; ans = x["answer"]
    a_neg = re.sub(rf"\b{tgt}\.$", f"{neg}.", ans); assert a_neg != ans
    node = [s.rstrip(".").split()[-1] for s in steps]                     # hop t's node = the CONCEPT NAME (shared vocabulary; Theory 168-7 point 6)
    return {"q": ENC.encode(x["question"]), "pre": ENC.encode(x["question"] + "\n"), "a1": ENC.encode("### " + ans), "a2": ENC.encode("### " + a_neg),
            "depth": len(steps), "steps": steps, "step_ids": [ENC.encode(" " + s) for s in steps], "node": node, "opt_true": tgt, "opt_false": neg}


def load(path):
    return [encode(x) for x in json.load(open(path))]


def relabel(x, rng, worst=False):
    """§191 RELABEL (Theory's draft): a raw TRAINING item (load_train_raw) -> (relabelled raw item, options swapped). Injective random maps of the item's entity
    names into ENT17 and its concept names into CON38, applied (one regex pass) to question, answer and steps (so to every aux target) and idx_to_symbol
    (edges / root / target / neg_target are indices: the graph is unchanged); premises shuffled; the question's two options swapped with p 0.5.
    worst=True (memory check): no shuffle / swap; the longest-token names go to the most frequent symbols."""
    assert x.get("_split") == "train", "§191: only training items are relabelled"
    sym = _lit(x["idx_to_symbol"]); steps = _lit(x["steps"]); ents = [w for w in sym if w[0].isupper()]; cons = [w for w in sym if not w[0].isupper()]
    assert set(ents) <= set(ENT17) and set(cons) <= set(CON38) and len(ents) + len(cons) == len(set(sym)), "§191: a name outside p8's 17 / 38 vocabulary"
    mq = Q_RE.fullmatch(x["question"]); assert mq, "§191: question format"
    prem = [p_.strip() for p_ in re.findall(r"[^.]+\.", mq.group(1))]; assert " ".join(prem) == mq.group(1), "§191: premise split"
    if worst:
        txt = x["question"] + " " + " ".join(steps); cnt = {w: len(re.findall(rf"\b{w}\b", txt)) for w in sym}; tl = lambda w: len(ENC.encode(" " + w))   # noqa: E731
        mp = {}
        for grp, V in ((ents, ENT17), (cons, CON38)):
            mp.update(zip(sorted(grp, key=lambda w: -cnt[w]), sorted(V, key=lambda w: -tl(w))))
        sw = False
    else:
        mp = dict(zip(ents, [ENT17[i] for i in rng.permutation(len(ENT17))[:len(ents)]])); mp.update(zip(cons, [CON38[i] for i in rng.permutation(len(CON38))[:len(cons)]]))
        prem = [prem[i] for i in rng.permutation(len(prem))]; sw = bool(rng.random() < 0.5)
    o1, o2 = (mq.group(4), mq.group(3)) if sw else (mq.group(3), mq.group(4))
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, mp), key=len, reverse=True)) + r")\b"); sub = lambda t_: pat.sub(lambda m_: mp[m_.group(1)], t_)   # noqa: E731
    y = dict(x); y.update(question=sub(" ".join(prem) + " Is %s a %s or %s?" % (mq.group(2), o1, o2)), answer=sub(x["answer"]), steps=[sub(t_) for t_ in steps], idx_to_symbol=[mp[w] for w in sym])
    return y, sw


def load_train_raw(path):
    out = []
    for x in json.load(open(path)):
        x["_split"] = "train"; out.append(x)
    return out


def pad(seqs, fill=EOS):
    T = max(len(q) for q in seqs); ids = np.full((len(seqs), T), fill); m = np.zeros((len(seqs), T), bool)
    for r, q in enumerate(seqs):
        ids[r, :len(q)] = q; m[r, :len(q)] = True
    return ids, m


class ProsMem(torch.nn.Module):
    """GPT-2 + §170's DeltaMem-R; think = [EOT, question]; decoder = [EOT, question, "\\n"] + answer, read-only; aux = §175's step decoder."""
    ans_bn = False                                                          # §208 AB (set_ab_qo): the answer decoder reads [EOT, '###'] + memory only
    qo = False                                                              # §208 QO: think passes >= 2 read the fixed filler x_f, not the question
    qo_ids = None

    def __init__(self, g):
        super().__init__(); self.g = g; self.mem = DM.DeltaMem(g, 6, "top")

    def set_ab_qo(self, ans_bn=False, qo=False, filler_ids=None):
        """§208 (GSM8K only; FactorMem inherits it): AB = no question tokens in the answer decoder; QO = the question at pass 1 only."""
        assert not qo or ans_bn, "--question_once needs --ans_bottleneck"
        self.ans_bn, self.qo = bool(ans_bn), bool(qo); self.qo_ids = [int(t_) for t_ in filler_ids] if qo else None; return self

    def pass_inputs(self, x, mask, h_low=None):
        """§208 the one think-input helper -> f(t) = (x_t, mask_t, h_low_t). Without QO: the inputs unchanged (the same objects) at every t. QO: t = 1 the
        question; t >= 2 the filler x_f broadcast to the batch rows, mask all true, its lower blocks computed once per call (when h_low is used)."""
        if not self.qo:
            return lambda t: (x, mask, h_low)
        xf = torch.tensor([self.qo_ids] * x.shape[0], device=x.device); mf = torch.ones(xf.shape, dtype=mask.dtype, device=x.device); hf = None
        if h_low is not None:
            self._hl_src_f = xf; hf = DM.gpt_lower(self.g, self.g.embed(xf), self.mem.layer)
        return lambda t: (x, mask, h_low) if t == 1 else (xf, mf, hf)

    def think_inputs(self, its, dev):
        ids_, m_ = pad([[EOT] + it["q"] for it in its]); return torch.tensor(ids_, device=dev), torch.tensor(m_, device=dev)

    def write(self, S, x, mask, h_low=None):
        assert h_low is None or x is self._hl_src or (self.qo and x is getattr(self, "_hl_src_f", None)), \
            "--cache_lower: every pass must see the identical input the lower blocks were cached on (§208 QO: per source, question / filler)"
        return DM.gpt_forward_mem(self.g, self.g.embed(x), self.mem, S, mask=mask, h_low=h_low)[1]

    def lower(self, x):
        """--cache_lower: blocks 0..5 on [EOT, question] once (every pass has the identical input here: no pass index on this path)."""
        self._hl_src = x; return DM.gpt_lower(self.g, self.g.embed(x), self.mem.layer)

    def think(self, S, x, mask, K, cache=False):
        hl = self.lower(x) if (cache and K) else None
        if self.qo and K:                                                   # §208 QO: pass t's input from pass_inputs (question at 1, filler after)
            pin = self.pass_inputs(x, mask, hl)
            for t in range(1, K + 1):
                S = self.write(S, *pin(t))
            return S
        for _ in range(K):
            S = self.write(S, x, mask, hl)
        return S

    def dec_loss(self, S, its, dev):
        seqs = [[EOT] + it["a1"] + [EOS] for it in its] if self.ans_bn else [[EOT] + it["pre"] + it["a1"] + [EOS] for it in its]   # §208 AB: no question
        assert not self.ans_bn or all(q[1] == ANS and len(q) == len(it["a1"]) + 2 for q, it in zip(seqs, its)), "§208 AB: decoder rows are [EOT, ANS, answer, EOS]"
        ids_, _ = pad(seqs); x = torch.tensor(ids_, device=dev)
        lab = np.full(ids_.shape, -100)
        for r, (it, q) in enumerate(zip(its, seqs)):
            n0 = 1 if self.ans_bn else 1 + len(it["pre"]); lab[r, n0:len(q)] = q[n0:]
        h = fwd_read(self.g, self.g.embed(x), self.mem, S); lg = self.g.logits(h[:, :-1]).float()
        return Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), torch.tensor(lab, device=dev)[:, 1:].reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def margins(self, S, its, dev):
        """log p(a1) - log p(a2) after [EOT, question, "\\n"], reading S (each item's memory used for both answers)."""
        assert not self.ans_bn, "§208 --ans_bottleneck is GSM8K-only"
        seqs, spans = [], []
        for it in its:
            for a_ in (it["a1"], it["a2"]):
                q = [EOT] + it["pre"] + a_; seqs.append(q); spans.append((1 + len(it["pre"]), len(q)))
        ids_, _ = pad(seqs); x = torch.tensor(ids_, device=dev); h = fwd_read(self.g, self.g.embed(x), self.mem, S.repeat_interleave(2, 0))
        rows = torch.cat([torch.full((b - a,), r, dtype=torch.long) for r, (a, b) in enumerate(spans)]).to(dev)
        pos = torch.cat([torch.arange(a - 1, b - 1) for a, b in spans]).to(dev); tok = torch.cat([x[r, a:b] for r, (a, b) in enumerate(spans)])
        lp = self.g.logits(h[rows, pos]).float().log_softmax(-1).gather(-1, tok[:, None])[:, 0]; sc = []; o_ = 0
        for a, b in spans:
            sc.append(float(lp[o_:o_ + b - a].sum())); o_ += b - a
        return np.array([sc[k] - sc[k + 1] for k in range(0, len(sc), 2)])

    def aux_logits(self, S, its, t, dev):
        """§175 on ProsQA: [EOT, " Step", " t", " :"] + steps[t-1] (teacher-forced) -> (logits predicting steps[t-1] + EOT, targets). No question token."""
        pre = [EOT, STEP_ID, pass_tok(t), COLON_ID]; seqs, tg = [], []
        for it in its:
            assert t <= it["depth"], "§168s2: no aux target past the last hop"
            tgt = it["step_ids"][t - 1] + [EOT]; seqs.append(pre + tgt[:-1]); tg.append(tgt)
        x_, _ = pad(seqs); y_, _ = pad(tg, -100); x = torch.tensor(x_, device=dev); y = torch.tensor(y_, device=dev)
        assert x.shape[1] == len(pre) + y.shape[1] - 1 and bool((x[:, :4] == torch.tensor(pre, device=dev)).all())
        h = fwd_read(self.g, self.g.embed(x), self.mem, S)
        return self.g.logits(h[:, len(pre) - 1:]).float(), y

    @torch.no_grad()
    def aux_greedy(self, S, t, L):
        x = torch.tensor([[EOT, STEP_ID, pass_tok(t), COLON_ID]] * S.shape[0], device=S.device)
        for _ in range(L):
            nt = self.g.logits(fwd_read(self.g, self.g.embed(x), self.mem, S)[:, -1]).float().argmax(-1); x = torch.cat([x, nt[:, None]], 1)
        return x[:, 4:].cpu().numpy()


def text_seq(it, k_drop):
    """Coconut's CoT text with the first k_drop steps removed: question\\n + steps[k:] (each + "\\n") + "### " + answer + EOS -> (ids, n_prefix)."""
    body = it["pre"] + [t for s in it["steps"][k_drop:] for t in ENC.encode(s + "\n")]; return body + it["a1"] + [EOS], len(it["pre"])


@torch.no_grad()
def generate_cot(g, ids, max_new, dev, amp):
    seq = list(ids)
    for _ in range(max_new):
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            h, _ = g(g.embed(torch.tensor([seq], device=dev)), None)
        t = int(g.logits(h[0, -1:]).float().argmax(-1))
        if t in (ANS, EOS):
            break
        seq.append(t)
    return seq


@torch.no_grad()
def generate_cot_batched(g, pres, max_new, dev, amp, bs):
    """generate_cot for many prompts (Audit 2026-10-10: batch 1 without a KV cache was ~3 h): prompts are grouped by length (no padding, so every
    row sees the positions and causal context generate_cot gives it), KV-cached greedy decoding; each row is cut at its first ANS / EOS -> the same
    sequences as generate_cot (gate: scratch/gate_cot_batched.py)."""
    out = [None] * len(pres); by_len = {}
    for i, p in enumerate(pres):
        by_len.setdefault(len(p), []).append(i)
    for L_, idx in sorted(by_len.items()):
        for s in range(0, len(idx), bs):
            ii = idx[s:s + bs]; x = torch.tensor([pres[i] for i in ii], device=dev); gen = []; done = torch.zeros(len(ii), dtype=torch.bool, device=dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                h, past = g(g.embed(x), None)
                for _ in range(max_new):
                    t = g.logits(h[:, -1]).float().argmax(-1); gen.append(t); done |= (t == ANS) | (t == EOS)
                    if bool(done.all()) or len(gen) == max_new:
                        break
                    h, past = g(g.embed(t[:, None]), past)
            G = torch.stack(gen, 1).cpu().numpy()
            for r, i in enumerate(ii):
                row = []
                for v in G[r]:
                    if int(v) in (ANS, EOS):
                        break
                    row.append(int(v))
                out[i] = list(pres[i]) + row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["stepaux", "currdirect", "textcot"]); ap.add_argument("--data", default="na", choices=["na"])
    ap.add_argument("--cache_lower", action="store_true", help="stepaux: blocks 0..5 on the (identical) think input computed once per step and shared by the K passes (exact; Theory / Audit 2026-10-10)")
    ap.add_argument("--ckpt_passes", action="store_true", help="stepaux: torch.utils.checkpoint around each think pass (memory; Audit 2026-10-10)")
    ap.add_argument("--think_k", type=int, default=10); ap.add_argument("--aux_step", type=float, default=1.0); ap.add_argument("--aux_fade", default="12000,12000")
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--micro", type=int, default=1); ap.add_argument("--auto_micro", action="store_true")
    ap.add_argument("--mem_budget", type=float, default=21.0); ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--time_steps", type=int, default=0)
    ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000); ap.add_argument("--fast_add", action="store_true")
    ap.add_argument("--stages", type=int, default=10, help="currdirect: stage k = 0..stages drops the first k steps; then k = stages again (12 stages of steps / 12)"); ap.add_argument("--max_new", type=int, default=220); ap.add_argument("--cot_eval_bs", type=int, default=64, help="textcot eval: KV-cached greedy CoT batch over same-length prompts (1 = generate_cot per item)")
    ap.add_argument("--ckpt_every", type=int, default=1000); ap.add_argument("--log_every", type=int, default=100); ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--train_n", type=int, default=0, help="smokes"); ap.add_argument("--test_n", type=int, default=0, help="smokes"); ap.add_argument("--extras_n", type=int, default=500)
    ap.add_argument("--probe_n_fit", type=int, default=4000); ap.add_argument("--fp32", action="store_true"); ap.add_argument("--sched_scale", type=float, default=1.0, help="smokes")
    ap.add_argument("--chunk_delta", type=int, default=0, help="stepaux: DM.CHUNK_DELTA - the delta heads' writes in chunks of this many tokens (WY form, exact in exact arithmetic; needs --fast_add; gate_chunk_delta.json)")
    ap.add_argument("--aux_reweight", action="store_true", help="note 168-12 STEP-AUX-REWEIGHT: the aux CE of hop t weighted by AUX_W[t-1] inside the per-item mean (stepaux)")
    ap.add_argument("--relabel", action="store_true", help="§191 PROSQA-RELABEL: each training item drawn is relabelled afresh (relabel(); own rng seed + 19100); test never; eval adds train_fit_relabel")
    ap.add_argument("--test500_every", type=int, default=0, help="§191: test500_acc (fixed rng 19101 500-item subset of the test set, the arm's eval path) every N steps and at the last step")
    ap.add_argument("--eval_train_n", type=int, default=0, help="note 168-16: with --eval_only, evaluate (acc / acc_by_depth / aux_exact_by_hop) on this many TRAINING items (rng 16816) -> eval_trainitems.json")
    ap.add_argument("--eval_only", action="store_true", help="evaluate <out>/model.pt (no training; args_eval.json; model.pt / args.json / log.jsonl untouched)")
    ap.add_argument("--resume", default=None, help="a ckpt_last.pt / kept ckpt of this run (model, opt, np_rng, step, elapsed)")
    ap.add_argument("--no_normcap", action="store_true", help="stepaux: skip note 185-1's inference-only NORM-CAP eval (smokes)")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); DM.FAST_ADD = bool(a.fast_add); DM.CHUNK_DELTA = int(a.chunk_delta)
    sc = a.sched_scale; SA = a.arm == "stepaux"; K = a.think_k if SA else 0; f0_, f1_ = [float(v) for v in a.aux_fade.split(",")]
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else (lambda: torch.autocast("cpu", enabled=False))
    pr = os.path.join(ROOT, "prosqa", "p8data"); Dtr = load(os.path.join(pr, "p8_na_train.json")); Dte = load(os.path.join(pr, "p8_na_test.json")); Dctl = load(os.path.join(pr, "p8_ctl.json"))
    Dtr = Dtr[:a.train_n] if a.train_n else Dtr; Dte = Dte[:a.test_n] if a.test_n else Dte
    if a.eval_train_n:                                                      # note 168-16: the eval path on the run's own training items (eval-path sanity check)
        assert a.eval_only, "--eval_train_n needs --eval_only"; Dte = [Dtr[i] for i in np.random.default_rng(16816).permutation(len(Dtr))[:a.eval_train_n]]
    EVAL_FN = "eval_trainitems.json" if a.eval_train_n else "eval.json"; Dctl = Dctl[:a.test_n] if a.test_n else Dctl
    if a.relabel:                                                           # §191: the raw training items (relabelled per draw); test / ctl stay as loaded
        Rtr = load_train_raw(os.path.join(pr, "p8_na_train.json")); Rtr = Rtr[:a.train_n] if a.train_n else Rtr; assert len(Rtr) == len(Dtr)
        rl_rng = np.random.default_rng(a.seed + 19100); rl_n = [0, 0]      # [items relabelled, options swapped]
        if not a.train_n:
            assert {w for x in Rtr for w in _lit(x["idx_to_symbol"])} == set(ENT17) | set(CON38), "§191: the train vocabulary = ENT17 + CON38"
        for D_ in (Dte, Dctl):                                              # Audit: test names come from the same 17 / 38 vocabulary (no OOD names)
            assert all(set(re.findall(r"\b[A-Z][a-z]+\b", ENC.decode(it["q"]))) <= set(ENT17) | {"Every", "Is"} for it in D_), "§191: a test entity name outside ENT17"
            assert all(set(re.findall(r"\b[a-z]+pus\b", ENC.decode(it["q"]))) <= set(CON38) for it in D_), "§191: a test concept name outside CON38"

        def rl_item(i):
            y, sw = relabel(Rtr[i], rl_rng); rl_n[0] += 1; rl_n[1] += int(sw); return encode(y)
    g = s5.load_gpt2(dev); m = ProsMem(g).to(dev) if SA else None; params = m if SA else g
    longest = max(range(len(Dtr)), key=lambda i: len(Dtr[i]["q"]) + sum(len(s) for s in Dtr[i]["step_ids"]))
    it_long = Dtr[longest]
    if a.relabel:                                                           # §191: the memory check on the worst relabel of the 200 longest items
        cand = sorted(range(len(Dtr)), key=lambda i: -(len(Dtr[i]["q"]) + sum(len(s) for s in Dtr[i]["step_ids"])))[:200]
        it_long = max((encode(relabel(Rtr[i], None, worst=True)[0]) for i in cand), key=lambda it: len(it["q"]) + sum(len(s) for s in it["step_ids"]))
    L_STAGE = max(1, a.steps // (a.stages + 2))                             # Theory 168-7 point 3: 12 stages of 1000 at 12k

    def stage(step):
        return min(a.stages, (step - 1) // L_STAGE)                          # k = 0..10, then 10 again for the last stage (training depths <= 10: no CoT left)

    def lam(step):
        return 0.0 if not SA else (a.aux_step if step <= f0_ * sc else a.aux_step * max(0.0, (f1_ * sc - step) / max(1.0, (f1_ - f0_) * sc)))

    aux_hist = []

    def loss_items(its, step):
        if SA:
            x, msk = m.think_inputs(its, dev); per = [[] for _ in its]; lam_ = lam(step)
            with ac():
                S = m.mem.zero_state(len(its), dev); hl = m.lower(x) if a.cache_lower else None   # --cache_lower: blocks 0..5 once per step
                for t in range(1, K + 1):
                    if a.ckpt_passes and torch.is_grad_enabled():           # Audit: per-pass activation checkpointing (the K x 721-token activations)
                        from torch.utils.checkpoint import checkpoint
                        S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False)
                    else:
                        S = m.write(S, x, msk, hl)
                    idx = [i for i, it in enumerate(its) if t <= it["depth"]]
                    if lam_ and idx:
                        lg_a, y_a = m.aux_logits(S[idx], [its[i] for i in idx], t, dev)
                        ce = Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100, reduction="none").sum(1) / (y_a != -100).sum(1)
                        for r_, i in enumerate(idx):
                            per[i].append(ce[r_] * AUX_W[t - 1] if a.aux_reweight else ce[r_])   # 168-12: w_t inside the per-item mean
                la = m.dec_loss(S, its, dev)
            if not lam_:
                return la
            aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean(); aux_hist.append((float(la), float(aux))); return la + lam_ * aux
        kd = (stage(step) if a.arm == "currdirect" else 0); seqs, n0s = zip(*[text_seq(it, kd) for it in its])
        ids_, _ = pad(list(seqs)); lab = np.full(ids_.shape, -100)
        for r, (q, n0) in enumerate(zip(seqs, n0s)):
            lab[r, n0:len(q)] = q[n0:]
        x = torch.tensor(ids_, device=dev)
        with ac():
            h, _ = g(g.embed(x), None)
        lg = g.logits(h[:, :-1]).float(); return Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), torch.tensor(lab, device=dev)[:, 1:].reshape(-1), ignore_index=-100)

    def mem_check(micro):
        torch.cuda.reset_peak_memory_stats(dev); its = [it_long] * (a.bs // micro); params.train()
        loss_items(its, 1).backward(); params.zero_grad(set_to_none=True)
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30 + 3 * sum(p_.numel() * 4 for p_ in params.parameters()) / 2 ** 30
    if a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.eval_only):
        mc = []
        while True:
            try:
                pk = mem_check(a.micro)
            except torch.OutOfMemoryError:
                pk = float("inf"); params.zero_grad(set_to_none=True)
            torch.cuda.empty_cache(); mc.append({"micro": a.micro, "peak_gib": pk}); print("MEM", a.micro, pk, flush=True)
            if pk <= a.mem_budget or a.micro >= a.bs or a.mem_check:
                break
            a.micro *= 2
        json.dump({"checks": mc, "micro": a.micro, "peak_gib": mc[-1]["peak_gib"], "longest_question_tokens": len(it_long["q"]), "relabel_worst": bool(a.relabel), "K": K, "aux": bool(SA and a.aux_step)},
                  open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return
    info = {**vars(a), "task": "prosqa_na", "train_items": len(Dtr), "test_items": len(Dte), "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
            "train_driver": driver_version(), **env_info(), "stage_len": L_STAGE if a.arm == "currdirect" else None}
    json.dump(info, open(os.path.join(a.out, "args_eval.json" if a.eval_only else (("args_resume_%d.json" % int(time.time())) if a.resume else "args.json")), "w", newline="\n"), indent=1)
    if SA and not a.resume and not a.eval_only:                                                 # note 147-2: g_add calibration at step 0 (a resume loads the calibrated s_add)
        its = [Dtr[i] for i in np.random.default_rng(16801).permutation(len(Dtr))[:32]]; x, msk = m.think_inputs(its, dev); m.mem.norms = []
        with torch.no_grad(), ac():
            m.think(m.mem.zero_state(len(its), dev), x, msk, min(K, 4))
        nr = m.mem.norms[1:]; m.mem.norms = None; r0 = float(torch.stack([d_ / a_.clamp_min(1e-12) for d_, a_ in nr]).median()) if nr else 1.0
        m.mem.s_add.data.fill_(math.log(max(r0, 1e-12))); json.dump({"g_add_init": r0}, open(os.path.join(a.out, "s168s2_init.json"), "w", newline="\n"))
    if a.aux_reweight:
        assert SA, "--aux_reweight is a stepaux option"
        if not a.train_n:
            from collections import Counter
            assert dict(Counter(it["depth"] for it in Dtr)) == HIST_168, "168-12: the train histogram the weights were derived from"
    opt = torch.optim.AdamW(params.parameters(), lr=a.lr, weight_decay=0.01); hist = []; n_nonfinite = 0; wg = []; gn_hist = []; clip_src = "run"
    log = open(os.devnull, "w") if a.eval_only else open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time(); step0 = 1
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); params.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); rng.bit_generator.state = ck["np_rng"]
        if a.relabel:
            rl_rng.bit_generator.state = ck["rl_rng"]; rl_n[:] = ck["rl_n"]
        step0 = ck["step"] + 1; n_nonfinite = ck.get("n_nonfinite", 0); gn_hist = list(ck.get("gn_hist", [])); t0 = time.time() - ck["elapsed"]; log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    last_ = a.steps if not a.time_steps else min(a.steps, a.time_steps); tt0 = time.time()
    if a.eval_only:                                                         # the trained model; no training step runs
        params.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev, weights_only=False)); last_ = 0
        cp_ = os.path.join(a.out, "ckpt_last.pt")                           # Theory: the clip rate from the kept checkpoint's gn_hist when it has one
        if os.path.exists(cp_):
            ck_ = torch.load(cp_, map_location="cpu", weights_only=False); gn_hist = list(ck_.get("gn_hist", []))
            clip_src = ("ckpt_last.pt gn_hist (steps 1..%s)" % ck_.get("step")) if gn_hist else "not logged"; del ck_
        else:
            clip_src = "not logged"
    T500 = [Dte[i] for i in np.random.default_rng(19101).permutation(len(Dte))[:500]] if a.test500_every else []

    def test500():
        """§191: candidate accuracy on T500 by the arm's eval path (stepaux: K passes; textcot: greedy CoT first). No rng is drawn (training unchanged)."""
        params.eval()
        with torch.no_grad():
            if SA:
                Mx = []
                for i in range(0, len(T500), a.eval_bs):
                    ch = T500[i:i + a.eval_bs]; x, msk = m.think_inputs(ch, dev)
                    with ac():
                        Mx.append(m.margins(m.think(m.mem.zero_state(len(ch), dev), x, msk, K, cache=a.cache_lower), ch, dev))
                Mx = np.concatenate(Mx)
            elif a.arm == "currdirect":
                Mx = candidate_margins(g, [(it["pre"], it["a1"], it["a2"], it["depth"]) for it in T500], dev, amp, a.eval_bs)
            else:
                Q = generate_cot_batched(g, [it["pre"] for it in T500], a.max_new, dev, amp, a.cot_eval_bs) if a.cot_eval_bs > 1 else [generate_cot(g, it["pre"], a.max_new, dev, amp) for it in T500]
                Mx = candidate_margins(g, [(q, it["a1"], it["a2"], it["depth"]) for q, it in zip(Q, T500)], dev, amp, a.eval_bs)
        params.train(); return float((Mx > 0).mean())
    for step in range(step0, last_ + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        params.train()
        if SA:
            m.mem.fade.fill_(min(1.0, step / max(1, int(500 * sc))))
            m.mem.gamma.fill_(0.0 if step <= a.write_mix_start * sc else min(1.0, (step - a.write_mix_start * sc) / max(1.0, (a.write_mix_end - a.write_mix_start) * sc)))
        idx = rng.integers(0, len(Dtr), a.bs); opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro
        for k in range(a.micro):
            lk = loss_items([rl_item(i) if a.relabel else Dtr[i] for i in idx[k * ch:(k + 1) * ch]], step); (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
        if SA:
            wg.append(write_grad_norm(m))
        gn = torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0); gn_hist.append(float(gn))   # pre-clip total norm (168-12: clip rate)
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            n_nonfinite += 1; opt.zero_grad(); log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush()
        else:
            opt.step(); hist.append(tot)
        if step % a.log_every == 0 or step == last_:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "sec": round(time.time() - t0)}; hist = []
            if SA:
                rec.update({"aux_lambda": lam(step), "answer_ce": float(np.mean([x_[0] for x_ in aux_hist])) if aux_hist else None,
                            "aux_ce": float(np.mean([x_[1] for x_ in aux_hist])) if aux_hist else None, "gamma": float(m.mem.gamma), "fade": float(m.mem.fade),
                            "write_grad_norm": float(np.mean(wg)) if wg else None}); aux_hist.clear(); wg = []
            if a.arm == "currdirect":
                rec["stage"] = stage(step)
            if gn_hist:                                                     # pre-clip total grad norm at this step + the clip share of the last 100 steps (log only)
                rec.update({"gn_preclip": gn_hist[-1], "clip_frac_100": float(np.mean([v_ > 1.0 for v_ in gn_hist[-100:]]))})
            if a.relabel:                                                   # Audit: the option-swap rate (p 0.5), cumulative over the run
                rec.update({"relabel_n": rl_n[0], "relabel_swap_rate": rl_n[1] / max(1, rl_n[0])})
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if a.test500_every and not a.time_steps and (step % a.test500_every == 0 or step == last_):
            tq_ = time.time(); r5 = {"step": step, "test500_acc": test500(), "test500_sec": round(time.time() - tq_, 1)}; log.write(json.dumps(r5) + "\n"); log.flush(); print(r5, flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps and not a.time_steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": params.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state, "step": step, "elapsed": time.time() - t0, "n_nonfinite": n_nonfinite, "gn_hist": gn_hist,
                        **({"rl_rng": rl_rng.bit_generator.state, "rl_n": list(rl_n)} if a.relabel else {})}, cp)
    if a.time_steps:
        sps = (time.time() - tt0) / max(1, last_); tj = {"arm": a.arm, "think_k": K, "steps_timed": last_, "sec_per_step": sps, "steps": a.steps,
                                                        "est_hours_steps": sps * a.steps / 3600, "micro": a.micro, "gpu": info["train_gpu"]}
        for f_ in ("timing.json", "eval.json"):
            json.dump(tj, open(os.path.join(a.out, f_), "w", newline="\n"), indent=1)
        print("TIMING", sps, flush=True); return
    if a.eval_only:
        lg_ = [json.loads(l_) for l_ in open(os.path.join(a.out, "log.jsonl")) if '"sec"' in l_]; train_sec = lg_[-1]["sec"] if lg_ else None
    else:
        train_sec = round(time.time() - t0); torch.save(params.state_dict(), os.path.join(a.out, "model.pt"))
    opt.zero_grad(set_to_none=True); opt.state.clear()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    params.eval(); t1 = time.time(); bs = a.eval_bs; dep = np.array([it["depth"] for it in Dte])
    by_depth = lambda M_, d_: {str(d): float((M_[d_ == d] > 0).mean()) for d in range(3, 15) if (d_ == d).any()}   # noqa: E731

    def cap_add(S, C):
        """note 185-1: the additive heads' state rescaled to norm C when its per-item norm exceeds C (direction kept; delta heads untouched)."""
        ha = m.mem.h_add; cur = S[:, :ha].float().flatten(1).norm(dim=1)
        f = torch.where(cur > C, C / cur.clamp_min(1e-12), torch.ones_like(cur)).to(S.dtype)
        return torch.cat([S[:, :ha] * f[:, None, None, None], S[:, ha:]], 1)

    def sa_margins(D, Kp, M0=None, C=None):
        out = []
        for i in range(0, len(D), bs):
            ch = D[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            with torch.no_grad(), ac():
                S = m.mem.zero_state(len(ch), dev) if M0 is None else torch.tensor(M0[i:i + bs], device=dev, dtype=m.mem.zero_state(1, dev).dtype)
                if C is None:
                    S = m.think(S, x, msk, Kp, cache=a.cache_lower)
                else:                                                       # NORM-CAP: the cap after every write
                    hl = m.lower(x) if (a.cache_lower and Kp) else None
                    for _ in range(Kp):
                        S = cap_add(m.write(S, x, msk, hl), C)
                out.append(m.margins(S, ch, dev))
        return np.concatenate(out)

    def states(D, T):
        """the flattened memory after passes 1..T: {t: (N, H, DH, DH) float numpy}."""
        out = {t: [] for t in range(1, T + 1)}
        for i in range(0, len(D), bs):
            ch = D[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            with torch.no_grad(), ac():
                S = m.mem.zero_state(len(ch), dev)
                for t in range(1, T + 1):
                    S = m.write(S, x, msk); out[t].append(S.float().cpu().numpy())
        return {t: np.concatenate(v) for t, v in out.items()}

    if SA:
        M = sa_margins(Dte, K); Mc = sa_margins(Dctl, K)
    elif a.arm == "currdirect":
        M = candidate_margins(g, [(it["pre"], it["a1"], it["a2"], it["depth"]) for it in Dte], dev, amp, bs)
        Mc = candidate_margins(g, [(it["pre"], it["a1"], it["a2"], it["depth"]) for it in Dctl], dev, amp, bs)
    else:                                                                   # textcot: greedy CoT, then candidate scoring after it
        cot_len = []

        def cot_items(D):
            Q = generate_cot_batched(g, [it["pre"] for it in D], a.max_new, dev, amp, a.cot_eval_bs) if a.cot_eval_bs > 1 else [generate_cot(g, it["pre"], a.max_new, dev, amp) for it in D]
            out = []
            for q, it in zip(Q, D):
                cot_len.append(len(q) - len(it["pre"])); out.append((q, it["a1"], it["a2"], it["depth"]))
            return out
        M = candidate_margins(g, cot_items(Dte), dev, amp, bs); Mc = candidate_margins(g, cot_items(Dctl), dev, amp, bs)
    trg_, trd_ = info["train_gpu"], info.get("train_driver")
    if a.eval_only and os.path.exists(os.path.join(a.out, "args.json")):      # Theory: an eval-only run reports the TRAINING card from args.json (eval_gpu = this card)
        ta_ = json.load(open(os.path.join(a.out, "args.json"))); trg_, trd_ = ta_.get("train_gpu", trg_), ta_.get("train_driver", trd_)
    res = {"arm": a.arm, "data": "na", "seed": a.seed, "steps": a.steps, "think_k": K, "acc": float((M > 0).mean()), "n": int(len(M)), "acc_by_depth": by_depth(M, dep),
           "n_by_depth": {str(d): int((dep == d).sum()) for d in range(3, 15) if (dep == d).any()}, "acc_ctl": float((Mc > 0).mean()), "n_ctl": int(len(Mc)),
           "train_sec": train_sec, "n_nonfinite": n_nonfinite, "micro": a.micro, "eval_dtype": str(amp), "gpu": trg_, "train_gpu": trg_, "train_driver": trd_,
           "acc_def": "candidate accuracy: log p('### ' + correct) > log p(the negative's) after the arm's prefix (stepaux: + the memory after K passes)"}
    if a.arm == "textcot":
        res["mean_cot_tokens"] = float(np.mean(cot_len)) if cot_len else None
    res["eval_gpu"] = torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"
    res["eval_only"] = bool(a.eval_only)

    def _partial(tag, s_=None):                                             # Audit 2026-10-11 (b263 lost its eval to a late crash): eval.json after each block
        if s_ is not None:
            res["s168s2"] = s_
        res["eval_partial_through"] = tag; res["eval_partial"] = True; json.dump(res, open(os.path.join(a.out, EVAL_FN), "w", newline="\n"), indent=1)
    _partial("base")
    if a.relabel and not a.eval_train_n:                                    # §191: train fit on 1,000 relabelled TRAIN items (rng 16816 picks, rng 19102 relabels)
        fr_ = np.random.default_rng(19102); Dfit = [encode(relabel(Rtr[i], fr_)[0]) for i in np.random.default_rng(16816).permutation(len(Rtr))[:1000]]
        if SA:
            tf_ = {str(kp): float((sa_margins(Dfit, kp) > 0).mean()) for kp in sorted({0, 1, 10, K})}
        elif a.arm == "currdirect":
            tf_ = {"acc": float((candidate_margins(g, [(it["pre"], it["a1"], it["a2"], it["depth"]) for it in Dfit], dev, amp, bs) > 0).mean())}
        else:
            Qf = generate_cot_batched(g, [it["pre"] for it in Dfit], a.max_new, dev, amp, a.cot_eval_bs) if a.cot_eval_bs > 1 else [generate_cot(g, it["pre"], a.max_new, dev, amp) for it in Dfit]
            tf_ = {"acc": float((candidate_margins(g, [(q, it["a1"], it["a2"], it["depth"]) for q, it in zip(Qf, Dfit)], dev, amp, bs) > 0).mean())}
        res["train_fit_relabel"] = {"acc_by_kprime" if SA else "acc_cot" if a.arm == "textcot" else "acc_direct": tf_, "n": len(Dfit),
                                    "items": "TRAIN items (rng 16816 permutation[:1000]) relabelled once (rng 19102)", "relabel_n": rl_n[0] or None,
                                    "relabel_swap_rate": rl_n[1] / rl_n[0] if rl_n[0] else None, "relabel_counts_def": "training draws of this process (None under --eval_only: see log.jsonl)"}
        _partial("train_fit_relabel")
    if SA:
        s = {"acc_by_depth_kprime": {str(kp): by_depth(sa_margins(Dte, kp), dep) for kp in KPRIMES}}; _partial("acc_by_depth_kprime", s)
        ex = [Dte[i] for i in np.random.default_rng(16802).permutation(len(Dte))[:a.extras_n]]; exd = np.array([it["depth"] for it in ex])
        hop = {}                                                            # per-hop aux exact match (greedy steps[t-1] + EOT), t <= min(depth, K)
        for i in range(0, len(ex), bs):
            ch = ex[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            with torch.no_grad(), ac():
                S = m.mem.zero_state(len(ch), dev)
                for t in range(1, K + 1):
                    S = m.write(S, x, msk); ix = [j for j, it in enumerate(ch) if t <= it["depth"]]
                    if ix:
                        L_ = max(len(ch[j]["step_ids"][t - 1]) + 1 for j in ix); gen = m.aux_greedy(S[ix], t, L_)
                        for r_, j in enumerate(ix):
                            tgt = ch[j]["step_ids"][t - 1] + [EOT]; hop.setdefault(t, []).append(bool(list(gen[r_, :len(tgt)]) == tgt))
        s["aux_exact_by_hop"] = {str(t): float(np.mean(v)) for t, v in sorted(hop.items())}; s["aux_n_by_hop"] = {str(t): len(v) for t, v in sorted(hop.items())}; _partial("aux_exact_by_hop", s)
        if a.eval_train_n:                                                  # 168-16: train items need only acc + K' sweep + aux hops; done
            res.update({"eval_partial": False, "items": "TRAIN items (rng 16816 permutation[:%d]) of p8_na_train" % a.eval_train_n, "eval_sec": round(time.time() - t1)}); res.pop("eval_partial_through", None)
            json.dump(res, open(os.path.join(a.out, EVAL_FN), "w", newline="\n"), indent=1); print("EVAL_TRAIN", res["acc"], s["aux_exact_by_hop"], flush=True); return
        St = states(ex, 2 * K)                                              # 173-1's statistic: mean over items of cos(Sd_t, Sd_t-1), Sd = the delta-head state
        def dcos(St_):
            o_ = {}
            for t in range(1, 2 * K + 1):
                u = St_[t][:, DM.H_ADD:].reshape(len(ex), -1); v = St_[t - 1][:, DM.H_ADD:].reshape(len(ex), -1) if t > 1 else np.zeros_like(u)
                nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1); ok = (nu > 0) & (nv > 0)
                o_[str(t)] = float(((u[ok] * v[ok]).sum(1) / (nu[ok] * nv[ok])).mean()) if ok.any() else None
            return o_
        s["delta_cos"] = dcos(St)
        torch.manual_seed(a.seed); m_keep = m; m = ProsMem(s5.load_gpt2(dev)).to(dev)   # cos at init: pretrained trunk + fresh memory (seed), gamma 1 (halt173 init)
        with torch.no_grad():
            m.mem.gamma.fill_(1.0); m.mem.fade.fill_(1.0)
        m.eval(); s["delta_cos_at_init"] = dcos(states(ex, 2 * K)); del m; m = m_keep; _partial("delta_cos", s)
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        donor = np.roll(np.arange(len(ex)), 1); S8 = states(ex, 8)[8][donor]  # PATH-INDEP (as §182): K' = 16 from 0 vs from another item's 8-pass memory
        M0 = sa_margins(ex, 16); M1 = sa_margins(ex, 16, S8)
        s["path_indep"] = {"answer_agree": float(((M0 > 0) == (M1 > 0)).mean()), "acc_from_zero": float((M0 > 0).mean()), "acc_from_donor": float((M1 > 0).mean()),
                           "n_items": len(ex), "kprime": 16, "donor_passes": 8, "derangement": "np.roll(sample order, 1)", "items": "test, rng 16802 permutation[:extras_n]"}; _partial("path_indep", s)
        Sall = states(ex, 16); h = np.full(len(ex), 16); hit = np.zeros(len(ex), bool); pick = np.zeros(len(ex), bool)   # AUX-DONE on ProsQA (Theory 168-7 point 1)
        for t in range(1, 17):
            for i in range(0, len(ex), bs):
                ix = np.arange(i, min(i + bs, len(ex))); Sx = torch.tensor(Sall[t][ix], device=dev, dtype=m.mem.zero_state(1, dev).dtype)
                with torch.no_grad(), ac():
                    gen = m.aux_greedy(Sx, t, 24)
                for r_, j in enumerate(ix):
                    w_ = ENC.decode([int(v) for v in gen[r_] if int(v) != EOT][:24]).strip().split(".")[0].split(); obj = w_[-1] if w_ else ""
                    if not hit[j] and obj in (ex[j]["opt_true"], ex[j]["opt_false"]):
                        hit[j] = True; h[j] = t; pick[j] = obj == ex[j]["opt_true"]
        Mh = np.array([sa_margins([ex[j]], int(h[j]))[0] for j in range(len(ex))]); d_ = h - exd
        s["aux_done"] = {"acc_at_halt": float((Mh > 0).mean()), "aux_pick_acc": float(pick[hit].mean()) if hit.any() else None, "mean_halt": float(h.mean()),
                         "mean_halt_minus_depth": float(d_.mean()), "frac_capped": float((~hit).mean()), "n_items": len(ex),
                         "halt_minus_depth_share": {"early": float((d_ < 0).mean()), "at_depth": float((d_ == 0).mean()), "+1": float((d_ == 1).mean()), "+2": float((d_ == 2).mean()), "later": float((d_ > 2).mean())},
                         "acc_at_halt_by_bucket": {k_: (float((Mh[v_] > 0).mean()) if v_.any() else None) for k_, v_ in {"early": d_ < 0, "at_depth": d_ == 0, "+1": d_ == 1, "later": d_ > 1}.items()},
                         "rule": "halt at the first pass t (1..16) whose greedy aux step's object (last word) is EITHER option named in the question; cap 16 (descriptive; T-SHIFT's pass-token caveat applies)"}; _partial("aux_done", s)
        ext_ = [it for it in Dte if 11 <= it["depth"] <= 14][:a.extras_n]; hx = {}   # aux_acc_by_pass_ext (Theory point 2): depths 11..14 to K' 16
        for i in range(0, len(ext_), bs):
            ch = ext_[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            with torch.no_grad(), ac():
                S = m.mem.zero_state(len(ch), dev)
                for t in range(1, 17):
                    S = m.write(S, x, msk); ix = [j for j, it in enumerate(ch) if t <= it["depth"]]
                    if ix:
                        gen = m.aux_greedy(S[ix], t, max(len(ch[j]["step_ids"][t - 1]) + 1 for j in ix))
                        for r_, j in enumerate(ix):
                            tg_ = ch[j]["step_ids"][t - 1] + [EOT]; hx.setdefault(t, []).append(bool(list(gen[r_, :len(tg_)]) == tg_))
        s["aux_acc_by_pass_ext"] = {str(t): float(np.mean(v)) for t, v in sorted(hx.items())}; s["n_by_pass_ext"] = {str(t): len(v) for t, v in sorted(hx.items())}; _partial("aux_acc_by_pass_ext", s)
        from boxes.nested170 import logreg_acc                              # STEP-RECOVER: logreg M_t (delta state) -> hop t's node id
        tr = [Dtr[i] for i in np.random.default_rng(16803).permutation(len(Dtr))[:a.probe_n_fit]]; Str = states(tr, K); sr = {}
        for t in range(1, K + 1):
            fi = [i for i, it in enumerate(tr) if t <= it["depth"]]; ti = [i for i, it in enumerate(ex) if t <= it["depth"]]
            if len(fi) > 50 and ti:
                Xf = Str[t][fi][:, DM.H_ADD:].reshape(len(fi), -1); Xt = St[t][ti][:, DM.H_ADD:].reshape(len(ti), -1)
                VOC = {nm: c_ for c_, nm in enumerate(sorted({it["node"][t - 1] for it in tr + ex if t <= it["depth"]}))}
                yf_ = np.array([VOC[tr[i]["node"][t - 1]] for i in fi]); yt_ = np.array([VOC[ex[i]["node"][t - 1]] for i in ti])
                assert yf_.max() < len(VOC) and yt_.max() < len(VOC), "STEP-RECOVER labels exceed n_cls"   # Audit: the probe is sized to len(VOC)
                sr[str(t)] = logreg_acc(Xf, yf_, Xt, yt_, dev, n_cls=len(VOC))   # b263: > 11 classes (nested170.logreg_acc defaulted to P = 11)
        s["step_recover"] = sr; s["step_recover_def"] = "classes = concept NAMES (shared vocabulary); multinomial logreg (nested170.logreg_acc) from the flattened delta-head state after pass t to hop t's node id; fit on probe_n_fit train items"; _partial("step_recover", s)
        if not a.no_normcap:                                                # note 185-1 (descriptive, no rule reads it): inference-only NORM-CAP, C from TRAIN items
            trc = [Dtr[i] for i in np.random.default_rng(18502).permutation(len(Dtr))[:1000]]; nrm = []
            for i in range(0, len(trc), bs):
                ch = trc[i:i + bs]; x, msk = m.think_inputs(ch, dev)
                with torch.no_grad(), ac():
                    S = m.think(m.mem.zero_state(len(ch), dev), x, msk, K, cache=a.cache_lower)
                nrm.append(S[:, :m.mem.h_add].float().flatten(1).norm(dim=1).cpu().numpy())
            C = float(np.median(np.concatenate(nrm)))
            s["normcap"] = {"C": C, "C_source": "median ||S_add|| after K = %d passes over %d train items (rng 18502 permutation[:1000])" % (K, len(trc)),
                            "acc_by_depth_kprime": {str(kp): by_depth(sa_margins(Dte, kp, C=C), dep) for kp in KPRIMES},
                            "def": "inference-only: after every write, S_add rescaled to norm C when ||S_add|| > C (per item, direction kept; delta untouched) - note 185-1; uncapped = s168s2.acc_by_depth_kprime"}
        res["s168s2"] = s
    g_ = np.array([v for v in gn_hist if math.isfinite(v)]) if gn_hist else np.zeros(0)   # 168-12: clip rate (every arm; s168s2.clip for stepaux)
    res["clip"] = {"clip_rate": float((g_ > 1.0).mean()) if len(g_) else None, "grad_norm_preclip": {"median": float(np.median(g_)) if len(g_) else None,
                   "p90": float(np.percentile(g_, 90)) if len(g_) else None, "n_steps": int(len(g_))}, "aux_reweight": bool(a.aux_reweight),
                   "weights": list(AUX_W) if a.aux_reweight else None, "source": clip_src, "def": "clip_grad_norm_(..., 1.0)'s pre-clip total norm per step; clip fires when > 1.0"}
    if SA:
        res["s168s2"]["clip"] = res["clip"]
    if a.eval_train_n:                                                      # 168-17: the label for the non-stepaux arms (stepaux returned above with it)
        res["items"] = "TRAIN items (rng 16816 permutation[:%d]) of p8_na_train" % a.eval_train_n
    res.pop("eval_partial_through", None); res["eval_partial"] = False       # the complete eval (Theory: score_s168s2 reads eval_partial true as running)
    res["eval_sec"] = round(time.time() - t1); json.dump(res, open(os.path.join(a.out, EVAL_FN), "w", newline="\n"), indent=1)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({"acc": res["acc"], "acc_by_depth": res["acc_by_depth"]}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
