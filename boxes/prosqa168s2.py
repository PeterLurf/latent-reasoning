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


def pass_tok(t):
    tk = ENC.encode(" %d" % t); assert len(tk) == 1; return tk[0]


def _lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def load(path):
    """-> list of dicts: q (question ids), pre (question + "\\n" ids), a1 / a2 ("### " + correct / negative answer ids), depth, steps (texts), step_ids
    (" " + step ids), node (hop t's node id = the object of steps[t-1], an index into idx_to_symbol)."""
    out = []
    for x in json.load(open(path)):
        sym = _lit(x["idx_to_symbol"]); steps = _lit(x["steps"]); tgt, neg = sym[int(x["target"])], sym[int(x["neg_target"])]; ans = x["answer"]
        a_neg = re.sub(rf"\b{tgt}\.$", f"{neg}.", ans); assert a_neg != ans
        node = [s.rstrip(".").split()[-1] for s in steps]                 # hop t's node = the CONCEPT NAME (shared vocabulary; Theory 168-7 point 6)
        out.append({"q": ENC.encode(x["question"]), "pre": ENC.encode(x["question"] + "\n"), "a1": ENC.encode("### " + ans), "a2": ENC.encode("### " + a_neg),
                    "depth": len(steps), "steps": steps, "step_ids": [ENC.encode(" " + s) for s in steps], "node": node, "opt_true": tgt, "opt_false": neg})
    return out


def pad(seqs, fill=EOS):
    T = max(len(q) for q in seqs); ids = np.full((len(seqs), T), fill); m = np.zeros((len(seqs), T), bool)
    for r, q in enumerate(seqs):
        ids[r, :len(q)] = q; m[r, :len(q)] = True
    return ids, m


class ProsMem(torch.nn.Module):
    """GPT-2 + §170's DeltaMem-R; think = [EOT, question]; decoder = [EOT, question, "\\n"] + answer, read-only; aux = §175's step decoder."""
    def __init__(self, g):
        super().__init__(); self.g = g; self.mem = DM.DeltaMem(g, 6, "top")

    def think_inputs(self, its, dev):
        ids_, m_ = pad([[EOT] + it["q"] for it in its]); return torch.tensor(ids_, device=dev), torch.tensor(m_, device=dev)

    def write(self, S, x, mask, h_low=None):
        assert h_low is None or x is self._hl_src, "--cache_lower: every pass must see the identical input the lower blocks were cached on"
        return DM.gpt_forward_mem(self.g, self.g.embed(x), self.mem, S, mask=mask, h_low=h_low)[1]

    def lower(self, x):
        """--cache_lower: blocks 0..5 on [EOT, question] once (every pass has the identical input here: no pass index on this path)."""
        self._hl_src = x; return DM.gpt_lower(self.g, self.g.embed(x), self.mem.layer)

    def think(self, S, x, mask, K, cache=False):
        hl = self.lower(x) if (cache and K) else None
        for _ in range(K):
            S = self.write(S, x, mask, hl)
        return S

    def dec_loss(self, S, its, dev):
        seqs = [[EOT] + it["pre"] + it["a1"] + [EOS] for it in its]; ids_, _ = pad(seqs); x = torch.tensor(ids_, device=dev)
        lab = np.full(ids_.shape, -100)
        for r, (it, q) in enumerate(zip(its, seqs)):
            n0 = 1 + len(it["pre"]); lab[r, n0:len(q)] = q[n0:]
        h = fwd_read(self.g, self.g.embed(x), self.mem, S); lg = self.g.logits(h[:, :-1]).float()
        return Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), torch.tensor(lab, device=dev)[:, 1:].reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def margins(self, S, its, dev):
        """log p(a1) - log p(a2) after [EOT, question, "\\n"], reading S (each item's memory used for both answers)."""
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
    ap.add_argument("--resume", default=None, help="a ckpt_last.pt / kept ckpt of this run (model, opt, np_rng, step, elapsed)")
    ap.add_argument("--no_normcap", action="store_true", help="stepaux: skip note 185-1's inference-only NORM-CAP eval (smokes)")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); DM.FAST_ADD = bool(a.fast_add); DM.CHUNK_DELTA = int(a.chunk_delta)
    sc = a.sched_scale; SA = a.arm == "stepaux"; K = a.think_k if SA else 0; f0_, f1_ = [float(v) for v in a.aux_fade.split(",")]
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else (lambda: torch.autocast("cpu", enabled=False))
    pr = os.path.join(ROOT, "prosqa", "p8data"); Dtr = load(os.path.join(pr, "p8_na_train.json")); Dte = load(os.path.join(pr, "p8_na_test.json")); Dctl = load(os.path.join(pr, "p8_ctl.json"))
    Dtr = Dtr[:a.train_n] if a.train_n else Dtr; Dte = Dte[:a.test_n] if a.test_n else Dte; Dctl = Dctl[:a.test_n] if a.test_n else Dctl
    g = s5.load_gpt2(dev); m = ProsMem(g).to(dev) if SA else None; params = m if SA else g
    longest = max(range(len(Dtr)), key=lambda i: len(Dtr[i]["q"]) + sum(len(s) for s in Dtr[i]["step_ids"]))
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
        torch.cuda.reset_peak_memory_stats(dev); its = [Dtr[longest]] * (a.bs // micro); params.train()
        loss_items(its, 1).backward(); params.zero_grad(set_to_none=True)
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30 + 3 * sum(p_.numel() * 4 for p_ in params.parameters()) / 2 ** 30
    if a.mem_check or (a.auto_micro and dev.startswith("cuda")):
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
        json.dump({"checks": mc, "micro": a.micro, "peak_gib": mc[-1]["peak_gib"], "longest_question_tokens": len(Dtr[longest]["q"]), "K": K, "aux": bool(SA and a.aux_step)},
                  open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return
    info = {**vars(a), "task": "prosqa_na", "train_items": len(Dtr), "test_items": len(Dte), "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
            "train_driver": driver_version(), **env_info(), "stage_len": L_STAGE if a.arm == "currdirect" else None}
    json.dump(info, open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    if SA and not a.resume:                                                 # note 147-2: g_add calibration at step 0 (a resume loads the calibrated s_add)
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
    opt = torch.optim.AdamW(params.parameters(), lr=a.lr, weight_decay=0.01); hist = []; n_nonfinite = 0; wg = []; gn_hist = []
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time(); step0 = 1
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); params.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); rng.bit_generator.state = ck["np_rng"]
        step0 = ck["step"] + 1; n_nonfinite = ck.get("n_nonfinite", 0); gn_hist = list(ck.get("gn_hist", [])); t0 = time.time() - ck["elapsed"]; log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    last_ = a.steps if not a.time_steps else min(a.steps, a.time_steps); tt0 = time.time()
    for step in range(step0, last_ + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        params.train()
        if SA:
            m.mem.fade.fill_(min(1.0, step / max(1, int(500 * sc))))
            m.mem.gamma.fill_(0.0 if step <= a.write_mix_start * sc else min(1.0, (step - a.write_mix_start * sc) / max(1.0, (a.write_mix_end - a.write_mix_start) * sc)))
        idx = rng.integers(0, len(Dtr), a.bs); opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro
        for k in range(a.micro):
            lk = loss_items([Dtr[i] for i in idx[k * ch:(k + 1) * ch]], step); (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
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
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps and not a.time_steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": params.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state, "step": step, "elapsed": time.time() - t0, "n_nonfinite": n_nonfinite, "gn_hist": gn_hist}, cp)
    if a.time_steps:
        sps = (time.time() - tt0) / max(1, last_); tj = {"arm": a.arm, "think_k": K, "steps_timed": last_, "sec_per_step": sps, "steps": a.steps,
                                                        "est_hours_steps": sps * a.steps / 3600, "micro": a.micro, "gpu": info["train_gpu"]}
        for f_ in ("timing.json", "eval.json"):
            json.dump(tj, open(os.path.join(a.out, f_), "w", newline="\n"), indent=1)
        print("TIMING", sps, flush=True); return
    train_sec = round(time.time() - t0); torch.save(params.state_dict(), os.path.join(a.out, "model.pt")); opt.zero_grad(set_to_none=True); opt.state.clear()
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
    res = {"arm": a.arm, "data": "na", "seed": a.seed, "steps": a.steps, "think_k": K, "acc": float((M > 0).mean()), "n": int(len(M)), "acc_by_depth": by_depth(M, dep),
           "n_by_depth": {str(d): int((dep == d).sum()) for d in range(3, 15) if (dep == d).any()}, "acc_ctl": float((Mc > 0).mean()), "n_ctl": int(len(Mc)),
           "train_sec": train_sec, "n_nonfinite": n_nonfinite, "micro": a.micro, "eval_dtype": str(amp), "gpu": info["train_gpu"], "train_gpu": info["train_gpu"],
           "acc_def": "candidate accuracy: log p('### ' + correct) > log p(the negative's) after the arm's prefix (stepaux: + the memory after K passes)"}
    if a.arm == "textcot":
        res["mean_cot_tokens"] = float(np.mean(cot_len)) if cot_len else None
    if SA:
        s = {"acc_by_depth_kprime": {str(kp): by_depth(sa_margins(Dte, kp), dep) for kp in KPRIMES}}
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
        s["aux_exact_by_hop"] = {str(t): float(np.mean(v)) for t, v in sorted(hop.items())}; s["aux_n_by_hop"] = {str(t): len(v) for t, v in sorted(hop.items())}
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
        m.eval(); s["delta_cos_at_init"] = dcos(states(ex, 2 * K)); del m; m = m_keep
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        donor = np.roll(np.arange(len(ex)), 1); S8 = states(ex, 8)[8][donor]  # PATH-INDEP (as §182): K' = 16 from 0 vs from another item's 8-pass memory
        M0 = sa_margins(ex, 16); M1 = sa_margins(ex, 16, S8)
        s["path_indep"] = {"answer_agree": float(((M0 > 0) == (M1 > 0)).mean()), "acc_from_zero": float((M0 > 0).mean()), "acc_from_donor": float((M1 > 0).mean()),
                           "n_items": len(ex), "kprime": 16, "donor_passes": 8, "derangement": "np.roll(sample order, 1)", "items": "test, rng 16802 permutation[:extras_n]"}
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
                         "rule": "halt at the first pass t (1..16) whose greedy aux step's object (last word) is EITHER option named in the question; cap 16 (descriptive; T-SHIFT's pass-token caveat applies)"}
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
        s["aux_acc_by_pass_ext"] = {str(t): float(np.mean(v)) for t, v in sorted(hx.items())}; s["n_by_pass_ext"] = {str(t): len(v) for t, v in sorted(hx.items())}
        from boxes.nested170 import logreg_acc                              # STEP-RECOVER: logreg M_t (delta state) -> hop t's node id
        tr = [Dtr[i] for i in np.random.default_rng(16803).permutation(len(Dtr))[:a.probe_n_fit]]; Str = states(tr, K); sr = {}
        for t in range(1, K + 1):
            fi = [i for i, it in enumerate(tr) if t <= it["depth"]]; ti = [i for i, it in enumerate(ex) if t <= it["depth"]]
            if len(fi) > 50 and ti:
                Xf = Str[t][fi][:, DM.H_ADD:].reshape(len(fi), -1); Xt = St[t][ti][:, DM.H_ADD:].reshape(len(ti), -1)
                VOC = {nm: c_ for c_, nm in enumerate(sorted({it["node"][t - 1] for it in tr + ex if t <= it["depth"]}))}
                sr[str(t)] = logreg_acc(Xf, np.array([VOC[tr[i]["node"][t - 1]] for i in fi]), Xt, np.array([VOC[ex[i]["node"][t - 1]] for i in ti]), dev)
        s["step_recover"] = sr; s["step_recover_def"] = "classes = concept NAMES (shared vocabulary); multinomial logreg (nested170.logreg_acc) from the flattened delta-head state after pass t to hop t's node id; fit on probe_n_fit train items"
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
                   "weights": list(AUX_W) if a.aux_reweight else None, "def": "clip_grad_norm_(..., 1.0)'s pre-clip total norm per step; clip fires when > 1.0"}
    if SA:
        res["s168s2"]["clip"] = res["clip"]
    res["eval_sec"] = round(time.time() - t1); json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({"acc": res["acc"], "acc_by_depth": res["acc_by_depth"]}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
