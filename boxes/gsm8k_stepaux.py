"""THEORY §200 GSM8K-AUG PILOT (58947b26; design docs/THEORY_GSM8K_TEMPLATE_DRAFT_2026-10-13.md §3; scorer theory/score_s200.py). GPT-2 small on
GSM8K-Aug (boxes/gsm8k_data.py: train filtered of wrong / unparsable equations; test = GSM8K's 1,319; valid = Aug valid 500). Two arms (--arm):
  stepaux : §191 A's recipe through prosqa168s2.ProsMem (DeltaMem-R 6 + 6 heads after block 6, top writes, gamma 2k -> 6k, W_o fade 500, g_add
            calibrated at step 0, fast_add): K think passes over [EOT, question] (--cache_lower: blocks 0..5 once per step); after pass t <= min(m, K)
            the read-only aux decoder [EOT, " Step", " t", " :"] + " <<e_t>>" + EOT (teacher-forced, pass token kept) reading M_t; the answer decoder
            [EOT, question, "\\n", "### " + answer, EOS] reading M_K (CE on "### " + answer + EOS). loss = answer CE + lambda aux (lambda 1 to the end).
  textcot : GPT-2 on question\\n + "<<e_t>>\\n" per equation + "### " + answer + EOS.
Answers (both arms): greedy after "###" (<= 12 tokens, stop at EOS or a newline token), scored by bench.tasks.norm_number exact match; the text arm
first generates its CoT greedily until "###" (no "###" within --max_new -> no answer).
Per draw (train only; test / valid never): --resample_numbers (R_x, then) --relabel_names (R_nu); every transformed question is asserted not to be a
test / valid question. Aug valid accuracy every --valid_every steps (the valid-best weights kept as model_validbest.pt; test never used for selection).
eval.json (score_s200 fields): acc, n, acc_by_m, acc_valid_best, valid_best_step, acc_test_validsel, fit {fit_fix, fit_rs}, struct {gap_mstrat,
seen_acc, unseen_acc}, filler {base, k16, k32}, dedupe {n_flagged, acc_excl_flagged}; stepaux adds acc_by_kprime, idle {m: {n, k_m2, k10}}, path_indep,
aux_exact_by_pass. symbolic / cost come from separate eval-only passes (--symbolic_dir; scratch eval_timing) and are null here.
§202 WARM-START (30401c78): --init_from <trunk state dict> (strict, into g before ProsMem(g); sha256 + path in args.json), --cot_valid (stepaux: the
plain trunk's TEXT-COT accuracy on Aug valid at each valid eval -> cot_valid_acc in log.jsonl), and eval.json aux_exact_by_grounding (stepaux).
    python -m boxes.gsm8k_stepaux --arm stepaux --think_k 10 --aux_fade 24000,24000 --cache_lower --fast_add --chunk_delta 64 --steps 24000 --out <dir>"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
from boxes.deltas5 import fwd_read, write_grad_norm  # noqa: E402
from boxes.fsq120 import driver_version, env_info, lr_at  # noqa: E402
from boxes.prosqa168s2 import ANS, ENC, EOS, EOT, ProsMem, pad  # noqa: E402
from bench.tasks import norm_number  # noqa: E402

KPRIMES = (0, 1, 2, 4, 6, 8, 10, 12, 16)
NL_TOKS = {t for t in range(ENC.n_vocab) if "\n" in ENC.decode([t])} if hasattr(ENC, "n_vocab") else set()
ANS_MAX = 12


def ans_ok(pred, gold):
    return pred is not None and norm_number(pred) == norm_number(gold)


@torch.no_grad()
def greedy_after(fwd, seqs, dev, ac, bs):
    """greedy continuation of each prefix (right-padded batches; each row read at its own last position, so padding never enters a real position's
    context), <= ANS_MAX tokens, stopping at EOS or a newline token -> decoded strings. fwd(x, rows) -> hidden (b, T, D)."""
    out = [None] * len(seqs)
    for i in range(0, len(seqs), bs):
        rows = list(range(i, min(i + bs, len(seqs)))); cur = [list(seqs[r]) for r in rows]; gen = [[] for _ in rows]; done = [False] * len(rows)
        for _ in range(ANS_MAX):
            ids_, _ = pad(cur); x = torch.tensor(ids_, device=dev); last = torch.tensor([len(c) - 1 for c in cur], device=dev)
            with ac():
                h = fwd(x, rows)
            nt = fwd.logits(h[torch.arange(len(cur), device=dev), last]).float().argmax(-1).tolist()
            for k_, t in enumerate(nt):
                if done[k_]:
                    continue
                if t == EOS or t in NL_TOKS:
                    done[k_] = True
                else:
                    gen[k_].append(t); cur[k_].append(t)
            if all(done):
                break
        for k_, r in enumerate(rows):
            out[r] = ENC.decode(gen[k_])
    return out


@torch.no_grad()
def cot_batched(g, pres, max_new, dev, amp, bs):
    """the text arm's CoT: KV-cached greedy over same-length prompts (prosqa168s2.generate_cot_batched's grouping) until "###" or EOS ->
    [(prefix + cot tokens, stopped_at_ans)]."""
    out = [None] * len(pres); by_len = {}
    for i, p in enumerate(pres):
        by_len.setdefault(len(p), []).append(i)
    for _, idx in sorted(by_len.items()):
        for s in range(0, len(idx), bs):
            ii = idx[s:s + bs]; x = torch.tensor([pres[i] for i in ii], device=dev); gen = []; done = torch.zeros(len(ii), dtype=torch.bool, device=dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                h, past = g(g.embed(x), None)
                for _ in range(max_new):
                    t = g.logits(h[:, -1]).float().argmax(-1); gen.append(t); done |= (t == ANS) | (t == EOS)
                    if bool(done.all()):
                        break
                    h, past = g(g.embed(t[:, None]), past)
            Gm = torch.stack(gen, 1).cpu().numpy()
            for r, i in enumerate(ii):
                row, hit = [], False
                for v in Gm[r]:
                    if int(v) == ANS:
                        hit = True; break
                    if int(v) == EOS:
                        break
                    row.append(int(v))
                out[i] = (list(pres[i]) + row, hit)
    return out


class Fwd:
    """a callable (x, rows) -> hidden with .logits, for greedy_after: the memory read (S rows) or the plain trunk."""
    def __init__(self, g, mem=None, S=None):
        self.g, self.mem, self.S, self.logits = g, mem, S, g.logits

    def __call__(self, x, rows):
        if self.mem is None:
            return self.g(self.g.embed(x), None)[0]
        return fwd_read(self.g, self.g.embed(x), self.mem, self.S[rows])


def with_prefix(it, pfx):
    """FILLER-N: the question with a fixed string prepended (re-encoded; eval only)."""
    raw = {"q": pfx + it["_q"], "eqs": it["eqs"], "ans": it["ans"]}; y = G.encode(raw, ENC, EOS); y["_q"] = raw["q"]; return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["stepaux", "textcot"])
    ap.add_argument("--cache_lower", action="store_true", help="stepaux: blocks 0..5 on the (identical) think input once per step, shared by the K passes (exact)")
    ap.add_argument("--ckpt_passes", action="store_true", help="stepaux: torch.utils.checkpoint around each think pass (memory)")
    ap.add_argument("--think_k", type=int, default=10); ap.add_argument("--aux_step", type=float, default=1.0); ap.add_argument("--aux_fade", default="24000,24000")
    ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--micro", type=int, default=1); ap.add_argument("--auto_micro", action="store_true")
    ap.add_argument("--mem_budget", type=float, default=21.0); ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--time_steps", type=int, default=0)
    ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000); ap.add_argument("--fast_add", action="store_true")
    ap.add_argument("--chunk_delta", type=int, default=0, help="stepaux: DM.CHUNK_DELTA (WY form; needs --fast_add)")
    ap.add_argument("--max_new", type=int, default=256, help="textcot: CoT tokens before '###'"); ap.add_argument("--cot_eval_bs", type=int, default=64)
    ap.add_argument("--ckpt_every", type=int, default=1000); ap.add_argument("--log_every", type=int, default=100); ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--valid_every", type=int, default=2000, help="Aug valid accuracy every N steps (and at the last step); the best weights -> model_validbest.pt")
    ap.add_argument("--relabel_names", action="store_true", help="§200 R_nu per draw (train only)"); ap.add_argument("--resample_numbers", action="store_true", help="§200 R_x per draw (train only)")
    ap.add_argument("--train_n", type=int, default=0, help="smokes"); ap.add_argument("--test_n", type=int, default=0, help="smokes"); ap.add_argument("--valid_n", type=int, default=0, help="smokes")
    ap.add_argument("--fit_n", type=int, default=1000); ap.add_argument("--extras_n", type=int, default=500); ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--sched_scale", type=float, default=1.0, help="smokes"); ap.add_argument("--symbolic_dir", default="", help="eval-only: GSM-Symbolic main / p1 / p2 jsonl dir (fetched at eval time, never committed)")
    ap.add_argument("--eval_only", action="store_true", help="evaluate <out>/model.pt (no training; args_eval.json)"); ap.add_argument("--eval_model", default="model.pt")
    ap.add_argument("--init_from", default="", help="§202: a GPT-2 trunk state dict (e.g. a textcot run's model.pt) loaded strictly into g before ProsMem(g)")
    ap.add_argument("--cot_valid", action="store_true", help="§202 stepaux: also the plain trunk's TEXT-COT accuracy on Aug valid at each valid eval (cot_valid_acc)")
    ap.add_argument("--aux_target", default="string", choices=["string", "value", "factor"], help="§204: the aux target (string = today's path; boxes/gsm8k_factor.py)")
    ap.add_argument("--aux_nopass", action="store_true", help="§204: the aux prompt [EOT, ' Step', ' :'] (no pass token) in training and every aux eval")
    ap.add_argument("--add_norm", default="", choices=["", "rms"], help="§204 F-NR: DeltaMem's additive reads RMS-normalised per head (the §185 path); '' = off")
    ap.add_argument("--eval_out", default="eval.json", help="§204: the eval file name (e.g. eval_s204ref.json for a comparator's eval-only re-run); its item files "
                    "then get the same stem (<stem>_items.json, <stem>_items_kprime.json, <stem>_args.json) so a landed run's files are never overwritten")
    ap.add_argument("--op_vocab", default="", help="§204 --aux_target factor: the frozen operator-signature list (default bench/data/gsm8k_aug/s204_op_vocab.json)")
    ap.add_argument("--resume", default=None); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    DM.FAST_ADD = bool(a.fast_add); DM.CHUNK_DELTA = int(a.chunk_delta); sc = a.sched_scale; SA = a.arm == "stepaux"; K = a.think_k if SA else 0
    f0_, f1_ = [float(v) for v in a.aux_fade.split(",")]
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else (lambda: torch.autocast("cpu", enabled=False))
    Rtr, fcnt = G.load_aug("train"); Rte, _ = G.load_aug("test", filter_bad=False); Rva, _ = G.load_aug("valid", filter_bad=False)
    assert len(Rte) == 1319 and len(Rva) == 500, (len(Rte), len(Rva))
    Rtr = Rtr[:a.train_n] if a.train_n else Rtr; Rte = Rte[:a.test_n] if a.test_n else Rte; Rva = Rva[:a.valid_n] if a.valid_n else Rva
    HOLD = {r["q"] for r in Rte + Rva}                                      # Audit: no transformed training question may equal a test / valid question
    enc = lambda r: {**G.encode(r, ENC, EOS), "_q": r["q"]}                 # noqa: E731
    Dte = [enc(r) for r in Rte]; Dva = [enc(r) for r in Rva]; meta = json.load(open(os.path.join(G.AUG, "s200_test_meta.json"), encoding="utf-8"))
    pools = G.load_names() if a.relabel_names else None; TPL = None; tr_n = {"draws": 0, "rs_tried": 0, "rs_accepted": 0, "rl_items": 0, "rl_names": 0}
    xr_rng = np.random.default_rng(a.seed + 20000)                          # the transforms' own rng (training batches unchanged by the flags)
    if a.resample_numbers:
        TPL = [G.parse_recomputable(r)[0] for r in Rtr]
    FT = SA and a.aux_target in ("value", "factor"); GF = None; OPV, opv_sha = [], None   # §204: factor / value targets (training items only when FT)
    if SA:
        import boxes.gsm8k_factor as GF
        OPV, opv_sha = GF.load_op_vocab(a.op_vocab or None)

    def with_ft(y, r):
        """§204: attach the per-step factor targets (computed per draw on the drawn - possibly transformed - raw item)."""
        y["ft"] = GF.factor_targets(r, G.parse_recomputable(r)[0], OPV); return y

    def draw(i):
        r = Rtr[i]; tr_n["draws"] += 1
        if TPL is not None and TPL[i] is not None:
            r, ok_ = G.resample(r, TPL[i], xr_rng, test_q=HOLD); tr_n["rs_tried"] += 1; tr_n["rs_accepted"] += int(ok_)
        if pools is not None:
            r, k_ = G.relabel_names(r, pools, xr_rng); tr_n["rl_items"] += int(k_ > 0); tr_n["rl_names"] += k_
        if TPL is not None or pools is not None:
            assert r["q"] not in HOLD, "§200: a transformed training question equals a test / valid question"
        return with_ft(enc(r), r) if FT else enc(r)
    g = s5.load_gpt2(dev); init_meta = None
    if a.init_from:                                                         # §202: before ProsMem(g), so DeltaMem's copies (block 6 ln_1 / c_attn,
        import hashlib                                                      # block 11 ln_1) come from the loaded trunk
        h_ = hashlib.sha256()
        with open(a.init_from, "rb") as f_:
            for b_ in iter(lambda: f_.read(1 << 24), b""):
                h_.update(b_)
        g.load_state_dict(torch.load(a.init_from, map_location=dev, weights_only=False), strict=True); init_meta = {"init_from": a.init_from, "init_sha256": h_.hexdigest()}
    NEW = SA and (a.aux_target != "string" or a.aux_nopass or bool(a.add_norm))   # §204: FactorMem only when a new flag is on (else the ProsMem call, unchanged)
    m = (GF.FactorMem(g, a.add_norm or None, a.aux_nopass, a.aux_target == "factor", a.seed).to(dev) if NEW else ProsMem(g).to(dev)) if SA else None; params = m if SA else g
    if a.eval_only and os.path.exists(os.path.join(a.out, "args.json")):   # §204 §8.2: a strict load needs the training run's add_norm / target / nopass
        a0_ = json.load(open(os.path.join(a.out, "args.json")))
        assert (a0_.get("add_norm") or "") == a.add_norm and (a0_.get("aux_target") or "string") == a.aux_target and bool(a0_.get("aux_nopass")) == a.aux_nopass, \
            "--eval_only: --add_norm / --aux_target / --aux_nopass must match the run's args.json"
    if SA:                                                                  # §204 §8.4: the eval items carry the factor targets (all stepaux arms, for the paired diagnostics)
        for y_, r_ in zip(Dte, Rte):
            with_ft(y_, r_)
        for y_, r_ in zip(Dva, Rva):
            with_ft(y_, r_)
    Lq = [len(v) for v in ENC.encode_ordinary_batch([r["q"] for r in Rtr], num_threads=4)]
    La = [len(v) for v in ENC.encode_ordinary_batch(["".join(" <<%s>>" % e for e in r["eqs"][:K]) if SA else "".join("<<%s>>\n" % e for e in r["eqs"]) for r in Rtr], num_threads=4)]
    if SA:                                                                  # the memory check's worst case: the longest question (K passes) carrying the
        iq, ie = int(np.argmax(Lq)), int(np.argmax(La))                     # deepest / longest aux targets (t <= K) - a composite upper bound
        it_long = {**enc(Rtr[iq]), **{k_: v_ for k_, v_ in enc(Rtr[ie]).items() if k_ in ("step_ids", "depth", "m", "eqs", "cot")}}
        if FT:                                                              # §204: the longest aux item's factor targets ride along
            with_ft(it_long, Rtr[ie])
    else:
        it_long = enc(Rtr[int(np.argmax([lq + la for lq, la in zip(Lq, La)]))])

    def lam(step):
        return 0.0 if not SA else (a.aux_step if step <= f0_ * sc else a.aux_step * max(0.0, (f1_ * sc - step) / max(1.0, (f1_ - f0_) * sc)))
    aux_hist = []; aux_terms = {}                                           # §204: per-term aux CE sums / counts between log records

    def loss_items(its, step):
        if SA:
            x, msk = m.think_inputs(its, dev); per = [[] for _ in its]; lam_ = lam(step)
            with ac():
                S = m.mem.zero_state(len(its), dev); hl = m.lower(x) if a.cache_lower else None
                for t in range(1, K + 1):
                    if a.ckpt_passes and torch.is_grad_enabled():
                        from torch.utils.checkpoint import checkpoint
                        S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False)
                    else:
                        S = m.write(S, x, msk, hl)
                    idx = [i for i, it in enumerate(its) if t <= it["depth"]]
                    if lam_ and idx and FT:                                 # §204: L_aux,t = the mean of the present factor / value terms
                        lt_, lg_t = GF.factor_loss(m, S[idx], [its[i] for i in idx], t, dev, a.aux_target)
                        for r_, i in enumerate(idx):
                            if lt_[r_] is not None:
                                per[i].append(lt_[r_])
                        for k_, (s_, n_) in lg_t.items():
                            aux_terms.setdefault(k_, [0.0, 0]); aux_terms[k_][0] += s_; aux_terms[k_][1] += n_
                    elif lam_ and idx:
                        lg_a, y_a = m.aux_logits(S[idx], [its[i] for i in idx], t, dev)
                        ce = Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100, reduction="none").sum(1) / (y_a != -100).sum(1)
                        for r_, i in enumerate(idx):
                            per[i].append(ce[r_])
                la = m.dec_loss(S, its, dev)
            if not lam_ or not any(per):
                aux_hist.append((float(la), None)); return la
            aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean(); aux_hist.append((float(la.detach()), float(aux.detach()))); return la + lam_ * aux
        seqs = [it["pre"] + it["cot"] + it["a1"] + [EOS] for it in its]; ids_, _ = pad(seqs); lab = np.full(ids_.shape, -100)
        for r, (q, it) in enumerate(zip(seqs, its)):
            n0 = len(it["pre"]); lab[r, n0:len(q)] = q[n0:]
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
        json.dump({"checks": mc, "micro": a.micro, "peak_gib": mc[-1]["peak_gib"], "longest_question_tokens": len(it_long["q"]), "longest_m": it_long["m"], "K": K,
                   "aux_tokens": sum(len(v_) for v_ in it_long["step_ids"][:K]) if SA else len(it_long["cot"]), "max_question_tokens": max(Lq), "max_aux_tokens": max(La),
                   "aux": bool(SA and a.aux_step)}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return
    info = {**vars(a), "task": "gsm8k_aug", "train_items": len(Rtr), "train_filter": fcnt, "test_items": len(Dte), "valid_items": len(Dva),
            "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info(),
            "recomputable_items": int(sum(t is not None for t in TPL)) if TPL is not None else None, **(init_meta or {}),
            **({"op_vocab_path": a.op_vocab or os.path.relpath(GF.OPV_PATH, ROOT).replace(os.sep, "/"), "op_vocab_sha256": opv_sha} if SA else {})}
    STEM = None if a.eval_out == "eval.json" else a.eval_out[:-5] if a.eval_out.endswith(".json") else a.eval_out
    assert STEM is None or a.eval_only, "--eval_out is for --eval_only re-runs"
    json.dump(info, open(os.path.join(a.out, (STEM + "_args.json") if STEM else "args_eval.json" if a.eval_only else (("args_resume_%d.json" % int(time.time())) if a.resume else "args.json")), "w", newline="\n"), indent=1)
    if SA and not a.resume and not a.eval_only:                             # note 147-2: g_add calibration at step 0
        its = [enc(Rtr[i]) for i in np.random.default_rng(16801).permutation(len(Rtr))[:32]]; x, msk = m.think_inputs(its, dev); m.mem.norms = []
        with torch.no_grad(), ac():
            m.think(m.mem.zero_state(len(its), dev), x, msk, min(K, 4))
        nr = m.mem.norms[1:]; m.mem.norms = None; r0 = float(torch.stack([d_ / a_.clamp_min(1e-12) for d_, a_ in nr]).median()) if nr else 1.0
        m.mem.s_add.data.fill_(math.log(max(r0, 1e-12))); json.dump({"g_add_init": r0}, open(os.path.join(a.out, "s200_init.json"), "w", newline="\n"))
    opt = torch.optim.AdamW(params.parameters(), lr=a.lr, weight_decay=0.01); hist = []; n_nonfinite = 0; wg = []; gn_hist = []; vhist = []
    log = open(os.devnull, "w") if a.eval_only else open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time(); step0 = 1
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); params.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); rng.bit_generator.state = ck["np_rng"]
        xr_rng.bit_generator.state = ck["xr_rng"]; tr_n.update(ck["tr_n"]); vhist = list(ck.get("vhist", []))
        step0 = ck["step"] + 1; n_nonfinite = ck.get("n_nonfinite", 0); gn_hist = list(ck.get("gn_hist", [])); t0 = time.time() - ck["elapsed"]
        log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    last_ = a.steps if not a.time_steps else min(a.steps, a.time_steps); tt0 = time.time()
    if a.eval_only:
        params.load_state_dict(torch.load(os.path.join(a.out, a.eval_model), map_location=dev, weights_only=False)); last_ = 0
    bs = a.eval_bs

    def think_states(D, Kp, M0=None, C=None):
        out = []
        for i in range(0, len(D), bs):
            ch = D[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            with torch.no_grad(), ac():
                S = m.mem.zero_state(len(ch), dev) if M0 is None else M0[i:i + bs].to(dev)
                S = m.think(S, x, msk, Kp, cache=a.cache_lower)
            out.append(S)
        return torch.cat(out)

    def answers(D, Kp=None, M0=None):
        """predicted answer strings: stepaux = greedy after [EOT, question, \\n, ###] reading M_Kp; textcot = greedy CoT to '###' then greedy."""
        if SA:
            S = think_states(D, K if Kp is None else Kp, M0)
            return greedy_after(Fwd(g, m.mem, S), [[EOT] + it["pre"] + [ANS] for it in D], dev, ac, bs)
        return cot_answers(D)

    def cot_answers(D):
        """the plain trunk g's TEXT-COT answers (the textcot arm's path; §202 --cot_valid reads it on a stepaux model's shared trunk)."""
        co = cot_batched(g, [it["pre"] for it in D], a.max_new, dev, amp, a.cot_eval_bs); out = [None] * len(D); hit = [i for i, (_, h_) in enumerate(co) if h_]
        if hit:
            pr = greedy_after(Fwd(g), [co[i][0] + [ANS] for i in hit], dev, ac, bs)
            for i, p_ in zip(hit, pr):
                out[i] = p_
        answers.cot_len = [len(c) - len(it["pre"]) for (c, _), it in zip(co, D)]
        return out

    def cot_valid_acc():
        params.eval(); v = acc_of(Dva, cot_answers(Dva)); params.train(); return v

    def acc_of(D, P):
        return float(np.mean([ans_ok(p_, it["ans"]) for p_, it in zip(P, D)])) if D else None

    def valid_acc():
        params.eval(); v = acc_of(Dva, answers(Dva)); params.train(); return v
    for step in range(step0, last_ + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        params.train()
        if SA:
            m.mem.fade.fill_(min(1.0, step / max(1, int(500 * sc))))
            m.mem.gamma.fill_(0.0 if step <= a.write_mix_start * sc else min(1.0, (step - a.write_mix_start * sc) / max(1.0, (a.write_mix_end - a.write_mix_start) * sc)))
        idx = rng.integers(0, len(Rtr), a.bs); opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro
        for k in range(a.micro):
            lk = loss_items([draw(i) for i in idx[k * ch:(k + 1) * ch]], step); (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
        if SA:
            wg.append(write_grad_norm(m))
        gn = torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0); gn_hist.append(float(gn))
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            n_nonfinite += 1; opt.zero_grad(); log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush()
        else:
            opt.step(); hist.append(tot)
        if step % a.log_every == 0 or step == last_:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "sec": round(time.time() - t0)}; hist = []
            if SA:
                rec.update({"aux_lambda": lam(step), "answer_ce": float(np.mean([x_[0] for x_ in aux_hist])) if aux_hist else None,
                            "aux_ce": float(np.mean([x_[1] for x_ in aux_hist if x_[1] is not None])) if any(x_[1] is not None for x_ in aux_hist) else None,
                            "gamma": float(m.mem.gamma), "fade": float(m.mem.fade), "write_grad_norm": float(np.mean(wg)) if wg else None}); aux_hist.clear(); wg = []
                if FT:                                                      # §204: the per-term aux CE (item-mean over the record's draws)
                    rec.update({"aux_%s_ce" % k_: (s_ / n_ if n_ else None) for k_, (s_, n_) in aux_terms.items()}); aux_terms.clear()
            if gn_hist:
                rec.update({"gn_preclip": gn_hist[-1], "clip_frac_100": float(np.mean([v_ > 1.0 for v_ in gn_hist[-100:]]))})
            if a.relabel_names or a.resample_numbers:
                rec.update({"tr_" + k_: v_ for k_, v_ in tr_n.items()})
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if a.valid_every and not a.time_steps and (step % a.valid_every == 0 or step == last_):
            tv = time.time(); va = valid_acc(); vhist.append((step, va)); r5 = {"step": step, "valid_acc": va, "valid_sec": round(time.time() - tv, 1)}
            if SA and a.cot_valid:                                          # §202 COT-RETAIN(t): descriptive, never used for selection
                tc = time.time(); r5.update({"cot_valid_acc": cot_valid_acc(), "cot_valid_sec": round(time.time() - tc, 1)})
            if va >= max(v for _, v in vhist):                               # ties: the later step
                torch.save(params.state_dict(), os.path.join(a.out, "model_validbest.pt")); r5["valid_best"] = True
            log.write(json.dumps(r5) + "\n"); log.flush(); print(r5, flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps and not a.time_steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": params.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state, "xr_rng": xr_rng.bit_generator.state, "tr_n": dict(tr_n),
                        "vhist": vhist, "step": step, "elapsed": time.time() - t0, "n_nonfinite": n_nonfinite, "gn_hist": gn_hist}, cp)
    if a.time_steps:
        sps = (time.time() - tt0) / max(1, last_ - step0 + 1); tj = {"arm": a.arm, "think_k": K, "steps_timed": last_ - step0 + 1, "sec_per_step": sps, "steps": a.steps,
                                                                     "est_hours_steps": sps * a.steps / 3600, "micro": a.micro, "gpu": info["train_gpu"]}
        for f_ in ("timing.json", "eval.json"):
            json.dump(tj, open(os.path.join(a.out, f_), "w", newline="\n"), indent=1)
        print("TIMING", sps, flush=True); return
    if a.eval_only:
        lg_ = [json.loads(l_) for l_ in open(os.path.join(a.out, "log.jsonl")) if '"sec"' in l_]; train_sec = lg_[-1]["sec"] if lg_ else None
        vhist = [(json.loads(l_)["step"], json.loads(l_)["valid_acc"]) for l_ in open(os.path.join(a.out, "log.jsonl")) if '"valid_acc"' in l_]
    else:
        train_sec = round(time.time() - t0); torch.save(params.state_dict(), os.path.join(a.out, "model.pt"))
    opt.zero_grad(set_to_none=True); opt.state.clear()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    params.eval(); t1 = time.time(); mt = np.array([it["m"] for it in Dte])
    res = {"arm": a.arm, "data": "gsm8k_aug", "seed": a.seed, "steps": a.steps, "think_k": K, "n": len(Dte), "train_sec": train_sec, "n_nonfinite": n_nonfinite,
           "micro": a.micro, "eval_dtype": str(amp), "train_gpu": info["train_gpu"], "train_driver": info["train_driver"],
           "eval_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "eval_only": bool(a.eval_only), "transforms": dict(tr_n),
           "acc_def": "greedy after '###' (<= %d tokens, stop at EOS / newline), bench.tasks.norm_number exact match; the text arm generates its CoT to '###' first" % ANS_MAX,
           "symbolic": None, "cost": None}
    EVF = a.eval_out

    def _partial(tag):
        res["eval_partial_through"] = tag; res["eval_partial"] = True; json.dump(res, open(os.path.join(a.out, EVF), "w", newline="\n"), indent=1)
    P = answers(Dte); C = np.array([ans_ok(p_, it["ans"]) for p_, it in zip(P, Dte)]); res["acc"] = float(C.mean())
    res["acc_by_m"] = {str(k_): float(C[mt == k_].mean()) for k_ in sorted(set(mt.tolist()))}; res["n_by_m"] = {str(k_): int((mt == k_).sum()) for k_ in sorted(set(mt.tolist()))}
    if not SA:
        res["mean_cot_tokens"] = float(np.mean(answers.cot_len))
    json.dump([{"i": i, "pred": p_, "gold": it["ans"], "ok": bool(c_)} for i, (p_, it, c_) in enumerate(zip(P, Dte, C))],
              open(os.path.join(a.out, (STEM + "_items.json") if STEM else "eval_items.json"), "w", newline="\n"), indent=0); _partial("acc")
    if vhist:
        bst = max(vhist, key=lambda sv: (sv[1], sv[0])); res.update({"acc_valid_best": bst[1], "valid_best_step": bst[0], "valid_hist": vhist})
        vb = os.path.join(a.out, "model_validbest.pt")
        if os.path.exists(vb) and bst[0] != a.steps:
            keep = {k_: v_.clone() for k_, v_ in params.state_dict().items()}; params.load_state_dict(torch.load(vb, map_location=dev, weights_only=False)); params.eval()
            res["acc_test_validsel"] = acc_of(Dte, answers(Dte)); params.load_state_dict(keep); params.eval(); del keep
        else:
            res["acc_test_validsel"] = res["acc"]
        _partial("validsel")
    if not a.test_n:                                                        # meta is indexed on the full 1,319
        st = np.array(meta["struct"]); seen, unseen = (st == "seen"), (st == "unseen"); num = den = 0.0
        for k_ in sorted(set(mt.tolist())):
            s_, u_ = C[seen & (mt == k_)], C[unseen & (mt == k_)]
            if len(s_) and len(u_):
                w_ = len(s_) * len(u_) / (len(s_) + len(u_)); num += w_ * (s_.mean() - u_.mean()); den += w_
        res["struct"] = {"gap_mstrat": float(num / den) if den else None, "seen_acc": float(C[seen].mean()), "unseen_acc": float(C[unseen].mean()),
                         "n_seen": int(seen.sum()), "n_unseen": int(unseen.sum()), "n_nokey": int((st == "nokey").sum()), "def": meta["def"]["struct"]}
        fl = np.array(meta["dedupe_flag"]); res["dedupe"] = {"n_flagged": int(fl.sum()), "acc_excl_flagged": float(C[~fl].mean()), "def": meta["def"]["dedupe"]}
        _partial("struct_dedupe")
    res["filler"] = {"base": res["acc"], "def": meta["def"]["filler"]}
    for k_, s_ in meta["filler"].items():                                   # FILLER-N: the fixed string prepended to every test question
        Df_ = [with_prefix(it, s_) for it in Dte]; res["filler"][k_] = acc_of(Df_, answers(Df_))
    _partial("filler")
    fr = np.random.default_rng(20010); fit_items = [Rtr[i] for i in fr.permutation(len(Rtr))[:a.fit_n]]; Dfit = [enc(r) for r in fit_items]
    fit = {"fit_fix": acc_of(Dfit, answers(Dfit)), "n_fix": len(Dfit)}
    rr = np.random.default_rng(20011); Drs = []
    for r in fit_items:
        tp, _ = G.parse_recomputable(r)
        if tp is not None:
            y, ok_ = G.resample(r, tp, rr, test_q=HOLD)
            if ok_:
                Drs.append(enc(y))
    fit.update({"fit_rs": acc_of(Drs, answers(Drs)), "n_rs": len(Drs), "def": "fit_fix: %d train items (rng 20010) as drawn; fit_rs: one fresh R_x draw (rng 20011) of "
                "each recomputable one, accepted draws only" % len(Dfit)})
    res["fit"] = fit; _partial("fit")
    if SA:
        Ck = {kp: (C if kp == K else np.array([ans_ok(p_, it["ans"]) for p_, it in zip(answers(Dte, kp), Dte)])) for kp in KPRIMES}
        res["acc_by_kprime"] = {str(kp): (res["acc"] if kp == K else float(np.mean(Ck[kp].tolist()))) for kp in KPRIMES}; _partial("acc_by_kprime")
        res["acc_by_m_kprime"] = {str(kp): {str(k_): float(Ck[kp][mt == k_].mean()) for k_ in sorted(set(mt.tolist()))} for kp in KPRIMES}   # §204 §8.4 / note 202-1
        json.dump([{"i": i, "m": int(mt[i]), **{"k%d" % kp: bool(Ck[kp][i]) for kp in KPRIMES}} for i in range(len(Dte))],
                  open(os.path.join(a.out, (STEM + "_items_kprime.json") if STEM else "eval_items_kprime.json"), "w", newline="\n"), indent=0)
        idle = {}
        for k_ in sorted(set(mt.tolist())):
            Dm = [it for it in Dte if it["m"] == k_]
            idle[str(k_)] = {"n": len(Dm), "k_m2": acc_of(Dm, answers(Dm, k_ + 2)), "k10": float(C[mt == k_].mean())}
        res["idle"] = idle; _partial("idle")
        ex_i = np.random.default_rng(16802).permutation(len(Dte))[:a.extras_n]; ex = [Dte[i] for i in ex_i]
        hop = {}; hop_gr = {}; grd = []                                      # §202 P-W8: step t's operand grounding from gsm8k_data.parse_recomputable:
        for i in ex_i:                                                      # "R" = any operand is a previous result, "Q" = question literals /
            tp_, _ = G.parse_recomputable(Rte[i])                           # constants only, "unparsed" = the item is not recomputable (no labels)
            grd.append([("R" if any(o_[3] == "r" for o_ in e_["ops"]) else "Q") for e_ in tp_["eqt"]] if tp_ is not None else None)
        for i in range(0, len(ex), bs):
            ch = ex[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            with torch.no_grad(), ac():
                S = m.mem.zero_state(len(ch), dev)
                for t in range(1, K + 1):
                    S = m.write(S, x, msk); ix = [j for j, it in enumerate(ch) if t <= it["depth"]]
                    if ix:
                        gen = m.aux_greedy(S[ix], t, max(len(ch[j]["step_ids"][t - 1]) + 1 for j in ix))
                        for r_, j in enumerate(ix):
                            tg_ = ch[j]["step_ids"][t - 1] + [EOT]; ok_ = bool(list(gen[r_, :len(tg_)]) == tg_); hop.setdefault(t, []).append(ok_)
                            gl_ = grd[i + j]; hop_gr.setdefault(gl_[t - 1] if gl_ is not None else "unparsed", {}).setdefault(t, []).append(ok_)
        res["aux_exact_by_pass"] = {str(t): float(np.mean(v)) for t, v in sorted(hop.items())}; res["aux_n_by_pass"] = {str(t): len(v) for t, v in sorted(hop.items())}
        res["aux_exact_by_grounding"] = {c_: {"acc": float(np.mean([x_ for v in d_.values() for x_ in v])), "n": sum(len(v) for v in d_.values()),
                                              "by_pass": {str(t): float(np.mean(v)) for t, v in sorted(d_.items())}, "n_by_pass": {str(t): len(v) for t, v in sorted(d_.items())}}
                                         for c_, d_ in sorted(hop_gr.items())}
        res["aux_exact_by_grounding"]["def"] = ("§202 P-W8: the aux_exact_by_pass items split by step t's operand grounding (gsm8k_data.parse_recomputable on the test raw item): "
                                                "R = any operand is a previous result, Q = question literals / constants only, unparsed = item not recomputable")
        S8 = think_states(ex, 8); donor = torch.tensor(np.roll(np.arange(len(ex)), 1)); S0 = think_states(ex, 16); S1 = think_states(ex, 16, S8[donor].cpu())
        P0 = greedy_after(Fwd(g, m.mem, S0), [[EOT] + it["pre"] + [ANS] for it in ex], dev, ac, bs); P1 = greedy_after(Fwd(g, m.mem, S1), [[EOT] + it["pre"] + [ANS] for it in ex], dev, ac, bs)
        hd = slice(m.mem.h_add, m.mem.h_add + m.mem.h_del)
        res["path_indep"] = {"answer_agree": float(np.mean([norm_number(p_ or "") == norm_number(q_ or "") for p_, q_ in zip(P0, P1)])), "acc_from_zero": acc_of(ex, P0),
                             "acc_from_donor": acc_of(ex, P1), "delta_cos_final": float(Fn.cosine_similarity(S0[:, hd].float().flatten(1), S1[:, hd].float().flatten(1), dim=1).mean()),
                             "n_items": len(ex), "kprime": 16, "donor_passes": 8, "derangement": "np.roll(sample order, 1)", "items": "test, rng 16802 permutation[:extras_n]"}
        _partial("path_indep")
        res["aux_suite"] = {"test_extras": GF.aux_suite(m, ex, lambda its_, k_: think_states(its_, k_), K, dev, a.aux_target, OPV, bs=bs, lotus=True),   # §204 §8.4
                            "aug_valid": GF.aux_suite(m, Dva, lambda its_, k_: think_states(its_, k_), K, dev, a.aux_target, OPV, bs=bs),
                            "items": {"test_extras": "test, rng 16802 permutation[:extras_n] (n %d)" % len(ex), "aug_valid": "Aug valid, all (n %d)" % len(Dva)},
                            "arm": a.aux_target, "nopass": bool(a.aux_nopass), "tmax": 8,
                            "def": "§204 §8.4: per source (Mt / M1 (pass-token arms) / M0 / COUNTER-SWAP cs_*), by pass t <= min(m, 8, K): value_exact (value / factor arms; "
                                   "greedy value after the aux prompt, decoded and compared with the gold RHS as a STRING (note 204-3); value_exact_tokid = token-id "
                                   "equality), op_acc / ptr1_acc / ptr2_acc (factor heads), or the string arm's value_in_string / op_in_string / ptr1_in_string from its "
                                   "greedy ' <<lhs=rhs>>'; ptr*_nocollide = items whose candidate set has no first-BPE-token collision; *_by_grounding = R / Q / unparsed"}
        res["aux_suite"]["lotus_sep"] = res["aux_suite"]["test_extras"].pop("lotus_sep")   # note 204-5: aux_suite["lotus_sep"] (test extras)
        _partial("aux_suite")
    if a.symbolic_dir:                                                      # eval-only: GSM-Symbolic (fetched at eval time; never committed)
        sy = {}
        for nm in ("main", "p1", "p2"):
            fp_ = os.path.join(a.symbolic_dir, nm + ".jsonl")
            if os.path.exists(fp_):
                X = [json.loads(l_) for l_ in open(fp_, encoding="utf-8") if l_.strip()]
                Ds = [{**G.encode({"q": x["question"], "eqs": [], "ans": x["answer"].split("####")[-1].strip()}, ENC, EOS), "_q": x["question"], "tid": x.get("id")} for x in X]
                Cs = np.array([ans_ok(p_, it["ans"]) for p_, it in zip(answers(Ds), Ds)]); by_t = {}
                for it, c_ in zip(Ds, Cs):
                    by_t.setdefault(it["tid"], []).append(c_)
                sy[nm] = {"acc": float(Cs.mean()), "n": len(Ds), "sd_over_templates": float(np.std([np.mean(v) for v in by_t.values()])) if by_t else None}
        res["symbolic"] = sy; _partial("symbolic")
    g_ = np.array([v for v in gn_hist if math.isfinite(v)]) if gn_hist else np.zeros(0)
    res["clip"] = {"clip_rate": float((g_ > 1.0).mean()) if len(g_) else None, "n_steps": int(len(g_))}
    res.pop("eval_partial_through", None); res["eval_partial"] = False; res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, EVF), "w", newline="\n"), indent=1)
    if not a.eval_only:
        for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
            if os.path.exists(os.path.join(a.out, ck_)):
                os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({"acc": res["acc"], "acc_by_m": res["acc_by_m"]}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
