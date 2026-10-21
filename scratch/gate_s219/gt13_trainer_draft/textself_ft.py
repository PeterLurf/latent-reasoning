"""§219 TEXT-SELF trainer (TS_v1.md at 146a25d0: §2.5, §2.6 T1, V1, T3): fine-tuning a §170 TEXT-COT GPT-2 on the matched self-training mixture.
boxes/nested170.py is imported read-only and never edited (sha256 pinned, G-T7). Its parser is copied verbatim (the 57 add_argument calls of
nested170.main, l.635-658; asserted by AST in scratch/gate_s219.py) plus the FT keys; its textcot loss / lr schedule (fsq120.lr_at + the warm-up branch
of l.1039) / AdamW wd 0.01 (fsq120.make_opt) / clip 1.0 / bf16 autocast / checkpointing are copied.
  plain stream: rng = default_rng(seed); every step draws nested170.train_items(rng, bs, n_max_train, EXCL) first (EXCL = test_set(per_n) n >= excl_min_n
    + SERIAL, the base's law), so each round replays the base's first `steps` plain batches.
  coin stream (--self_pool): rng_s = default_rng(seed + 21800); per step coin = rng_s.random() < p_self; on a self step the next bs pool records (cursor)
    replace the plain batch and bs scalar rng_s.integers(0, 5) are drawn and discarded (vbank.draw_step's self-U draws, self_kextra 4), so the self
    steps fall at the latent rounds' step indices (PROOFS P4).
  self item target: textself.compose_T(syms, values) (shape + the labeller's values; no root key exists); loss = the base's textcot CE (prompt masked)
    on every target token. self_value_ce / self_copy_ce are logged from the same logits (no gradient).
  micro: plain --micro (the base's auto_micro, train part only), self --self_micro fixed (T1 2, T2 4; never changed).
  seal (T3): --no_final_eval is REQUIRED and this module has no generation / eval code path at all: no cot_answers, no eval*.json; --stop_at returns
    before saving model.pt; --time_steps writes timing.json only; --mem_check / --mem_check_self write mem_check*.json only.
Ambiguities resolved here (TS_v1 silent; nearest precedent followed):
  A1 --no_final_eval is asserted true rather than switching an eval off (there is no eval code to switch); the plain auto-micro check keeps nested170's
     train part (longest of 4 x bs draws, forward + backward, + the AdamW estimate) and drops its eval / eval-memory generation (V3).
  A2 FT-key defaults: init_from "", self_pool "", p_self 0.0, self_micro 2 (vbank's), no_final_eval false (store_true; required at run time),
     mem_check_self false, mem_check_steps 2 (G-T9's two optimiser steps), stop_at 0, draws_log_steps 0, args_ref "".
  A3 Features the textcot path does not train (memory arms, anneal, csr, iit, grokfast, aux, k_window, probe, eval_at, eval_only, kprimes, path_indep,
     fast_add) are asserted at their TEXT_NEUTRAL_NEW / off values; scratch and warmup are honoured exactly as nested170 (make_trunk, l.1039).
  A4 --init_from: strict load; the sha256 is printed and written to init_from.json (not args.json: a new args key would read ARGS_BAD); asserted ==
     TS_v1 V6's pin when the path is s170_cot_s1/model.pt; T2's pin (T1's model.pt sha) is the stage script's / scorer's check.
  A5 --keep_ckpt_steps accepts 0 = a model-only copy right after init (vbank's convention; G-T4's step-0 check).
  A6 draws_log record = {step, plain_hash = sha1(json.dumps([syms of the 64 plain items])), coin, n (plain) | self_n + U (self)}; the latent's U values
     are logged so G-T3 can compare every U as PROOFS P4 did.
  A7 --time_steps (G-T10) alternates self (odd steps) and plain (even) and writes timing.json only. Note 219-4 (3): the forced self batch is the 64
     LONGEST records of the round's deepest n (T1 n 12, T2 n 16) at the round's fixed self micro, every odd step (deep_self, the same batch as G-T9),
     so the projection is the worst self shape and never an under-projection from pool order. Timing only:
     (never eval.json: SEAL-BREACH reads any eval.json in a cell dir); the stage scripts must use timing.json / mem_check_self.json / train_done.json as
     their done markers.
  A8 TRAIN-TOKENS (Addendum 2 item 2): cumulative per type in every log record and train_done.json: target tokens, token-positions (prompt + target,
     unpadded) and padded positions (rows x batch-max length).
  A10 Each log record also carries loss_last (that step's own loss; G-T4 compares the resumed step 3 at the registered --log_every 100).
  A9 A finished run writes train_done.json {steps, train_sec, n_nonfinite, n_self_steps, pool_cursor, self_reuse, train_tokens, self_n_counts} after
     model.pt and deletes ckpt_last / ckpt_prev (vbank's order); off-host mirroring is the stage script's job (G-T11).
    python -m boxes.textself_ft --arm textcot --init_from boxes/runs_fsq/s170_cot_s1/model.pt --steps 12000 --lr 3e-5 --self_pool <pool> --p_self 0.5
      --self_micro 2 --no_final_eval --draws_log_steps 12000 --auto_micro --args_ref boxes/runs_fsq/s170_cot_s1/args.json --out boxes/runs_fsq/s219_t1_s1"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.textself as TS  # noqa: E402

EOT = N.EOT
TID = N.TID
FT_KEYS = ("init_from", "self_pool", "p_self", "self_micro", "no_final_eval", "mem_check_self", "mem_check_steps", "stop_at", "draws_log_steps", "args_ref")
INERT = {"n_slots": 8, "anneal": "", "probe_loss_every": 0, "mem_layer": 6, "cache_lower": False, "anneal_gate": "", "csr": "", "iit": "", "iit_tau": 1,
         "iit_nmax": 0, "grokfast": "", "eval_only": False, "aux_step": 0.0, "kmatch_impl": "masked", "add_norm": "", "aux_nopass": False,
         "aux_reweight": False, "aux_hold": False, "kprimes": "", "path_indep": False, "k_window": "", "fast_add": False, "eval_at": 0, "think_k": 0}
TEXTCOT_FORMAT = "[EOT, expr, ' ='] then ' e2 = e3 = ... = t' EOT (Feng; loss on all generated tokens)"


def build_parser():
    # ---- verbatim copy of boxes/nested170.py main() l.635-658 (sha256 1961bd36...; asserted by AST in scratch/gate_s219.py) ----
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=["direct", "inject", "textcot", "readslots"]); ap.add_argument("--n_slots", type=int, default=8); ap.add_argument("--think_k", type=int, default=0)
    ap.add_argument("--steps", type=int, default=48000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_max_train", type=int, default=8); ap.add_argument("--micro", type=int, default=1); ap.add_argument("--ckpt_passes", action="store_true")
    ap.add_argument("--mem_write", default="top", choices=["top"]); ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000)
    ap.add_argument("--anneal", default="", help="note 196-1 BUDGET-ANNEAL: 'nmax:K:until,...' stages (n_max_train, think_k, last step); each continues the previous "
                    "stage's weights / optimiser; the last stage must equal --n_max_train / --think_k / --steps ('' = off)")
    ap.add_argument("--probe_loss_every", type=int, default=0, help="note 196-1: every N steps, the mean answer / aux CE on 256 FIXED full-task items (n <= --n_max_train, "
                    "K = --think_k; own rng seed + 19700; no_grad, eval mode; logged as probe_loss / probe_aux; 0 = off)")
    ap.add_argument("--mem_layer", type=int, default=6, help="§198 MEM-HIGH: the block the memory branch enters (pretrained trunk; default 6 = §170); --cache_lower caches 0..mem_layer-1")
    ap.add_argument("--cache_lower", action="store_true", help="training only: blocks 0..mem.layer-1 on the (identical) think input once per batch, shared by the K "
                    "passes (exact in exact arithmetic; gate_cache_lower_n170). Held: never compare a cached cell with an uncached comparator (Theory)")
    ap.add_argument("--anneal_gate", default="", help="note 196-1 entry-gated --anneal: 'thr:need:max_steps[:bin]'; --anneal stages 'nmax:K:G' (gated) and a final "
                    "'nmax:K:final_len'; --steps must equal the cap (#gated x max_steps + final_len); LR = the cosine over that cap; '' = fixed-step --anneal")
    ap.add_argument("--csr", default="", help="§199 CSR (Theory 199-A): 'w,lam_min,lam_post,tau,start' - the contraction-schedule regulariser on the aux path's per-pass read R_t, active from step start ('' = off)")
    ap.add_argument("--keep_ckpt_steps", default="", help="§199: comma steps at which a model-only copy ckpt_model_<step>.pt is kept (for the local path_indep probe)")
    ap.add_argument("--iit", default="", help="IIT-SERIAL (boxes/iit_serial.py): 'w_I,w_S,B_I' - the gamma-weighted interchange (IIT) + same-state (SIIT) terms, "
                    "one separate graph after the main backward, only while gamma(step) > 0 ('' = off)")
    ap.add_argument("--iit_tau", type=int, default=1, help="IIT-SERIAL: the swap pass tau"); ap.add_argument("--iit_nmax", type=int, default=0, help="IIT-SERIAL: pairs only from items n <= nmax (0 = all)")
    ap.add_argument("--grokfast", default="", choices=["", "ema"], help="§196: Grokfast EMA gradient filter (grokfast_ema) after backward, before the clip; '' = off")
    ap.add_argument("--gf_alpha", type=float, default=0.98); ap.add_argument("--gf_lamb", type=float, default=2.0)
    ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", default=None); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--cot_eval_bs", type=int, default=64, help="TEXT-COT generation batch (KV cache; b223 OOMed at 250 on n = 16 chains)"); ap.add_argument("--eval_only", action="store_true", help="evaluate <out>/model.pt (no training; writes args_eval.json; model.pt / args.json / log untouched)"); ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--per_n", type=int, default=1000); ap.add_argument("--serial_pool", type=int, default=2200); ap.add_argument("--auto_micro", action="store_true")
    ap.add_argument("--mem_budget", type=float, default=21.0); ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--time_steps", type=int, default=0, help="cost check: train this many steps, report s/step, exit")
    ap.add_argument("--eval_at", type=int, default=0, help="also evaluate acc_by_n at this step -> eval_<k>k.json"); ap.add_argument("--aux_step", type=float, default=0.0, help="§175: lambda of the read-only step decoder CE (0 = off: the §170 path)"); ap.add_argument("--aux_fade", default="36000,48000", help="§175: lambda constant to the first step, then linear to 0 at the second"); ap.add_argument("--kmatch_impl", default="masked", choices=["masked", "bucketed"], help="§177 (f): note 177-5's masked single batch, or the per-K groups"); ap.add_argument("--add_norm", default="", choices=["", "den", "rms"], help="§185 ADD-NORM: the additive reads normalised by phi(q)^T z (den) or per-head RMS (rms); '' = off"); ap.add_argument("--aux_nopass", action="store_true", help="§187 AUX-NOPASS: the aux prompt [EOT, ' Step', ' :'] (no pass token) in training and every aux eval"); ap.add_argument("--aux_reweight", action="store_true", help="§184: the aux CE of pass t <= n weighted by AUX_W[t-1] (equal expected weight per pass position)"); ap.add_argument("--aux_hold", action="store_true", help="§182 HOLD: aux target e_{n+1} on passes n < t <= K (with --k_window and --aux_step)"); ap.add_argument("--kprimes", default="", help="§182: the K' set for s170.acc_by_n_kprime ('' = {0, 1, K/2, K, 2K})"); ap.add_argument("--path_indep", action="store_true", help="§182: s182.path_indep (K' = 16 from M_0 = 0 vs another item's 8-pass memory)"); ap.add_argument("--excl_min_n", type=int, default=1, help="note 178-3 (s178b): exclude test items from training only for n >= this (1 = §170's protocol)"); ap.add_argument("--scratch", default="", help="§178: 'L,D,H' -> a RANDOM-INIT GPT-2 trunk (no pretrained weights), memory after block L // 2 with D / 128 + D / 128 heads"); ap.add_argument("--warmup", type=int, default=100, help="LR warm-up steps (§178: 1000)"); ap.add_argument("--k_window", default="", help="§177 (f): 'lo,hi' -> each n-bucketed micro-batch runs K = n + U{lo..hi} think passes (inject)"); ap.add_argument("--fast_add", action="store_true", help="deltamem.FAST_ADD: the additive heads' pass writes as one sum (Audit 2026-10-09; scratch/gate_fastadd.py)"); ap.add_argument("--sched_scale", type=float, default=1.0, help="smokes only"); ap.add_argument("--fp32", action="store_true"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    # ---- end of the verbatim copy; the §219 FT keys follow ----
    ap.add_argument("--init_from", default="", help="§219: the base model.pt (strict load into the GPT-2 trunk, sha256 printed); a fresh AdamW")
    ap.add_argument("--self_pool", default="", help="§219: the round's pool jsonl (pool_meta.json beside it; sha + key set asserted); empty = no self steps")
    ap.add_argument("--p_self", type=float, default=0.0); ap.add_argument("--self_micro", type=int, default=2, help="§219: micro on self steps (fixed: T1 2, T2 4)")
    ap.add_argument("--no_final_eval", action="store_true", help="§219 T3 (seal): REQUIRED; the trainer has no generation / eval path")
    ap.add_argument("--mem_check_self", action="store_true", help="§219 G-T9: the 64 longest pool records of the deepest n, --mem_check_steps optimiser steps "
                    "at --self_micro, peak incl. AdamW -> mem_check_self.json, exit")
    ap.add_argument("--mem_check_steps", type=int, default=2); ap.add_argument("--stop_at", type=int, default=0, help="smokes: stop after this step (no model.pt)")
    ap.add_argument("--draws_log_steps", type=int, default=0, help="§219 G-T3: steps 1..N logged to draws_log.json (registered: 12000); 0 = off")
    ap.add_argument("--args_ref", default="", help="an args.json to diff against (printed as ARGS-DIFF, written to args_diff.json)")
    return ap


def plain_hash(its):
    return hashlib.sha1(json.dumps([it["syms"] for it in its]).encode()).hexdigest()


def atomic_json(obj, path, **kw):
    """json to path + '.tmp', then os.replace: a crash leaves the previous complete file (G-T13 item 4)."""
    with open(path + ".tmp", "w", newline="\n") as fh:
        json.dump(obj, fh, **kw)
    os.replace(path + ".tmp", path)


def deep_self(pool, bs):
    """the deepest self shape (G-T9 and, Note 219-4 (3), G-T10): the bs longest records of the pool's deepest n. -> (records, n_top)."""
    n_top = max(r_["n"] for r_ in pool)                                     # once (O(N); b310's lesson)
    return sorted([r_ for r_ in pool if r_["n"] == n_top], key=lambda r_: -len(r_["syms"]))[:bs], n_top


def coin_step(rng_s, p_self, bs):
    """one step of the coin stream (vbank.draw_step's consumption): -> (coin, the bs discarded self-U draws or None)."""
    coin = bool(rng_s.random() < p_self)
    return coin, ([int(rng_s.integers(0, 5)) for _ in range(bs)] if coin else None)


def make_loss(g, dev, ac):
    """the base's textcot CE (nested170 l.697-704 verbatim for plain items); a self item (key "values", no root) takes compose_T as its target.
    st (dict or None) collects the value / copy CE split and TRAIN-TOKENS counters (no gradient)."""
    def loss_items(items, st=None):
        seqs, lab, vpos = [], [], []
        for it in items:
            pre = [EOT] + N.ids(it["syms"]) + [TID[" ="]]
            if "values" in it:                                              # §219 self item: shape + the labeller's values; never N.cot_ids
                tgt, vp = TS.compose_T(it["syms"], it["values"], slots=True); vpos.append([len(pre) + q for q in vp])
            else:
                tgt = N.cot_ids(it["root"])
            seqs.append(pre + tgt); lab.append([-100] * len(pre) + tgt)
        ids_, _ = N.pad(seqs); lb, _ = N.pad(lab, -100); x = torch.tensor(ids_, device=dev); y = torch.tensor(lb, device=dev)
        with ac():
            h, _ = g(g.embed(x), None)
        lg = g.logits(h[:, :-1]).float(); L = Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), y[:, 1:].reshape(-1), ignore_index=-100)
        if st is not None:
            with torch.no_grad():
                ntok = int((y[:, 1:] != -100).sum()); kind = "self" if vpos else "plain"; tk = st.setdefault("tok", {})
                c_ = tk.setdefault(kind, [0, 0, 0]); c_[0] += sum(len(q) - lab_.count(-100) for q, lab_ in zip(seqs, lab)); c_[1] += sum(len(q) for q in seqs); c_[2] += int(x.numel())
                if vpos:
                    rr = torch.tensor([r for r, vp in enumerate(vpos) for _ in vp], device=dev); cc = torch.tensor([q for vp in vpos for q in vp], device=dev)
                    vs = float(Fn.cross_entropy(lg[rr, cc - 1], y[rr, cc], reduction="sum")); nv = int(rr.numel())
                    st["value_sum"] = st.get("value_sum", 0.0) + vs; st["n_value"] = st.get("n_value", 0) + nv
                    st["copy_sum"] = st.get("copy_sum", 0.0) + float(L) * ntok - vs; st["n_copy"] = st.get("n_copy", 0) + ntok - nv
        return L
    return loss_items


def main(argv=None):
    ap = build_parser(); a = ap.parse_args(argv); a.task = "nested"
    assert a.arm == "textcot", "§219: --arm textcot only"
    assert a.no_final_eval, "§219 T3 (seal): --no_final_eval is required (this trainer has no eval path)"
    for k_, v_ in INERT.items():
        assert getattr(a, k_) == v_, ("§219: --%s is not trained on the text path; it must stay %r" % (k_, v_), getattr(a, k_))
    assert a.bs % a.micro == 0 and a.bs % a.self_micro == 0 and 0.0 <= a.p_self <= 1.0
    if a.arm not in ("inject", "readslots"):
        a.mem_write = a.write_mix_start = a.write_mix_end = None; a.think_k = 0
    os.makedirs(a.out, exist_ok=True); dev = a.device
    from boxes.fsq120 import make_opt, lr_at, driver_version, env_info
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    TE = N.test_set(a.per_n); EXCL = {tuple(it["syms"]) for it in TE if it["n"] >= a.excl_min_n}; g = N.make_trunk(vars(a), dev); del TE
    r_sp = np.random.default_rng(17300)                                     # note 170-1: the dedicated n = 8 SERIAL pool (excluded from training)
    for _ in range(a.serial_pool):
        r__, t__ = N.generate(r_sp, 8); EXCL.add(tuple(N.render(r__)))
    r_ex = np.random.default_rng(17899)                                     # note 178-3 (nested170 l.675-676, own stream; args only)
    a.excl_survive_by_n = {str(n_): round(float(np.mean([tuple(N.render(N.generate(r_ex, n_)[0])) not in EXCL for _ in range(2000)])), 4) for n_ in range(1, a.n_max_train + 1)}
    pool, rng_s, selfst = None, None, {"cursor": 0, "n_self_steps": 0, "self_reuse": 0}
    if a.self_pool:
        pool, pool_meta = TS.load_pool(a.self_pool, os.path.join(os.path.dirname(a.self_pool), "pool_meta.json")); rng_s = np.random.default_rng(a.seed + 21800)
        print("SELF_POOL", a.self_pool, len(pool), "records, sha", pool_meta["sha256"][:16], flush=True)
    assert a.p_self == 0.0 or pool is not None, "--p_self > 0 needs --self_pool"
    assert not a.mem_check_self or pool is not None, "--mem_check_self needs --self_pool"
    loss_items = make_loss(g, dev, ac)
    if a.init_from and not a.resume:                                        # FT: strict load, sha printed (A4), fresh AdamW below
        sha = TS.file_sha256(a.init_from)
        if os.path.normpath(a.init_from).replace("\\", "/").endswith("s170_cot_s1/model.pt"):
            assert sha == TS.BASE_SHA256, ("§219 V6: the base model.pt sha256", sha)
        sd = torch.load(a.init_from, map_location=dev, weights_only=True); mk = g.load_state_dict(sd, strict=True); del sd
        assert mk.missing_keys == [] and mk.unexpected_keys == [], mk
        print("INIT_FROM %s missing %s unexpected %s sha256 %s" % (a.init_from, mk.missing_keys, mk.unexpected_keys, sha), flush=True)
        json.dump({"init_from": a.init_from, "sha256": sha, "missing": mk.missing_keys, "unexpected": mk.unexpected_keys},
                  open(os.path.join(a.out, "init_from.json"), "w", newline="\n"), indent=1)

    def mem_check(micro):                                                   # nested170.mem_check's train part (A1: no evaluate_n)
        torch.cuda.reset_peak_memory_stats(dev); r_ = np.random.default_rng(0); its = [it for it in N.train_items(r_, 4 * a.bs, a.n_max_train, set()) if True]
        its = sorted(its, key=lambda it: -len(it["syms"]))[:a.bs // micro]
        l_ = loss_items(its); l_.backward(); g.zero_grad(set_to_none=True)
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30

    if a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.resume and not a.mem_check_self):
        mc = []
        while True:
            try:
                pk = mem_check(a.micro)
            except torch.OutOfMemoryError:
                pk = float("inf")
            if pk != float("inf"):
                pk += 2 * sum(p_.numel() * 4 for p_ in g.parameters() if p_.requires_grad) / 2 ** 30
            torch.cuda.empty_cache()
            if pk == float("inf"):
                g.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            mc.append({"micro": a.micro, "peak_gib": pk}); print("MEM", a.micro, pk, flush=True)
            if pk <= a.mem_budget or a.micro >= a.bs or a.mem_check:
                break
            a.micro *= 2
        json.dump({"checks": mc, "micro": a.micro, "peak_gib": mc[-1]["peak_gib"], "train": "the longest of 4 bs draws at n <= %d" % a.n_max_train,
                   "eval": "none (§219 seal: no generation in the trainer)"}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return
        assert mc[-1]["peak_gib"] <= a.mem_budget
    info = {**vars(a), "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info(),
            "hostprobe_sps": os.environ.get("HOSTPROBE_SPS"), "format": TEXTCOT_FORMAT}
    json.dump(info, open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    if a.args_ref:                                                          # the launch log's ARGS-DIFF (vbank precedent; the scorer's rule is score_s219)
        ref = json.load(open(a.args_ref)); dif = {k: [ref.get(k), info.get(k)] for k in sorted(set(ref) | set(info)) if k != "args_ref" and ref.get(k) != info.get(k)}
        json.dump({"ref": a.args_ref, "diff": dif}, open(os.path.join(a.out, "args_diff.json"), "w", newline="\n"), indent=1); print("ARGS-DIFF", json.dumps(dif), flush=True)
    params = g; opt = make_opt(params, a.lr)
    if a.mem_check_self:                                                    # G-T9: the deepest self shape, fixed self micro, real optimiser steps
        deep, n_top = deep_self(pool, a.bs)
        its = [TS.self_item(r_) for r_ in deep]; ch = len(its) // a.self_micro; t_max = max(len(r_["syms"]) + 2 + TS.c_struct(r_["syms"]) for r_ in deep)
        if dev.startswith("cuda"):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        ok_, t1_ = True, time.time()
        try:
            for _ in range(a.mem_check_steps):
                opt.zero_grad(set_to_none=True)
                for k_ in range(a.self_micro):
                    lk = loss_items(its[k_ * ch:(k_ + 1) * ch]); (lk / a.self_micro).backward(); del lk
                torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0); opt.step()
        except torch.OutOfMemoryError:
            ok_ = False
        pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else None
        mj = {"peak_gib": pk if ok_ else "OOM", "pass": bool(ok_ and (pk is None or pk <= a.mem_budget)), "mem_budget": a.mem_budget, "self_micro": a.self_micro,
              "bs": a.bs, "steps": a.mem_check_steps, "n_top": n_top, "T_max": t_max, "gpu": info["train_gpu"], "amp": str(amp), "sec": round(time.time() - t1_, 1),
              "def": "peak after the optimiser steps (incl. AdamW's moments) on the 64 longest pool records of the deepest n at the fixed self micro"}
        json.dump(mj, open(os.path.join(a.out, "mem_check_self.json"), "w", newline="\n"), indent=1); print("MEMCHECK_SELF", json.dumps(mj), flush=True)
        return
    step0 = 1; n_nonfinite = 0; hist = []; gn_log = []; tok = {"plain": [0, 0, 0], "self": [0, 0, 0]}; self_n_counts = {}
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time(); draws_log = []
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); params.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu()); step0 = ck["step"] + 1; n_nonfinite = ck["n_nonfinite"]; gn_log = list(ck.get("gn_log", [])); t0 = time.time() - ck["elapsed"]
        tok = ck["tok"]; self_n_counts = ck["self_n_counts"]
        assert (rng_s is None) == (ck.get("rng_s") is None), "§219: resume with / without a pool must match the checkpoint"
        if rng_s is not None:
            rng_s.bit_generator.state = ck["rng_s"]; selfst.update(ck["self_state"])
        if os.path.exists(os.path.join(a.out, "draws_log.json")):
            draws_log = [d for d in json.load(open(os.path.join(a.out, "draws_log.json"))) if d["step"] < step0]
        if a.draws_log_steps and not a.time_steps:                          # G-T13 item 4: draws_log is written BEFORE each checkpoint, so it covers 1..step0-1
            assert [d["step"] for d in draws_log] == list(range(1, min(step0 - 1, a.draws_log_steps) + 1)), "§219: draws_log.json does not cover the resumed range"
        log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush(); print("RESUME", a.resume, "step", ck["step"], flush=True); del ck
    KEEP_CK = {int(v_) for v_ in a.keep_ckpt_steps.split(",")} if a.keep_ckpt_steps else set()
    if 0 in KEEP_CK and step0 == 1:                                         # A5: the model right after init
        torch.save({"model": params.state_dict(), "step": 0}, os.path.join(a.out, "ckpt_model_0.pt"))

    if a.time_steps and pool is not None:                                   # Note 219-4 (3): G-T10's forced self batch = the deepest self shape
        deep_recs, deep_n = deep_self(pool, a.bs); deep_its = [TS.self_item(r_) for r_ in deep_recs]
        deep_T = max(len(r_["syms"]) + 2 + TS.c_struct(r_["syms"]) for r_ in deep_recs)

    def draw_step(step):
        """the plain batch first (every step, the base's stream), then the coin; a self step takes the next bs pool records (no reuse). -> its, is_self, dl."""
        its = N.train_items(rng, a.bs, a.n_max_train, EXCL)
        lg_ = step <= a.draws_log_steps and not a.time_steps
        dl = {"step": step, "plain_hash": plain_hash(its)} if lg_ else None
        if pool is None:
            if lg_:
                dl.update({"coin": False, "n": [it["n"] for it in its]})
            return its, False, dl
        if a.time_steps:                                                    # A7 / Note 219-4 (3): forced alternation; odd steps = the deepest self shape
            coin = step % 2 == 1
            if coin:
                U = [int(rng_s.integers(0, 5)) for _ in range(a.bs)]            # (drawn as vbank's G-TIME; no cursor: the deep batch every odd step)
                return deep_its, True, None
            U = None
        else:
            coin, U = coin_step(rng_s, a.p_self, a.bs)
        if not coin:
            if lg_:
                dl.update({"coin": False, "n": [it["n"] for it in its]})
            return its, False, dl
        c0 = selfst["cursor"]
        if c0 + a.bs > len(pool):
            raise RuntimeError("§219: the self pool is exhausted (no reuse allowed)")
        recs = pool[c0:c0 + a.bs]; selfst["cursor"] = c0 + a.bs; selfst["n_self_steps"] += 1
        for r_ in recs:
            self_n_counts[str(r_["n"])] = self_n_counts.get(str(r_["n"]), 0) + 1
        if lg_:
            dl.update({"coin": True, "self_n": [r_["n"] for r_ in recs], "U": U})
        return [TS.self_item(r_) for r_ in recs], True, dl

    last = a.steps if not a.time_steps else min(a.steps, step0 + a.time_steps - 1)
    if a.stop_at:
        last = min(last, a.stop_at)
    win = {"plain": [], "self": [], "st": {}}; tself = [0.0, 0, 0.0, 0]; tt0 = time.time()
    for step in range(step0, last + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps) if a.warmup == 100 else lr_at(a.lr, gr, step, a.steps) / min(1, step / 100) * min(1, step / a.warmup)   # nested170 l.1039
        params.train(); ts0 = time.time()
        its, is_self, dl = draw_step(step); opt.zero_grad(); tot = 0.0
        if dl is not None:
            draws_log.append(dl)
        mic = a.self_micro if is_self else a.micro; ch = a.bs // mic; st = {}
        for k_ in range(mic):
            lk = loss_items(its[k_ * ch:(k_ + 1) * ch], st); (lk / mic).backward(); tot += float(lk.item()) / mic
        gn = torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0); gn_log.append(float(gn))
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            n_nonfinite += 1; opt.zero_grad(); log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush()
        else:
            opt.step(); hist.append(tot); win["self" if is_self else "plain"].append(tot)
        for kind, c_ in st.get("tok", {}).items():
            for q in range(3):
                tok[kind][q] += c_[q]
        for k2 in ("value_sum", "n_value", "copy_sum", "n_copy"):
            if k2 in st:
                win["st"][k2] = win["st"].get(k2, 0) + st[k2]
        if dev.startswith("cuda") and a.time_steps:
            torch.cuda.synchronize(dev)
        tself[0 if is_self else 2] += time.time() - ts0; tself[1 if is_self else 3] += 1
        if step % a.log_every == 0 or step == a.steps or step == last:
            ws = win["st"]
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "loss_last": tot, "sec": round(time.time() - t0),
                   "plain_loss": float(np.mean(win["plain"])) if win["plain"] else None, "self_loss": float(np.mean(win["self"])) if win["self"] else None,
                   "self_value_ce": ws["value_sum"] / ws["n_value"] if ws.get("n_value") else None, "self_copy_ce": ws["copy_sum"] / ws["n_copy"] if ws.get("n_copy") else None,
                   "self_frac": len(win["self"]) / max(1, len(win["self"]) + len(win["plain"])), "pool_cursor": selfst["cursor"], "n_self_steps": selfst["n_self_steps"],
                   "self_reuse": selfst["self_reuse"], "lr": opt.param_groups[0]["lr"], "gn_preclip": gn_log[-1], "clip_frac_100": float(np.mean([v_ > 1.0 for v_ in gn_log[-100:]])),
                   "n_nonfinite": n_nonfinite, "train_tokens": {k: {"target_tokens": v[0], "token_positions": v[1], "padded_positions": v[2]} for k, v in tok.items()}}
            hist = []; win = {"plain": [], "self": [], "st": {}}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if step in KEEP_CK:
            torch.save({"model": params.state_dict(), "step": step}, os.path.join(a.out, "ckpt_model_%d.pt" % step))
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps and not a.time_steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            if draws_log:                                                   # G-T13 item 4: draws_log FIRST (atomic), then the checkpoint (tmp + os.replace), so
                atomic_json(draws_log, os.path.join(a.out, "draws_log.json"))   # every checkpoint on disk is covered by the draws_log on disk
            torch.save({"model": params.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "step": step,
                        "n_nonfinite": n_nonfinite, "gn_log": gn_log, "tok": tok, "self_n_counts": self_n_counts, "elapsed": time.time() - t0,
                        **({"rng_s": rng_s.bit_generator.state, "self_state": dict(selfst)} if rng_s is not None else {})}, cp + ".tmp")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            os.replace(cp + ".tmp", cp)
    if draws_log:
        atomic_json(draws_log, os.path.join(a.out, "draws_log.json"))
    if a.time_steps:                                                        # G-T10 (A7): timing.json only
        sps_self = tself[0] / max(1, tself[1]); sps_plain = tself[2] / max(1, tself[3])
        sps = (a.p_self * sps_self + (1 - a.p_self) * sps_plain) if pool is not None else (time.time() - tt0) / max(1, last - step0 + 1)
        tj = {"arm": a.arm, "steps_timed": last - step0 + 1, "sec_per_step": sps, "est_hours": sps * a.steps / 3600, "sec_per_step_self": sps_self if tself[1] else None,
              "sec_per_step_plain": sps_plain if tself[3] else None, "self_micro": a.self_micro, "micro": a.micro, "p_self": a.p_self, "gpu": info["train_gpu"],
              "forced": ("odd steps self = the %d longest records of the deepest n %d (T_max %d; Note 219-4 (3)), even steps plain; sec_per_step = p_self-weighted "
                         "mix" % (len(deep_its), deep_n, deep_T)) if pool is not None else "plain", **({"deep_n": deep_n, "deep_T_max": deep_T} if pool is not None else {})}
        json.dump(tj, open(os.path.join(a.out, "timing.json"), "w", newline="\n"), indent=1); print("TIMING", sps, flush=True); return
    if a.stop_at and last < a.steps:
        print("STOPPED_AT", last, flush=True); return
    train_sec = round(time.time() - t0); torch.save(params.state_dict(), os.path.join(a.out, "model.pt"))
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    json.dump({"steps": a.steps, "train_sec": train_sec, "n_nonfinite": n_nonfinite, "n_self_steps": selfst["n_self_steps"], "pool_cursor": selfst["cursor"],
               "self_reuse": selfst["self_reuse"], "train_tokens": {k: {"target_tokens": v[0], "token_positions": v[1], "padded_positions": v[2]} for k, v in tok.items()},
               "self_n_counts": self_n_counts, "model_sha256": TS.file_sha256(os.path.join(a.out, "model.pt"))},
              open(os.path.join(a.out, "train_done.json"), "w", newline="\n"), indent=1)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
