"""Theory (seed-stability trajectory, Audit-routed): nested170's s182.path_indep EXACTLY (500 test items n 3..8, rng 18201 permutation[:500]; K' 16 from
M_0 = 0 vs from a derangement donor's (np.roll by 1) 8-pass memory; answer_agree, delta_cos_final (mean cos of the final delta-head states),
acc_from_zero, acc_from_donor) on a checkpoint or model.pt of a run, plus s175 aux_acc_by_pass / aux_acc_by_role (nested170.aux_eval's definitions, on
s175's n <= 8 set: rng 17022 permutation[:1000], t <= K). Eval-only; the run's args.json builds the model (inject or readslots); eval batch = args
eval_bs; bf16 autocast on Ampere+, fp32 otherwise (the card is recorded).
Theory / Audit 2026-10-13 additions: --stage nmax:K | auto = the STAGE-MATCHED probe for curriculum / annealed checkpoints (test items n <= nmax, the same
rng 18201 permutation[:500] of that pool, K' = 2K from M_0 = 0 vs a K-pass donor; auto = the --anneal 'nmax:K:until' stage holding at the ckpt step) ->
pathindep_stage_<step>.json, by_n included; --log <log.jsonl> (default <run>/log.jsonl) = the training-loss bin at the step (mean loss / answer_ce over the
records in (step - 1000, step]); --loss_seen yes|no = whether that run's training loss at this step had been looked at before the probe (else "unrecorded").
    python scratch/probe_pathindep.py <run dir> <ckpt.pt | model.pt> [device] [--no_aux] [--stage nmax:K|auto] [--log path] [--loss_seen yes|no]
    -> <run>/pathindep_<step>.json (pathindep_stage_<step>.json with --stage)"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402


@torch.no_grad()
def main():
    run, ckp = sys.argv[1], sys.argv[2]; dev = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "cuda:1"; no_aux = "--no_aux" in sys.argv
    opt = lambda k_, d_=None: sys.argv[sys.argv.index(k_) + 1] if k_ in sys.argv else d_  # noqa: E731
    stage_arg, log_p, seen = opt("--stage"), opt("--log", os.path.join(run, "log.jsonl")), opt("--loss_seen", "unrecorded")
    assert seen in ("yes", "no", "unrecorded"), seen
    a = json.load(open(os.path.join(run, "args.json"))); DM.FAST_ADD = bool(a.get("fast_add")); K = int(a["think_k"]); bs = int(a.get("eval_bs") or 250)
    g = N.make_trunk(a, dev); ML, HA, HD = N.mem_geometry(a)
    m = (N.ReadSlotsN(g, ML, "top", HA, HD, a.get("add_norm") or None, int(a.get("n_slots") or 8)) if a["arm"] == "readslots"
         else N.InjectN(g, ML, "top", HA, HD, a.get("add_norm") or None)).to(dev)
    m.nopass = bool(a.get("aux_nopass")); ck = torch.load(ckp, map_location=dev, weights_only=False); sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    step = int(ck.get("step")) if isinstance(ck, dict) and "step" in ck else None; m.load_state_dict(sd); m.eval(); del ck
    amp = None if (not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    TE = N.test_set(int(a.get("per_n") or 1000))
    n_lo, n_hi, KD, KP, stage = 3, 8, 8, 16, None                           # the standard probe: n 3..8, an 8-pass donor, K' 16
    if stage_arg:
        if stage_arg == "auto":                                             # --anneal 'nmax:K:until,...': the first stage whose until >= the ckpt step
            assert a.get("anneal") and step is not None, "--stage auto needs a fixed-step --anneal run and a ckpt step"
            st_ = [tuple(int(v_) for v_ in s_.split(":")) for s_ in a["anneal"].split(",")]; nm_, k_ = next((nm, kk) for nm, kk, un in st_ if step <= un)
        else:
            nm_, k_ = (int(v_) for v_ in stage_arg.split(":"))
        n_lo, n_hi, KD, KP, stage = 1, nm_, k_, 2 * k_, {"n_max": nm_, "K": k_, "spec": stage_arg}

    def states(items, t):
        out = []
        for i in range(0, len(items), bs):
            ch = items[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            with ac():
                S = m.mem.zero_state(len(ch), dev)
                for _ in range(t):
                    S = m.write(S, x, msk)
            out.append(S.float().cpu().numpy())
        return np.concatenate(out)
    pool = [it for it in TE if n_lo <= it["n"] <= n_hi]; its_ = [pool[i] for i in np.random.default_rng(18201).permutation(len(pool))[:500]]
    der = np.roll(np.arange(len(its_)), 1); S8 = states(its_, KD)[der]; A0, A1, F0, F1 = [], [], [], []
    for i in range(0, len(its_), bs):
        ch = its_[i:i + bs]; x, msk = m.think_inputs(ch, dev)
        for S0, Ao, Fo in ((None, A0, F0), (S8[i:i + bs], A1, F1)):
            with ac():
                Sz = m.mem.zero_state(len(ch), dev); S = Sz if S0 is None else torch.tensor(S0, device=dev, dtype=Sz.dtype)
                for _ in range(KP):
                    S = m.write(S, x, msk)
                lg = m.dec_logits(S, ch, dev)
            Ao.append(lg[:, N.NUM_IDS].argmax(-1).cpu().numpy()); Fo.append(S[:, m.mem.h_add:m.mem.h_add + m.mem.h_del].float().flatten(1).cpu())
    A0, A1 = np.concatenate(A0), np.concatenate(A1); F0, F1 = torch.cat(F0), torch.cat(F1); tt_ = np.array([it["t"] for it in its_])
    cs_ = Fn.cosine_similarity(F0, F1, dim=1).numpy(); nn_ = np.array([it["n"] for it in its_])
    by_n = {str(n_): {"n_items": int((nn_ == n_).sum()), "answer_agree": float((A0 == A1)[nn_ == n_].mean()), "delta_cos_final": float(cs_[nn_ == n_].mean())}
            for n_ in sorted(set(nn_.tolist()))}
    tl = None
    if step is not None and os.path.exists(log_p):                          # read AFTER the probe is computed
        rr_ = [json.loads(l_) for l_ in open(log_p) if l_.strip()]; rr_ = [r_ for r_ in rr_ if "loss" in r_ and step - 1000 < r_["step"] <= step]
        tl = {"window": [step - 1000, step], "n_records": len(rr_), "loss_mean": float(np.mean([r_["loss"] for r_ in rr_])) if rr_ else None,
              "answer_ce_mean": float(np.mean([r_["answer_ce"] for r_ in rr_ if r_.get("answer_ce") is not None])) if any(r_.get("answer_ce") is not None for r_ in rr_) else None}
    out = {"step": step, "ckpt": os.path.basename(ckp), "path_indep": {"answer_agree": float((A0 == A1).mean()), "delta_cos_final": float(cs_.mean()),
           "n_items": len(its_), "acc_from_zero": float((A0 == tt_).mean()), "acc_from_donor": float((A1 == tt_).mean()), "kprime": KP, "donor_passes": KD,
           "items_n": [n_lo, n_hi], "by_n": by_n}, "stage": stage, "train_loss_bin_1k": tl, "loss_seen_before_probe": seen,
           "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "eval_dtype": str(amp), "run": run,
           "def": ("nested170 s182.path_indep reproduced (items test n 3..8 rng 18201 [:500]; derangement np.roll 1; K' 16; delta_cos = mean cos of final delta-head states)"
                   if stage is None else "STAGE-MATCHED path_indep (test n %d..%d rng 18201 [:500]; derangement np.roll 1; %d-pass donor; K' %d; delta_cos as s182)" % (n_lo, n_hi, KD, KP))}
    if not no_aux:                                                          # s175 aux_acc_by_pass / by_role (nested170.aux_eval) on rng 17022 n <= 8 [:1000], t <= K
        p8 = [it for it in TE if it["n"] <= 8]; te8 = [p8[i] for i in np.random.default_rng(17022).permutation(len(p8))[:1000]]; ex, role = {}, {}
        for i in range(0, len(te8), bs):
            ch = te8[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            with ac():
                S = m.mem.zero_state(len(ch), dev)
                for t in range(1, K + 1):
                    S = m.write(S, x, msk); idx = [j for j, it in enumerate(ch) if len(N.aux_of(it)) >= t]
                    if not idx:
                        continue
                    sub = [ch[j] for j in idx]; lg_a, y_a = m.aux_logits(S[idx], sub, t, dev); pr = lg_a.argmax(-1).cpu().numpy(); yy = y_a.cpu().numpy()
                    gen = m.aux_greedy(S[idx], t, int((yy != -100).sum(1).max()))
                    for r_, it in enumerate(sub):
                        sy, hi = N.aux_of(it)[t - 1]; L_ = len(sy); tgt = N.ids(sy) + [N.EOT]; ok_ = pr[r_, :L_] == yy[r_, :L_]
                        rr = role.setdefault(t, {"handle": [], "copy": []}); rr["handle"].append(bool(ok_[hi])); rr["copy"].extend(bool(ok_[k_]) for k_ in range(L_) if k_ != hi)
                        ex.setdefault(t, []).append(bool(list(gen[r_, :len(tgt)]) == tgt))
        out["s175"] = {"aux_acc_by_pass": {str(t): float(np.mean(v)) for t, v in sorted(ex.items())},
                       "aux_acc_by_role": {str(t): {k_: (float(np.mean(v)) if v else None) for k_, v in d.items()} for t, d in sorted(role.items())}}
    fn = os.path.join(run, "pathindep_%s%s.json" % ("stage_" if stage else "", step if step is not None else "final"))
    json.dump(out, open(fn, "w", newline="\n"), indent=1); print("PATHINDEP", run, step, out["path_indep"], flush=True)


if __name__ == "__main__":
    main()
