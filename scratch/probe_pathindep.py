"""Theory (seed-stability trajectory, Audit-routed): nested170's s182.path_indep EXACTLY (500 test items n 3..8, rng 18201 permutation[:500]; K' 16 from
M_0 = 0 vs from a derangement donor's (np.roll by 1) 8-pass memory; answer_agree, delta_cos_final (mean cos of the final delta-head states),
acc_from_zero, acc_from_donor) on a checkpoint or model.pt of a run, plus s175 aux_acc_by_pass / aux_acc_by_role (nested170.aux_eval's definitions, on
s175's n <= 8 set: rng 17022 permutation[:1000], t <= K). Eval-only; the run's args.json builds the model (inject or readslots); eval batch = args
eval_bs; bf16 autocast on Ampere+, fp32 otherwise (the card is recorded).
    python scratch/probe_pathindep.py <run dir> <ckpt.pt | model.pt> [device] [--no_aux]   -> <run>/pathindep_<step>.json"""
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
    a = json.load(open(os.path.join(run, "args.json"))); DM.FAST_ADD = bool(a.get("fast_add")); K = int(a["think_k"]); bs = int(a.get("eval_bs") or 250)
    g = N.make_trunk(a, dev); ML, HA, HD = N.mem_geometry(a)
    m = (N.ReadSlotsN(g, ML, "top", HA, HD, a.get("add_norm") or None, int(a.get("n_slots") or 8)) if a["arm"] == "readslots"
         else N.InjectN(g, ML, "top", HA, HD, a.get("add_norm") or None)).to(dev)
    m.nopass = bool(a.get("aux_nopass")); ck = torch.load(ckp, map_location=dev, weights_only=False); sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    step = int(ck.get("step")) if isinstance(ck, dict) and "step" in ck else None; m.load_state_dict(sd); m.eval(); del ck
    amp = None if (not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    TE = N.test_set(int(a.get("per_n") or 1000))

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
    pool = [it for it in TE if 3 <= it["n"] <= 8]; its_ = [pool[i] for i in np.random.default_rng(18201).permutation(len(pool))[:500]]
    der = np.roll(np.arange(len(its_)), 1); S8 = states(its_, 8)[der]; A0, A1, F0, F1 = [], [], [], []
    for i in range(0, len(its_), bs):
        ch = its_[i:i + bs]; x, msk = m.think_inputs(ch, dev)
        for S0, Ao, Fo in ((None, A0, F0), (S8[i:i + bs], A1, F1)):
            with ac():
                Sz = m.mem.zero_state(len(ch), dev); S = Sz if S0 is None else torch.tensor(S0, device=dev, dtype=Sz.dtype)
                for _ in range(16):
                    S = m.write(S, x, msk)
                lg = m.dec_logits(S, ch, dev)
            Ao.append(lg[:, N.NUM_IDS].argmax(-1).cpu().numpy()); Fo.append(S[:, m.mem.h_add:m.mem.h_add + m.mem.h_del].float().flatten(1).cpu())
    A0, A1 = np.concatenate(A0), np.concatenate(A1); F0, F1 = torch.cat(F0), torch.cat(F1); tt_ = np.array([it["t"] for it in its_])
    out = {"step": step, "ckpt": os.path.basename(ckp), "path_indep": {"answer_agree": float((A0 == A1).mean()), "delta_cos_final": float(Fn.cosine_similarity(F0, F1, dim=1).mean()),
           "n_items": len(its_), "acc_from_zero": float((A0 == tt_).mean()), "acc_from_donor": float((A1 == tt_).mean()), "kprime": 16, "donor_passes": 8},
           "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "eval_dtype": str(amp), "run": run,
           "def": "nested170 s182.path_indep reproduced (items test n 3..8 rng 18201 [:500]; derangement np.roll 1; K' 16; delta_cos = mean cos of final delta-head states)"}
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
    fn = os.path.join(run, "pathindep_%s.json" % (step if step is not None else "final"))
    json.dump(out, open(fn, "w", newline="\n"), indent=1); print("PATHINDEP", run, step, out["path_indep"], flush=True)


if __name__ == "__main__":
    main()
