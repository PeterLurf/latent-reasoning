"""note 212-15 hook gates (eval-only vbank hooks slot_mask / v0 / gather_hist_ablate; block_bank kmask), before any S0-1..3 output:
  g1 defaults: v0 = the default prelude passed explicitly == default rollout (bitwise); slot_mask returning all-False == default (max |diff|, coda logits);
     gather_hist_ablate at K = 2 (no mid versions) == default (PA).
  g2 UNREADABLE masked slots ("removed from every reader"): with a fixed slot set S removed from t0 on, perturbing S (t0 = 0: the input tokens at S, via the
     masked prelude; t0 = 1: the stored entries / carry of S at passes >= 1, via edit) leaves the answer logits at K' bitwise unchanged; N1 (CARRY) and N2 (PA).
     Sanity: the same perturbation WITHOUT the mask changes them.
  g3 STAB-HIST (gather weights recomputed at t = 6, real query slots): "self" -> mid (1 <= s <= t-2) own-slot mass exactly 0, mid cross-slot mass > 0;
     "other" -> mid cross-slot mass exactly 0, mid own-slot mass > 0; intact mid own mass > 0 (the arm is exposed).
  g4 GAPPED prelude: all-False interior mask vs deltamem.gpt_lower (max |diff|; explicit-mask vs is_causal kernels).
-> scratch/gate_s212_d3.json.   python scratch/gate_s212_d3.py --device cuda:2"""
import argparse
import json
import sys

sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_diag as VD  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:2"); a = ap.parse_args(); dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    TE = N.test_set(1000); out = {}
    its = [it for it in TE if it["n"] == 10][:32]; x, pad, lens = VB.batch_x(its, dev); B, T = x.shape; K = 12; rows = torch.arange(B, device=dev)
    lg = lambda m, r: m.answer_logits(r["coda"][K][1], lens)  # noqa: E731
    for name in ("s211_vb_carry_s1", "s211_vb_pa_s1"):
        m, _ = VE.load("boxes/runs_fsq/" + name, dev); o = {}
        r0 = m.rollout(x, pad, K, coda_rows={K: rows}); L0 = lg(m, r0); VK0 = r0["V"][K]; V00 = r0["V"][0]; del r0
        r1 = m.rollout(x, pad, K, coda_rows={K: rows}, v0=V00.clone()); o["g1_v0_explicit_bitwise"] = bool(torch.equal(lg(m, r1), L0) and torch.equal(r1["V"][K], VK0)); del r1
        m.slot_mask = lambda t, V: torch.zeros(V.shape[:2], dtype=torch.bool, device=V.device)
        r2 = m.rollout(x, pad, K, coda_rows={K: rows}); m.slot_mask = None
        o["g1_zero_mask_maxdiff_logits"] = float((lg(m, r2) - L0).abs().max()); o["g1_zero_mask_maxdiff_VK"] = float((r2["V"][K] - VK0).abs().max()); del r2
        if m.mode == "pa":
            r0b = m.rollout(x, pad, 2, coda_rows={2: rows}); m.gather_hist_ablate = "self"; r3 = m.rollout(x, pad, 2, coda_rows={2: rows}); m.gather_hist_ablate = None
            o["g1_hist_ablate_K2_bitwise"] = bool(torch.equal(r3["V"][2], r0b["V"][2])); del r0b, r3
        # g2: S = 3 interior real slots per item (positions 3, 5, 7: never EOT / ' =' at n 10)
        S = torch.zeros_like(pad); S[:, [3, 5, 7]] = True; assert bool((S <= pad).all()) and not bool(S[rows, lens - 1].any())
        g = torch.Generator(device="cpu").manual_seed(21215)
        # t0 = 0: perturb the input tokens at S, masked prelude + slot_mask from V_0
        x2 = x.clone(); x2[S] = torch.tensor(N.NUM_IDS, device=dev)[torch.randint(0, len(N.NUM_IDS), (int(S.sum()),), generator=g).to(dev)]
        res = []
        for xx in (x, x2):
            m.slot_mask = lambda t, V: S if t == 0 else torch.zeros_like(S)
            r_ = m.rollout(xx, pad, K, coda_rows={K: rows}, v0=VD._lower_masked(m.g, xx, S)); m.slot_mask = None; res.append(lg(m, r_)); del r_
        o["g2_t0_masked_invariant_bitwise"] = bool(torch.equal(res[0], res[1]))
        ru = [lg(m, m.rollout(xx, pad, K, coda_rows={K: rows})) for xx in (x, x2)]; o["g2_t0_unmasked_changes_maxdiff"] = float((ru[0] - ru[1]).abs().max())
        # t0 = 1: removed after pass 1; perturb the stored entries / carry at S for passes >= 1
        noise = torch.randn(B, T, VK0.shape[-1], generator=g).to(dev) * 5.0

        def ed(t, V):
            return torch.where(S[..., None], V + noise, V)
        res = []
        for e_ in (None, ed):
            m.slot_mask = lambda t, V: S if t >= 1 else torch.zeros_like(S)
            r_ = m.rollout(x, pad, K, coda_rows={K: rows}, edit=e_); m.slot_mask = None; res.append(lg(m, r_)); del r_
        o["g2_t1_masked_invariant_bitwise"] = bool(torch.equal(res[0], res[1]))
        ru = [lg(m, m.rollout(x, pad, K, coda_rows={K: rows}, edit=e_)) for e_ in (None, ed)]; o["g2_t1_unmasked_changes_maxdiff"] = float((ru[0] - ru[1]).abs().max())
        # MASKVOID smoke: no NaN, exposure
        cnt = [0]

        def mv(t, V):
            z = (m.rec_logits(V).argmax(-1) == VB.VOID) & pad if t >= 1 else torch.zeros_like(pad); cnt[0] += int(z.sum()); return z
        m.slot_mask = mv; r_ = m.rollout(x, pad, K, coda_rows={K: rows}); m.slot_mask = None
        o["maskvoid_smoke_finite"] = bool(torch.isfinite(lg(m, r_)).all()); o["maskvoid_smoke_void_slot_events"] = cnt[0]; del r_
        if m.mode == "pa":                                                  # g3
            for how in ("self", "other"):
                m.gather_hist_ablate = how; m.gather_split_log = []; r_ = m.rollout(x, pad, K, coda_rows={K: rows})
                sl = m.gather_split_log; m.gather_split_log = None; m.gather_hist_ablate = None
                o["g3_%s_split_older_same_slot_incl_V0_max_t3plus" % how] = float(max(float(z[2].max()) for z in sl if z[0] >= 3))   # includes V_0[j] (kept)
                o["g3_%s_finite" % how] = bool(torch.isfinite(lg(m, r_)).all()); del r_
            # recompute the gather weights at t = 6 on the intact bank per arm: mid (1 <= s <= t-2) own-slot (s, j) and cross-slot (s, i != j) mass
            r_ = m.rollout(x, pad, 6); t = 6; gk = [m.gkey(V) for V in r_["S"][:t]]; rms0 = r_["V"][0].pow(2).mean(-1, keepdim=True).sqrt()
            NV = [m.nrm(V, rms0) for V in r_["S"][:t]]; orig_sm = torch.softmax; del r_
            s_idx = torch.arange(t, device=dev).repeat_interleave(T); i_idx = torch.arange(T, device=dev).repeat(t)
            mid = (s_idx >= 1) & (s_idx <= t - 2); same = i_idx[None, :] == torch.arange(T, device=dev)[:, None]
            for how in (None, "self", "other"):
                cap = {}

                def spy(z, d, cap=cap):
                    w = orig_sm(z, d); cap["w"] = w; return w
                m.gather_hist_ablate = how; torch.softmax = spy
                try:
                    m.gather(t, gk, NV, pad)
                finally:
                    torch.softmax = orig_sm; m.gather_hist_ablate = None
                w = cap["w"]; rm_ = pad[:, None, :, None].to(w.dtype)                     # real query slots only
                own = (w * (mid[None, :] & same).to(w.dtype)[None, None]).sum(-1); crs = (w * (mid[None, :] & ~same).to(w.dtype)[None, None]).sum(-1)
                own = own * pad[:, None, :]; crs = crs * pad[:, None, :]
                o["g3_%s_mid_own_mass_max" % how] = float(own.max()); o["g3_%s_mid_cross_mass_max" % how] = float(crs.max())
        o["g4_gapped_prelude_maxdiff"] = float((VD._lower_masked(m.g, x, torch.zeros_like(pad)) - VB.DM_().gpt_lower(m.g, m.g.embed(x), VB.CORE[0])).abs().max())
        out[name] = o; print(name, o, flush=True); del m; torch.cuda.empty_cache()
    ok = all(out[k]["g1_v0_explicit_bitwise"] and out[k]["g2_t0_masked_invariant_bitwise"] and out[k]["g2_t1_masked_invariant_bitwise"] and out[k]["maskvoid_smoke_finite"]
             and out[k]["g2_t0_unmasked_changes_maxdiff"] > 0 and out[k]["g2_t1_unmasked_changes_maxdiff"] > 0 for k in out)
    p = out["s211_vb_pa_s1"]; ok = (ok and p["g1_hist_ablate_K2_bitwise"] and p["g3_self_mid_own_mass_max"] == 0.0 and p["g3_self_mid_cross_mass_max"] > 0
                                    and p["g3_other_mid_cross_mass_max"] == 0.0 and p["g3_other_mid_own_mass_max"] > 0 and p["g3_None_mid_own_mass_max"] > 0)
    out["pass"] = bool(ok); out["device"] = torch.cuda.get_device_name(dev)
    json.dump(out, open("D:/Latent/scratch/gate_s212_d3.json", "w", newline="\n"), indent=1); print("GATE pass", ok, flush=True)


if __name__ == "__main__":
    main()
