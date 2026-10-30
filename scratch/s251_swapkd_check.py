"""Audit 10-30 $0 bug check (part_swap_kd SWAP-K identical across aex_s1 / aex_s2 / rcb_plaw4_s2): on the first 50 d 4 pairs (swap_pairs_d(a, 4), REG law),
per model: (1) sw = dec_with(m, R, SD) per item; (2) dec_with with SD := zeros / random (per-item Frobenius norm matched, seed 0) / the recipient's own S_K;
(3) grab_states norms (per-item ||S_t||, t = swap_t(4), K) + cross-model equality of SD. Plus the intact answers at K. -> scratch/s251_swapkd_check/check.json
    python -B scratch/s251_swapkd_check.py [--n 50 --bs 50 --device cuda:0]"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
from boxes import inject_aw_eval as E  # noqa: E402

RUNS = tuple(os.environ.get("RUNS", "aex_s1,aex_s2,rcb_plaw4_s2").split(","))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50); ap.add_argument("--bs", type=int, default=50); ap.add_argument("--device", default="cuda:0")
    o = ap.parse_args(); dev = o.device; out = {"n": o.n, "runs": {}}; SDs = {}
    for run in RUNS:
        a, m, g = E.load(os.path.join(ROOT, "boxes/runs_fsq", run), dev); K = a["think_k"]
        R, Dn, kept, tried, t = E.swap_pairs_d(a, 4); R, Dn, kept = R[:o.n], Dn[:o.n], kept[:o.n]
        own = np.array([p_[0]["answer"] for p_ in kept]); dfin = np.array([p_[1]["answer"] for p_ in kept])
        with torch.no_grad():
            S = E.grab_states(m, Dn, dev, o.bs, (t, K)); SD = S[K]; SR = E.grab_states(m, R, dev, o.bs, (K,))[K]
            I, _ = E.passes(m, R, dev, o.bs, {K})
            sw = E.dec_with(m, R, SD, dev, o.bs); z = E.dec_with(m, R, torch.zeros_like(SD), dev, o.bs)
            gen = torch.Generator().manual_seed(0); rn = torch.randn(SD.shape, generator=gen)
            rn = rn * (SD.flatten(1).norm(dim=1) / rn.flatten(1).norm(dim=1)).view(-1, *[1] * (SD.dim() - 1))
            r = E.dec_with(m, R, rn, dev, o.bs); ow = E.dec_with(m, R, SR, dev, o.bs)
        SDs[run] = SD
        out["runs"][run] = {"k": K, "t": t, "pair_sigs_head": [p_[0]["sig"] for p_ in kept[:3]], "sw": sw.tolist(), "zeros": z.tolist(), "random": r.tolist(), "own_SK": ow.tolist(),
                            "intact_K": I[K].tolist(), "eq_sw_zeros": int((sw == z).sum()), "eq_sw_random": int((sw == r).sum()), "eq_sw_own": int((sw == ow).sum()),
                            "eq_own_intact": int((ow == I[K]).sum()), "sw_follow_dfin": int((sw == dfin).sum()), "sw_stay_own": int((sw == own).sum()),
                            "norm_SD_K": [float(x) for x in SD.flatten(1).norm(dim=1)[:10]], "norm_SD_t": [float(x) for x in S[t].flatten(1).norm(dim=1)[:10]],
                            "norm_SR_K": [float(x) for x in SR.flatten(1).norm(dim=1)[:10]], "SD_K_eq_SD_t": bool(torch.equal(SD, S[t]))}
        del m, g; torch.cuda.empty_cache()
    rs = list(RUNS); out["cross"] = {}
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            x, y = out["runs"][rs[i]], out["runs"][rs[j]]
            out["cross"]["%s|%s" % (rs[i], rs[j])] = {"sw_equal_items": int(sum(p == q for p, q in zip(x["sw"], y["sw"]))),
                                                      "intact_equal_items": int(sum(p == q for p, q in zip(x["intact_K"], y["intact_K"]))),
                                                      "SD_max_abs_diff": float((SDs[rs[i]] - SDs[rs[j]]).abs().max()), "same_pairs": x["pair_sigs_head"] == y["pair_sigs_head"]}
    os.makedirs("scratch/s251_swapkd_check", exist_ok=True)
    json.dump(out, open("scratch/s251_swapkd_check/" + os.environ.get("OUTN", "check") + ".json", "w", newline="\n"), indent=1)
    print(json.dumps({"cross": out["cross"], "per_run": {r_: {k: v for k, v in out["runs"][r_].items() if k.startswith(("eq_", "SD_K_eq", "norm_SD_K"))} for r_ in rs}}, indent=1))


if __name__ == "__main__":
    main()
