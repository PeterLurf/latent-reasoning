"""$0 OC block for OW_draft.md §6: P(pass) of R1 / R2 / R3 and the joint C1-LANG label at one seed, by simulation.
Binomial item noise; the design effect 1.3 deflates n for every proportion; arms drawn independently (conservative for the paired R2).
Writes ow_oc.json next to this file.   python -B docs/research_ordered_words_2026-10-23/ow_oc.py"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json  # noqa: E402

import numpy as np  # noqa: E402

REPS, DEFF = 20000, 1.3
N_BAND = 3000           # 3 band depths x 1,000 test items (the minimum band)
N_PAIRS = 1000          # the minimum kept pairs for R3 (band depths >= 4)
BAR_R1_GAP, BAR_R1_ABS, BAR_R2, BAR_R3 = 0.30, 0.60, 0.30, 0.30

SCEN = {   # true values: lat = LAT at K = d + 2, dir = DIRECT, k0 / k1 = LAT at K' 0 / 1, cf = cf-follow, fl = cf floor (the model's floor arm)
    "no_effect": dict(lat=0.30, dir=0.30, k0=0.30, k1=0.30, cf=0.10, fl=0.10),
    "effect_strong": dict(lat=0.80, dir=0.30, k0=0.25, k1=0.30, cf=0.55, fl=0.10),
    "effect_marginal": dict(lat=0.64, dir=0.30, k0=0.30, k1=0.32, cf=0.43, fl=0.10),
    "effect_at_bars": dict(lat=0.60, dir=0.30, k0=0.30, k1=0.30, cf=0.40, fl=0.10),
    "pause_compute_no_state": dict(lat=0.70, dir=0.30, k0=0.30, k1=0.35, cf=0.10, fl=0.10),
    "relay_or_precompute": dict(lat=0.75, dir=0.30, k0=0.30, k1=0.30, cf=0.10, fl=0.10),
    "unused_latents": dict(lat=0.32, dir=0.30, k0=0.31, k1=0.31, cf=0.10, fl=0.10),
}


def draw(rng, p, n):
    ne = int(round(n / DEFF)); return rng.binomial(ne, p, REPS) / ne


def main():
    rng = np.random.default_rng(23001); out = {"reps": REPS, "design_effect": DEFF, "n_band_items": N_BAND, "n_pairs": N_PAIRS,
                                               "bars": {"R1_gap": BAR_R1_GAP, "R1_abs": BAR_R1_ABS, "R2_gap": BAR_R2, "R3_over_floor": BAR_R3}, "scenarios": {}}
    for nm, s in SCEN.items():
        lat, dr, k0, k1 = (draw(rng, s[k], N_BAND) for k in ("lat", "dir", "k0", "k1")); cf, fl = draw(rng, s["cf"], N_PAIRS), draw(rng, s["fl"], N_PAIRS)
        r1 = (lat - dr >= BAR_R1_GAP) & (lat >= BAR_R1_ABS); r2 = (lat - k0 >= BAR_R2) & (lat - k1 >= BAR_R2); r3 = cf - fl >= BAR_R3
        out["scenarios"][nm] = {"truth": s, "P_R1": float(r1.mean()), "P_R2": float(r2.mean()), "P_R3": float(r3.mean()), "P_C1_LANG": float((r1 & r2 & r3).mean()),
                                "P_DEEP_NOT_SERIAL": float((r1 & r2 & ~r3).mean())}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ow_oc.json"), "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    for nm, r in out["scenarios"].items():
        print("%-24s R1 %.3f  R2 %.3f  R3 %.3f  C1-LANG %.3f  DEEP-NOT-SERIAL %.3f" % (nm, r["P_R1"], r["P_R2"], r["P_R3"], r["P_C1_LANG"], r["P_DEEP_NOT_SERIAL"]))


if __name__ == "__main__":
    main()
