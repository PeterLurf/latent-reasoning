"""WARM-COT draft calcs (CPU, seconds): (1) the per-pass routing-fidelity model acc(rho) on C's acc_by_m; (2) the tilt ratio JOINT / WARM in
Part II's swept-gain model with b_P proportional to gamma(t) * pi(t); (3) OC of the single-seed readings; (4) cost. Reads §200 C's eval.json
and the b286 timing (read-only). Writes calc_warm.json next to this script."""
import json
import math
import os

OUT = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(r"D:\Latent\boxes\runs_fsq\gsm_txt_aug_s1\eval.json"))
TL = json.load(open(r"D:\Latent\boxes\runs_fsq\s200_time_lat_aug\timing.json"))
TT = json.load(open(r"D:\Latent\boxes\runs_fsq\s200_time_txt_aug\timing.json"))
res = {}

# (1) acc(rho) = sum_m n_m acc_C(m) rho^m / N   (inherit C's per-item plan; each pass routes with fidelity rho, independent)
n = {int(k): v for k, v in E["n_by_m"].items()}; a = {int(k): v for k, v in E["acc_by_m"].items()}; N = sum(n.values())
res["acc_rho"] = {str(r): sum(n[m] * a[m] * r ** max(m, 1) for m in n) / N for r in (0.80, 0.85, 0.90, 0.95, 0.98, 1.0)}

# (2) tilt ratio. gamma ramps 2k -> 6k; a(t) = a_max * gamma(t), 1/a_max = 3.3k (Part II calibration); b_P(t) = k * gamma(t) * pi(t).
#     mu_Z ∝ ∫ exp(-A(s)) gamma(s) pi(s) ds. pi_W(s) = pi_C * r (retention r constant); pi_J(s) = C's valid curve (shared-trunk proxy).
vh = [(0, 0.0)] + [tuple(x) for x in E["valid_hist"]]; piC = 0.35


def piJ(s):
    for (s0, v0), (s1, v1) in zip(vh, vh[1:]):
        if s <= s1:
            return v0 + (v1 - v0) * (s - s0) / (s1 - s0)
    return vh[-1][1]


def gam(s):
    return 0.0 if s <= 2000 else min(1.0, (s - 2000) / 4000)


amax = 1 / 3300.0; ds = 10.0; A = 0.0; num_J = num_W = 0.0; s = 2000.0; partial = {}
while s < 60000:
    g_ = gam(s); A += amax * g_ * ds; w = math.exp(-A) * g_ * ds
    num_J += w * piJ(s); num_W += w * piC
    for cut in (6000, 8000, 12000):
        if abs(s - cut) < ds / 2:
            partial[str(cut)] = num_J / num_W
    s += ds
res["tilt_ratio_J_over_W_r1"] = num_J / num_W; res["tilt_ratio_partial"] = partial
res["tilt_note"] = "z_J / z_W at equal sigma_Z, WARM retention r = 1; with retention r, z_W scales by r, so J > W iff r < ratio"

# (3) OC. Single-seed binary readings; paired accuracy readings at +-2.8 points.
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
oc = {}
for pw, pa in ((0.9, 0.3), (0.9, 0.6), (0.7, 0.3), (0.5, 0.5), (0.3, 0.3)):
    oc["Pwarm%.1f_Praw%.1f" % (pw, pa)] = {"P(W formed, A not)": pw * (1 - pa), "P(W formed)": pw}
res["oc_loop"] = oc
pr = {}
for se in (1.25, 1.5):
    for d in (0.0, -2.0, -4.0, -6.0, +4.0):
        pr["se%.2f_delta%+.0f" % (se, d)] = {"P(matches)": Phi((2.8 - d) / se) - Phi((-2.8 - d) / se), "P(below)": Phi((-2.8 - d) / se), "P(above)": 1 - Phi((2.8 - d) / se)}
res["oc_paired"] = pr

# (4) cost. Anchors: b286 (3090 / 595.84): latent A config 1.482 s/step at micro 4; text C config timed 0.265; C realised 6105 s / 24k.
lat = TL["sec_per_step"]; txt_t = TT["sec_per_step"]; txt_r = 6105 / 24000; ph = 0.255; host = 0.10
res["anchors"] = {"latent_s_per_step_b286": lat, "text_s_per_step_b286": txt_t, "text_s_per_step_C_realised": txt_r}
cells = {"WARM 12k": (12000, lat, 0.05), "WARM 8k": (8000, lat, 0.05), "C+ 12k (text continued)": (12000, txt_r, 0.05),
         "JOINT 24k": (24000, lat + txt_r, 0.05), "JOINT 12k": (12000, lat + txt_r, 0.05)}
cost = {}
for k, (st, sps, ev_h) in cells.items():
    h = st * sps / 3600 + ev_h; cost[k] = {"hours_central": round(h, 2), "usd_central": round(h * ph + host, 2), "hours_high": round(h * 1.35, 2), "usd_high": round(h * 1.35 * ph + host, 2)}
res["cost"] = cost
json.dump(res, open(os.path.join(OUT, "calc_warm.json"), "w", newline="\n"), indent=1); print(json.dumps(res, indent=1))
