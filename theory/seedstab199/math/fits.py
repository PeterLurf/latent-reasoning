"""N2: kappa implied by s3's numbers under the margin model p_t = Phi(m0 kappa^t / sigma) (Theorem 1 + Gaussian logit noise), vs the
step-ratio and delta-cos statistics. Pure arithmetic on numbers quoted in FACTS.md / eval.json (s185_rms_s3)."""
import json, math
import numpy as np
from statistics import NormalDist
N = NormalDist()
def fit(ts, ps):
    y = np.log([N.inv_cdf(p) for p in ps]); A = np.c_[np.ones(len(ts)), ts]; c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ c; return float(np.exp(c[1])), float(np.exp(c[0])), float(np.sqrt(np.mean(res ** 2)))
out = {}
# (i) aux conditional per-step success (FACTS: 0.97, 0.92, 0.90, 0.86, 0.73, 0.72), taken as passes t = 3..8
k, a, r = fit(np.arange(3, 9), [0.97, 0.92, 0.90, 0.86, 0.73, 0.72]); out["aux_conditional_t3_8"] = dict(kappa=k, m0_over_sigma=a, rms_resid_log=r)
# (ii) answer accuracy by n at K' = 8 (eval.json s185_rms_s3 acc_by_n_kprime['8']): per-step ratio acc(n)/acc(n-1), n = 3..8
acc = {1: 1.0, 2: 1.0, 3: 0.962, 4: 0.845, 5: 0.646, 6: 0.488, 7: 0.359, 8: 0.231}
ps = [acc[n] / acc[n - 1] for n in range(3, 9)]; k2, a2, r2 = fit(np.arange(3, 9), ps); out["answer_ratio_n3_8"] = dict(ratios=[round(p, 3) for p in ps], kappa=k2, m0_over_sigma=a2, rms_resid_log=r2)
# (iii) step ratio (whole state, lyap192): log ratios -0.175, -0.150, -0.095 at t = 2,3,4
out["step_ratio"] = dict(kappa_t2=math.exp(-0.175), kappa_t3=math.exp(-0.150), kappa_t4=math.exp(-0.095))
# (iv) delta cos after 16 passes (zero vs donor start): if the donor's offset decays as kappa^16 and |S_donor| ~ |S*|, then 1 - cos ~ kappa^32 (1 - c0)
#      with c0 = cos between two different items' states; c0 unknown -> report kappa for c0 in {0, 0.3, 0.5}
for c0 in (0.0, 0.3, 0.5):
    out["delta_cos_kappa_c0_%.1f" % c0] = (((1 - 0.978) / (1 - c0)) ** (1 / 32))
# (v) Theorem-1 horizon N = log(L diam / mu)/log(1/kappa) for the fitted kappa: passes until the margin halves / falls 10x
for nm, kk in (("aux", k), ("answer", k2)):
    out["halving_passes_" + nm] = math.log(2) / math.log(1 / kk); out["tenfold_passes_" + nm] = math.log(10) / math.log(1 / kk)
print(json.dumps(out, indent=1)); json.dump(out, open("fits.json", "w", newline="\n"), indent=1)
