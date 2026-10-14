"""SEEDSTAB2 rev 2 (POST-HOC data): step_recover pooled over t >= 2, weighted by each t's test count, for the 16 D7 models; and the count of
converged (model, t) cells at t = 5..7 with step_recover >= 0.30. Reads eval.json s175.step_recover / step_recover_info only. Output: pooled_t2p.json."""
import json, os

R = "D:/Latent/boxes/runs_fsq/"
conv = ["s185_rms_s1", "s185_rms_s2", "s190_rmsnopass_s1", "s190_rmsnopass_s2", "s187_nopass_s1", "s187_nopass_s2", "s182_window_s1", "s182_window_s2", "s184_reweight_s2"]
fail = ["s185_rms_s3", "s199_csr_s1", "s185_den_s1", "s188_rwwin_s1", "s196_rmsgf_s3", "s182_hold_s1", "s182_nofade_s1"]


def find(d, k):
    if isinstance(d, dict):
        for k_, v in d.items():
            if k_ == k:
                return v
            r = find(v, k)
            if r is not None:
                return r


out = {}; cells = []
for r in conv + fail:
    e = json.load(open(R + r + "/eval.json")); sr = find(e, "step_recover"); info = find(e, "step_recover_info")
    ts = [t for t in sr if int(t) >= 2]; w = {t: info["n"][t]["test"] for t in ts}
    out[r] = {"pooled_t2p": sum(sr[t] * w[t] for t in ts) / sum(w.values()), "class": "conv" if r in conv else "fail"}
    if r in conv:
        cells += [(r, t, sr[t]) for t in ("5", "6", "7")]
for r, v in out.items():
    print("%-20s %s %.3f" % (r, v["class"], v["pooled_t2p"]))
below = [c for c in cells if c[2] < 0.30]
print("converged t5-7 cells >= 0.30: %d of %d; below: %s" % (len(cells) - len(below), len(cells), below))
out["_t57_cells_ge_030"] = [len(cells) - len(below), len(cells)]
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pooled_t2p.json"), "w", newline="\n"), indent=1)
