"""Exploratory (theory stream): front trajectory per model from the frozen far_sweep arrays (seed 2033).
Edge index e = 0..11 -> edge into distance e+1 (0..5 chain A = near, 6..11 chain B = far).
Front f_k = argmax_e excess at step k (None if max < thr). Amplitude a_k = that max."""
import json, sys
import numpy as np

def load_sets():
    out = {}
    for fs, lab in (("results/far_sweep_p30_s2033.json", "results/p30.json"), ("results/far_sweep_l_s2033.json", "results/p_l.json")):
        F = json.load(open(fs)); rows = json.load(open(lab))["rows"]
        for k, v in F.items():
            r = rows.get(k.split("/")[-1])
            if r is None: continue
            X = np.hstack([np.array(v["near"]), np.array(v["far"])])     # (12 steps, 12 edges)
            out[k.split("/")[-1]] = dict(X=X, K=v["K"], **{kk: r[kk] for kk in ("P7", "extrap", "d7", "EA", "in_peak")})
    return out

def traj(X, thr=0.01):
    f, a = [], []
    for k in range(X.shape[0]):
        e = int(np.argmax(X[k])); m = float(X[k, e])
        f.append(e + 1 if m >= thr else None); a.append(m)
    return f, a

if __name__ == "__main__":
    D = load_sets()
    for name, d in sorted(D.items(), key=lambda kv: (kv[1]["extrap"], kv[1]["P7"])):
        f, a = traj(d["X"])
        fs = " ".join("%2s" % (x if x is not None else ".") for x in f)
        print("%-10s K%d %s P7 %.2f d7 %.3f | front(dist) %s | amp %s" % (name, d["K"], "EXT" if d["extrap"] else "   ", d["P7"], d["d7"], fs,
              " ".join("%3d" % round(1000 * x) for x in a)))
