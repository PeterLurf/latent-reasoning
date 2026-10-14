# read-only extraction of K' profiles and acc_by_t from existing eval files (CPU, no torch)
import json, os, sys
R = "D:/Latent/boxes/runs_fsq"
runs = ["s175_auxk8_s1","s182_nofade_s1","s182_hold_s1","s182_window_s1","s182_window_s2","s184_reweight_s1","s184_reweight_s2",
        "s185_den_s1","s185_rms_s1","s185_rms_s2","s185_rms_s3","s187_nopass_s1","s187_nopass_s2","s188_rwwin_s1","s190_rmsnopass_s1","s190_rmsnopass_s2"]
out = {}
for r in runs:
    p = os.path.join(R, r, "eval.json")
    if not os.path.exists(p): print(r, "no eval.json"); continue
    e = json.load(open(p)); a = e.get("s170", {}).get("acc_by_n_kprime", {})
    tk = e.get("think_k"); kw = None
    try:
        args = json.load(open(os.path.join(R, r, "args.json"))); kw = args.get("k_window"); 
    except Exception: args = {}
    s7 = e.get("s177") or {}
    out[r] = {"think_k": tk, "k_window": kw, "add_norm": args.get("add_norm"), "aux_nopass": args.get("aux_nopass"), "kp": a}
    print("==", r, "K", tk, "win", kw, "norm", args.get("add_norm"), "nopass", args.get("aux_nopass"))
    keys = [k for k in a.keys()]
    for n in range(2, 13):
        row = []
        for k in keys:
            v = a[k].get(str(n)); row.append("%s:%s" % (k, "%.3f" % v if v is not None else "-"))
        print("  n%2d " % n + " ".join(row))
    hp = os.path.join(R, r, "eval_s173_halt.json")
    if os.path.exists(hp):
        h = json.load(open(hp)); print("  halt cap", h.get("cap"), "think_k", h.get("think_k"))
        for n, v in h["acc_by_t"].items():
            print("   acc_by_t n%s: " % n + " ".join("%.2f" % x for x in v))
        out[r]["acc_by_t"] = h["acc_by_t"]; out[r]["halt"] = h["halt"]; out[r]["halt_fixed_k"] = h["halt_fixed_k"]
json.dump(out, open(os.path.join(os.path.dirname(__file__), "kprofile.json"), "w"), indent=1)
