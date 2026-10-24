"""HOLD-C1 (HC_draft.md) $0 check K1: checkpoint availability and args identity for every HC model (no torch, no GPU; idle priority).
Writes hc_ckpt_check.out.json next to this script (LF).
    python -B docs/research_holdc1_2026-10-24/hc_ckpt_check.py [--sha]     (--sha adds a full sha256 of each model.pt, ~4 GB read)
Per model: model.pt present / size / mtime (/ sha256), args.json seed / init_from / lr / steps / far / fh / fh_lambda_fixed, and which eval_vb.json
parts exist (acc / probe / vswap / stall) with their card. Args diffs (shared keys, identity keys removed):
  FT-CTRL vs HC and RD-FT vs HC (expected: the far / fh flags only), s224_fhft_s2 vs s230_fhftfix_s1 (expected: seed, init_from, fh_lambda_fixed),
  and the PLANNED Stage-B cell s232_fhftfix_s2 = s230_fhftfix_s1's args with seed 2, init_from s216_d3t_s2 (expected diff: seed + init_from)."""
import hashlib
import json
import os
import sys
import time

try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if hasattr(psutil, "IDLE_PRIORITY_CLASS") else 19)
except Exception:                                                          # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R = os.path.join(ROOT, "boxes", "runs_fsq")
ROLES = {"s1": {"BASE": "s213_d3t_s1", "FT-CTRL": "s215_ftctrl_s1", "RD-FT": "s224_rdft_s1", "HC": "s230_fhftfix_s1"},
         "s2": {"BASE": "s216_d3t_s2", "FT-CTRL": "s224_ftctrl_s2", "RD-FT": "s224_rdft_s2", "HC-near": "s224_fhft_s2"}}
IDENT = {"out", "args_ref", "train_gpu", "train_driver", "torch_version", "cuda_version", "cudnn_version", "python_version", "image_tag", "device"}
FIELDS = ("seed", "init_from", "lr", "steps", "far", "fh", "fh_lambda_fixed", "far_t", "far_n", "mode", "cur", "c_sched")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def adiff(a, b):
    return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if k not in IDENT and a.get(k) != b.get(k)}


def main():
    do_sha = "--sha" in sys.argv; out = {"models": {}, "diffs": {}}
    for lin, roles in ROLES.items():
        for role, run in roles.items():
            d = os.path.join(R, run); mp = os.path.join(d, "model.pt"); o = {"lineage": lin, "role": role, "model_pt": os.path.exists(mp)}
            if o["model_pt"]:
                st = os.stat(mp); o.update(size=st.st_size, mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)))
                if do_sha:
                    o["sha256"] = sha(mp)
            a = json.load(open(os.path.join(d, "args.json"))); o["args"] = {k: a.get(k) for k in FIELDS}
            ev = os.path.join(d, "eval_vb.json")
            if os.path.exists(ev):
                e = json.load(open(ev)); o["eval_vb_parts"] = [p for p in ("acc_n2", "probe", "vswap", "stall") if p in e]; o["eval_vb_card"] = e.get("card")
            o["eval_khorizon"] = os.path.exists(os.path.join(d, "eval_khorizon.json")); out["models"][run] = o
    A = lambda r: json.load(open(os.path.join(R, r, "args.json")))  # noqa: E731
    hc1 = A("s230_fhftfix_s1")
    out["diffs"]["FT-CTRL s1 vs HC s1"] = adiff(A("s215_ftctrl_s1"), hc1)
    out["diffs"]["RD-FT s1 vs HC s1"] = adiff(A("s224_rdft_s1"), hc1)
    out["diffs"]["FT-CTRL s2 vs HC-near s2"] = adiff(A("s224_ftctrl_s2"), A("s224_fhft_s2"))
    out["diffs"]["RD-FT s2 vs HC-near s2"] = adiff(A("s224_rdft_s2"), A("s224_fhft_s2"))
    out["diffs"]["HC-near s2 vs HC s1"] = adiff(A("s224_fhft_s2"), hc1)
    plan = dict(hc1); plan.update(seed=2, init_from="boxes/runs_fsq/s216_d3t_s2/model.pt", out="boxes/runs_fsq/s232_fhftfix_s2")
    out["diffs"]["PLANNED s232_fhftfix_s2 vs HC s1"] = adiff(plan, hc1)
    out["diffs"]["PLANNED s232_fhftfix_s2 vs FT-CTRL s2"] = adiff(plan, A("s224_ftctrl_s2"))
    out["ok"] = {"all_model_pt": all(v["model_pt"] for v in out["models"].values()),
                 "planned_diff_is_seed_init": set(out["diffs"]["PLANNED s232_fhftfix_s2 vs HC s1"]) == {"seed", "init_from"}}
    fo = os.path.join(HERE, "hc_ckpt_check.out.json")
    with open(fo, "w", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    for r, o in out["models"].items():
        print("%-18s %-3s %-8s pt=%s %s parts=%s card=%s khor=%s" % (r, o["lineage"], o["role"], o["model_pt"], o.get("mtime", ""), o.get("eval_vb_parts"),
                                                                     (o.get("eval_vb_card") or "")[-8:], o["eval_khorizon"]))
    for k, v in out["diffs"].items():
        print(k, v)
    print("ok", out["ok"], "wrote", fo)


if __name__ == "__main__":
    sys.exit(main())
