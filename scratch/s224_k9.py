"""§224 K9 (FH_v1 §6, fix 9): TREE's part_khorizon to K' 128 on the RTX 3070 at eval_bs 32 (the 216-14 card / bs) on s216_r2_s1 (G-IH0 bitwise in
its committed eval_khorizon.json) -> run A with the default mem_frac 0.85: every arm's K' <= 64 answers == the committed file's, per n (bitwise);
no K' dropped (ks 24..128 all present). The eval_bs fallback must be exercised: if A did not split, run B with mem_frac set midway between A's
measured peaks at (32, 64) and (32, 128) -> plan [(32, <= 64), (b2 < 32, 96 / 128)], K' <= 64 again bitwise vs the committed file, K' 96 / 128
answers vs A reported as a share (another eval_bs; descriptive). Accuracies are not printed (only gate fields) -> scratch/gate_s224/k9.json.
    python -B scratch/s224_k9.py [--tree DIR]"""
import json
import os
import re
import sys
import time

TREE = "D:/Latent/scratch/_tmp/fh224_tree"
for i, v in enumerate(sys.argv):
    if v == "--tree":
        TREE = sys.argv[i + 1]
sys.path.insert(0, TREE)
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank_diag as VD  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402

RUN = "D:/Latent/boxes/runs_fsq/s216_r2_s1"; DEV = "cuda:1"; OUT = "D:/Latent/scratch/gate_s224/k9.json"


def cmp_committed(res, ref):
    eq = {}
    for arm in ref["items"]:
        if arm in ("y", "idx"):
            continue
        for n, byk in ref["items"][arm].items():
            for k, v in byk.items():
                eq["%s|%s|%s" % (arm, n, k)] = res["items"][arm][n][k] == v
    return {"n_compared": len(eq), "n_equal": sum(eq.values()), "unequal": [k for k, v in eq.items() if not v][:20],
            "y_equal": res["items"]["y"] == ref["items"]["y"], "idx_equal": res["items"]["idx"] == ref["items"]["idx"]}


def peak(res, b_, k):
    for b, kk, s in res["k_tried"]:
        if b == b_ and kk == k:
            m_ = re.search(r"peak ([0-9.]+)", s); return float(m_.group(1)) if m_ else None
    return None


def main():
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert "3070" in torch.cuda.get_device_name(DEV), torch.cuda.get_device_name(DEV)
    ref = json.load(open(os.path.join(RUN, "eval_khorizon.json"))); m, _ = VE.load(RUN, DEV); TE = N.test_set(1000); o = {"run": RUN, "card": torch.cuda.get_device_name(DEV)}
    t0 = time.time(); A = VD.part_khorizon(m, TE, DEV, 32, RUN); o["A"] = {"plan": A["plan"], "k_tried": A["k_tried"], "ks": A["ks"], "sec": round(time.time() - t0),
                                                                          "vs_committed": cmp_committed(A, ref), "G-IH0_bitwise": A["gate_IH0"]["bitwise"]}
    json.dump(o, open(OUT, "w", newline="\n"), indent=1)
    if len(A["plan"]) == 1:                                                 # the fallback was not needed: force it
        p64, p128 = peak(A, 32, 64), peak(A, 32, 128)
        if p64 is None:                                                     # A fitted 128 at once: measure (32, 64) inside B's own probe instead
            p64 = 0.0
        mf = (p64 + p128) / 2 if p64 else p128 * 0.95; o["B_mem_frac"] = mf
        t0 = time.time(); B = VD.part_khorizon(m, TE, DEV, 32, RUN, mem_frac=mf)
        hi = [k for b_, ks_ in [(d["eval_bs"], d["ks"]) for d in B["plan"]] if b_ < 32 for k in ks_]
        sh = {}
        for arm in A["items"]:
            if arm in ("y", "idx"):
                continue
            for n in A["items"][arm]:
                for k in hi:
                    a_, b__ = A["items"][arm][n][str(k)], B["items"][arm][n][str(k)]; sh.setdefault(str(k), []).extend(int(x == y) for x, y in zip(a_, b__))
        o["B"] = {"plan": B["plan"], "k_tried": B["k_tried"], "ks": B["ks"], "sec": round(time.time() - t0), "vs_committed": cmp_committed(B, ref),
                  "hi_ks_share_equal_vs_A": {k: sum(v) / len(v) for k, v in sh.items()}}
    fb = o.get("B", o["A"]); ok_a = o["A"]["vs_committed"]["n_equal"] == o["A"]["vs_committed"]["n_compared"] > 0 and o["A"]["ks"] == [24, 32, 48, 64, 96, 128]
    ok_fb = len(fb["plan"]) == 2 and fb["plan"][0]["eval_bs"] == 32 and fb["plan"][1]["eval_bs"] < 32 and fb["vs_committed"]["n_equal"] == fb["vs_committed"]["n_compared"] and fb["ks"] == [24, 32, 48, 64, 96, 128]
    o["pass"] = bool(ok_a and ok_fb and o["A"]["vs_committed"]["y_equal"] and o["A"]["vs_committed"]["idx_equal"])
    json.dump(o, open(OUT, "w", newline="\n"), indent=1); print("K9 pass", o["pass"], flush=True)


if __name__ == "__main__":
    main()
