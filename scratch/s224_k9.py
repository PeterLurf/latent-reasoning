"""§224 K9 (FH_v1 §6, fix 9), take 2: boxes.vbank_diag.part_khorizon (ks to 128) on the RTX 3070 at eval_bs 32 (the 216-14 card / bs) on
s216_r2_s1 (G-IH0 bitwise in its committed eval_khorizon.json), in two parts so both memory paths are exercised (take 1, the full n 1..8 part,
OOMed at 0:56 in the 3070's halved-bs K' 128 rollouts - fragmentation, 2.2 GiB reserved-unallocated - and on an 8 GB card never tests the
single-rollout path the 3090 evals use):
  a  the test items with n 1..4 (their per-n chunks are the full run's: chunks are formed per n in test order): K' 128 at bs 32 must fit -> plan =
     ONE rollout per chunk to 128 (the 3090 path); every arm's K' <= 64 answers == the committed file's, per n (bitwise);
  b  n 7..8 (the deepest chunks): K' 128 at bs 32 must NOT fit -> plan [(32, K' <= k_lo), (b2 < 32, the rest)], no K' dropped; K' <= 64 answers
     bitwise vs the committed file where they ran at bs 32; any K' <= 64 run at b2 is reported as a share (another eval_bs). Run with
     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 (allocator only; no numeric change).
Accuracies are not printed (only gate fields) -> scratch/gate_s224/k9.json.   python -B scratch/s224_k9.py a|b [--tree DIR]"""
import json
import os
import sys
import time

TREE = "D:/Latent"
for i, v in enumerate(sys.argv):
    if v == "--tree":
        TREE = sys.argv[i + 1]
sys.path.insert(0, TREE)
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank_diag as VD  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402

RUN = "D:/Latent/boxes/runs_fsq/s216_r2_s1"; DEV = "cuda:1"; OUT = "D:/Latent/scratch/gate_s224/k9.json"
NS = {"a": (1, 2, 3, 4), "b": (7, 8)}


def main():
    part = sys.argv[1]; assert part in NS
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert "3070" in torch.cuda.get_device_name(DEV), torch.cuda.get_device_name(DEV)
    ref = json.load(open(os.path.join(RUN, "eval_khorizon.json"))); m, _ = VE.load(RUN, DEV)
    TE = [it for it in N.test_set(1000) if it["n"] in NS[part]]
    t0 = time.time(); R = VD.part_khorizon(m, TE, DEV, 32, RUN); bs_by_k = R["eval_bs_by_k"]
    eq, sh = {}, {}
    for arm in ref["items"]:
        if arm in ("y", "idx"):
            continue
        for n in [str(v) for v in NS[part]]:
            for k, v in ref["items"][arm][n].items():
                same = [int(a == b) for a, b in zip(R["items"][arm][n][k], v)]
                if bs_by_k[k] == 32:
                    eq["%s|%s|%s" % (arm, n, k)] = sum(same) == len(same) == len(v)
                else:
                    sh.setdefault(k, []).extend(same)
    y_ok = all(R["items"]["y"][str(n)] == ref["items"]["y"][str(n)] for n in NS[part])
    o = json.load(open(OUT)) if os.path.exists(OUT) else {}
    o[part] = {"ns": NS[part], "plan": R["plan"], "k_tried": R["k_tried"], "ks": R["ks"], "eval_bs_by_k": bs_by_k, "sec": round(time.time() - t0),
               "n_bitwise_compared": len(eq), "n_bitwise_equal": sum(eq.values()), "unequal": [k for k, v in eq.items() if not v][:20],
               "share_equal_at_other_bs": {k: sum(v) / len(v) for k, v in sh.items()}, "y_equal": y_ok,
               "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}
    single = len(R["plan"]) == 1 and R["plan"][0]["eval_bs"] == 32
    split = len(R["plan"]) == 2 and R["plan"][0]["eval_bs"] == 32 and R["plan"][1]["eval_bs"] < 32
    o[part]["pass"] = bool((single if part == "a" else split) and R["ks"] == [24, 32, 48, 64, 96, 128] and len(eq) > 0 and all(eq.values()) and y_ok)
    if "a" in o and "b" in o:
        o["pass"] = o["a"]["pass"] and o["b"]["pass"]
    json.dump(o, open(OUT, "w", newline="\n"), indent=1); print("K9", part, "pass", o[part]["pass"], flush=True)


if __name__ == "__main__":
    main()
