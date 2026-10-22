"""§224 K9 (FH_v1 §6, fix 9), take 3: OLD vs NEW part_khorizon on ONE card (the RTX 4060, cuda:0; Theory 10-22: the 3070 stays held until the
hardware is checked, so the committed 3070 file cannot be the reference). Model s216_r2_s1, eval_bs 32, fp32. OLD = the pre-§224 boxes/vbank_diag.py
(4e56c8ca, ks 24..64; tree scratch/_tmp/pre224_tree), NEW = the committed one (ks 24..128). Each runs in its own process (two module versions)
on the same test-item subset; chunks are formed per n in test order, so they are the full run's chunks.
  a  n 1..4: NEW must take the single-rollout plan [(32, 24..128)] (the 3090 path); every arm's K' <= 64 answers == OLD's (bitwise).
  b  n 7..8 (deepest): NEW must split [(32, K' <= k_lo), (b2 < 32, the rest)], no K' dropped; K' <= k_lo answers == OLD's (bitwise); K' <= 64 run at
     b2 reported as a share. Both processes with PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 (allocator only; take 1 OOMed by fragmentation).
Take 1 (3070, full n 1..8) OOMed; take 2 (3070, committed-file reference) was stopped by Theory's hold. Accuracies are never printed.
    python -B scratch/s224_k9.py a|b|b64  (driver; K9_DEV=cuda:N picks the card)      python -B scratch/s224_k9.py run <tree> <part> <out.json>   (worker)"""
import json
import os
import subprocess
import sys
import time

RUN = "D:/Latent/boxes/runs_fsq/s216_r2_s1"; DEV = os.environ.get("K9_DEV", "cuda:0"); OUT = "D:/Latent/scratch/gate_s224/k9.json"; TMP = "D:/Latent/scratch/_tmp/s224_k9"
NS = {"a": (1, 2, 3, 4), "b": (7, 8), "b64": (7, 8)}; LIM = {"b64": 64}; TREES = {"old": "D:/Latent/scratch/_tmp/pre224_tree", "new": "D:/Latent"}


def worker(tree, part, out):
    sys.path.insert(0, tree)
    import torch
    import boxes.nested170 as N
    import boxes.vbank_diag as VD
    import boxes.vbank_eval as VE
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    m, _ = VE.load(RUN, DEV); TE = [it for it in N.test_set(1000) if it["n"] in NS[part]]; t0 = time.time()
    if part in LIM:                                                       # b64: the first LIM items per n in test order (= the full run's first chunks)
        cnt = {}; TE = [it for it in TE if cnt.__setitem__(it["n"], cnt.get(it["n"], 0) + 1) or cnt[it["n"]] <= LIM[part]]
    R = VD.part_khorizon(m, TE, DEV, 32, RUN)
    keep = {k: R.get(k) for k in ("plan", "k_tried", "ks", "k_max", "eval_bs_by_k")}
    json.dump({"tree": tree, "file": VD.__file__, "items": R["items"], "meta": keep, "sec": round(time.time() - t0), "card": torch.cuda.get_device_name(DEV)},
              open(out, "w", newline="\n"))


def driver(part):
    os.makedirs(TMP, exist_ok=True); env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    if part in ("b", "b64"):
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    res = {}
    for tag, tree in TREES.items():
        f = "%s/%s_%s.json" % (TMP, part, tag)
        r = subprocess.run([sys.executable, "-B", __file__, "run", tree, part, f], capture_output=True, text=True, env=env, cwd="D:/Latent")
        if r.returncode != 0:
            res[tag] = {"rc": r.returncode, "stderr": r.stderr[-1500:]}; continue
        res[tag] = json.load(open(f))
    o = json.load(open(OUT)) if os.path.exists(OUT) and "take" in json.load(open(OUT)) else {"take": 3}
    if "items" not in res.get("old", {}) or "items" not in res.get("new", {}):
        o[part] = {"pass": False, "error": {k: v for k, v in res.items() if "items" not in v}}
    else:
        O, W = res["old"], res["new"]; bs_by_k = W["meta"]["eval_bs_by_k"]; eq, sh = {}, {}
        for arm in O["items"]:
            if arm in ("y", "idx"):
                continue
            for n in O["items"][arm]:
                for k, v in O["items"][arm][n].items():
                    same = [int(a == b) for a, b in zip(W["items"][arm][n][k], v)]
                    if bs_by_k[k] == 32:
                        eq["%s|%s|%s" % (arm, n, k)] = len(same) == len(v) and sum(same) == len(v)
                    else:
                        sh.setdefault(k, []).extend(same)
        plan = W["meta"]["plan"]; single = len(plan) == 1 and plan[0]["eval_bs"] == 32
        split = len(plan) == 2 and plan[0]["eval_bs"] == 32 and plan[1]["eval_bs"] < 32
        o[part] = {"ns": NS[part], "new_meta": W["meta"], "old_meta": O["meta"], "sec": {"old": O["sec"], "new": W["sec"]}, "card": W["card"],
                   "files": {"old": O["file"], "new": W["file"]}, "n_bitwise_compared": len(eq), "n_bitwise_equal": sum(eq.values()),
                   "unequal": [k for k, v in eq.items() if not v][:20], "share_equal_at_other_bs": {k: sum(v) / len(v) for k, v in sh.items()},
                   "y_equal": O["items"]["y"] == W["items"]["y"], "idx_equal": O["items"]["idx"] == W["items"]["idx"], "alloc_conf": env.get("PYTORCH_CUDA_ALLOC_CONF")}
        o[part]["pass"] = bool((single if part == "a" else split) and W["meta"]["ks"] == [24, 32, 48, 64, 96, 128] and len(eq) > 0 and all(eq.values())
                               and o[part]["y_equal"] and o[part]["idx_equal"])
    if "a" in o and ("b" in o or "b64" in o):                            # take 3b (10-22): b was stopped at 7.1 h (launch-bound bs fallback); b64 replaces it
        o["pass"] = o["a"]["pass"] and (o.get("b64", {}).get("pass") or o.get("b", {}).get("pass", False))
    json.dump(o, open(OUT, "w", newline="\n"), indent=1); print("K9", part, "pass", o[part]["pass"], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        worker(*sys.argv[2:5])
    else:
        driver(sys.argv[1])
