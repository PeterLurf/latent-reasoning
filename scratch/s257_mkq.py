"""§256 DISTRACT runs (DS_v1 §2; Audit: cell $1.4, guard $2.9): queues b391 dis_s1 / b392 dis_s2 + frozen args + pre-deploy check.
Each queue (one RTX 3090): K7 mem line + G-TIME line (the training flags, micro 2), then the EVAL-ONLY line rcb_plaw4_s<k>_ds256 (ws_deep + insert_ho on the init, through
scratch.s256_guarded --mode evalonly: blob + model sha asserted), then the TRAIN line dis_s<k> (s256_guarded --mode train: blob + init sha asserted, the eval-only outputs required)
with rcb_plaw4_s<k>'s flags (gate_s251.RCB, seed k) + --init_model the unbuffered-read rcb_plaw4_s<k> copy + --distract_p 1, micro 2 / eval_bs 250 pinned, 16k, --keep_final_ckpt,
eval_parts = rcb's + test_perm, insert_ho, ws_deep. Pre-deploy: every asserted file's git blob id recomputed from an UNBUFFERED read == HEAD; init copies' unbuffered sha256.
    python -B scratch/s257_mkq.py"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s251 as G1  # noqa: E402
import s228_args_check as C  # noqa: E402
from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN, sha_nobuf  # noqa: E402
import ctypes  # noqa: E402
import ctypes.wintypes as W  # noqa: E402


def nobuf_bytes(p):
    size = os.path.getsize(p); h = k32.CreateFileW(os.path.abspath(p), GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, None)
    CH = 1 << 24; buf = k32.VirtualAlloc(None, CH, 0x3000, 0x04); got = W.DWORD(0); left = size; out = b""
    try:
        while left > 0:
            assert k32.ReadFile(h, buf, CH, ctypes.byref(got), None); n = min(got.value, left)
            if n == 0:
                break
            out += ctypes.string_at(buf, n); left -= n
    finally:
        k32.CloseHandle(h)
    return out

EVAL = "acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws,lp5h,swap_kd,test_perm,insert_ho,ws_deep"
INIT = {1: ("boxes/runs_fsq/rcb_plaw4_s1_init/model.pt", "67775e2492044a8a"), 2: ("boxes/runs_fsq/rcb_plaw4_s2_init/model.pt", "a1d93d77fff86d93")}
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/insert_render.py", "boxes/perm_render.py", "scratch/s256_guarded.py", "scratch/s256_ho_shas.json",
        "scratch/gate_s255d.py", "scratch/gate_s228.py", "scratch/gate_s251.py", "scratch/s228_args_check.py", "scratch/s222_det_wrap.py", "boxes/_inject_aw_s252.py"]
GSHIP = "scratch/gate_s255d.py scratch/gate_s228.py scratch/gate_s251.py scratch/s228_args_check.py scratch/s222_det_wrap.py boxes/_inject_aw_s252.py boxes/runs_fsq/aw_plaw2_s1/model.pt"


def flags(k):
    fl = list(G1.RCB); i = fl.index("--init_model"); fl[i + 1] = INIT[k][0]; j = fl.index("--init_model_sha"); fl[j + 1] = INIT[k][1]
    return fl + (["--seed", "2"] if k == 2 else []) + ["--distract_p", "1", "--micro", "2", "--eval_bs", "250"]


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}; pre = {}
for p in BLOB:
    b = nobuf_bytes(p); pre[p] = {"head": head[p], "disk": hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()}
inits = {k: sha_nobuf(os.path.abspath(v[0])) for k, v in INIT.items()}
ok = all(v["head"] == v["disk"] for v in pre.values()) and all(inits[k].startswith(INIT[k][1]) for k in INIT)
os.makedirs("scratch/gate_s257", exist_ok=True); out = {"blobs": pre, "init_unbuffered_sha256": inits, "predeploy_ok": ok, "args": {}}
assert ok, ("pre-deploy", pre, inits)
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB)
for k, n in ((1, 391), (2, 392)):
    run = "dis_s%d" % k; fl = flags(k); tail = ["--keep_final_ckpt", "--ckpt_every", "1000", "--eval_parts", EVAL]
    d = G8.fresh(ROOT + "/scratch/_tmp/s257/args_" + run)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + tail + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, rc
    new = json.load(open(d + "/args.json")); json.dump(new, open("scratch/s257_args_frozen_%s.json" % run, "w", newline="\n"), indent=1)
    ref = json.load(open("boxes/runs_fsq/rcb_plaw4_s%d/args.json" % k)); dif = {kk: v for kk, v in C.diff(ref, new).items() if kk != "pair_stats"}
    out["args"][run] = {"keys": sorted(dif), "diff": dif, "micro": new["micro"], "auto_micro": new["auto_micro"], "eval_bs": new["eval_bs"], "seed": new["seed"]}
    ev = "boxes/runs_fsq/rcb_plaw4_s%d_ds256" % k; fs = " ".join(fl)
    q = ("# THEORY §256 DISTRACT (DS_v1; Audit cell $1.4 / guard $2.9): %s = rcb_plaw4_s%d's flags + init rcb_plaw4_s%d + --distract_p 1, 16k, micro 2 pinned, --keep_final_ckpt; "
         "first the eval-only ws_deep + insert_ho on the init (rcb_plaw4_s%d_ds256); host blob / sha asserts (scratch.s256_guarded). Frozen args scratch/s257_args_frozen_%s.json.\n"
         "# HOSTPROBE\n# P=1\n"
         "boxes/runs_fsq/%s_mem boxes.inject_aw %s --mem_check\n"
         "boxes/runs_fsq/%s_gtime boxes.inject_aw %s --time_steps 60\n"
         "%s scratch.s256_guarded --mode evalonly --blob %s --model %s=%s --args boxes/runs_fsq/rcb_plaw4_s%d/args.json --parts ws_deep,insert_ho --gates --ship scratch/s256_ho_shas.json %s\n"
         "boxes/runs_fsq/%s scratch.s256_guarded --mode train --blob %s --init %s=%s --require %s/eval_ws_deep.json %s/eval_insert_ho.json %s/gates_pass.json -- %s %s --args_ref "
         "scratch/s257_args_frozen_%s.json\n") % (run, k, k, k, run, run, fs, run, fs, ev, blob, INIT[k][0], INIT[k][1], k, GSHIP, run, blob, INIT[k][0], INIT[k][1], ev, ev, ev, fs,
                                                  " ".join(tail), run)
    open("scratch/gen_q_b%d.txt" % n, "w", encoding="utf-8", newline="\n").write(q)
json.dump(out, open("scratch/gate_s257/predeploy_args.json", "w", newline="\n"), indent=1, default=str)
print({r: (v["keys"], v["micro"], v["auto_micro"], v["eval_bs"], v["seed"]) for r, v in out["args"].items()})
