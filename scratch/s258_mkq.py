"""§257 DISTRACT-CHAIN runs (DC2_v1 §2; Theory: cell $1.2, guard $2.6): queues b393 dch_s1 / b394 dch_s2 + frozen args + pre-deploy check (= s257_mkq.py's §256 recipe):
Each queue (one RTX 3090): K7 mem line + G-TIME line (the dch flags, micro 2), then the EVAL-ONLY line dis_s<k>_dc257 (insert_ho = §256 families + the §257 held-out
families on the dis_s<k> init, through scratch.s257_guarded --mode evalonly --gates: blob + model sha asserted, the §257 host gates offbit / offbit_d / acts / smoke / gradflow),
then the TRAIN line dch_s<k> (s257_guarded --mode train: blob + init sha asserted, eval_insert_ho.json + gates_pass.json required) = gate_s257.dch_flags(k): dis_s<k>'s flags
(rcb_plaw4_s<k>'s + --distract_p 1, micro 2 / eval_bs 250, seed k) with --init_model the unbuffered-read dis_s<k> copy (scratch/s257d_build/mk_init_copies.py: read == majority
sha and == ckpt_final's model), --distract_mix s256,chain,givop, --steps 8000 (fresh AdamW, 100-step warm-up, cosine over 8k from lr 1e-4), --keep_final_ckpt, eval_parts = dis's.
Pre-deploy: every asserted file's git blob id from an UNBUFFERED read == HEAD; init copies' unbuffered sha256 (3 reads, all equal).
    python -B scratch/s258_mkq.py"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s257 as G7  # noqa: E402
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
INIT = {1: ("boxes/runs_fsq/dis_s1_init/model.pt", "35813f1a41ea8085"), 2: ("boxes/runs_fsq/dis_s2_init/model.pt", "a68cb45e4b4d153d")}
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/insert_render.py", "boxes/perm_render.py", "scratch/s257_guarded.py", "scratch/s256_ho_shas.json",
        "scratch/s257_ho_shas.json", "scratch/gate_s257.py", "scratch/gate_s228.py", "scratch/gate_s251.py", "scratch/s228_args_check.py", "scratch/s222_det_wrap.py",
        "boxes/_inject_aw_s256.py"]
GSHIP = ("scratch/gate_s257.py scratch/gate_s228.py scratch/gate_s251.py scratch/s228_args_check.py scratch/s222_det_wrap.py boxes/_inject_aw_s256.py "
         "boxes/runs_fsq/aw_plaw2_s1/model.pt scratch/s256_ho_shas.json scratch/s257_ho_shas.json")

head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}; pre = {}
for p in BLOB:
    b = nobuf_bytes(p); pre[p] = {"head": head[p], "disk": hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()}
inits = {k: [sha_nobuf(os.path.abspath(v[0])) for _ in range(3)] for k, v in INIT.items()}
ok = all(v["head"] == v["disk"] for v in pre.values()) and all(all(s_.startswith(INIT[k][1]) for s_ in inits[k]) for k in INIT)
os.makedirs("scratch/gate_s257d", exist_ok=True); out = {"blobs": pre, "init_unbuffered_sha256_x3": inits, "predeploy_ok": ok, "args": {}}
assert ok, ("pre-deploy", pre, inits)
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB)
for k, n in ((1, 393), (2, 394)):
    run = "dch_s%d" % k; fl = G7.dch_flags(k, *INIT[k]); tail = ["--keep_final_ckpt", "--ckpt_every", "1000", "--eval_parts", EVAL]
    d = G8.fresh(ROOT + "/scratch/_tmp/s258/args_" + run)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + tail + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, rc
    new = json.load(open(d + "/args.json")); json.dump(new, open("scratch/s258_args_frozen_%s.json" % run, "w", newline="\n"), indent=1)
    ref = json.load(open("boxes/runs_fsq/dis_s%d/args.json" % k)); dif = {kk: v for kk, v in C.diff(ref, new).items() if kk != "pair_stats"}
    out["args"][run] = {"keys": sorted(dif), "diff": dif, "micro": new["micro"], "auto_micro": new["auto_micro"], "eval_bs": new["eval_bs"], "seed": new["seed"], "steps": new["steps"]}
    ev = "boxes/runs_fsq/dis_s%d_dc257" % k; fs = " ".join(fl)
    q = ("# THEORY §257 DISTRACT-CHAIN (DC2_v1; cell $1.2 / guard $2.6): %s = 8k FT from dis_s%d (unbuffered copy, sha on host) with dis_s%d's flags + --distract_mix s256,chain,givop, "
         "--steps 8000, micro 2 pinned, --keep_final_ckpt; first the eval-only insert_ho (+ §257 families) on the init + the §257 host gates (dis_s%d_dc257); scratch.s257_guarded. "
         "Frozen args scratch/s258_args_frozen_%s.json.\n"
         "# HOSTPROBE\n# P=1\n"
         "boxes/runs_fsq/%s_mem boxes.inject_aw %s --mem_check\n"
         "boxes/runs_fsq/%s_gtime boxes.inject_aw %s --time_steps 60\n"
         "%s scratch.s257_guarded --mode evalonly --blob %s --model %s=%s --args boxes/runs_fsq/dis_s%d/args.json --parts insert_ho --gates --pilot_k %d --ship %s\n"
         "boxes/runs_fsq/%s scratch.s257_guarded --mode train --blob %s --init %s=%s --require %s/eval_insert_ho.json %s/gates_pass.json -- %s %s --args_ref "
         "scratch/s258_args_frozen_%s.json\n") % (run, k, k, k, run, run, fs, run, fs, ev, blob, INIT[k][0], INIT[k][1], k, k, GSHIP, run, blob, INIT[k][0], INIT[k][1],
                                                  ev, ev, fs, " ".join(tail), run)
    open("scratch/gen_q_b%d.txt.hold" % n, "w", encoding="utf-8", newline="\n").write(q)
json.dump(out, open("scratch/gate_s257d/predeploy_args.json", "w", newline="\n"), indent=1, default=str)
print({r: (v["keys"], v["micro"], v["auto_micro"], v["eval_bs"], v["seed"], v["steps"]) for r, v in out["args"].items()})
