"""§258 B BRIDGE (LB2_v1 §2): queue b396 bridge_s1 (one RTX 3090) = K7 mem + G-TIME probe lines, then the TRAIN line through scratch.s257_guarded --mode train (git blob
ids + the init's sha asserted on the host; host_shas.json after): rcb_plaw4_s1's flags (gate_s251.RCB: REG d ~ U{1..4}, 16k) with --init_model the unbuffered-read
aw_plaw2_s1 copy (RCB's own init, sha 484dab1eab9e2c12) + --distract_p 1 (rcb and §256 merged into one stage), micro 2 / eval_bs 250 pinned, --keep_final_ckpt,
eval_parts = dis's (acc incl. acc_kprime, swap_kd, lp5h, cue_only, ws_deep, insert_ho, test_perm, ...). Frozen args + diffs vs dis_s1 and rcb_plaw4_s1.
Pre-deploy: unbuffered blob ids == HEAD; the init copy's unbuffered sha256, majority of 3 reads (single local reads flip; the host assert binds).
    python -B scratch/s260_mkq.py"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s251 as G1  # noqa: E402
import s228_args_check as C  # noqa: E402
from nobuf_sha import sha_nobuf  # noqa: E402

EVAL = "acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws,lp5h,swap_kd,test_perm,insert_ho,ws_deep"
INIT = ("boxes/runs_fsq/aw_plaw2_s1_init/model.pt", "484dab1eab9e2c12")
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/insert_render.py", "boxes/perm_render.py", "scratch/s257_guarded.py", "scratch/s256_ho_shas.json",
        "scratch/s257_ho_shas.json"]


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def flags():
    fl = list(G1.RCB); i = fl.index("--init_model"); fl[i + 1] = INIT[0]; j = fl.index("--init_model_sha"); fl[j + 1] = INIT[1]
    return fl + ["--distract_p", "1", "--micro", "2", "--eval_bs", "250"]


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}
pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}; inits = [sha_nobuf(os.path.abspath(INIT[0])) for _ in range(3)]
ok = all(v["head"] == v["disk"] for v in pre.values()) and sum(s_.startswith(INIT[1]) for s_ in inits) >= 2   # 11-01 RAM: single local reads flip; majority of 3 (the host assert is binding)
out = {"blobs": pre, "init_unbuffered_sha256_x3": inits, "predeploy_ok": ok}
assert ok, ("pre-deploy", pre, inits)
run = "bridge_s1"; fl = flags(); tail = ["--keep_final_ckpt", "--ckpt_every", "1000", "--eval_parts", EVAL]
d = G8.fresh(ROOT + "/scratch/_tmp/s260/args_" + run)
rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + tail + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, rc
new = json.load(open(d + "/args.json")); json.dump(new, open("scratch/s260_args_frozen_%s.json" % run, "w", newline="\n"), indent=1)
for ref in ("dis_s1", "rcb_plaw4_s1"):
    dif = {k: v for k, v in C.diff(json.load(open("boxes/runs_fsq/%s/args.json" % ref)), new).items() if k != "pair_stats"}; out["diff_vs_" + ref] = {"keys": sorted(dif), "diff": dif}
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); fs = " ".join(fl)
q = ("# THEORY §258 B BRIDGE (LB2_v1 §2; ~$1.1): bridge_s1 = rcb_plaw4_s1's flags (REG d U{1..4}, 16k) from aw_plaw2_s1 (unbuffered copy, sha on host) + --distract_p 1, "
     "micro 2 pinned, --keep_final_ckpt, dis's eval parts; scratch.s257_guarded. Frozen args scratch/s260_args_frozen_bridge_s1.json.\n# HOSTPROBE\n# P=1\n"
     "boxes/runs_fsq/%s_mem boxes.inject_aw %s --mem_check\n"
     "boxes/runs_fsq/%s_gtime boxes.inject_aw %s --time_steps 60\n"
     "boxes/runs_fsq/%s scratch.s257_guarded --mode train --blob %s --init %s=%s --ship scratch/s256_ho_shas.json scratch/s257_ho_shas.json -- %s %s --args_ref "
     "scratch/s260_args_frozen_%s.json\n") % (run, fs, run, fs, run, blob, INIT[0], INIT[1], fs, " ".join(tail), run)
open("scratch/gen_q_b396.txt.hold", "w", encoding="utf-8", newline="\n").write(q)
os.makedirs("scratch/gate_s260", exist_ok=True); json.dump(out, open("scratch/gate_s260/predeploy_b396.json", "w", newline="\n"), indent=1, default=str)
print({k: v["keys"] for k, v in out.items() if k.startswith("diff")})
