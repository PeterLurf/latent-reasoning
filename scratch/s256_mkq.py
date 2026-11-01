"""Note 255-6 DISTRACT Vast cell kit: scratch/s256_args_frozen_pilot_dis_s1.json (--args_only, the cell's training flags), scratch/gen_q_b390.txt (K7 mem line + G-TIME line with
the pilot flags + --distract_p 1, micro 2; then the cell line scratch.s256_distract_cell with --init the unbuffered-read copy, --blob <path>=<HEAD git blob id> for every shipped code
file, --ship the helper scripts / rcb_plaw4_s1's args.json), and scratch/gate_s256/predeploy.json: every shipped file's git blob id recomputed from an UNBUFFERED disk read ==
HEAD's (Audit 11-01: the deploy tar reads through the page cache), the init copy's unbuffered sha256.
    python -B scratch/s256_mkq.py"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN, sha_nobuf  # noqa: E402
import ctypes  # noqa: E402
import ctypes.wintypes as W  # noqa: E402

RCB = ("--arm inject --think_k 8 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add --cache_lower --d_max 4 --steps 16000 --cue step --write_src answer "
       "--init_trunk boxes/runs_fsq/s233_askname_s1/model.pt --init_sha c00f7ec0bd7977fe --write_mix_start 0 --write_mix_end 0 --mul_s_p 0").split()
INIT = "boxes/runs_fsq/rcb_plaw4_s1_init/model.pt"; PF = RCB + ["--init_model", INIT, "--init_model_sha", "67775e2492044a8a", "--distract_p", "1"]
TRAIN_TAIL = ["--micro", "2", "--eval_bs", "250", "--stop_at", "2000", "--ckpt_every", "1000", "--log_every", "100"]
CODE = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/insert_render.py", "boxes/perm_render.py", "boxes/_inject_aw_s252.py", "scratch/s256_distract_cell.py",
        "scratch/gate_s255d.py", "scratch/gate_s228.py", "scratch/gate_s251.py", "scratch/s228_args_check.py", "scratch/s222_det_wrap.py"]
SHIP = ["scratch/gate_s255d.py", "scratch/gate_s228.py", "scratch/gate_s251.py", "scratch/s228_args_check.py", "scratch/s222_det_wrap.py", "boxes/_inject_aw_s252.py",
        "boxes/runs_fsq/rcb_plaw4_s1/args.json", "boxes/runs_fsq/aw_plaw2_s1/model.pt"]   # the host gates' RCB flags init from aw_plaw2_s1 (sha asserted by the trainer)


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


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in CODE}
pre = {}
for p in CODE:
    b = nobuf_bytes(p); pre[p] = {"head": head[p], "disk": hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()}
pre_ok = all(v["head"] == v["disk"] for v in pre.values()); init_sha = sha_nobuf(os.path.abspath(INIT))
aw2_sha = sha_nobuf(os.path.abspath("boxes/runs_fsq/aw_plaw2_s1/model.pt")); assert aw2_sha.startswith("484dab1eab9e2c12"), ("aw_plaw2_s1 disk sha", aw2_sha)
d = G8.fresh(ROOT + "/scratch/_tmp/s256/args"); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + PF + TRAIN_TAIL + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
assert rc == 0, rc; json.dump(json.load(open(d + "/args.json")), open("scratch/s256_args_frozen_pilot_dis_s1.json", "w", newline="\n"), indent=1)
os.makedirs("scratch/gate_s256", exist_ok=True)
json.dump({"blobs": pre, "all_equal": pre_ok, "init_unbuffered_sha256": init_sha, "init_ok": init_sha.startswith("67775e2492044a8a"), "aw_plaw2_s1_unbuffered_sha256": aw2_sha},
          open("scratch/gate_s256/predeploy.json", "w", newline="\n"), indent=1)
assert pre_ok and init_sha.startswith("67775e2492044a8a"), ("pre-deploy", pre_ok, init_sha)
blob = " ".join("%s=%s" % (p, head[p]) for p in CODE)
q = ("# THEORY Note 255-6 DISTRACT pilot (Vast, cap $0.5): one cell = host asserts (init sha, git blobs), host build gates (gate_s255d offbit / tiny / acts / smoke / gradflow), host "
     "step-0 INSERT on rcb_plaw4_s1, micro 2 2-insert mem check, train pilot_dis_s1 (rcb_plaw4_s1 flags + init rcb_plaw4_s1 + --distract_p 1, 16k schedule, stop 2000), "
     "INSERT at 1k / 2k. Frozen args scratch/s256_args_frozen_pilot_dis_s1.json.\n# HOSTPROBE\n# P=1\n"
     "boxes/runs_fsq/pilot_dis_s1_mem boxes.inject_aw %s --micro 2 --mem_check\n"
     "boxes/runs_fsq/pilot_dis_s1_gtime boxes.inject_aw %s --micro 2 --time_steps 60\n"
     "boxes/runs_fsq/pilot_dis_s1 scratch.s256_distract_cell --init %s --blob %s --ship %s\n") % (" ".join(PF), " ".join(PF), INIT, blob, " ".join(SHIP))
open("scratch/gen_q_b390.txt", "w", encoding="utf-8", newline="\n").write(q); print("predeploy ok; written b390")
