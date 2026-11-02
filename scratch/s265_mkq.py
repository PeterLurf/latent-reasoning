"""§260 GENERIC-BUN pilot (score_gb): queue b409 on one RTX 3090 =
  pilot_gb_s1_mem / _gtime  K7 mem_check + G-TIME (the pilot flags; a crash refuses the host);
  pilot_gb_s1_step0         scratch.s259_guarded --mode evalonly: the bun_s1 init (streaming copy, sha on host) under the GENERIC cue (--args = the pilot's frozen args,
                            cue generic) -> eval_acc + eval_ws (no training);
  pilot_gb_s1               scratch.s257_guarded --mode train: bun_s1's flags (dch_flags + --hold_inc 1.6 --hold_log; the §257 insert mix) with --init_model the bun_s1 copy,
                            --cue generic (the §252 build), --steps 8000 --stop_at 2000 (ckpt at 2000), micro 2 pinned;
  pilot_gb_s1_2k            scratch.s258s2_ws6k --run pilot_gb_s1 --step 2000: eval acc + ws on the 2k ckpt under the generic cue (its own dir: the eval stub log must not
                            overwrite the training log.jsonl); the landing copies eval_acc / eval_ws into pilot_gb_s1/.
Frozen args + diff vs bun_s1/args.json. Pre-deploy: unbuffered blob ids == HEAD; the init's streaming sha (majority of 3); ships checked.
    python -B scratch/s265_mkq.py"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s257 as G7  # noqa: E402
import s228_args_check as C  # noqa: E402
from nobuf_sha import sha_nobuf  # noqa: E402

INIT = ("boxes/runs_fsq/bun_s1_init/model.pt", "1ad3517b7635fddf")
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/insert_render.py", "boxes/perm_render.py", "boxes/s5.py", "boxes/nested170.py", "boxes/inject_ow.py",
        "boxes/deltamem.py", "boxes/core.py", "scratch/s257_guarded.py", "scratch/s259_guarded.py", "scratch/s258s2_ws6k.py", "scratch/s256_ho_shas.json",
        "scratch/s257_ho_shas.json"]


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def flags():
    fl = G7.dch_flags(1, *INIT) + ["--hold_inc", "1.6", "--hold_log"]; i = fl.index("--cue"); fl[i + 1] = "generic"; return fl


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}
pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}; inits = [sha_nobuf(os.path.abspath(INIT[0])) for _ in range(3)]
ok = all(v["head"] == v["disk"] for v in pre.values()) and sum(s_.startswith(INIT[1]) for s_ in inits) >= 2; out = {"blobs": pre, "init_unbuffered_sha256_x3": inits, "predeploy_ok": ok}
assert ok, ("pre-deploy", pre, inits)
run = "pilot_gb_s1"; fl = flags(); tail = ["--ckpt_every", "1000", "--stop_at", "2000", "--eval_parts", "acc,ws"]
d = G8.fresh(ROOT + "/scratch/_tmp/s265/args_" + run)
rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + tail + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, rc
shutil.copy(d + "/args.json", "scratch/s265_args_frozen_%s.json" % run)
shutil.copy(d + "/args.json", "scratch/s265_args_frozen_%s_step0.json" % run)                # the step-0 eval's args (cue generic; the eval rebuilds from them)
dif = {k: v for k, v in C.diff(json.load(open("boxes/runs_fsq/bun_s1/args.json")), json.load(open(d + "/args.json"))).items() if k != "pair_stats"}; out["diff_vs_bun_s1"] = dif
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); fs = " ".join(fl); W = "boxes/runs_fsq/" + run
q = ("# THEORY §260 GENERIC-BUN pilot (score_gb): pilot_gb_s1 = bun_s1 (streaming copy, sha on host) + bun_s1's args + --cue generic, 8k schedule stopped at 2000; "
     "step-0 and 2k evals (acc + ws) under the generic cue. Frozen args scratch/s265_args_frozen_pilot_gb_s1.json.\n# HOSTPROBE\n# P=1\n"
     "%s_mem boxes.inject_aw %s --mem_check\n"
     "%s_gtime boxes.inject_aw %s --time_steps 60\n"
     "%s_step0 scratch.s259_guarded --mode evalonly --blob %s --model %s=%s --args scratch/s265_args_frozen_%s_step0.json --parts acc,ws --ship %s scratch/s265_args_frozen_%s_step0.json scratch/s256_ho_shas.json scratch/s257_ho_shas.json\n"
     "%s scratch.s257_guarded --mode train --blob %s --init %s=%s --ship %s scratch/s256_ho_shas.json scratch/s257_ho_shas.json -- %s %s --args_ref scratch/s265_args_frozen_%s.json\n"
     "%s_2k scratch.s258s2_ws6k --run %s --step 2000\n") % (W, fs, W, fs, W, blob, INIT[0], INIT[1], run, INIT[0], run, W, blob, INIT[0], INIT[1], INIT[0], fs, " ".join(tail), run, W, W)
open("scratch/gen_q_b409.txt.hold", "w", encoding="utf-8", newline="\n").write(q)
assert subprocess.call([sys.executable, "scratch/check_queue_ships.py", "scratch/gen_q_b409.txt.hold"]) == 0
os.makedirs("scratch/gate_s265", exist_ok=True); json.dump(out, open("scratch/gate_s265/predeploy.json", "w", newline="\n"), indent=1, default=str); print(sorted(dif))
