"""§260 GENERIC-BUN full stage (Note 260-2; score_gbf): queues b410 gb_s1 / b411 gb_s2, one RTX 3090 each (micro 2 pinned) =
  <run>_mem / _gtime  K7 mem_check + G-TIME (the gb flags);
  <run>               scratch.s257_guarded --mode train (git blobs + the bun_s<k> streaming copy's sha on the host): bun_s<k>'s flags (dch_flags + --hold_inc 1.6 --hold_log,
                      the §257 insert mix) with --init_model the bun_s<k> copy, --cue generic (§252), --steps 8000; eval_parts acc,swap_kd,lp5h,insert_ho,ws,ws_deep (under the
                      generic cue: the run's args); no --keep_final_ckpt (no later stage registered);
  <run> (long)        scratch.s262_long: run_long K' 8 / 64 / 128 on d 1-6 -> eval_long_k128.json;
  <run> (plan-free)   scratch.s266_planfree: cue generic + identical pass inputs + passes()' PLAN-FREE assert -> gates_pass.json {"plan_free": true}.
Frozen args + diff vs bun_s<k>/args.json. Pre-deploy: unbuffered blob ids == HEAD; the inits' streaming shas (majority of 3); ships checked.
    python -B scratch/s266_mkq.py"""
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

INIT = {1: ("boxes/runs_fsq/bun_s1_init/model.pt", "1ad3517b7635fddf"), 2: ("boxes/runs_fsq/bun_s2_init/model.pt", "09337f63aec57959")}
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/insert_render.py", "boxes/perm_render.py", "boxes/s5.py", "boxes/nested170.py", "boxes/inject_ow.py",
        "boxes/deltamem.py", "boxes/core.py", "scratch/s257_guarded.py", "scratch/s262_long.py", "scratch/s266_planfree.py", "scratch/s256_ho_shas.json",
        "scratch/s257_ho_shas.json"]
EVAL = "acc,swap_kd,lp5h,insert_ho,ws,ws_deep,stab"   # Note 260-3: + stab (TEST2 d 1..7, K' 0..14)


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def flags(k):
    fl = G7.dch_flags(k, *INIT[k]) + ["--hold_inc", "1.6", "--hold_log"]; i = fl.index("--cue"); fl[i + 1] = "generic"; return fl


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}; pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}
inits = {k: [sha_nobuf(os.path.abspath(v[0])) for _ in range(3)] for k, v in INIT.items()}
ok = all(v["head"] == v["disk"] for v in pre.values()) and all(sum(s_.startswith(INIT[k][1]) for s_ in inits[k]) >= 2 for k in INIT)
out = {"blobs": pre, "init_unbuffered_sha256_x3": inits, "predeploy_ok": ok, "args": {}}; assert ok, ("pre-deploy", pre, inits)
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); SH = "scratch/s256_ho_shas.json scratch/s257_ho_shas.json"
for n, k in ((410, 1), (411, 2)):
    run = "gb_s%d" % k; fl = flags(k); tail = ["--ckpt_every", "1000", "--eval_parts", EVAL]; d = G8.fresh(ROOT + "/scratch/_tmp/s266/args_" + run)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + tail + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, (run, rc)
    shutil.copy(d + "/args.json", "scratch/s265_args_frozen_%s.json" % run)
    dif = {kk: v for kk, v in C.diff(json.load(open("boxes/runs_fsq/bun_s%d/args.json" % k)), json.load(open(d + "/args.json"))).items() if kk != "pair_stats"}
    out["args"][run] = {"keys": sorted(dif), "diff": dif}; fs = " ".join(fl); W = "boxes/runs_fsq/" + run
    q = ("# THEORY §260 GENERIC-BUN (Note 260-2; score_gbf): %s = bun_s%d (streaming copy, sha on host) + bun's args + --cue generic, 8k, micro 2 pinned; evals under the "
         "generic cue + run_long K' 8/64/128 d 1-6 + the plan-free gate. Frozen args scratch/s265_args_frozen_%s.json.\n# HOSTPROBE\n# P=1\n"
         "%s_mem boxes.inject_aw %s --mem_check\n"
         "%s_gtime boxes.inject_aw %s --time_steps 60\n"
         "%s scratch.s257_guarded --mode train --blob %s --init %s=%s --ship %s %s -- %s %s --args_ref scratch/s265_args_frozen_%s.json\n"
         "%s scratch.s262_long --kprime_list 8,64,128 --long_ds 1,2,3,4,5,6 --long_name eval_long_k128.json\n"
         "%s scratch.s266_planfree\n") % (run, k, run, W, fs, W, fs, W, blob, INIT[k][0], INIT[k][1], INIT[k][0], SH, fs, " ".join(tail), run, W, W)
    open("scratch/gen_q_b%d.txt.hold" % n, "w", encoding="utf-8", newline="\n").write(q)
    assert subprocess.call([sys.executable, "scratch/check_queue_ships.py", "scratch/gen_q_b%d.txt.hold" % n]) == 0
os.makedirs("scratch/gate_s266", exist_ok=True); json.dump(out, open("scratch/gate_s266/predeploy.json", "w", newline="\n"), indent=1, default=str)
print({r: v["keys"] for r, v in out["args"].items()})
