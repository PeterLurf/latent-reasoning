"""§261 TARGET-CUE pilot (score_tc): queue b412 on one RTX 3090 (filler family s267, drain gen10) =
  pilot_tc_s1_mem / _gtime  K7 mem_check + G-TIME (the pilot flags; a crash refuses the host);
  pilot_tc_s1_gates         scratch.s268_gates (Audit G-D b412 (1)): gate_s268 offbit_aw / offbit_gen (flags-off bitwise vs the pre-§261 tree scratch/s268_ref.tar), smoke_plain,
                            smoke_tgt, gradflow on this host -> gates_pass.json (the TRAIN line --require's it);
  pilot_tc_s1_step0         scratch.s259_guarded --mode evalonly: the bun_s1 init (streaming copy, sha on host) under the TARGET cue (--args = the pilot's frozen args)
                            -> eval_acc + eval_ws + eval_insert_ho (no training); its model.pt copy is NOFETCH (gen10);
  pilot_tc_s1               scratch.s257_guarded --mode train: bun_s1's flags (dch_flags + --hold_inc 1.6 --hold_log; the §257 insert mix) with --init_model the bun_s1 copy,
                            --cue target (§261), --steps 8000 --stop_at 2000 (ckpt at 2000, exit 9), micro 2 pinned; requires pilot_tc_s1_gates/gates_pass.json;
  pilot_tc_s1 (2k evals)    scratch.s268_stopland --step 2000 --parts acc,ws,insert_ho: model.pt from the 2k ckpt + the evals IN the run dir (a gen10 RUN dir; the training
                            log.jsonl kept);
  pilot_tc_s1 (gate)        scratch.s268_target: cue target + identical pass inputs (passes()' assert armed) + the tail names S_d and no other entity -> gates_pass.json.
Frozen args + diff vs bun_s1/args.json. Pre-deploy: unbuffered blob ids == HEAD; the init's streaming sha (majority of 3); ships checked.
    python -B scratch/s268_mkq.py"""
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
        "boxes/deltamem.py", "boxes/core.py", "boxes/ow_words.py", "boxes/nested_words.py", "scratch/s257_guarded.py", "scratch/s259_guarded.py", "scratch/s268_stopland.py",
        "scratch/s268_target.py", "scratch/s268_gates.py", "scratch/gate_s268.py", "scratch/gate_s257.py", "scratch/gate_s251.py", "scratch/gate_s228.py",
        "scratch/s222_det_wrap.py", "scratch/s268_ref.tar", "scratch/s256_ho_shas.json", "scratch/s257_ho_shas.json"]
GSHIP = ("scratch/s268_ref.tar scratch/gate_s268.py scratch/gate_s257.py scratch/gate_s251.py scratch/gate_s228.py scratch/s228_args_check.py scratch/s222_det_wrap.py "
         "scratch/s257_guarded.py boxes/runs_fsq/bun_s1_init/model.pt scratch/s256_ho_shas.json scratch/s257_ho_shas.json")
PARTS = "acc,ws,insert_ho"


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def flags():
    fl = G7.dch_flags(1, *INIT) + ["--hold_inc", "1.6", "--hold_log"]; i = fl.index("--cue"); fl[i + 1] = "target"; return fl


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}
pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}; inits = [sha_nobuf(os.path.abspath(INIT[0])) for _ in range(3)]
ok = all(v["head"] == v["disk"] for v in pre.values()) and sum(s_.startswith(INIT[1]) for s_ in inits) >= 2; out = {"blobs": pre, "init_unbuffered_sha256_x3": inits, "predeploy_ok": ok}
assert ok, ("pre-deploy", pre, inits)
run = "pilot_tc_s1"; fl = flags(); tail = ["--ckpt_every", "1000", "--stop_at", "2000", "--eval_parts", PARTS]
d = G8.fresh(ROOT + "/scratch/_tmp/s268/args_" + run)
rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + tail + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, rc
shutil.copy(d + "/args.json", "scratch/s267_args_frozen_%s.json" % run)
shutil.copy(d + "/args.json", "scratch/s267_args_frozen_%s_step0.json" % run)                # the step-0 eval's args (cue target; the eval rebuilds from them)
dif = {k: v for k, v in C.diff(json.load(open("boxes/runs_fsq/bun_s1/args.json")), json.load(open(d + "/args.json"))).items() if k != "pair_stats"}; out["diff_vs_bun_s1"] = dif
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); fs = " ".join(fl); W = "boxes/runs_fsq/" + run; SH = "scratch/s256_ho_shas.json scratch/s257_ho_shas.json"
q = ("# THEORY §261 TARGET-CUE pilot (score_tc): pilot_tc_s1 = bun_s1 (streaming copy, sha on host) + bun_s1's args + --cue target, 8k schedule stopped at 2000; "
     "host gates first (gates_pass.json required); step-0 and 2k evals (acc + ws + insert_ho) under the target cue + the target-only gate. Frozen args scratch/s267_args_frozen_pilot_tc_s1.json.\n# HOSTPROBE\n# P=1\n"
     "# NOFETCH %s_step0/model.pt\n"
     "%s_mem boxes.inject_aw %s --mem_check\n"
     "%s_gtime boxes.inject_aw %s --time_steps 60\n"
     "%s_gates scratch.s268_gates --blob %s --gates offbit_aw,offbit_gen,smoke_plain,smoke_tgt,gradflow --ship %s\n"
     "%s_step0 scratch.s259_guarded --mode evalonly --blob %s --model %s=%s --args scratch/s267_args_frozen_%s_step0.json --parts %s --ship %s scratch/s267_args_frozen_%s_step0.json %s\n"
     "%s scratch.s257_guarded --mode train --blob %s --init %s=%s --require %s_gates/gates_pass.json --ship %s %s -- %s %s --args_ref scratch/s267_args_frozen_%s.json\n"
     "%s scratch.s268_stopland --step 2000 --parts %s\n"
     "%s scratch.s268_target\n") % (W, W, fs, W, fs, W, blob, GSHIP, W, blob, INIT[0], INIT[1], run, PARTS, INIT[0], run, SH, W, blob, INIT[0], INIT[1], W, INIT[0], SH, fs, " ".join(tail), run,
                                    W, PARTS, W)
if os.environ.get("S268_NO_STEP0") == "1":                                   # attempt 4 (11-03): the step-0 evals landed in attempt 3 (host shas OK) - not re-run
    ls_ = [l_ for l_ in q.split("\n") if "_step0" not in l_]; assert len(q.split("\n")) - len(ls_) == 2, "NO_STEP0: the NOFETCH + step0 lines"; q = "\n".join(ls_)
open("scratch/gen_q_b412.txt.hold", "w", encoding="utf-8", newline="\n").write(q)
assert subprocess.call([sys.executable, "scratch/check_queue_ships.py", "scratch/gen_q_b412.txt.hold"]) == 0
os.makedirs("scratch/gate_s268", exist_ok=True); json.dump(out, open("scratch/gate_s268/predeploy.json", "w", newline="\n"), indent=1, default=str); print(sorted(dif))
