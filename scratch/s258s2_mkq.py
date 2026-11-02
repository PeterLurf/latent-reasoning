"""§258 S2 P2-M (LB2_v1 §2; Note 258-3: S1 winner askm_lr3e5): queue b401 p2m_s1 on one RTX A6000 (the pinned class; HOSTPROBE 1.15) =
  p2m_s1_mem / _gtime   K7 mem_check + G-TIME (the filler's admit measure);
  s2g                   the S0 host gates on this host + the tested-resume gate (gate_s258m resume: GPT-2 small 60 straight == 30 + resume 30, ckpt equal);
  p2m_s1 (segment 1)    scratch.s257_guarded --mode train --init askm_lr3e5_init=<sha> --require s2g/gates_pass.json -- aw_plaw2_s1's flags with --base gpt2-medium,
                        --init_trunk the unbuffered askm_lr3e5 copy (+ --init_sha), --lr 3e-5, micro 2 pinned, --keep_final_ckpt, --stop_at 6000 (ckpt at 6000);
  p2m_s1_ws6k           scratch.s258s2_ws6k: eval acc + ws on the step-6000 ckpt (the S2 formation rule's 6k read);
  p2m_s1 (segment 2)    the same command + --resume boxes/runs_fsq/p2m_s1/ckpt_last.pt (no --stop_at) -> 12k + the full aw_plaw2 eval set (eval_acc, eval_ws, ...).
Frozen args (--args_only) + diff vs aw_plaw2_s1/args.json. Pre-deploy: unbuffered blob ids == HEAD; the init copy's unbuffered sha (majority of 3); every path=sha model shipped.
    python -B scratch/s258s2_mkq.py"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import s228_args_check as C  # noqa: E402
from nobuf_sha import sha_nobuf  # noqa: E402

INIT = ("boxes/runs_fsq/askm_lr3e5_init/model.pt", "b5c68ace4e2e60c8")
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/insert_render.py", "boxes/perm_render.py", "boxes/s5.py", "boxes/nested170.py", "boxes/inject_ow.py",
        "boxes/deltamem.py", "boxes/core.py", "scratch/s257_guarded.py", "scratch/s258m_host.py", "scratch/gate_s258m.py", "scratch/s258_askname.py", "scratch/s258s2_ws6k.py",
        "scratch/s256_ho_shas.json", "scratch/s257_ho_shas.json", "scratch/s258m_ref.tar"]
GSHIP = ("scratch/s258m_ref.tar scratch/gate_s258m.py scratch/gate_s228.py scratch/gate_s251.py scratch/s222_det_wrap.py scratch/s258_askname.py scratch/s257_guarded.py "
         "boxes/runs_fsq/aw_plaw2_s1/model.pt boxes/runs_fsq/s233_askname_s1/model.pt scratch/s256_ho_shas.json scratch/s257_ho_shas.json")
FL = ["--arm", "inject", "--think_k", "8", "--d_max", "2", "--steps", "12000", "--bs", "64", "--lr", "3e-5", "--seed", "1", "--mem_write", "top", "--write_mix_start", "0",
      "--write_mix_end", "0", "--cache_lower", "--aux_step", "1", "--aux_fade", "48000,48000", "--aux_reweight", "--fast_add", "--ckpt_every", "1000", "--log_every", "100",
      "--eval_bs", "250", "--eval_parts", "acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws", "--cue", "step", "--write_src", "answer",
      "--init_trunk", INIT[0], "--init_sha", INIT[1], "--base", "gpt2-medium", "--micro", "2", "--keep_final_ckpt"]


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}
pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}; inits = [sha_nobuf(os.path.abspath(INIT[0])) for _ in range(3)]
ok = all(v["head"] == v["disk"] for v in pre.values()) and sum(s_.startswith(INIT[1]) for s_ in inits) >= 2; out = {"blobs": pre, "init_unbuffered_sha256_x3": inits, "predeploy_ok": ok}
assert ok, ("pre-deploy", pre, inits)
run = "p2m_s1"; d = G8.fresh(ROOT + "/scratch/_tmp/s258s2/args_" + run)
rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + FL + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, rc
shutil.copy(d + "/args.json", "scratch/s260_args_frozen_%s.json" % run)
dif = {k: v for k, v in C.diff(json.load(open("boxes/runs_fsq/aw_plaw2_s1/args.json")), json.load(open(d + "/args.json"))).items() if k != "pair_stats"}; out["diff_vs_aw_plaw2_s1"] = dif
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); fs = " ".join(FL); W = "boxes/runs_fsq/" + run
q = ("# THEORY §258 S2 P2-M (LB2_v1 §2; Note 258-3): p2m_s1 = aw_plaw2_s1's args on gpt2-medium from askm_lr3e5 (lr 3e-5, micro 2 pinned), 12k in two segments "
     "(stop 6000 -> WS-6k eval -> resume = the tested resume); this host's gates first (+ the resume gate). Frozen args scratch/s260_args_frozen_p2m_s1.json.\n# HOSTPROBE 1.15\n# P=1\n"
     "%s_mem boxes.inject_aw %s --mem_check\n"
     "%s_gtime boxes.inject_aw %s --time_steps 60\n"
     "boxes/runs_fsq/s2g scratch.s258m_host --blob %s --gates offbit_aw,offbit_ow,step0,smoke_aw,smoke_ow,resume --ship %s\n"
     "%s scratch.s257_guarded --mode train --blob %s --init %s=%s --require boxes/runs_fsq/s2g/gates_pass.json --ship %s -- %s --stop_at 6000 --args_ref scratch/s260_args_frozen_%s.json\n"
     "%s_ws6k scratch.s258s2_ws6k --run %s --step 6000\n"
     "%s scratch.s257_guarded --mode train --blob %s --init %s=%s --require boxes/runs_fsq/s2g/gates_pass.json --ship %s -- %s --resume %s/ckpt_last.pt\n") % (
    W, fs, W, fs, blob, GSHIP, W, blob, INIT[0], INIT[1], INIT[0], fs, run, W, W, W, blob, INIT[0], INIT[1], INIT[0], fs, W)
open("scratch/gen_q_b401.txt.hold", "w", encoding="utf-8", newline="\n").write(q)
json.dump(out, open("scratch/gate_s258m/predeploy_s2.json", "w", newline="\n"), indent=1, default=str); print(sorted(dif))
