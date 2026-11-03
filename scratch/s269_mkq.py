"""§262 NAME-KEYED pilot (NK_v1 §4 + Notes 262-1 / 262-2; score_nk_pilot): queues b413 pilot_nk_s1 / b414 pilot_nkc_s1, one RTX 3090 each (filler family s269, drain gen10) =
  <run>_mem / _gtime   K7 mem_check + G-TIME (the cell's flags; a crash refuses the host);
  <run>_gates          scratch.s269_gates: gate_s269 offbit_aw / offbit_perm (flags-off bitwise vs the pre-§262 tree scratch/s269_ref.tar), step0 (CTRL == NK bitwise with
                       the heads present, incl. the hold stats on S[:, :NB]), ids, smoke_nk, smoke_ctrl, gradflow on this host -> gates_pass.json (TRAIN --require's it);
  <run>                scratch.s257_guarded --mode train: bun_s1's flags (dch_flags + --hold_inc 1.6 --hold_log; the §257 insert mix) + --perm_p 0.5 (perm-then-insert)
                       + --ask_p 0.25 (one-hop ask items) [+ --nk 1], --init_model the bun_s1 copy, --steps 8000 --stop_at 2000 (ckpt at 2000), micro 2 pinned;
  <run> (2k evals)     scratch.s268_stopland --step 2000 --parts acc,nk_pilot: model.pt from the 2k ckpt + eval_acc (TEST2, Note 262-1 (a)) + eval_nk_pilot in the run dir.
Frozen args (scratch/s269_args_frozen_<run>.json) + diff vs bun_s1/args.json. Pre-deploy: unbuffered blob ids == HEAD; the init's streaming sha (majority of 3); ships checked.
    python -B scratch/s269_mkq.py"""
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
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/nk.py", "boxes/ask_render.py", "boxes/insert_render.py", "boxes/perm_render.py", "boxes/s5.py",
        "boxes/nested170.py", "boxes/inject_ow.py", "boxes/inject_st.py", "boxes/deltamem.py", "boxes/deltas5.py", "boxes/core.py", "boxes/ow_words.py", "boxes/nested_words.py",
        "scratch/s257_guarded.py", "scratch/s268_stopland.py", "scratch/s269_gates.py", "scratch/gate_s269.py", "scratch/gate_s257.py", "scratch/gate_s251.py",
        "scratch/gate_s228.py", "scratch/s222_det_wrap.py", "scratch/s269_ref.tar", "scratch/s256_ho_shas.json", "scratch/s257_ho_shas.json"]
GSHIP = ("scratch/s269_ref.tar scratch/gate_s269.py scratch/gate_s257.py scratch/gate_s251.py scratch/gate_s228.py scratch/s228_args_check.py scratch/s222_det_wrap.py "
         "scratch/s257_guarded.py boxes/runs_fsq/bun_s1_init/model.pt boxes/runs_fsq/s233_askname_s1/model.pt scratch/s256_ho_shas.json scratch/s257_ho_shas.json")
PARTS = "acc,nk_pilot"
CELLS = ((413, "pilot_nk_s1", 1), (414, "pilot_nkc_s1", 0))


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def flags(nk):
    return G7.dch_flags(1, *INIT) + ["--hold_inc", "1.6", "--hold_log", "--perm_p", "0.5", "--ask_p", "0.25"] + (["--nk", "1"] if nk else [])


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}
pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}; inits = [sha_nobuf(os.path.abspath(INIT[0])) for _ in range(3)]
ok = all(v["head"] == v["disk"] for v in pre.values()) and sum(s_.startswith(INIT[1]) for s_ in inits) >= 2; out = {"blobs": pre, "init_unbuffered_sha256_x3": inits, "predeploy_ok": ok, "args": {}}
assert ok, ("pre-deploy", pre, inits)
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); SH = "scratch/s256_ho_shas.json scratch/s257_ho_shas.json"
for n, run, nk in CELLS:
    fl = flags(nk); tail = ["--ckpt_every", "1000", "--stop_at", "2000", "--eval_parts", PARTS]; d = G8.fresh(ROOT + "/scratch/_tmp/s269/args_" + run)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + tail + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, (run, rc)
    shutil.copy(d + "/args.json", "scratch/s269_args_frozen_%s.json" % run)
    dif = {k: v for k, v in C.diff(json.load(open("boxes/runs_fsq/bun_s1/args.json")), json.load(open(d + "/args.json"))).items() if k != "pair_stats"}
    out["args"][run] = {"keys": sorted(dif), "diff": dif}; fs = " ".join(fl); W = "boxes/runs_fsq/" + run
    q = ("# THEORY §262 NAME-KEYED pilot (NK_v1 §4; score_nk_pilot): %s = bun_s1 (streaming copy, sha on host) + bun_s1's args + perm-then-insert (--perm_p 0.5) + one-hop "
         "ask items (--ask_p 0.25)%s, 8k schedule stopped at 2000; host gates first (gates_pass.json required); 2k evals acc + nk_pilot. Frozen args "
         "scratch/s269_args_frozen_%s.json.\n# HOSTPROBE\n# P=1\n"
         "%s_mem boxes.inject_aw %s --mem_check\n"
         "%s_gtime boxes.inject_aw %s --time_steps 60\n"
         "%s_gates scratch.s269_gates --blob %s --gates offbit_aw,offbit_perm,smoke_nk,smoke_ctrl,gradflow,step0,ids --ship %s\n"
         "%s scratch.s257_guarded --mode train --blob %s --init %s=%s --require %s_gates/gates_pass.json --ship %s %s -- %s %s --args_ref scratch/s269_args_frozen_%s.json\n"
         "%s scratch.s268_stopland --step 2000 --parts %s\n") % (run, " + the NK head (--nk 1)" if nk else " (CTRL: no NK head)", run, W, fs, W, fs, W, blob, GSHIP,
                                                                  W, blob, INIT[0], INIT[1], W, INIT[0], SH, fs, " ".join(tail), run, W, PARTS)
    if os.environ.get("S269_NO_GATES") == "1":                               # Audit 11-03 (b414 CTRL rerun): no gates line; TRAIN --require's b413's landed gates_pass.json (same blobs)
        GP = "boxes/runs_fsq/pilot_nk_s1_gates/gates_pass.json"; assert os.path.exists(GP)
        ls_ = [l_ for l_ in q.split("\n") if " scratch.s269_gates " not in l_]; assert len(q.split("\n")) - len(ls_) == 1
        q = "\n".join(ls_).replace("--require %s_gates/gates_pass.json --ship " % W, "--require %s --ship %s " % (GP, GP)); assert q.count("--require %s --ship %s " % (GP, GP)) == 1
        q = q.replace("host gates first (gates_pass.json required)", "no gates line: TRAIN requires b413's gates_pass.json (same blobs; Audit 11-03 CTRL rerun)")
    if os.environ.get("S269_ONLY") and str(n) != os.environ["S269_ONLY"]:
        continue
    open("scratch/gen_q_b%d.txt.hold" % n, "w", encoding="utf-8", newline="\n").write(q)
    assert subprocess.call([sys.executable, "scratch/check_queue_ships.py", "scratch/gen_q_b%d.txt.hold" % n]) == 0
os.makedirs("scratch/gate_s269", exist_ok=True); json.dump(out, open("scratch/gate_s269/predeploy.json", "w", newline="\n"), indent=1, default=str)
print({r: v["keys"] for r, v in out["args"].items()})
