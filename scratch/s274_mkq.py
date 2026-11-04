"""§271 N1 SHUFFLED-OBJECTS stage S1 queues (N1_v1 §5 S1 + Notes 271-1 / 271-4 / 271-5; build serial s274, filler family s274; guard $3.0; RTX 3090 pinned):
  b425 nloop_s1  LOOP-SI seed 1 = of_s1's recipe (of_s1_init streaming copy 9453d5ff, sha on host) with the task flags --src so --cue index (no OW perm / ask / distract
                 flags), --steps 8000 --stop_at 4000 (the pilot = the first half of seed 1's 8k cosine); host gates first (G3 gradflow loop; G4 mem_check; G6 time probe
                 300 steps); at 4000 scratch.s274_stopland evals the gate subset acc_so / ws_so / swap_so / kcurve_so (inject_so_eval).
  b426 ndof_s1..3  DIRECT-OF seeds 1-3, one host, sequential: --arm direct --src so --cue index from of_s1's trunk (of_s1_trunk f7c2a5a3, the g.* tensors), same
                 steps / bs / lr / d_max; G3 gradflow direct_of first; full eval battery in-process (inject_so_eval, every part; loop-only parts write n/a).
  b427 ndsi_s1..3  DIRECT-SI seeds 1-3: DIRECT-OF + --so_direct_si 1 --aux_step 1 --aux_reweight --aux_fade 48000,48000 (Note 271-1; CE x 1/n_tok, Note 271-5).
Ships: the inits, the floors file (boxes/runs_fsq/n1_floors/floors_so.json, sha 1db54c39 asserted here) and the 4 BBH files. Frozen args scratch/s274_args_frozen_<run>.json
(local --args_only) + the diffs (LOOP-SI vs of_s1; DIRECT-SI vs DIRECT-OF) -> scratch/gate_s274/predeploy.json. Pre-deploy: blob ids == HEAD; init shas (majority of 3).
    python -B scratch/s274_mkq.py"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
from nobuf_sha import sha_nobuf  # noqa: E402
import caps_check as CC  # noqa: E402
import gate_s228 as G8  # noqa: E402
import gate_s257 as G7  # noqa: E402
import s228_args_check as C  # noqa: E402

OFI = ("boxes/runs_fsq/of_s1_init/model.pt", "9453d5ff26015853"); TRK = ("boxes/runs_fsq/of_s1_trunk/model.pt", "f7c2a5a3ff7427ff")
FLOORS = ("boxes/runs_fsq/n1_floors/floors_so.json", "1db54c39016c8d23")
BBH = ["scratch/n1_build/bbh/bbh_sha.json"] + ["scratch/n1_build/bbh/tracking_shuffled_objects_%s_objects.json" % s for s in ("three", "five", "seven")]
BLOB = ["boxes/inject_so.py", "boxes/inject_so_eval.py", "boxes/so_words.py", "boxes/so_tok.py", "boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/inject_ow.py",
        "boxes/inject_st.py", "boxes/nested170.py", "boxes/deltamem.py", "boxes/deltas5.py", "boxes/core.py", "boxes/s5.py", "boxes/ow_words.py", "boxes/nested_words.py",
        "scratch/s257_guarded.py", "scratch/s274_guarded.py", "scratch/s274_stopland.py", "scratch/gate_s274.py", "scratch/gate_s251.py"]
ALL_PARTS = "acc_so,ws_so,swap_so,kcurve_so,long_so,perm_label,lookup,bbh,beta_so,floors"; GATE_PARTS = "acc_so,ws_so,swap_so,kcurve_so"


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def loop_flags():
    fl = G7.dch_flags(1, *OFI) + ["--hold_inc", "1.6", "--hold_log"]
    i = fl.index("--distract_p"); del fl[i:i + 2]; i = fl.index("--distract_mix"); del fl[i:i + 2]
    i = fl.index("--cue"); fl[i + 1] = "index"; return fl + ["--src", "so"]


def direct_flags(k, si):
    fl = ["--arm", "direct", "--src", "so", "--cue", "index", "--d_max", "4", "--steps", "8000", "--mul_s_p", "0", "--micro", "1", "--eval_bs", "250",
          "--init_trunk", TRK[0], "--init_sha", TRK[1]] + (["--seed", str(k)] if k != 1 else [])
    return fl + (["--so_direct_si", "1", "--aux_step", "1", "--aux_reweight", "--aux_fade", "48000,48000"] if si else [])


def noinit(fl):
    fl = list(fl)
    for f in ("--init_model", "--init_model_sha", "--init_trunk", "--init_sha"):
        if f in fl:
            i = fl.index(f); del fl[i:i + 2]
    return fl


def frozen(run, fl, tail, ref=None):
    d = G8.fresh(ROOT + "/scratch/_tmp/s274/args_" + run)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_so"] + fl + tail + ["--device", "cpu", "--args_only"] + (["--args_ref", ref] if ref else []) + ["--out", d],
               d + "/stdout.log"); assert rc == 0, (run, rc, open(d + "/stdout.log").read()[-1500:])
    shutil.copy(d + "/args.json", "scratch/s274_args_frozen_%s.json" % run); return json.load(open(d + "/args.json"))


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}; pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}
ish = {p: [sha_nobuf(os.path.abspath(p)) for _ in range(3)] for p, _ in (OFI, TRK)}
ok = (all(v["head"] == v["disk"] for v in pre.values()) and all(sum(s_.startswith(s) for s_ in ish[p]) >= 2 for p, s in (OFI, TRK))
      and sha_nobuf(os.path.abspath(FLOORS[0])).startswith(FLOORS[1]) and all(os.path.exists(p) for p in BBH))
out = {"blobs": pre, "init_unbuffered_sha256_x3": ish, "predeploy_ok": ok, "args": {}}; assert ok, ("pre-deploy", out)
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); DATA = " ".join([FLOORS[0]] + BBH)
# ---- frozen args + diffs
lf = loop_flags(); ltail = ["--stop_at", "4000", "--ckpt_every", "1000", "--eval_parts", GATE_PARTS]
al = frozen("nloop_s1", lf, ltail, "boxes/runs_fsq/of_s1/args.json")
out["args"]["nloop_s1_vs_of_s1"] = {kk: v for kk, v in C.diff(json.load(open("boxes/runs_fsq/of_s1/args.json")), al).items() if kk != "pair_stats"}
ad = {}
for k in (1, 2, 3):
    for si, nm in ((0, "ndof"), (1, "ndsi")):
        ad["%s_s%d" % (nm, k)] = frozen("%s_s%d" % (nm, k), direct_flags(k, si), ["--ckpt_every", "1000", "--eval_parts", ALL_PARTS])
dsi = {kk: v for kk, v in C.diff(ad["ndof_s1"], ad["ndsi_s1"]).items() if kk not in ("pair_stats", "out")}
out["args"]["ndsi_s1_vs_ndof_s1"] = dsi
assert set(dsi) - {"format"} <= {"so_direct_si", "aux_step", "aux_reweight", "aux_fade"}, ("Audit (2): DIRECT-SI vs DIRECT-OF", sorted(dsi))
# 'format' = the trainer's descriptive string (names the SI loss when --so_direct_si 1); kept in the recorded diff, excluded from the assert (11-04)
assert "format" not in dsi or dsi["format"][1].startswith(dsi["format"][0]), ("format: the SI text extends the OF text", dsi["format"])
eq = {r: {k_: a_.get(k_) for k_ in ("steps", "lr", "bs", "d_max")} for r, a_ in [("nloop_s1", al)] + list(ad.items())}
assert len({json.dumps(v, sort_keys=True) for v in eq.values()}) == 1, ("Audit (3): steps / lr / bs / d_max equal across arms", eq)
out["steps_lr_bs_equal"] = eq
# ---- queues
caps_loop = CC.caps_line([0.1, 0.25, 0.25, 2.3, 0.6], 0.6, 0.25, 1.9, 0.854, "RTX_3090", 0.1)


def hard_wall(wallw, eval_h, gates_h):
    """Audit 11-04 LOW: '# HARD_WALL_H' = 1.5 x the projected host hours (setup 0.4 + lines + eval + gates), asserted >= 1.3 x; host idle stop backstop."""
    proj = 0.4 + sum(wallw) + eval_h + gates_h; hw = round(1.5 * proj, 2); assert hw >= 1.3 * proj
    return "\n# HARD_WALL_H %s\n# IDLE_H 1.5" % hw


caps_loop += hard_wall([0.1, 0.25, 0.25, 2.3, 0.6], 0.6, 0.25)
W = "boxes/runs_fsq/nloop_s1"; fs = " ".join(lf); GP = W + "_gates/gates_pass.json"
q = ("# THEORY §271 N1 S1 LOOP-SI seed 1 (score_n1 S1 GO gate): of_s1's recipe + --src so --cue index, 8k cosine stopped at 4000; host gates G3 gradflow (loop), G4 mem, "
     "G6 time (300 steps) before TRAIN; stopland evals %s (inject_so_eval). Frozen args scratch/s274_args_frozen_nloop_s1.json.\n# HOSTPROBE\n# P=1\n%s\n"
     "%s_mem boxes.inject_so %s --mem_check\n%s_gtime boxes.inject_so %s --time_steps 300\n"
     "%s_gates scratch.gate_s274 --arms loop -- %s ;; x\n"
     "%s scratch.s274_guarded --mode train --blob %s --init %s=%s --require %s --ship %s %s scratch/s257_guarded.py scratch/gate_s251.py -- %s %s --args_ref scratch/s274_args_frozen_nloop_s1.json\n"
     "%s scratch.s274_stopland --step 4000 --parts %s\n") % (GATE_PARTS, caps_loop, W, fs, W, fs, W, " ".join(noinit(lf)), W, blob, OFI[0], OFI[1], GP, OFI[0], DATA, fs,
                                                          " ".join(ltail), W, GATE_PARTS)
open("scratch/gen_q_b425.txt.hold", "w", encoding="utf-8", newline="\n").write(q)
for n, nm, si in ((426, "ndof", 0), (427, "ndsi", 1)):
    W1 = "boxes/runs_fsq/%s_s1" % nm; f1 = " ".join(direct_flags(1, si)); GP = W1 + "_gates/gates_pass.json"
    body = ("%s_mem boxes.inject_so %s --mem_check\n%s_gtime boxes.inject_so %s --time_steps 300\n%s_gates scratch.gate_s274 --arms %s -- x ;; %s\n") % (
        W1, f1, W1, f1, W1, "direct_si" if si else "direct_of", " ".join(noinit(direct_flags(1, si))))
    for k in (1, 2, 3):
        Wk = "boxes/runs_fsq/%s_s%d" % (nm, k); fk = " ".join(direct_flags(k, si))
        body += "%s scratch.s274_guarded --mode train --blob %s --init %s=%s --require %s --ship %s %s scratch/s257_guarded.py scratch/gate_s251.py scratch/s274_stopland.py -- %s --ckpt_every 1000 --eval_parts %s --args_ref scratch/s274_args_frozen_%s_s%d.json\n" % (
            Wk, blob, TRK[0], TRK[1], GP, TRK[0], DATA, fk, ALL_PARTS, nm, k)
    caps = CC.caps_line([0.1, 0.1, 0.1, 1.0, 1.0, 1.0], 0.5, 0.1, 0.3, 0.854, "RTX_3090", 0.1) + hard_wall([0.1, 0.1, 0.1, 1.0, 1.0, 1.0], 0.5, 0.1)
    q = ("# THEORY §271 N1 S1 %s seeds 1-3 (one host, sequential; score_n1 D-L1 / X1 / DIRECT gate): one pass from of_s1's trunk (of_s1_trunk %s)%s; G3 gradflow %s first; "
         "full eval battery in-process. Frozen args scratch/s274_args_frozen_%s_s<k>.json.\n# HOSTPROBE\n# P=1\n%s\n%s") % (
        "DIRECT-SI" if si else "DIRECT-OF", TRK[1], " + --so_direct_si 1 --aux_step 1 --aux_reweight --aux_fade 48000,48000 (Notes 271-1 / 271-5)" if si else "",
        "direct_si" if si else "direct_of", nm, caps, body)
    open("scratch/gen_q_b%d.txt.hold" % n, "w", encoding="utf-8", newline="\n").write(q)
for n in (425, 426, 427):
    p = "scratch/gen_q_b%d.txt.hold" % n
    assert subprocess.call([sys.executable, "scratch/check_queue_ships.py", p]) == 0, p
    assert subprocess.call([sys.executable, "-B", "scratch/queue_dryrun.py", p]) == 0, p
os.makedirs("scratch/gate_s274", exist_ok=True); json.dump(out, open("scratch/gate_s274/predeploy.json", "w", newline="\n"), indent=1, default=str)
print(json.dumps({"loop_vs_of_s1": sorted(out["args"]["nloop_s1_vs_of_s1"]), "dsi_vs_dof": out["args"]["ndsi_s1_vs_ndof_s1"], "eq": eq["nloop_s1"]}))
