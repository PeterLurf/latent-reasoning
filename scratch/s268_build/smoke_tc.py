"""§261 TARGET-CUE CPU smoke -> scratch/gate_s268/smoke_tc.json (no readings kept):
  (1) boxes.inject_aw --args_only with the pilot flags (cue target) -> args.json; (2) scratch.s268_target on a temp run dir (those args + the bun_s1 copy): rc 0 and
  gates_pass.json target_only; (3) every §256 + §257 held-out insert item (TEST2 d 4) tokenizes under the target cue: identical cues, tail == tgt_cue(raw) names S_d, and
  rh_input(it, t, '') == think_cues[t - 1]; (4) scratch.s268_stopland on a fake 2000-step ckpt_last: model.pt + ckpt_stop_sha256.json written (eval parts 'none_' -> nonzero
  eval rc expected; the eval is not run here), and it refuses a second call (model.pt present).
    python -B scratch/s268_build/smoke_tc.py"""
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s257 as G7  # noqa: E402

INIT = ("boxes/runs_fsq/bun_s1_init/model.pt", "1ad3517b7635fddf")
fl = G7.dch_flags(1, *INIT) + ["--hold_inc", "1.6", "--hold_log"]; fl[fl.index("--cue") + 1] = "target"
d = "scratch/_tmp/s268_smoke"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
rc0 = subprocess.call([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--ckpt_every", "1000", "--stop_at", "2000", "--eval_parts", "acc,ws,insert_ho", "--device", "cpu",
                                                                                "--args_only", "--out", d], stdout=open("scratch/_tmp/s268_args.log", "w"), stderr=subprocess.STDOUT)
a = json.load(open(d + "/args.json")); o = {"args_rc": rc0, "args_cue": a.get("cue"), "args_stop_at": a.get("stop_at"), "args_steps": a.get("steps")}
shutil.copy(INIT[0], d + "/model.pt")
o["target_rc"] = subprocess.call([sys.executable, "-B", "-m", "scratch.s268_target", "--device", "cpu", "--out", d], stdout=open("scratch/_tmp/s268_tgt.log", "w"),
                                 stderr=subprocess.STDOUT)
o["gates_pass"] = json.load(open(d + "/gates_pass.json")) if os.path.exists(d + "/gates_pass.json") else {}
from boxes import inject_aw as AW  # noqa: E402
from boxes import inject_aw_eval as EV  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402
import boxes.insert_render as IR  # noqa: E402
s = IO.build_sets(mul_s_p=0); HO = {**IR.held_out_sets(s["test2"][4])[0], **IR.held_out2_sets(s["test2"][4])[0]}; n = 0; bad = []
for fam, dd in HO.items():
    for pos, its in dd.items():
        for it in EV.tokd({"test2": {4: its}}, a, (4,))[4]:
            tc = it["think_cues"]; tail = AW.ENC.decode(tc[0][len(it["think"]):]); sd = it["raw"]["names"]["spine"][-1]; n += 1
            if not (all(q == tc[0] for q in tc) and tail == AW.tgt_cue(it["raw"]) + " The answer is:" and (" %s has" % sd) in tail
                    and all(EV.rh_input(it, t, "") == tc[t - 1] for t in range(1, len(tc) + 1))):
                bad.append("%s:%s:%s" % (fam, pos, it["sig"]))
o["ho_items"] = n; o["ho_bad"] = bad[:5]; o["ho_n_bad"] = len(bad); o["ho_fams"] = sorted(HO)
import torch  # noqa: E402
a2 = dict(a); json.dump(a2, open(d + "/args.json", "w")); os.remove(d + "/gates_pass.json") if os.path.exists(d + "/gates_pass.json") else None
sd_ = torch.load(d + "/model.pt", map_location="cpu", weights_only=False); os.remove(d + "/model.pt"); torch.save({"step": 2000, "model": sd_}, d + "/ckpt_last.pt"); del sd_
r1 = subprocess.call([sys.executable, "-B", "-m", "scratch.s268_stopland", "--step", "2000", "--parts", "none_", "--device", "cpu", "--out", d],
                     stdout=open("scratch/_tmp/s268_sl.log", "w"), stderr=subprocess.STDOUT)
o["stopland_rc_eval_skipped"] = r1; o["stopland_model"] = os.path.getsize(d + "/model.pt") if os.path.exists(d + "/model.pt") else 0
o["stopland_ckpt_json"] = os.path.exists(d + "/ckpt_stop_sha256.json"); o["stopland_host_shas"] = os.path.exists(d + "/host_shas.json")
r2 = subprocess.call([sys.executable, "-B", "-m", "scratch.s268_stopland", "--step", "2000", "--parts", "none_", "--device", "cpu", "--out", d],
                     stdout=open("scratch/_tmp/s268_sl2.log", "w"), stderr=subprocess.STDOUT); o["stopland_second_call_refused"] = r2 != 0 and "already holds" in open("scratch/_tmp/s268_sl2.log").read()
o["pass"] = (rc0 == 0 and o["args_cue"] == "target" and o["args_stop_at"] == 2000 and o["target_rc"] == 0 and o["gates_pass"].get("target_only") is True and n > 0
             and not bad and o["stopland_model"] > 0 and o["stopland_ckpt_json"] and o["stopland_host_shas"] and o["stopland_second_call_refused"])
os.makedirs("scratch/gate_s268", exist_ok=True); json.dump(o, open("scratch/gate_s268/smoke_tc.json", "w", newline="\n"), indent=1); print("smoke_tc pass", o["pass"]); shutil.rmtree(d)
