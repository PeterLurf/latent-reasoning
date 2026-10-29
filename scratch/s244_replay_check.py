"""§244 host replay check (AO_v1 §3; Audit G-D b7cb8630 rule): the b366 host's aw_plaw2_s1_rep_gtime (aw_plaw2_s1's exact flags + --time_steps 100, current stack,
RTX 3090) vs aw_plaw2_s1's logged step-100 record. The step-100 "loss" is the mean over steps 1-100 (the only record before step 100 in both logs).
PASS iff |rel diff| of that mean <= 1e-2 (and so within 3%); answer_ce / aux_ce / gn_preclip_mean100 reported. -> scratch/gate_s244/host_replay.json
    python -B scratch/s244_replay_check.py <host> <port>"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); h, p = sys.argv[1], sys.argv[2]
r = subprocess.run(["ssh", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new", "-i", os.path.expanduser("~/.ssh/runpod_ed25519"), "-p", p, "root@" + h,
                    "cat /workspace/Latent/boxes/runs_fsq/aw_plaw2_s1_rep_gtime/log.jsonl; echo ===; cat /workspace/Latent/boxes/runs_fsq/aw_plaw2_s1_rep_gtime/args.json"],
                   capture_output=True, text=True, timeout=120)
logtxt, _, argtxt = r.stdout.partition("===")
rep = [json.loads(l_) for l_ in logtxt.splitlines() if l_.strip().startswith("{")]; rep100 = [x for x in rep if x.get("step") == 100]
org = [json.loads(l_) for l_ in open(os.path.join(ROOT, "boxes", "runs_fsq", "aw_plaw2_s1", "log.jsonl"))]; org100 = [x for x in org if x.get("step") == 100][0]
o = {"host": "%s:%s" % (h, p), "orig_step100": {k: org100.get(k) for k in ("loss", "answer_ce", "aux_ce", "gn_preclip_mean100", "aux_lambda")}}
if rep100:
    x = rep100[0]; o["replay_step100"] = {k: x.get(k) for k in ("loss", "answer_ce", "aux_ce", "gn_preclip_mean100", "aux_lambda")}
    o["rel_diff"] = {k: (x[k] - org100[k]) / abs(org100[k]) for k in ("loss", "answer_ce", "aux_ce", "gn_preclip_mean100") if x.get(k) is not None and org100.get(k)}
    try:
        ra = json.loads(argtxt); o["replay_gpu"] = ra.get("train_gpu"); o["replay_micro"] = ra.get("micro"); o["replay_module_sha"] = (ra.get("module_sha256") or "")[:12]
    except ValueError:
        pass
    o["pass"] = abs(o["rel_diff"]["loss"]) <= 1e-2
else:
    o["pass"] = False; o["why"] = "no step-100 record on the host yet (or ssh failed: %s)" % r.stderr[-300:]
os.makedirs(os.path.join(ROOT, "scratch", "gate_s244"), exist_ok=True)
json.dump(o, open(os.path.join(ROOT, "scratch", "gate_s244", "host_replay.json"), "w", newline="\n"), indent=1); print(json.dumps(o))
