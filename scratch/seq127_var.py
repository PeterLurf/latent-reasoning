"""THEORY note 127-9: VAR Stage 2, sequenced for cost. S5 s4, then s3, then s1, one at a time (boxes/runs_fsq/s127_s5_var_s{seed}:
fsq120 §120 FINAL + --var_res --var_est batch, guard / train_driver via --probe127, --stop_plateau). Stop at the first one that
plateaus (no eval.json) or ends long_acc < 0.99: ENTRY-FIX is then false and the rest are not run. FF s1-2 (s127_ff_var_s{1,2})
only if all three S5 runs are exact.   python -m scratch.seq127_var --device D --out SUMMARYDIR [extra fsq120 args]"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def qlog(msg):
    qs = glob.glob(os.path.join(ROOT, "logs", "queue_gen_*.out"))
    if len(qs) == 1:
        with open(qs[0], "a") as f:
            f.write(time.strftime("%H:%M:%S") + " " + msg + "\n")


def run(fam, seed, dev, extra):
    out = "boxes/runs_fsq/s127_%s_var_s%d" % (fam, seed)
    cmd = [sys.executable, "-u", "-m", "boxes.fsq120", "--family", fam, "--arm", "final", "--seed", str(seed), "--var_res", "--var_est", "batch",
           "--probe127", "--stop_plateau"] + extra + ["--device", dev, "--out", out]
    rc = subprocess.call(cmd, stdout=open(os.path.join("logs", "gen_s127_%s_var_s%d.log" % (fam, seed)), "w"), stderr=subprocess.STDOUT)
    ev = os.path.join(out, "eval.json"); acc = json.load(open(ev)).get("long_acc") if os.path.exists(ev) else None
    qlog("evaluated %s rc %d%s" % (out, rc, "" if acc is not None else " no-eval"))
    return {"dir": out, "rc": rc, "long_acc": acc}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a, extra = ap.parse_known_args(); os.chdir(ROOT); os.makedirs(a.out, exist_ok=True); res = {"runs": [], "stopped_at": None}
    for seed in (4, 3, 1):
        r = run("s5", seed, a.device, extra); res["runs"].append(r)
        if r["long_acc"] is None or r["long_acc"] < 0.99:
            res["stopped_at"] = r["dir"]; break
    if res["stopped_at"] is None:
        for seed in (1, 2):
            res["runs"].append(run("ff", seed, a.device, extra))
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1); print("SEQ127", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
