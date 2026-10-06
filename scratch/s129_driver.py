"""THEORY §129 driver (one RTX 3090 host): python -m scratch.s129_driver --device D --out SUMMARYDIR [extra fsq120 args]
0. gradflow (scratch/gradflow_check.py --paths fsq120_s129, CUDA bf16 as training) -> SUMMARYDIR/gradflow_check_s129_host.json (fetched with the summary);
   a FAIL stops everything.
1. Phase 1 over the fixed seed order 2, 1, 3, 4, 101..104, 201..204, ...: boxes/runs_fsq/s129_s5_s{seed} = §120 FINAL S5 +
   127-3's guard + §127 probe (--probe127) with --save_at_flag. Not flagged -> stopped at 2k (dir kept, not in the panel); flagged
   -> ckpt_2k.pt saved -> PANEL (the first 4 flagged seeds; no more phase-1 launches after that).
2. For each panel seed, on this host: REDRAW (--resume_2k <ckpt> --redraw_head, same dir, log appended) and CLOCK-ONLY
   (--resume_2k <ckpt> --clock_only, dir s129_s5_s{seed}_clock), both with --stop_plateau (flag at clock 2k, else 24k + eval).
   The two continuations of a seed run together; at most one pair at a time beside the phase-1 scan (3090 memory). The 2k
   checkpoint (~1.5 GB) is deleted once both continuations of its seed have exited (recorded in the summary).
Every run gets an " evaluated <dir> rc ..." line in the host's queue log (drain); SUMMARYDIR/eval.json summarises."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = [2, 1, 3, 4] + [o + k for o in (100, 200, 300, 400) for k in (1, 2, 3, 4)]


def qlog(msg):
    qs = glob.glob(os.path.join(ROOT, "logs", "queue_gen_*.out"))
    if len(qs) == 1:
        with open(qs[0], "a") as f:
            f.write(time.strftime("%H:%M:%S") + " " + msg + "\n")


def last_marks(out):
    lp = os.path.join(out, "log.jsonl"); m = {}
    if os.path.exists(lp):
        for line in open(lp):
            for k in ("flag_fired", "stopped", "redraw", "clock_only"):
                if '"%s"' % k in line:
                    m[k] = True
    m["eval"] = os.path.exists(os.path.join(out, "eval.json"))
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    ap.add_argument("--runs_root", default=os.path.join("boxes", "runs_fsq")); ap.add_argument("--panel", type=int, default=4)
    ap.add_argument("--poll", type=int, default=30); ap.add_argument("--skip_gradflow", action="store_true", help="smoke only")
    a, extra = ap.parse_known_args(); os.chdir(ROOT); os.makedirs(a.out, exist_ok=True); summ = {"seeds": [], "panel": [], "deleted_ckpts": []}
    if not a.skip_gradflow:
        rc = subprocess.call([sys.executable, "-u", os.path.join("scratch", "gradflow_check.py"), "--paths", "fsq120_s129", "--device", a.device,
                              "--out", os.path.join(a.out, "gradflow_check_s129_host.json")])
        summ["gradflow_rc"] = rc
        if rc != 0:
            json.dump(summ, open(os.path.join(a.out, "slot_error.json"), "w"), indent=1); print("GRADFLOW FAIL", flush=True); sys.exit(1)
    base = [sys.executable, "-u", "-m", "boxes.fsq120", "--family", "s5", "--arm", "final", "--probe127"] + extra + ["--device", a.device]
    running = []                                                            # (seed, kind, out, Popen)
    pending = []                                                            # panel seeds whose pair has not started

    def launch_pair(seed):
        d = os.path.join(a.runs_root, "s129_s5_s%d" % seed).replace(os.sep, "/"); ck = d + "/ckpt_2k.pt"
        for kind, out, fl in (("redraw", d, ["--redraw_head"]), ("clock", d + "_clock", ["--clock_only"])):
            lg = open(os.path.join("logs", "gen_s129_s5_s%d%s.log" % (seed, "_redraw" if kind == "redraw" else "_clock")), "w")
            running.append((seed, kind, out, subprocess.Popen(base + ["--seed", str(seed), "--stop_plateau", "--resume_2k", ck] + fl + ["--out", out], stdout=lg, stderr=subprocess.STDOUT)))

    def reap():
        for r in list(running):
            seed, kind, out, p = r
            if p.poll() is not None:
                running.remove(r); mk = last_marks(out)
                qlog("evaluated %s rc %d%s" % (out, p.returncode, "" if mk["eval"] else " no-eval"))
                summ["seeds"].append({"seed": seed, "run": kind, "dir": out, "rc": p.returncode, "eval": mk["eval"], "stopped": mk.get("stopped", False)})
                if not any(s_ == seed for s_, _, _, _ in running):              # both continuations done -> drop the 2k checkpoint
                    ck = os.path.join(a.runs_root, "s129_s5_s%d" % seed, "ckpt_2k.pt")
                    if os.path.exists(ck):
                        os.remove(ck); summ["deleted_ckpts"].append(ck)
        if not running and pending:
            launch_pair(pending.pop(0))

    for seed in ORDER:
        if len(summ["panel"]) >= a.panel:
            break
        out = os.path.join(a.runs_root, "s129_s5_s%d" % seed).replace(os.sep, "/")
        lg = open(os.path.join("logs", "gen_s129_s5_s%d_phase1.log" % seed), "w")
        p = subprocess.Popen(base + ["--seed", str(seed), "--save_at_flag", "--out", out], stdout=lg, stderr=subprocess.STDOUT)
        while p.poll() is None:
            reap(); time.sleep(a.poll)
        mk = last_marks(out)
        if mk.get("flag_fired"):
            summ["panel"].append(seed); pending.append(seed); reap()
        elif mk.get("stopped"):
            qlog("evaluated %s rc %d no-eval" % (out, p.returncode)); summ["seeds"].append({"seed": seed, "run": "phase1", "dir": out, "flagged": False})
        else:
            qlog("evaluated %s rc %d no-eval" % (out, p.returncode)); summ["error"] = "phase 1 seed %d ended without a flag decision (rc %d)" % (seed, p.returncode); break
    while running or pending:
        reap(); time.sleep(a.poll)
    json.dump(summ, open(os.path.join(a.out, "eval.json" if "error" not in summ else "slot_error.json"), "w", newline="\n"), indent=1)
    print("S129", json.dumps(summ), flush=True); sys.exit(0 if "error" not in summ else 1)


if __name__ == "__main__":
    main()
