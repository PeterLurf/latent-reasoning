"""THEORY §128 restart-on-plateau driver (one seed slot): python -m scratch.restart128 --family F --slot k --device D --out SLOTDIR
Slot k tries seeds k, k+100, k+200, k+300 in order: boxes/runs_fsq/s128_{F}_s{seed} = §120 FINAL fsq120 (+ 127-3's guard and the
§127 probe / checks via --probe127) with --stop_plateau. An attempt whose 123-3 flag fires stops at 2k (log kept, no eval.json)
and the next seed runs; the slot's accepted run is its first non-plateau attempt. Held-out families (cyc / mod / rdfa, note 128-1):
the family's first flagged attempt on this host claims boxes/runs_fsq/.check128_{F} (atomic mkdir), keeps training to 24k as the
flag's false-positive check, and the next seed starts beside it. All four slots of a held-out family run on one host.
Every attempt gets its own " evaluated <dir> rc ..." line in the host's queue log, so the drain fetches it (a stopped attempt
has no eval.json: the drain keeps its args + log as <dir>.void_partial; rename back on fetch). SLOTDIR/eval.json summarises."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HELD = ("cyc", "mod", "rdfa")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def qlog_line(msg):
    qs = glob.glob(os.path.join(ROOT, "logs", "queue_gen_*.out"))
    if len(qs) == 1:                                                        # on a Vast host there is exactly one queue log
        with open(qs[0], "a") as f:
            f.write(time.strftime("%H:%M:%S") + " " + msg + "\n")


def status(out):
    if os.path.exists(os.path.join(out, "eval.json")) and os.path.getsize(os.path.join(out, "eval.json")) > 0:
        return "accepted"
    lp = os.path.join(out, "log.jsonl")
    if os.path.exists(lp):
        for line in open(lp):
            if '"stopped"' in line:
                return "plateau"
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", required=True); ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True); ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--runs_root", default=os.path.join("boxes", "runs_fsq"), help="smokes: a scratch dir, never the real run names (Audit)")
    a, extra = ap.parse_known_args(); os.chdir(ROOT); os.makedirs(a.out, exist_ok=True)
    marker = os.path.join(a.runs_root, ".check128_" + a.family) if a.family in HELD else None
    attempts = []; checks = []; accepted = None; err = None
    for seed in [a.slot + 100 * i for i in range(4)]:
        out = os.path.join(a.runs_root, "s128_%s_s%d" % (a.family, seed)).replace(os.sep, "/"); st = status(out)
        if st:                                                              # already decided (restart-safe)
            attempts.append({"seed": seed, "dir": out, "status": st})
            if st == "accepted":
                accepted = seed; break
            continue
        cmd = [sys.executable, "-u", "-m", "boxes.fsq120", "--family", a.family, "--arm", "final", "--seed", str(seed), "--micro", "2",
               "--no_cert", "--probe127", "--stop_plateau"] + (["--check_marker", marker] if marker else []) + extra + ["--device", a.device, "--out", out]
        lg = open(os.path.join("logs" if a.runs_root == os.path.join("boxes", "runs_fsq") else a.runs_root, "gen_s128_%s_s%d.log" % (a.family, seed)), "w"); t0 = time.time()
        p = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT)
        while True:
            rc = p.poll()
            if rc is not None:
                st = status(out)
                qlog_line("evaluated %s rc %d%s" % (out, rc, "" if st == "accepted" else " no-eval"))
                attempts.append({"seed": seed, "dir": out, "status": st or "error", "rc": rc, "sec": round(time.time() - t0)})
                break
            if os.path.exists(os.path.join(out, "CHECK_RUN")):              # note 128-1: keeps training beside the next seed
                checks.append((seed, out, p, t0)); attempts.append({"seed": seed, "dir": out, "status": "check_run"}); st = "check_run"
                break
            time.sleep(a.poll)
        if st == "accepted":
            accepted = seed; break
        if st not in ("plateau", "check_run"):
            err = "attempt seed %d ended without eval.json or a plateau stop (rc %s)" % (seed, attempts[-1].get("rc")); break
    for seed, out, p, t0 in checks:                                         # wait for the check runs (they are not the slot's runs)
        rc = p.wait(); st = status(out)
        qlog_line("evaluated %s rc %d%s" % (out, rc, "" if st == "accepted" else " no-eval"))
        for at in attempts:
            if at["seed"] == seed:
                at.update({"check_finished": st == "accepted", "rc": rc, "sec": round(time.time() - t0)})
    res = {"family": a.family, "slot": a.slot, "accepted_seed": accepted, "attempts": attempts, "n_restarts": sum(at["status"] in ("plateau", "check_run") for at in attempts),
           "error": err, "extra_args": extra}
    if accepted is None and err is None:
        res["slot_failed"] = "all four attempts plateaued"
    json.dump(res, open(os.path.join(a.out, "eval.json" if err is None else "slot_error.json"), "w", newline="\n"), indent=1)
    print("SLOT", json.dumps(res), flush=True)
    sys.exit(0 if err is None else 1)


if __name__ == "__main__":
    main()
