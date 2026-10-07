"""THEORY §144 RUNNING-SUM restart-on-plateau driver (one seed slot; b1d7741c): restart138 with §144's family / arms / names:
    python -m scratch.restart144 --arm f8|n64 --slot k [--sum_mod 10000] --device D --out SLOTDIR   -> boxes/runs_fsq/s144_{ARM}[_m{2,3}]_s{seed}
fsq120 --family sum4 --sum_mod M (the 4-digit running sum, random start text, two-digit items); the ladder (--sum_mod 100 / 1000) names m2 / m3.
§128's S5 recipe exactly (§120 FINAL, guard + probes via --probe127, --stop_plateau, --micro 2 applied at L >= 32 as in §128:
--micro_from 32), plus the user's checkpoints (--ckpt_every 2000) and only the carrier changed by the arm (args names per note 133-1).
Slot k tries seeds k, k+100, k+200, k+300; an attempt whose 123-3 flag fires stops at 2k (log kept, no eval.json) and the next seed runs.
status() counts an attempt as a plateau only when its "stopped" value starts with "plateau" (Audit 2026-10-06: other stop records,
e.g. a prototype's, must not advance a slot). SLOTDIR/eval.json summarises."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {"f8": ["--d_q", "8"], "n64": ["--d_q", "64", "--no_round", "--dither", "0.25"]}


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
                try:
                    if str(json.loads(line).get("stopped", "")).startswith("plateau"):
                        return "plateau"
                except ValueError:
                    pass
    return None


def memcheck(a, extra):
    """§138 placement (fa1720ff): W_down / W_up 768 x 768 plus the hinge's extra forward, checked at L 32 with the micro rule on the run's own
    card before the slot starts. The arm's exact flags at --sched_scale 0.05 (L_max 32 from step 700, the hinge from step 100), full bs 64,
    stopped at 900 (--stop_after: no eval, no model); peak = max logged mem_gib (torch max_memory_allocated, cumulative). PASS = clean
    STOP_AFTER and peak <= --mem_limit_gib (21 of the 3090's 24 GiB). Written under the slot dir, never under a run name."""
    d = os.path.join(a.out, "memcheck_run").replace(os.sep, "/"); t0 = time.time()
    cmd = [sys.executable, "-u", "-m", "boxes.fsq120", "--family", "sum4", "--sum_mod", str(a.sum_mod), "--arm", "final", "--seed", "9", "--micro", "2", "--micro_from", "32",
           "--no_cert", "--probe127", "--sched_scale", "0.05", "--stop_after", "900"] + ARMS[a.arm] + extra + ["--device", a.device, "--out", d]
    with open(os.path.join(a.out, "memcheck.log"), "w") as lg:
        rc = subprocess.call(cmd, stdout=lg, stderr=subprocess.STDOUT)
    mem, lmax = [], 0
    if os.path.exists(os.path.join(d, "log.jsonl")):
        for line in open(os.path.join(d, "log.jsonl")):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "mem_gib" in r:
                mem.append(r["mem_gib"]); lmax = max(lmax, r.get("L_max", 0))
    stopped = "STOP_AFTER" in open(os.path.join(a.out, "memcheck.log")).read(); peak = max(mem) if mem else None
    return {"arm": a.arm, "rc": rc, "stop_after_reached": stopped, "peak_gib": peak, "L_max_reached": lmax, "limit_gib": a.mem_limit_gib,
            "pass": bool(rc == 0 and stopped and peak is not None and lmax >= 32 and peak <= a.mem_limit_gib), "sec": round(time.time() - t0),
            "cmd": " ".join(cmd[1:])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=sorted(ARMS)); ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True); ap.add_argument("--poll", type=int, default=20); ap.add_argument("--sum_mod", type=int, default=10000)
    ap.add_argument("--runs_root", default=os.path.join("boxes", "runs_fsq"), help="smokes: a scratch dir, never the real run names (Audit)")
    ap.add_argument("--memcheck", action="store_true", help="§138 placement: peak memory at L 32 on this card (micro rule) before any attempt")
    ap.add_argument("--mem_limit_gib", type=float, default=21.0)
    a, extra = ap.parse_known_args(); os.chdir(ROOT); os.makedirs(a.out, exist_ok=True)
    if a.memcheck and not os.path.exists(os.path.join(a.out, "memcheck.json")):
        mc = memcheck(a, extra); json.dump(mc, open(os.path.join(a.out, "memcheck.json"), "w", newline="\n"), indent=1)
        if os.path.isdir("logs"):                                           # logs/ is always drained; a FAILed slot dir (no eval.json) is not
            json.dump(mc, open(os.path.join("logs", "memcheck144_%s_k%d.json" % (a.arm, a.slot)), "w", newline="\n"), indent=1)
        qlog_line("MEMCHECK %s %s peak %s GiB (limit %s)" % (a.arm, "PASS" if mc["pass"] else "FAIL", mc["peak_gib"], a.mem_limit_gib)); print("MEMCHECK", json.dumps(mc), flush=True)
    if a.memcheck and not json.load(open(os.path.join(a.out, "memcheck.json")))["pass"]:
        sys.exit(3)                                                         # no attempt runs; the slot dir keeps memcheck.json + the check's log
    attempts = []; accepted = None; err = None
    for seed in [a.slot + 100 * i for i in range(4)]:
        out = os.path.join(a.runs_root, "s144_%s%s_s%d" % (a.arm, {10000: "", 100: "_m2", 1000: "_m3"}[a.sum_mod], seed)).replace(os.sep, "/"); st = status(out)
        if st:                                                              # already decided (restart-safe)
            attempts.append({"seed": seed, "dir": out, "status": st})
            if st == "accepted":
                accepted = seed; break
            continue
        cmd = [sys.executable, "-u", "-m", "boxes.fsq120", "--family", "sum4", "--sum_mod", str(a.sum_mod), "--arm", "final", "--seed", str(seed), "--micro", "2", "--micro_from", "32",
               "--cert_time", "1800", "--probe127", "--stop_plateau", "--ckpt_every", "2000"] + ARMS[a.arm] + extra + ["--device", a.device, "--out", out]
        lg = open(os.path.join("logs" if a.runs_root == os.path.join("boxes", "runs_fsq") else a.runs_root, "gen_s144_%s%s_s%d.log" % (a.arm, {10000: "", 100: "_m2", 1000: "_m3"}[a.sum_mod], seed)), "w"); t0 = time.time()
        rc = subprocess.call(cmd, stdout=lg, stderr=subprocess.STDOUT); st = status(out)
        qlog_line("evaluated %s rc %d%s" % (out, rc, "" if st == "accepted" else " no-eval"))
        attempts.append({"seed": seed, "dir": out, "status": st or "error", "rc": rc, "sec": round(time.time() - t0)})
        if st == "accepted":
            accepted = seed; break
        if st != "plateau":
            err = "attempt seed %d ended without eval.json or a plateau stop (rc %s)" % (seed, rc); break
    res = {"arm": a.arm, "slot": a.slot, "accepted_seed": accepted, "attempts": attempts, "n_restarts": sum(at["status"] == "plateau" for at in attempts),
           "error": err, "arm_args": ARMS[a.arm], "extra_args": extra}
    if accepted is None and err is None:
        res["slot_failed"] = "all four attempts plateaued"
    json.dump(res, open(os.path.join(a.out, "eval.json" if err is None else "slot_error.json"), "w", newline="\n"), indent=1)
    print("SLOT", json.dumps(res), flush=True)
    sys.exit(0 if err is None else 1)


if __name__ == "__main__":
    main()
