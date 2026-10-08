"""THEORY §158 FREE-PAIRS (c94ee2bf + note 158-1): restart155 with the single arm free (--mem_write top --free_pairs 0.5; §155 R otherwise:
write_mix 2000 / 6000, 24k, slots k / k+100 / k+200 / k+300), dirs boxes/runs_fsq/s158_free_{task}_s{seed}; on-host gate155 (top) + mem_check before slot 1.
    python -m scratch.restart158 --arm free --task 4x4 --slot k --device D --out SLOTDIR
restart147's notes:
THEORY §140 restart-on-plateau driver (one seed slot; restart137 with §140's module and names):
    python -m scratch.restart147 --task 4x4 --slot k --device D --out SLOTDIR   -> boxes/runs_fsq/s147_delta_{task}_s{seed} via boxes.multpair
THEORY §147 DELTA-STATE (b1d7741c / note 147-1): restart140 with --carrier deltamem (the memory branch after block 6); the 2k flag with
loss(2k) < 1.0 is a misfire (note 147-1 (1)): multpair logs it and does not stop, so the attempt runs on and is accepted.
§128's recipe as registered (boxes.multpair defaults: FSQ d 8 / L 8, p_hard 2000 -> 4000, 24k, bs 64, lr 1e-4, the 127-3 guard) with
--stop_plateau (123-3's 2k flag) and --ckpt_every 2000. Slot k tries seeds k, k+100, k+200, k+300; an attempt whose flag fires stops at 2k
(log kept, no eval.json) and the next seed runs. Every attempt logs its flag decision + loss(100 / 1k / 2k) (note 140 as 137-1; score_s140 reads a
flag firing at loss(2k) < 1.0 as a misfire). status() counts an attempt as a plateau only when its "stopped" value starts with "plateau".
SLOTDIR/eval.json summarises."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {"free": ["--mem_write", "top", "--free_pairs", "0.5"]}   # §158 FREE-PAIRS (c94ee2bf): §155 R + --free_pairs 0.5


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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True, choices=["4x4", "5x5"]); ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS)); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    ap.add_argument("--runs_root", default=os.path.join("boxes", "runs_fsq"), help="smokes: a scratch dir, never the real run names (Audit)")
    a, extra = ap.parse_known_args(); os.chdir(ROOT); os.makedirs(a.out, exist_ok=True)
    gj = os.path.join(a.out, "gate158_%s.json" % a.arm)
    if not os.path.exists(gj):                                              # §155 gates on this host's GPU (bf16) + memory at L 17 + flush
        wr = "top"; sh = False
        rc1 = subprocess.call([sys.executable, "-u", "scratch/gate155.py", a.device, wr] + (["shift"] if sh else ["noshift"]) + [a.out], stdout=open(os.path.join(a.out, "gate155.log"), "w"), stderr=subprocess.STDOUT)
        md = os.path.join(a.out, "memcheck").replace(os.sep, "/")
        rc2 = subprocess.call([sys.executable, "-u", "-m", "boxes.multpair", "--family", "mult_pair", "--task", a.task, "--seed", "1", "--carrier", "deltamem"] + ARMS[a.arm] +
                              ["--mem_check", "--device", a.device, "--out", md], stdout=open(os.path.join(a.out, "memcheck.log"), "w"), stderr=subprocess.STDOUT)
        res_ = {"gate_rc": rc1, "memcheck_rc": rc2, "memcheck": json.load(open(os.path.join(md, "mem_check.json"))) if os.path.exists(os.path.join(md, "mem_check.json")) else None}
        res_["pass"] = rc1 == 0 and rc2 == 0 and res_["memcheck"] is not None and res_["memcheck"]["peak_gib"] <= 21.0
        json.dump(res_, open(gj, "w"), indent=1); qlog_line("GATE158 %s %s" % (a.arm, "PASS" if res_["pass"] else "FAIL %s" % res_))
    if not json.load(open(gj))["pass"]:
        sys.exit(3)
    attempts = []; accepted = None; err = None
    for seed in [a.slot + 100 * i for i in range(4)]:
        out = os.path.join(a.runs_root, "s158_%s_%s_s%d" % (a.arm, a.task, seed)).replace(os.sep, "/"); st = status(out)
        if st:                                                              # already decided (restart-safe)
            attempts.append({"seed": seed, "dir": out, "status": st})
            if st == "accepted":
                accepted = seed; break
            continue
        cmd = [sys.executable, "-u", "-m", "boxes.multpair", "--family", "mult_pair", "--task", a.task, "--seed", str(seed), "--stop_plateau", "--ckpt_every", "2000", "--carrier", "deltamem"] + ARMS[a.arm] + extra + \
              ["--device", a.device, "--out", out]
        lg = open(os.path.join("logs" if a.runs_root == os.path.join("boxes", "runs_fsq") else a.runs_root, "gen_s158_%s_%s_s%d.log" % (a.arm, a.task, seed)), "w"); t0 = time.time()
        rc = subprocess.call(cmd, stdout=lg, stderr=subprocess.STDOUT); st = status(out)
        qlog_line("evaluated %s rc %d%s" % (out, rc, "" if st == "accepted" else " no-eval"))
        attempts.append({"seed": seed, "dir": out, "status": st or "error", "rc": rc, "sec": round(time.time() - t0)})
        if st == "accepted":
            accepted = seed; break
        if st != "plateau":
            err = "attempt seed %d ended without eval.json or a plateau stop (rc %s)" % (seed, rc); break
    res = {"task": a.task, "slot": a.slot, "accepted_seed": accepted, "attempts": attempts, "n_restarts": sum(at["status"] == "plateau" for at in attempts),
           "error": err, "extra_args": extra}
    if accepted is None and err is None:
        res["slot_failed"] = "all four attempts plateaued"
    json.dump(res, open(os.path.join(a.out, "eval.json" if err is None else "slot_error.json"), "w", newline="\n"), indent=1)
    print("SLOT", json.dumps(res), flush=True)
    sys.exit(0 if err is None else 1)


if __name__ == "__main__":
    main()
