"""Memory gate in front of a queued training run (§143 placement: "memory at L 32 with d 768 plus COPY rows", on the run's own 3090).
    queue line:  <out> scratch.memgate --target boxes.fsq120 [--mc_force] [--mc_scale 0.05] [--mc_stop 900] [--limit_gib 21] <the run's args>
First the run's exact args at --sched_scale mc_scale, --stop_after mc_stop (and --grow_force_smoke with --mc_force, so both §143 widenings fire on
step alone and width 768 is reached before L_max 32), into boxes/runs_fsq/memcheck/<run name> (never a run name). PASS = clean STOP_AFTER,
L_max 32 reached, the final logged width (grow events) reached, and peak mem_gib (torch max_memory_allocated, logged per window) <= limit_gib.
PASS -> the real run (exit code passed through); FAIL -> memgate.json in the memcheck dir + logs/memgate_<run>.json, exit 3, nothing trained."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def qlog_line(msg):
    qs = glob.glob(os.path.join(ROOT, "logs", "queue_gen_*.out"))
    if len(qs) == 1:                                                        # on a Vast host there is exactly one queue log
        with open(qs[0], "a") as f:
            f.write(time.strftime("%H:%M:%S") + " " + msg + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True); ap.add_argument("--mc_force", action="store_true")
    ap.add_argument("--mc_scale", type=float, default=0.05); ap.add_argument("--mc_stop", type=int, default=900); ap.add_argument("--limit_gib", type=float, default=21.0)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a, rest = ap.parse_known_args(); os.chdir(ROOT); name = os.path.basename(a.out.rstrip("/"))
    md = os.path.join("boxes", "runs_fsq", "memcheck", name).replace(os.sep, "/"); os.makedirs(md, exist_ok=True); t0 = time.time()
    cmd = [sys.executable, "-u", "-m", a.target] + rest + ["--sched_scale", str(a.mc_scale), "--stop_after", str(a.mc_stop)] + \
          (["--grow_force_smoke"] if a.mc_force else []) + ["--device", a.device, "--out", md]
    with open(os.path.join(md, "memgate.log"), "w") as lg:
        rc = subprocess.call(cmd, stdout=lg, stderr=subprocess.STDOUT)
    mem, lmax, width = [], 0, None
    if os.path.exists(os.path.join(md, "log.jsonl")):
        for line in open(os.path.join(md, "log.jsonl")):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "mem_gib" in r:
                mem.append(r["mem_gib"]); lmax = max(lmax, r.get("L_max", 0))
            if "grow" in r:
                width = r["grow"].get("to")
    want = max([int(x) for x in (rest[rest.index("--grow2") + 1],)] if "--grow2" in rest else [0]) or None
    stopped = "STOP_AFTER" in open(os.path.join(md, "memgate.log")).read(); peak = max(mem) if mem else None
    res = {"run": name, "rc": rc, "stop_after_reached": stopped, "peak_gib": peak, "L_max_reached": lmax, "width_reached": width, "width_wanted": want,
           "limit_gib": a.limit_gib, "sec": round(time.time() - t0), "cmd": " ".join(cmd[1:])}
    res["pass"] = bool(rc == 0 and stopped and peak is not None and lmax >= 32 and peak <= a.limit_gib and (want is None or width == want))
    json.dump(res, open(os.path.join(md, "memgate.json"), "w", newline="\n"), indent=1)
    if os.path.isdir("logs"):
        json.dump(res, open(os.path.join("logs", "memgate_%s.json" % name), "w", newline="\n"), indent=1)
    qlog_line("MEMGATE %s %s peak %s GiB at L %s width %s (limit %s)" % (name, "PASS" if res["pass"] else "FAIL", peak, lmax, width, a.limit_gib))
    print("MEMGATE", json.dumps(res), flush=True)
    if not res["pass"]:
        sys.exit(3)
    sys.exit(subprocess.call([sys.executable, "-u", "-m", a.target] + rest + ["--device", a.device, "--out", a.out]))


if __name__ == "__main__":
    main()
