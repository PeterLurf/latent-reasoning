"""THEORY §160 R-ROWS on-host driver (Audit a283c617 approval: launch once the on-host gate160 and memcheck pass):
  1. scratch/gate160.py on this GPU (bf16 where supported; g0 is the CPU check against HEAD, skipped here) -> <out>/gate160/gate160_<mode>.json
  2. boxes.multrow --mem_check with the run's own arguments -> <out>/memcheck/mem_check.json (peak <= 21 GiB)
  3. the run: python -m boxes.multrow <args> --device D --out OUT. A failed gate or memcheck exits 3 with no training (logged to the queue log).
    python -m scratch.restart160 <boxes.multrow args incl. --mem_write> --device D --out OUT"""
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
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True); ap.add_argument("--mem_write", required=True)
    a, extra = ap.parse_known_args(); os.chdir(ROOT); os.makedirs(a.out, exist_ok=True); gj = os.path.join(a.out, "gate160_pre.json")
    if not os.path.exists(gj):
        gd = os.path.join(a.out, "gate160").replace(os.sep, "/"); md = os.path.join(a.out, "memcheck").replace(os.sep, "/")
        rc1 = subprocess.call([sys.executable, "-u", "scratch/gate160.py", a.device, a.mem_write, gd], stdout=open(os.path.join(a.out, "gate160.log"), "w"), stderr=subprocess.STDOUT)
        rc2 = subprocess.call([sys.executable, "-u", "-m", "boxes.multrow"] + extra + ["--mem_write", a.mem_write, "--mem_check", "--device", a.device, "--out", md],
                              stdout=open(os.path.join(a.out, "memcheck.log"), "w"), stderr=subprocess.STDOUT)
        mc = os.path.join(md, "mem_check.json"); gf = os.path.join(gd, "gate160_%s.json" % a.mem_write)
        res_ = {"gate_rc": rc1, "memcheck_rc": rc2, "gate": json.load(open(gf)) if os.path.exists(gf) else None, "memcheck": json.load(open(mc)) if os.path.exists(mc) else None}
        res_["pass"] = rc1 == 0 and rc2 == 0 and res_["memcheck"] is not None and res_["memcheck"]["peak_gib"] <= 21.0
        json.dump(res_, open(gj, "w", newline="\n"), indent=1); qlog_line("GATE160 %s %s" % (a.mem_write, "PASS" if res_["pass"] else "FAIL rc %s / %s" % (rc1, rc2)))
    if not json.load(open(gj))["pass"]:
        sys.exit(3)
    sys.exit(subprocess.call([sys.executable, "-u", "-m", "boxes.multrow"] + extra + ["--mem_write", a.mem_write, "--device", a.device, "--out", a.out]))


if __name__ == "__main__":
    main()
