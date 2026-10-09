"""THEORY note 177-6, on the b240 host: the (d) cell continues from the gate run's own 12k checkpoint instead of losing it to the drain
(drain_gen_partial excludes a finished run's ckpt_* and destroys the host). Started by hand next to b240's queue as
    nohup python -u -m scratch.s177d_gate_resume > logs/s177d_gate.out 2>&1 &
it is itself a `python -u -m` process, so drain_gen_partial reads the host as NOT_FINISHED (PROCS > 0) until it exits - no gap.
  1. wait for "evaluated boxes/runs_fsq/proto177_fadetidx12k_s1" + QUEUE_DONE in logs/queue_gen_b240.out
  2. gate (note 177-6): eval_12k.json (step 12000, p 0) full 4x4 exact match acc >= 0.5 -> PASS, else BELOW (exit; the ckpt may go; no 'FAIL' word - the watcher greps it)
  3. PASS: wait for scratch/s177d.go (Theory's go, written by Experiments over ssh) up to 3 h; scratch/s177d.nogo or a timeout -> exit
  4. go: inject169 --resume <gate run>/ckpt_last.pt --resumed_from proto177_fadetidx12k_s1@12000 -> boxes/runs_fsq/s177_fadetidx_s1, the gate
     run's own args (no --stop_at) and micro, HOSTPROBE_SPS from this host's probe; queue-log lines "start" / "evaluated ... rc" in the worker's format
  QUEUE_DONE is appended again on every exit path, so the drain then fetches both runs and destroys the host."""
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.environ.get("S177D_ROOT", "/workspace/Latent"); os.chdir(ROOT)   # env overrides: the local CPU test only
STEP = int(os.environ.get("S177D_STEP", "12000")); DEV = os.environ.get("S177D_DEVICE", "cuda:0"); EXTRA = os.environ.get("S177D_EXTRA", "").split()
GATE = "boxes/runs_fsq/proto177_fadetidx12k_s1"; CELL = "boxes/runs_fsq/s177_fadetidx_s1"; LOG = "logs/queue_gen_b240.out"
GO, NOGO = "scratch/s177d.go", "scratch/s177d.nogo"


def qlog(msg):
    with open(LOG, "a") as f_:
        f_.write(time.strftime("%H:%M:%S") + " " + msg + "\n")


def lines():
    return open(LOG).read().splitlines() if os.path.exists(LOG) else []


def main():
    try:
        while not (any((" evaluated " + GATE) in l_ for l_ in lines()) and lines() and lines()[-1].endswith("QUEUE_DONE")):
            time.sleep(20)
        f12 = os.path.join(GATE, "eval_%dk.json" % (STEP // 1000)); e = json.load(open(f12)) if os.path.exists(f12) else {}
        ok = e.get("step") == STEP and e.get("p") == 0.0 and e.get("acc") is not None and e["acc"] >= 0.5
        qlog("S177D_GATE %s acc_scaffold_12k %s acc_by_column %s (note 177-6: >= 0.5; %s)" % ("PASS" if ok else "BELOW", e.get("acc"), e.get("acc_by_column"), f12))
        if not ok:
            return
        t0 = time.time()
        while not (os.path.exists(GO) or os.path.exists(NOGO)) and time.time() - t0 < 3 * 3600:
            time.sleep(20)
        if not os.path.exists(GO):
            qlog("S177D no go (%s)" % ("nogo file" if os.path.exists(NOGO) else "3 h timeout")); return
        a0 = json.load(open(os.path.join(GATE, "args.json"))); ev = json.load(open(os.path.join(GATE, "eval.json")))
        micro = int(ev.get("micro") or a0.get("micro") or 1)
        sps = ([m_.group(1) for m_ in (re.search(r"HOSTPROBE ([0-9.]+) s/step", l_) for l_ in lines()) if m_] or [""])[0]
        cmd = [sys.executable, "-u", "-m", "boxes.inject169", "--think_k", "4n", "--fade", a0["fade"], "--pass_index", a0["pass_index"], "--fast_add",
               "--seed", str(a0["seed"]), "--steps", str(a0["steps"]), "--ckpt_every", "1000", "--micro", str(micro), "--bench", a0["bench"],
               "--resume", os.path.join(GATE, "ckpt_last.pt"), "--resumed_from", "proto177_fadetidx12k_s1@12000", "--device", DEV, "--out", CELL] + EXTRA
        qlog("start " + CELL); env = dict(os.environ, HOSTPROBE_SPS=sps)
        rc = subprocess.call(cmd, stdout=open("logs/gen_s177_fadetidx_s1.log", "w"), stderr=subprocess.STDOUT, env=env)
        ev_ = os.path.join(CELL, "eval.json"); qlog("evaluated %s rc %s%s" % (CELL, rc, "" if os.path.exists(ev_) and os.path.getsize(ev_) > 0 else " no-eval"))
    finally:
        qlog("QUEUE_DONE")


if __name__ == "__main__":
    main()
