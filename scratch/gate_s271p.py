"""§265 DENSE-DISTRACT pilot per-host gate (Audit 11-04 on b417's smoke_dd OOM: --micro 4 pinned in all pilot cells; RTX 3090). Queue line (the runner appends
--device <d> --out <dir>; <dir> = boxes/runs_fsq/<run>_gates):
  <dir> scratch.gate_s271p --mem boxes/runs_fsq/<run>_mem/mem_check.json -- <the cell's exact trainer flags>
  smoke  boxes.inject_aw <flags> --ckpt_every 0 --log_every 1 --stop_at 20 (the cell's own micro / nmax): rc 9, 20 finite losses, the stop bin's distract_n has some n > 2
         when the flags carry --distract_nmax > 2 (the nmax draw is live).
  mem    the same host's <run>_mem/mem_check.json (the trainer's --mem_check: the longest d 4 items with the nmax-insert shape at bs / micro + AdamW states; the eval probe):
         peak_gib_train <= 22.5 and peak_gib_eval <= 22.5 (1 GiB headroom, the §251 auto_micro lesson) and its micro == the flags' --micro.
-> <dir>/smoke.json, <dir>/mem.json, and <dir>/gates_pass.json only if both pass (the TRAIN line --require's it). Exit 0 iff pass."""
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); LIM = 22.5


def main():
    argv = sys.argv[1:]; i = argv.index("--"); own, fl = argv[:i], argv[i + 1:]
    dev = fl[fl.index("--device") + 1]; out = os.path.join(ROOT, fl[fl.index("--out") + 1])
    for f in ("--device", "--out"):
        k = fl.index(f); del fl[k:k + 2]
    memp = os.path.join(ROOT, own[own.index("--mem") + 1]); os.makedirs(out, exist_ok=True)
    micro = int(fl[fl.index("--micro") + 1]); nmax = int(fl[fl.index("--distract_nmax") + 1]) if "--distract_nmax" in fl else 2
    d = os.path.join(out, "smoke"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    with open(os.path.join(d, "stdout.log"), "w") as f_:
        rc = subprocess.run([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--ckpt_every", "0", "--log_every", "1", "--stop_at", "20", "--device", dev, "--out", d],
                            cwd=ROOT, stdout=f_, stderr=subprocess.STDOUT).returncode
    rs = [json.loads(l_) for l_ in open(os.path.join(d, "log.jsonl"))] if os.path.exists(os.path.join(d, "log.jsonl")) else []
    ls = [r["loss"] for r in rs if isinstance(r.get("step"), int) and r.get("loss") is not None]; br = [r for r in rs if "bin" in r]
    dn = br[-1].get("distract_n", {}) if br else {}
    sm = {"rc": rc, "n_loss": len(ls), "finite": bool(ls) and all(math.isfinite(x) for x in ls), "distract_n": dn, "micro": micro, "nmax": nmax}
    sm["pass"] = rc == 9 and sm["finite"] and len(ls) >= 20 and (nmax <= 2 or any(int(k) > 2 for k in dn))
    for f in os.listdir(d):
        if f.endswith(".pt"):
            os.remove(os.path.join(d, f))
    mc = json.load(open(memp)) if os.path.exists(memp) else {}
    me = {"path": own[own.index("--mem") + 1], "peak_gib_train": mc.get("peak_gib_train"), "peak_gib_eval": mc.get("peak_gib_eval"), "micro": mc.get("micro"), "limit_gib": LIM}
    me["pass"] = bool(mc) and mc["peak_gib_train"] <= LIM and mc["peak_gib_eval"] <= LIM and mc.get("micro") == micro
    json.dump(sm, open(os.path.join(out, "smoke.json"), "w"), indent=1); json.dump(me, open(os.path.join(out, "mem.json"), "w"), indent=1)
    ok = sm["pass"] and me["pass"]
    if ok:
        json.dump({"pass": True, "smoke": sm, "mem": me}, open(os.path.join(out, "gates_pass.json"), "w"), indent=1)
    print("GATE_S271P", json.dumps({"smoke": sm, "mem": me, "pass": ok}), flush=True)
    if not ok:                                                              # Note 265-3 / Audit: a config-deterministic failure (every 3090 fails it): stop, never BAD-host retry
        print("GATE-FAIL config (smoke %s, mem %s): no train, no redeploy; tell Theory (next: micro 8)" % (sm["pass"], me["pass"]), flush=True)
    sys.exit(0 if ok else 7)


if __name__ == "__main__":
    main()
