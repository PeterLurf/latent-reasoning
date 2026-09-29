"""One-shot status: training progress, log freshness, detached job tails, GPU state."""
import glob
import json
import os
import subprocess
import time

now = time.time()
for f in sorted(glob.glob("runs/*/log.jsonl")):
    age = now - os.path.getmtime(f)
    if age > 3 * 3600:
        continue
    lines = open(f).read().splitlines()
    ev = [json.loads(l) for l in lines if '"eval"' in l]
    last = json.loads(lines[-1]) if lines else {}
    cur = [json.loads(l) for l in lines if "curriculum_advance_to" in l]
    e = ev[-1] if ev else None
    s = f"{f.split(os.sep)[1] if os.sep in f else f:24s} step {last.get('step', '?'):>6} age {age:5.0f}s"
    if e:
        s += f"  eval@{e['step']} delayed {e['eval']['delayed']['bal_acc']:.3f} upfront {e['eval']['upfront']['bal_acc']:.3f}"
    if cur:
        s += f"  curr->{cur[-1]['curriculum_advance_to']}"
    print(s)
for f in sorted(glob.glob("logs/*.err")):
    age = now - os.path.getmtime(f)
    if age > 3 * 3600:
        continue
    tail = open(f, errors="replace").read().strip().splitlines()[-2:]
    out = f.replace(".err", ".out")
    otail = open(out, errors="replace").read().strip().splitlines()[-1:] if os.path.exists(out) else []
    print(f"{f:28s} out: {otail[0][:110] if otail else ''}")
    if tail and ("Error" in tail[-1] or "Traceback" in "".join(tail)):
        print("   ERR:", tail[-1][:160])
print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
