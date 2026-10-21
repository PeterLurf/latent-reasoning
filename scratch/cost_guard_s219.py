"""§219 cost guard = scratch/cost_guard.py unchanged EXCEPT last_rec (same CLI, same GUARD_STOP / GUARD_MARKS / GUARD_MIRROR environment, same log).
Why: cost_guard.last_rec greps '"step": N, "loss": X, "sec": N' adjacent in log.jsonl; boxes/textself_ft.py writes "loss_last" between "loss" and "sec"
(A10), so the stock regex never matches a §219 record, the guard never projects and never stops (checked on the G-T4 real log.jsonl). Here the host's last
log.jsonl line holding "sec" is parsed as JSON (training records every --log_every; the resume / nonfinite lines carry no "sec" and are skipped).
    GUARD_STOP=1 GUARD_MARKS=500,1000,2000,4000,8000 python scratch/cost_guard_s219.py <tag> <cap> <steps> <n>:<run>:<high $>:<steps> [...]"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_guard as CG  # noqa: E402   (module-level argv parsing: identical CLI; its main() is the stock loop)


def parse_last(text):
    """the last JSON line with integer step and sec -> (step, sec), else None."""
    for ln in reversed([l for l in text.splitlines() if l.strip()]):
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if isinstance(d, dict) and isinstance(d.get("step"), int) and isinstance(d.get("sec"), (int, float)):
            return int(d["step"]), int(d["sec"])
    return None


def last_rec(h, p, cell):
    cmd = "grep '\"sec\"' /workspace/Latent/boxes/runs_fsq/%s/log.jsonl 2>/dev/null | tail -n 3" % cell
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-i", CG.KEY, "-p", p, "root@" + h, cmd], capture_output=True, text=True, timeout=60).stdout
        return parse_last(r)
    except Exception:                                                       # noqa: BLE001
        return None


CG.last_rec = last_rec

if __name__ == "__main__":
    CG.main()
