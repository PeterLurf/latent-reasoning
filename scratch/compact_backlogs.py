"""Compact the local backlogs (2026-10-12: c1_backlog_worker2 spawns awk / cut / md5sum / python per line on every 2-min poll; ~500 lines x 3
workers kept ~40 % CPU busy and slowed every MSYS spawn, incl. gen_deploy's auto_ship -> stale offers). Lines whose done file exists (non-empty)
or whose claim lock exists (c1/runs/locks/bl_<md5 of the line + newline, first 12 hex> = already started once, never retried) move to
scratch/c1_backlog_<card>.archive.txt with comments; only runnable lines stay. Atomic replace, retried while a worker holds the file open.
    python scratch/compact_backlogs.py [--dry]"""
import hashlib
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
dry = "--dry" in sys.argv
for card in ("2070", "3070", "4060", "fp32"):
    p = "scratch/c1_backlog_%s.txt" % card
    if not os.path.exists(p):
        continue
    raw = open(p, encoding="utf-8", newline="").read(); L = raw.split("\n")
    keep, arch = [], []
    for l in L:
        if not l.strip():
            continue
        if l.startswith("#"):
            arch.append(l); continue
        f = l.split()
        done_f = f[1] if len(f) > 1 else ""
        h = hashlib.md5((l + "\n").encode("utf-8")).hexdigest()[:12]
        if (done_f and os.path.exists(done_f) and os.path.getsize(done_f) > 0) or os.path.isdir("c1/runs/locks/bl_" + h):
            arch.append(l)
        else:
            keep.append(l)
    print(card, "lines", len([l for l in L if l.strip() and not l.startswith("#")]), "keep", len(keep), "archive", len(arch))
    for l in keep:
        print("   keep:", l[:150])
    if dry:
        continue
    stamp = "# --- archived %s by compact_backlogs.py (done or claimed) ---" % time.strftime("%Y-%m-%d %H:%M")
    with open(p.replace(".txt", ".archive.txt"), "a", encoding="utf-8", newline="\n") as fa:
        fa.write(stamp + "\n" + "\n".join(arch) + "\n")
    tmp = p + ".new"; open(tmp, "w", encoding="utf-8", newline="\n").write("# compacted %s: done / claimed lines in %s\n" % (time.strftime("%Y-%m-%d %H:%M"), os.path.basename(p.replace(".txt", ".archive.txt"))) + "".join(l + "\n" for l in keep))
    for i in range(120):
        try:
            os.replace(tmp, p); break
        except PermissionError:
            time.sleep(1)
    else:
        print("  could not replace", p, "(held open); left", tmp)
