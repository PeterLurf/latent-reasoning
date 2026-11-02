"""Audit 11-02 (b395 lost dis_s2: its '--model <path>=<sha>' init never shipped - auto_ship takes plain path tokens only): the path of every '--model' / '--init'
'<path>=<hex>' value in a queue's non-comment lines must also appear somewhere in the queue as a plain token (so auto_ship tars it). (--blob code paths ship as
module dependencies and are not checked here.) Exit 1 + the missing paths, else 0.
    python scratch/check_queue_ships.py scratch/gen_q_b<n>.txt"""
import re
import sys

lines = [ln.split() for ln in open(sys.argv[1], encoding="utf-8") if ln.strip() and not ln.startswith("#")]
plain = {t for ln in lines for t in ln}
need = {ln[i + 1].split("=")[0] for ln in lines for i, t in enumerate(ln[:-1]) if t in ("--model", "--init") and re.match(r"^[^=]+=[0-9a-f]{8,}$", ln[i + 1])}
miss = sorted(need - plain)
if miss:
    print("NOT SHIPPED (path=sha only):", " ".join(miss)); sys.exit(1)
print("ships OK (%d model / init paths)" % len(need))
