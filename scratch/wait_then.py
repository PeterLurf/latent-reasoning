"""Queue wrapper (Audit 2026-10-06, §135 TSD hosts): wait for a file that is copied in host-to-host after the deploy (`vastai copy`),
verify its sha256, then run the module. gen_vast_job appends --device / --out, which are passed through.
    python -m scratch.wait_then --wait <file> --sha <sha256> [--timeout 5400] --module boxes.fsq120 <module args...>"""
import argparse
import hashlib
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--wait", required=True); ap.add_argument("--sha", required=True); ap.add_argument("--timeout", type=int, default=5400)
ap.add_argument("--module", required=True)
a, rest = ap.parse_known_args(); t0 = time.time(); last = -1
while True:
    if os.path.exists(a.wait):
        sz = os.path.getsize(a.wait)
        if sz == last and sz > 0:                                           # size stable over one poll: the copy has finished
            h = hashlib.sha256(open(a.wait, "rb").read()).hexdigest()
            if h == a.sha:
                print("WAIT_THEN: %s ok (%d bytes, sha verified) after %d s" % (a.wait, sz, time.time() - t0), flush=True); break
            print("WAIT_THEN: sha mismatch %s (%s), waiting for a re-copy" % (a.wait, h[:16]), flush=True)
        last = sz
    if time.time() - t0 > a.timeout:
        print("WAIT_THEN: timeout waiting for %s" % a.wait, flush=True); sys.exit(3)
    time.sleep(20)
sys.exit(subprocess.call([sys.executable, "-u", "-m", a.module] + rest, cwd=ROOT))
