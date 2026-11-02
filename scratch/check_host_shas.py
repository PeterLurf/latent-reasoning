"""Audit (§257 G-D note 1): every file a host listed in <dir>/host_shas.json, re-read locally UNBUFFERED (majority of 3; this box's single reads can flip) == the host's sha256.
    python -B scratch/check_host_shas.py <run dir> ...   -> one line per dir: HOST_SHAS OK n | MISMATCH <files> | NO host_shas.json; exit 1 on any mismatch"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ram_scan"))
from nobuf_sha import sha_nobuf  # noqa: E402

bad = 0
for d in sys.argv[1:]:
    p = os.path.join(d, "host_shas.json")
    if not os.path.exists(p):
        print(d, "NO host_shas.json"); bad = 1; continue
    hs = json.load(open(p)); mm = []
    for f, want in hs.items():
        q = os.path.abspath(os.path.join(d, f))
        if not os.path.exists(q):
            mm.append(f + "(absent)"); continue
        r = [sha_nobuf(q) for _ in range(3)]
        if sum(x == want for x in r) < 2:
            mm.append(f)
    print(d, "HOST_SHAS OK %d" % len(hs) if not mm else "MISMATCH " + " ".join(mm)); bad |= bool(mm)
sys.exit(1 if bad else 0)
