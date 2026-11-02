"""§258 B BRIDGE (LB2_v1 §2): the init = an unbuffered-read copy of aw_plaw2_s1/model.pt (RCB's init; expected sha256 prefix 484dab1eab9e2c12, committed in gate_s251.RCB
before the RAM faults). A read is written only if its sha256 matches; the copy is re-read unbuffered 3x -> scratch/gate_s260/bridge_init.json.
    python -B scratch/s260_build/mk_bridge_init.py"""
import hashlib
import json
import os
import sys

ROOT = "D:/Latent"; os.chdir(ROOT); sys.path.insert(0, ROOT + "/scratch/ram_scan")
from nobuf_sha import sha_nobuf  # noqa: E402

src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
     "import ctypes.wintypes as W\n" + src[i:j], g)
WANT = "484dab1eab9e2c12"; SRC = "boxes/runs_fsq/aw_plaw2_s1/model.pt"; DST = "boxes/runs_fsq/aw_plaw2_s1_init/model.pt"; o = {"want": WANT}
for att in range(8):
    b = g["nobuf_bytes"](SRC); sha = hashlib.sha256(b).hexdigest(); o["attempts"] = att + 1
    if sha.startswith(WANT):
        break
    del b
assert sha.startswith(WANT), ("no matching read in 8", sha)
os.makedirs(os.path.dirname(DST), exist_ok=True)
with open(DST, "wb") as f:
    f.write(b)
o["sha256"] = sha; o["copy_reads"] = [sha_nobuf(os.path.abspath(DST)) for _ in range(3)]; o["ok"] = all(x == sha for x in o["copy_reads"])
os.makedirs("scratch/gate_s260", exist_ok=True); json.dump(o, open("scratch/gate_s260/bridge_init.json", "w", newline="\n"), indent=1); print(o["ok"], o["attempts"], sha[:16])
