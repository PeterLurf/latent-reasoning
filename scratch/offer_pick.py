"""fill_generic's offer choice as a helper (2026-10-14, §199 dual filler): RTX 3090 offers with driver == argv[1] (or any when '-') and price <=
argv[2], minus excluded / live IPs, ranked (preferred IPs, driver 595.71, price). Prints "<offer id> <ip>" or nothing.
    python scratch/offer_pick.py <driver|-> <dph max>"""
import json
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
drv, dph = sys.argv[1], sys.argv[2]
q = "gpu_name=RTX_3090 num_gpus=1 reliability>0.97 dph<=%s disk_space>=40 rentable=true inet_down>=300 cuda_vers>=12.4" % dph
r = subprocess.run(["bash", "C:/Users/peter/.local/vastai/vast.sh", "search", "offers", q, "-o", "dph", "--raw", "--limit", "150"], capture_output=True, text=True)
try:
    d = json.loads(r.stdout)
except Exception:
    sys.stderr.write("offer_pick: search failed: %s\n" % (r.stdout[-300:] + r.stderr[-300:])); sys.exit(0)
ex = {l.split()[0] for l in open("scratch/vast_exclude_ips.txt") if l.strip() and not l.startswith("#")}
used = {l.split()[0] for l in open("scratch/live_hosts.txt") if l.strip()}
pf = [l.strip() for l in open("scratch/vast_prefer_ips.txt") if l.strip() and not l.startswith("#")] if os.path.exists("scratch/vast_prefer_ips.txt") else []
rows = [(0 if any(str(x.get("public_ipaddr", "")).startswith(p) for p in pf) else 1, 0 if str(x.get("driver_version", "")).startswith("595.71") else 1, x["dph_total"], x["id"], x.get("public_ipaddr"))
        for x in d if (drv == "-" or str(x.get("driver_version", "")) == drv) and x.get("public_ipaddr") not in ex and x.get("public_ipaddr") not in used
        and "oid:%s" % x.get("id") not in ex and "mid:%s" % x.get("machine_id") not in ex]
if rows:
    b = sorted(rows)[0]; print(b[3], b[4])
