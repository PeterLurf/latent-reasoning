"""RAM bit-flip triage (Audit 11-01): every eval_*.json / log.jsonl / args.json / *.json of the §251-§255 run dirs and the local result dirs, hashed through the normal (page
cache) read and through an unbuffered read (FILE_FLAG_NO_BUFFERING, scratchpad nobuf_sha.sha_nobuf). -> scratch/ram_scan/json_scan.json {n, mismatches [{path, cached, disk}]}.
    python -B scratch/ram_scan_json.py"""
import glob
import hashlib
import json
import os
import sys

ROOT = "D:/Latent"; os.chdir(ROOT)
sys.path.insert(0, ROOT + "/scratch/ram_scan")
from nobuf_sha import sha_nobuf  # noqa: E402

RUNS = ["aex_s1", "aex_s2", "aex0_s1", "aex0_s2", "aex_s3", "aex_s4", "rcb_plaw4_s1", "rcb_plaw4_s2", "rcb_plaw4_s1_re", "rcb_plaw4_s2_re", "aw_plaw2_s1_reg_s251", "pac_acue_s1",
        "pac_ctrl_s1", "pilot_np_s1", "pilot_gp_s1", "p2p_s1", "p2p_s2", "p2p_old_s1", "p2pc_s1", "lbr_s1", "lbr_s2", "lb_step0", "s252_askperm_s1", "pilot_cur_s1"]
DIRS = ["scratch/s251cc_s0", "scratch/s251ns_s0", "scratch/s252_scale0", "scratch/s252_askperm", "scratch/s255_insert", "scratch/s251_swapkd_check"]
files = []
for r in RUNS:
    files += [f for f in glob.glob("boxes/runs_fsq/%s/*" % r) if f.endswith((".json", ".jsonl"))]
for d in DIRS:
    files += glob.glob(d + "/*.json")
mm = []
for f in sorted(set(files)):
    c = hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]; d = sha_nobuf(os.path.abspath(f))[:16]
    if c != d:
        mm.append({"path": f, "cached": c, "disk": d})
os.makedirs("scratch/ram_scan", exist_ok=True)
json.dump({"n": len(set(files)), "mismatches": mm}, open("scratch/ram_scan/json_scan.json", "w", newline="\n"), indent=1)
print("scanned", len(set(files)), "mismatches", len(mm)); [print(m) for m in mm[:20]]
