"""§216 G-S1 EXCLUSION, post hoc on the built pool file (SD2_v1 §2.2 item 8 / §7 at e9ade2ec): pool ∩ (test_set(1000, n_max 24) ∪ SERIAL ∪ 213-37 n 9..24
∪ GATE-1 ∪ every s216 gate (incl. this lineage's) ∪ every other s216 pool ∪ the 440k round-1 pool) = ∅ by exact rendered syms, rebuilt from scratch with
B1's own exclusion_sets(); plus: the file's sha256 == pool_meta.json's, record keys == REC_KEYS, n == #operators(syms), per-n counts == pool_meta per_n,
no within-pool duplicate, and pool_meta's per-n draws / collisions / duplicates logged. CPU only.
    python scratch/s216_gs1.py --lineage s1 --round 2"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, "D:/Latent"); sys.path.insert(0, "D:/Latent/scratch"); os.chdir("D:/Latent")
import boxes.selfdepth as SD  # noqa: E402
from s216_build_pool import CFG, exclusion_sets  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--lineage", required=True); ap.add_argument("--round", type=int, required=True); a = ap.parse_args()
cfg = CFG[(a.lineage, a.round)]; fp = os.path.join(cfg["pool_dir"], cfg["pool_file"]); meta = json.load(open(os.path.join(cfg["pool_dir"], "pool_meta.json")))
recs, _ = SD.load_pool(fp, os.path.join(cfg["pool_dir"], "pool_meta.json"))   # sha256 + key set + n == #operators asserted inside
S = exclusion_sets(cfg, with_own_gates=True); syms = [tuple(r["syms"]) for r in recs]
hits = {k: sum(s in v for s in syms) for k, v in S.items()}; cnt = Counter(r["n"] for r in recs); dup = len(syms) - len(set(syms))
o = {"pool": fp, "n_records": len(recs), "sha256": meta["sha256"], "excl_sizes": {k: len(v) for k, v in S.items()}, "hits": hits, "per_n": {str(n): c for n, c in sorted(cnt.items())},
     "per_n_meta": meta["per_n"], "within_pool_dups": dup, "draws_n": meta["draws_n"], "collide_n": meta["collide_n"], "dup_n": meta["dup_n"]}
o["pass"] = all(v == 0 for v in hits.values()) and dup == 0 and {str(n): c for n, c in cnt.items()} == {k: int(v) for k, v in meta["per_n"].items()}
os.makedirs("scratch/gate_s216", exist_ok=True)
json.dump(o, open("scratch/gate_s216/gs1_%s_r%d.json" % (a.lineage, a.round), "w", newline="\n"), indent=1); print("G-S1", "PASS" if o["pass"] else "FAIL", json.dumps(o)[:900])
