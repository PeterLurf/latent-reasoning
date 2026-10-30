"""§250 G1 (FD_v1 §2): every existing set sha, snapshotted before / after the nested_words FRAMES append. -> scratch/gate_s250/G1_<tag>.json
build_sets(mul_s_p 0 / 1 / 2): every key of shas (test2, transfer, pairs2, floors2, s226_test, s227_decode, set_law), excl size; NW.test_set / NW.pair_candidates and
OW.test_set / OW.pair_candidates at their defaults (set_sha over texts; pairs: rec then don texts); the module file shas (informational).
    python -B scratch/s250_build/sha_snapshot.py before|after"""
import hashlib
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT); os.environ.setdefault("OMP_NUM_THREADS", "3")
import boxes.inject_ow as IO  # noqa: E402
import boxes.nested_words as NW  # noqa: E402
import boxes.ow_words as OW  # noqa: E402

tag = sys.argv[1]; o = {"build_sets": {}}
for law in (0, 1, 2):
    s = IO.build_sets(mul_s_p=law); o["build_sets"][str(law)] = {"shas": s["shas"], "excl_n": len(s["excl"])}
o["NW_test_set"] = NW.set_sha([it for d, v in NW.test_set().items() for it in v])
p = NW.pair_candidates(); o["NW_pair_candidates"] = NW.set_sha([x for d in p for pr in p[d] for x in pr])
o["OW_test_set"] = OW.set_sha([it for d, v in OW.test_set().items() for it in v])
p = OW.pair_candidates(); o["OW_pair_candidates"] = OW.set_sha([x for d in p for pr in p[d] for x in pr])
o["files"] = {f: hashlib.sha256(open("boxes/%s.py" % f, "rb").read()).hexdigest() for f in ("nested_words", "ow_words", "inject_ow", "inject_aw")}
os.makedirs("scratch/gate_s250", exist_ok=True); json.dump(o, open("scratch/gate_s250/G1_%s.json" % tag, "w", newline="\n"), indent=1); print("snapshot", tag, "written")
