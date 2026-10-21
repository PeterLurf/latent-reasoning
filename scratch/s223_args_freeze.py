"""§223 frozen args with single-change diffs (PS2_v1 Addendum A item 3 / Addendum B item 3): the trainer's own --args_only (boxes/codi_ps.py main(),
CPU) for the four cells on their queue tokens, then the key-by-key diffs (excluding codi_ps.META and the derived 'cell' label):
    PS vs M1-only  must differ ONLY in {bottleneck}
    PS vs M2-only  must differ ONLY in {recode, lam}
    PS vs DIRECT   must differ ONLY in {direct, recode, lam, bottleneck} (the latent-loop flags; DIRECT has no D head and no mask)
-> scratch/s223_args_frozen_<cell>.json.draft (for Theory's freeze by sha) + scratch/gate_s223/args_diffs.json.   python -B scratch/s223_args_freeze.py"""
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import boxes.codi_ps as PS  # noqa: E402

CELLS = {"PS": ("s223_ps_s1", ["--recode", "--lam", "1", "--bottleneck"]), "M1-only": ("s223_m1_s1", ["--recode", "--lam", "1"]),
         "M2-only": ("s223_m2_s1", ["--bottleneck"]), "DIRECT": ("s223_direct_s1", ["--direct"])}
ALLOWED = {"M1-only": {"bottleneck"}, "M2-only": {"recode", "lam"}, "DIRECT": {"direct", "recode", "lam", "bottleneck"}}
IGN = set(PS.META) | {"cell"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    tmp = "scratch/_tmp/s223_args"; shutil.rmtree(tmp, ignore_errors=True); A = {}
    for cell, (run, flags) in CELLS.items():
        od = os.path.join(tmp, "boxes/runs_fsq", run); argv = flags + ["--args_only", "--device", "cpu", "--out", od]   # gen_vast_job appends --device / --out
        PS.main(argv); A[cell] = json.load(open(os.path.join(od, "args.json")))
        dst = "scratch/s223_args_frozen_%s.json.draft" % run.replace("s223_", "").replace("_s1", ""); shutil.copyfile(os.path.join(od, "args.json"), dst)
        print("DRAFT", cell, dst, sha(dst)[:16], flush=True)
    res = {"ignored_keys": sorted(IGN), "diffs": {}, "pass": True}
    for cell, allowed in ALLOWED.items():
        dif = {k: [A["PS"].get(k), A[cell].get(k)] for k in sorted(set(A["PS"]) | set(A[cell])) if k not in IGN and A["PS"].get(k) != A[cell].get(k)}
        ok = set(dif) == allowed; res["diffs"]["PS vs " + cell] = {"keys": dif, "allowed": sorted(allowed), "ok": ok}; res["pass"] &= ok
        print("PS vs", cell, sorted(dif), "OK" if ok else "BAD", flush=True)
    res["drafts"] = {c: {"file": "scratch/s223_args_frozen_%s.json.draft" % r.replace("s223_", "").replace("_s1", ""),
                         "sha256": sha("scratch/s223_args_frozen_%s.json.draft" % r.replace("s223_", "").replace("_s1", ""))} for c, (r, _) in CELLS.items()}
    os.makedirs("scratch/gate_s223", exist_ok=True); json.dump(res, open("scratch/gate_s223/args_diffs.json", "w", newline="\n"), indent=1)
    print("ARGS-FREEZE", "PASS" if res["pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
