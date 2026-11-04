"""§270 FULL-PERM gate G5 SMOKE AT FULL DATA SCALE (FP_v1 §7; CPU, local) -> scratch/gate_s272/g5_smoke.json.
(a) the fp_s1 trainer line (of_s1's flags + --perm_scope all; the init models dropped: the local box's whole-file reads are unreliable, and data does not read them),
    real sets (EXCL n as the host), --stop_at 3: rc 9, losses finite, args.json perm_scope 'all'.
(b) part_full_perm on of_s1/model.pt (a streaming nobuf copy, majority-of-3 sha == of_s1/host_shas' model sha when listed) at d 2, all 1,000 items, through
    inject_aw_eval.load + the part (ds=(2,)): the by_d['2'] keys score_fp reads (acc, acc_k2, n, changed, by_flag) exist, n 1,000, values in [0, 1].
    The scratch copy and the .pt are deleted after (scratch-pt-cleanup).
    python -B scratch/s272_build/g5_smoke.py [a] [b]"""
import json
import math
import os
import shutil
import subprocess
import sys

NEW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")).replace("\\", "/"); MAIN = "D:/Latent"
sys.path.insert(0, NEW); sys.path.insert(0, NEW + "/scratch"); sys.path.insert(0, MAIN + "/scratch/ram_scan"); os.chdir(NEW)
OUT = NEW + "/scratch/gate_s272"; TMP = MAIN + "/scratch/_tmp/s272g5"; os.makedirs(OUT, exist_ok=True)


def part_a():
    av = sys.argv; sys.argv = [av[0]]                                      # g1_offbit reads sys.argv[1] at import
    import g1_offbit as G1
    sys.argv = av; d = TMP + "/a"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    fl = G1.flags(1) + ["--perm_scope", "all", "--ckpt_every", "1000", "--stop_at", "3", "--log_every", "1", "--eval_parts", "acc,test_perm,insert_ho,swap_kd,lp5h,stab,full_perm"]
    with open(d + "/stdout.log", "w") as f_:
        rc = subprocess.run([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", "cpu", "--out", d], cwd=NEW, stdout=f_, stderr=subprocess.STDOUT).returncode
    recs = [json.loads(l_) for l_ in open(d + "/log.jsonl")] if os.path.exists(d + "/log.jsonl") else []
    losses = [r["loss"] for r in recs if r.get("loss") is not None]; a = json.load(open(d + "/args.json")) if os.path.exists(d + "/args.json") else {}
    res = {"rc": rc, "n_loss": len(losses), "finite": all(math.isfinite(x) for x in losses), "perm_scope": a.get("perm_scope"), "excl_n": a.get("excl_n")}
    for f in os.listdir(d):
        if f.endswith(".pt"):
            os.remove(os.path.join(d, f))
    res["pass"] = rc == 9 and len(losses) >= 3 and res["finite"] and res["perm_scope"] == "all"; return res


def part_b():
    from nobuf_sha import sha_nobuf
    import boxes.fp_parts as FPP
    import boxes.inject_aw_eval as EV
    from boxes import inject_ow as IO
    d = TMP + "/b"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d); src = MAIN + "/boxes/runs_fsq/of_s1"
    subprocess.check_call([sys.executable, MAIN + "/scratch/ram_scan/nobuf_copy.py", src + "/model.pt", d + "/model.pt", sha_nobuf(src + "/model.pt")[:16]])
    shutil.copy(src + "/args.json", d + "/args.json")
    a, m, g = EV.load(d, "cpu"); sets = IO.build_sets(mul_s_p=EV.law(a))
    o = FPP.part_full_perm(m, g, sets, a, "cpu", 250, ds=(2,)); r = o["by_d"]["2"]
    os.remove(d + "/model.pt")
    vals = [r["acc"], r["acc_k2"], r["changed"]] + [v["acc"] for f in r["by_flag"].values() for v in f.values()]
    res = {"keys": sorted(r), "n": r["n"], "by_flag_keys": sorted(r["by_flag"]), "in_unit": all(0 <= v <= 1 for v in vals), "items_n": len(o["items"]["2"]["sig"])}
    res["pass"] = set(r) >= {"acc", "acc_k2", "n", "changed", "by_flag"} and r["n"] == 1000 and res["in_unit"] and res["items_n"] == 1000; return res


if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b"]; out = {}
    for w in which:
        out[w] = {"a": part_a, "b": part_b}[w](); print(w, json.dumps(out[w]), flush=True)
    prev = json.load(open(OUT + "/g5_smoke.json")) if os.path.exists(OUT + "/g5_smoke.json") else {}
    prev.update(out); prev["pass"] = all(prev.get(k, {}).get("pass") for k in ("a", "b"))
    json.dump(prev, open(OUT + "/g5_smoke.json", "w", newline="\n"), indent=1); sys.exit(0 if all(v["pass"] for v in out.values()) else 1)
