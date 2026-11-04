"""§270 gate G1b OFFBIT-265 (Audit 11-04, sequencing (b)): the §265 pilot's data batches are bitwise equal between the gated trainer (git d7e60d3d: boxes/inject_aw.py
blob bd1dcc6a, the blob b417's dd_gates ran on) and the merged §270 trainer, with the EXACT §265 pilot flags (s271_mkq_pilot.flags: bun's flags + --hold_inc 1.6
--hold_log; ddp = + --distract_nmax 8; ddc = CTRL, nmax default), seeds 1 / 2, steps 1..300; the inits dropped (data does not read them). Same instrumented-copy
method as g1_offbit (per-step sha of every item field except 'raw'). Also: inject_aw_eval.py d7e60d3d -> new touches only the full_perm registration (PARTS /
AW_ONLY / run dispatch). -> scratch/gate_s272/g1b_offbit265.json
    python -B scratch/s272_build/g1b_offbit265.py [N]"""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
av = sys.argv; sys.argv = [av[0]]
import g1_offbit as G1  # noqa: E402
sys.argv = av
import gate_s257 as G7  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300; REF = "d7e60d3d"; G1.N = N; TMP = G1.TMP + "b"; G1.TMP = TMP


def ref_tree():
    t = TMP + "/ref"
    if not os.path.isdir(t + "/boxes"):
        os.makedirs(t, exist_ok=True); tp = TMP + "/ref.tar"
        subprocess.check_call(["git", "archive", "-o", tp, REF, ":(glob)boxes/*.py"], cwd=G1.NEW); tarfile.open(tp).extractall(t)
    b = open(t + "/boxes/inject_aw.py", "rb").read(); assert hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest().startswith("bd1dcc6a"), "ref: the §265-gated blob"
    return t


def flags(k, nmax):
    fl = G7.dch_flags(k, "x", "x") + ["--hold_inc", "1.6", "--hold_log"] + (["--distract_nmax", str(nmax)] if nmax else [])
    for f in ("--init_model", "--init_model_sha", "--init_trunk", "--init_sha"):
        if f in fl:
            i = fl.index(f); del fl[i:i + 2]
    return fl


def run(tree, k, nmax, tag):
    """G1.run with the §265 pilot flags (G1.run reads G1.flags; not swapped, so jobs can run in parallel)."""
    d = G1.G8.fresh(TMP + "/" + tag); env = {**G1.G8.ENV, "G1_N": str(N), "G1_OUT": d + "/hashes.json"}
    with open(d + "/stdout.log", "w") as f_:
        rc = subprocess.run([sys.executable, "-B", "-m", "boxes._inject_aw_g1_s272"] + flags(k, nmax) + ["--device", "cpu", "--out", d], cwd=tree, env=env,
                            stdout=f_, stderr=subprocess.STDOUT).returncode
    h = json.load(open(d + "/hashes.json")) if os.path.exists(d + "/hashes.json") else []
    return tag, rc, h


def eval_diff():
    """the changed lines inject_aw_eval.py d7e60d3d -> new; ok iff every added line names full_perm and exactly the 3 edited lines (PARTS, AW_ONLY, the attn_fade dispatch line) are removed."""
    import difflib
    old = subprocess.check_output(["git", "show", REF + ":boxes/inject_aw_eval.py"], cwd=G1.NEW).decode("utf-8").splitlines()
    new = open(G1.NEW + "/boxes/inject_aw_eval.py", encoding="utf-8").read().splitlines()
    d = [ln for ln in difflib.unified_diff(old, new, lineterm="", n=0) if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    rm = [ln for ln in d if ln.startswith("-")]; add = [ln for ln in d if ln.startswith("+")]
    ok = len(rm) == 3 and rm[0].startswith("-PARTS = (") and rm[1].startswith("-AW_ONLY = (") and '"attn_fade"' in rm[2] and all("full_perm" in ln or "attn_fade" in ln for ln in add)
    return {"removed": [x[:120] for x in rm], "added": [x[:160] for x in add], "only_full_perm": ok}


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True); rt = ref_tree(); G1.junction(rt); G1.instrument(rt); G1.instrument(G1.NEW)
    jobs = [(t, k, nm, "%s_%s_s%d" % (lab, "ddp" if nm else "ddc", k)) for nm in (8, None) for k in (1, 2) for t, lab in ((rt, "ref"), (G1.NEW, "new"))]
    with ThreadPoolExecutor(3) as ex:                                       # <= 3 local CPU jobs
        res = {t: (rc, h) for t, rc, h in ex.map(lambda j: run(*j), jobs)}
    out = {"N": N, "ref_commit": REF, "runs": {t: {"rc": rc, "n": len(h), "sha": hashlib.sha256("".join(h).encode()).hexdigest()} for t, (rc, h) in res.items()}}
    ok = True
    for cfg in ("ddp", "ddc"):
        for k in (1, 2):
            r, n_ = res["ref_%s_s%d" % (cfg, k)], res["new_%s_s%d" % (cfg, k)]
            mis = next((i + 1 for i, (x, y) in enumerate(zip(r[1], n_[1])) if x != y), None); eq = r[0] == n_[0] == 0 and len(r[1]) == len(n_[1]) == N and mis is None
            out["runs"]["new_%s_s%d" % (cfg, k)].update({"equal_ref": eq, "first_mismatch_step": mis}); ok &= eq
    out["eval_diff"] = eval_diff(); new_blob = open(G1.NEW + "/boxes/inject_aw.py", "rb").read()
    out["new_inject_aw_blob"] = hashlib.sha1(b"blob %d\0" % len(new_blob) + new_blob).hexdigest()
    out["pass"] = ok and out["eval_diff"]["only_full_perm"]
    json.dump(out, open(G1.OUT + "/g1b_offbit265.json", "w", newline="\n"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "runs"} | {"eq": {t: v.get("equal_ref") for t, v in out["runs"].items()}}))
    sys.exit(0 if out["pass"] else 1)
