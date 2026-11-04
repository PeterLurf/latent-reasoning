"""§270 FULL-PERM gate G1 OFFBIT (FP_v1 §7; CPU, data only): the training batches (chain + ask items, every token field) of steps 1..N are bitwise equal between
the §263 module (git 7cfd033a: boxes/inject_aw.py sha256 4d185e6d...) and the new tree with --perm_scope omitted and --perm_scope op, seeds 1 / 2.
Both trees run a gate-only instrumented copy of boxes/inject_aw.py (boxes/_inject_aw_g1_s272.py): right after the step's batch is drawn it hashes every item's fields
except 'raw' (json, sorted keys) and skips the model step; it exits 0 after step N. The flags are of_s<k>'s (s270_mkq.flags) without the init models (data does not read
them; the local box's whole-file reads are unreliable), + --device cpu. -> scratch/gate_s272/g1_offbit.json (per-run sha + first mismatch step).
    python -B scratch/s272_build/g1_offbit.py [N]"""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

NEW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")).replace("\\", "/"); MAIN = "D:/Latent"
sys.path.insert(0, NEW + "/scratch"); os.chdir(NEW)
import gate_s228 as G8  # noqa: E402
import gate_s257 as G7  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300; REF = "7cfd033a"; TMP = NEW + "/scratch/_tmp/s272g1"; OUT = NEW + "/scratch/gate_s272"
ANCHOR = "        its = draw_items(d) + (ask_items(d) if NASK else []); assert len(its) == a.bs;"
HOOK = ("        _g1_hook(step, its)\n        continue\n")
HEAD = ('import json as _gj, hashlib as _gh, os as _go, sys as _gs\n_G1 = []\n'
        'def _g1_hook(step, its):\n'
        '    _G1.append(_gh.sha256(_gj.dumps([{k: v for k, v in it.items() if k != "raw"} for it in its], sort_keys=True, default=str).encode()).hexdigest())\n'
        '    if step >= int(_go.environ["G1_N"]):\n'
        '        _gj.dump(_G1, open(_go.environ["G1_OUT"], "w")); _gs.exit(0)\n')


def instrument(tree):
    src = open(tree + "/boxes/inject_aw.py", encoding="utf-8").read(); i = src.index(ANCHOR); j = src.index("\n", i) + 1
    assert src.count(ANCHOR) == 1; k = src.index("\nimport ") + 1
    open(tree + "/boxes/_inject_aw_g1_s272.py", "w", encoding="utf-8", newline="\n").write(src[:k] + HEAD + src[k:j] + HOOK + src[j:])


def junction(tree):
    """GPT-2 weights (gitignored prosqa/ckpt): link prosqa (absent) or prosqa/ckpt (a git tree's tracked prosqa/) to D:/Latent's."""
    q, src = (tree + "/prosqa", MAIN + "/prosqa") if not os.path.exists(tree + "/prosqa") else (tree + "/prosqa/ckpt", MAIN + "/prosqa/ckpt")
    if not os.path.exists(q):
        subprocess.check_call(["cmd", "/c", "mklink", "/J", q.replace("/", "\\"), src.replace("/", "\\")], stdout=subprocess.DEVNULL)
    assert os.path.exists(tree + "/prosqa/ckpt/gpt2_pretrained.bin"), tree


def ref_tree():
    t = TMP + "/ref"
    if not os.path.isdir(t + "/boxes"):
        os.makedirs(t, exist_ok=True); tp = TMP + "/ref.tar"
        subprocess.check_call(["git", "archive", "-o", tp, REF, "boxes"]); tarfile.open(tp).extractall(t)
    assert hashlib.sha256(open(t + "/boxes/inject_aw.py", "rb").read()).hexdigest().startswith("4d185e6d"), "ref: the §263 module"
    return t


def flags(k):
    fl = G7.dch_flags(k, "x", "x") + ["--hold_inc", "1.6", "--hold_log", "--perm_p", "0.5", "--ask_p", "0.25"]
    for f in ("--init_model", "--init_model_sha", "--init_trunk", "--init_sha"):
        if f in fl:
            i = fl.index(f); del fl[i:i + 2]
    return fl


def run(tree, k, extra, tag):
    d = G8.fresh(TMP + "/" + tag); env = {**G8.ENV, "G1_N": str(N), "G1_OUT": d + "/hashes.json"}
    with open(d + "/stdout.log", "w") as f_:
        rc = subprocess.run([sys.executable, "-B", "-m", "boxes._inject_aw_g1_s272"] + flags(k) + extra + ["--device", "cpu", "--out", d], cwd=tree, env=env,
                            stdout=f_, stderr=subprocess.STDOUT).returncode
    h = json.load(open(d + "/hashes.json")) if os.path.exists(d + "/hashes.json") else []
    return tag, rc, h


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True); os.makedirs(OUT, exist_ok=True); rt = ref_tree(); junction(rt); junction(NEW); instrument(rt); instrument(NEW)
    jobs = [(rt, k, [], "ref_s%d" % k) for k in (1, 2)] + [(NEW, k, [], "omit_s%d" % k) for k in (1, 2)] + [(NEW, k, ["--perm_scope", "op"], "op_s%d" % k) for k in (1, 2)]
    with ThreadPoolExecutor(3) as ex:                                       # <= 3 local CPU jobs (memory latent-cpu-contention)
        res = {t: (rc, h) for t, rc, h in ex.map(lambda j: run(*j), jobs)}
    out = {"N": N, "ref_commit": REF, "runs": {t: {"rc": rc, "n": len(h), "sha": hashlib.sha256("".join(h).encode()).hexdigest()} for t, (rc, h) in res.items()}}
    ok = True
    for k in (1, 2):
        r = res["ref_s%d" % k][1]
        for t in ("omit_s%d" % k, "op_s%d" % k):
            h = res[t][1]; mis = next((i + 1 for i, (x, y) in enumerate(zip(r, h)) if x != y), None)
            eq = res[t][0] == 0 and res["ref_s%d" % k][0] == 0 and len(h) == len(r) == N and mis is None; ok &= eq
            out["runs"][t]["equal_ref"] = eq; out["runs"][t]["first_mismatch_step"] = mis
    out["pass"] = ok; json.dump(out, open(OUT + "/g1_offbit.json", "w", newline="\n"), indent=1); print(json.dumps(out, indent=1)); sys.exit(0 if ok else 1)
