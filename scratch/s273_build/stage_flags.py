"""§268 LINEAGE-SEED (build serial s273; Note 268-1): per stage, the s1 stage's exact trainer version (a git archive of boxes/ at the commit whose trainer sha256 equals
the s1 args.json module_sha256) and the s1 args, with seed 3, the seed-3 init chain and out changed. The CLI is rebuilt from boxes/runs_fsq/<stage>_s1/args.json by
introspecting that version's own argparse (every action whose dest is in args.json and whose value differs from the parser default; store_true -> the bare flag), then
checked LOCALLY with that version's --args_only --args_ref <s1 args.json>: the diff (minus META / run-info keys) must be exactly {seed, init paths / shas, out}.
    python -B scratch/s273_build/stage_flags.py <stage> [--init_model <path>=<sha16>]     -> scratch/s273_flags_<stage>_s3.json {commit, module, argv, diff}
Stages (commit / module, Note 268-1): askname 515aa108 inject_ow via the §233 wrapper docs/research_steptold_2026-10-24/st_ask_name_local.py (sha 6eba1cb8;
the first commit holding both it and inject_ow 0695c354; the wrapper adds --arm direct itself) | aw_plaw2 56fe2c78 inject_aw | rcb_plaw4 50838f38 | dis 1deba3e5 | dch 18c0a06f | bun 923573e1."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

ROOT = "D:/Latent"; os.chdir(ROOT)
STAGES = {"askname": ("s233_askname", "515aa108", "boxes.inject_ow"), "aw_plaw2": ("aw_plaw2", "56fe2c78", "boxes.inject_aw"),
          "rcb_plaw4": ("rcb_plaw4", "50838f38", "boxes.inject_aw"), "dis": ("dis", "1deba3e5", "boxes.inject_aw"),
          "dch": ("dch", "18c0a06f", "boxes.inject_aw"), "bun": ("bun", "923573e1", "boxes.inject_aw")}
DROP = {"device", "out", "args_only", "args_ref", "resume", "init_trunk", "init_sha", "init_model", "init_model_sha", "seed"}
TRUNK_S3 = "boxes/runs_fsq/s233_askname_s3/model.pt"
WRAP = "docs/research_steptold_2026-10-24/st_ask_name_local.py"


def runner(stage, module):
    """the command prefix: the askname wrapper script (sha asserted by main) or -m <module>."""
    return [WRAP] if stage == "askname" else ["-m", module]


def tree(commit):
    """git archive <commit> boxes -> scratch/_tmp/s273/<commit>/ (boxes/runs_fsq and prosqa linked to D:/Latent's; never rmtree through a link)."""
    t = ROOT + "/scratch/_tmp/s273/" + commit
    if not os.path.isdir(t + "/boxes"):
        os.makedirs(t, exist_ok=True); tp = t + ".tar"; subprocess.check_call(["git", "archive", "-o", tp, commit, "boxes"] + ([WRAP] if commit == "515aa108" else []))
        tarfile.open(tp).extractall(t)
    for sub in ("prosqa", "boxes/runs_fsq"):
        q = t + "/" + sub
        if not os.path.exists(q):
            subprocess.check_call(["cmd", "/c", "mklink", "/J", q.replace("/", "\\"), (ROOT + "/" + sub).replace("/", "\\")], stdout=subprocess.DEVNULL)
    return t


def parser_actions(t, module):
    """[(option, dest, default, nargs0)] of the version's argparse (parse_args intercepted, nothing runs)."""
    code = ("import argparse, json, sys\n"
            "def _pa(self, args=None, namespace=None):\n"
            "    print('ACTIONS' + json.dumps([[a.option_strings[0], a.dest, a.default if isinstance(a.default, (int, float, str, bool, type(None))) else str(a.default), a.nargs == 0]"
            " for a in self._actions if a.option_strings and a.dest != 'help'])); sys.exit(0)\n"
            "argparse.ArgumentParser.parse_args = _pa\nimport runpy; runpy.run_module('%s', run_name='__main__')\n") % module
    o = subprocess.run([sys.executable, "-B", "-c", code], cwd=t, capture_output=True, text=True)
    line = [ln for ln in o.stdout.splitlines() if ln.startswith("ACTIONS")]; assert line, o.stderr[-2000:]
    return json.loads(line[0][7:])


def argv_from(a, acts):
    out = []
    for opt, dest, default, flag in acts:
        if dest in DROP or dest not in a or a[dest] == default or a[dest] is None:     # None: set after parsing (e.g. a direct arm's mem_write)
            continue
        if flag:
            assert isinstance(a[dest], bool), (dest, a[dest]); out += [opt] if a[dest] else []
        else:
            out += [opt, str(a[dest])]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=list(STAGES)); ap.add_argument("--init_model", default="")
    ap.add_argument("--trunk_sha", default="", help="sha16 of s233_askname_s3/model.pt (stages after askname)"); x = ap.parse_args()
    name, commit, module = STAGES[x.stage]; ref = "boxes/runs_fsq/%s_s1/args.json" % name; a = json.load(open(ref)); t = tree(commit)
    src = {"boxes.inject_ow": "boxes/inject_ow.py", "boxes.inject_aw": "boxes/inject_aw.py"}[module]
    assert hashlib.sha256(open(t + "/" + src, "rb").read()).hexdigest() == a["module_sha256"], "the stage's trainer version"
    argv = argv_from(a, parser_actions(t, module)) + ["--seed", "3"]
    if x.stage == "askname":                                                # the wrapper adds --arm direct
        assert hashlib.sha256(open(t + "/" + WRAP, "rb").read()).hexdigest().startswith("6eba1cb8"); i = argv.index("--arm"); del argv[i:i + 2]
    if a.get("init_trunk"):
        assert x.trunk_sha, "--trunk_sha"; argv += ["--init_trunk", TRUNK_S3, "--init_sha", x.trunk_sha]
    if a.get("init_model"):
        p, s = x.init_model.split("="); argv += ["--init_model", p, "--init_model_sha", s]
    d = ROOT + "/scratch/_tmp/s273/args_" + x.stage
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    o = subprocess.run([sys.executable, "-B"] + runner(x.stage, module) + argv + ["--device", "cpu", "--args_only", "--args_ref", os.path.abspath(ref), "--out", d], cwd=t,
                       capture_output=True, text=True)
    assert o.returncode == 0, o.stdout[-1500:] + o.stderr[-1500:]
    dif = json.load(open(d + "/args_diff.json"))["diff"]
    allowed = {"seed", "out", "init_trunk", "init_sha", "init_sha256", "init_model", "init_model_sha", "init_model_sha256", "args_ref", "args_only"}
    extra = sorted(set(dif) - allowed)
    res = {"stage": x.stage, "run": name + "_s3", "commit": commit, "module": module, "runner": runner(x.stage, module), "argv": argv, "diff": dif, "extra_diff_keys": extra, "ok": not extra}
    json.dump(res, open(ROOT + "/scratch/s273_flags_%s_s3.json" % x.stage, "w", newline="\n"), indent=1); print(json.dumps({k: res[k] for k in ("run", "extra_diff_keys", "ok")}))
    print(" ".join(argv)); sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
