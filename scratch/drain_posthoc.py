"""Model-skip rule for the drain (Theory + Audit, 2026-10-06), run ON THE HOST before a fetched run dir is tarred:
    python scratch/drain_posthoc.py <run dir> [device]   ->   one line: "KEEP <reason>" or "SKIP <reason> | posthoc: <steps>"
SKIP model.pt only if long_acc < 0.99 AND every ext < 0.99, and after every registered on-host post-hoc step has run:
fp32 --contraction_only for fsq120 runs (eval_contraction_fp32.json). skip_off / donor are already in eval.json; CERT / growth
belong to sections whose runs are always kept (§128) or are closed.
ALWAYS KEEP: any exact run (long_acc / cumulative 512 / any ext >= 0.99); every §128 run (accepted or 128-1 check runs); §119
Coconut (SIM-CoT inits) and any competent §119 run (intact_acc >= 0.90); runs matching scratch/model_keep_patterns.txt
(registration-named runs, controls, §129 / §130); anything this script cannot judge (no eval / unknown layout -> KEEP).
eval / log / args / preds_test.npy are always fetched (only model.pt is ever skipped)."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def accs(e):
    """Long accuracy and every extension accuracy found in an eval.json of any of our layouts."""
    out = {}
    if isinstance(e.get("long_acc"), (int, float)):
        out["long_acc"] = e["long_acc"]
    cum = e.get("cumulative")
    if isinstance(cum, dict) and cum.get("512") is not None:
        out["cumulative_512"] = cum["512"]
    for k in ("ext_2048", "ext_8192"):
        v = e.get(k)
        if isinstance(v, dict) and isinstance(v.get("cumulative"), (int, float)):
            out[k] = v["cumulative"]
    ex = e.get("extension_2048")
    if isinstance(ex, dict) and isinstance(ex.get("cumulative"), dict):
        for t, v in ex["cumulative"].items():
            if v is not None:
                out["extension_2048_" + t] = v
    return out


def patterns():
    p = os.path.join(ROOT, "scratch", "model_keep_patterns.txt")
    return [ln.strip() for ln in open(p) if ln.strip() and not ln.startswith("#")] if os.path.exists(p) else []


def main():
    d = sys.argv[1].rstrip("/"); dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"; name = os.path.basename(d)
    ep = os.path.join(d, "eval.json")
    if not os.path.exists(os.path.join(d, "model.pt")):
        print("KEEP no model.pt"); return
    if not os.path.exists(ep):
        print("KEEP no eval.json (partial / void: kept for investigation)"); return
    if name.startswith(("s130chk_", "s131x_")) and not os.path.exists(os.path.join(d, "eval_probe_posthoc.json")):
        # Audit 2026-10-06: the end-of-training probe (probe127 + opprobe) on the host before destroy; the json then comes back with the
        # small files (pass 1). Output to a log (the drain reads this script's LAST stdout line as the decision).
        mod = "boxes.fsq120_posthoc" if os.path.exists(os.path.join(ROOT, "boxes", "fsq120_posthoc.py")) else "boxes.fsq120"
        with open(os.path.join(ROOT, "logs", "probe_posthoc_%s.log" % name), "w") as lg:
            subprocess.call([sys.executable, "-u", "-m", mod, "--from_args", os.path.join(d, "args.json"), "--probe_only", "--device", dev, "--out", d],
                            cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT)
    if name.startswith(("s139_", "s142_")) and not os.path.exists(os.path.join(d, "eval_contraction_fp32.json")):   # §139 / §142: fp32 contraction on-host via boxes.multrow
        with open(os.path.join(ROOT, "logs", "posthoc_contraction_%s.log" % name), "w") as lg:
            subprocess.call([sys.executable, "-u", "-m", "boxes.multrow", "--from_args", os.path.join(d, "args.json"), "--contraction_only", "--fp32", "--device", dev, "--out", d],
                            cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT)
    if name.startswith(("s140_pair_", "s145_", "s150_")) and not os.path.exists(os.path.join(d, "eval_contraction_fp32.json")):   # §140: fp32 contraction on-host via boxes.multpair
        with open(os.path.join(ROOT, "logs", "posthoc_contraction_%s.log" % name), "w") as lg:
            subprocess.call([sys.executable, "-u", "-m", "boxes.multpair", "--from_args", os.path.join(d, "args.json"), "--contraction_only", "--fp32", "--device", dev, "--out", d],
                            cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT)
    if name.startswith(("s137_col_", "s141_")):                             # §137 (9c9b1899) / §141 (9e45b92b): fp32 contraction + prefix on-host via boxes.multcol
        for flag, outf in ((["--contraction_only", "--fp32"], "eval_contraction_fp32.json"), (["--prefix_only"], "eval_prefix.json")):
            if not os.path.exists(os.path.join(d, outf)):
                with open(os.path.join(ROOT, "logs", "posthoc_%s_%s.log" % (outf.split(".")[0], name)), "w") as lg:
                    subprocess.call([sys.executable, "-u", "-m", "boxes.multcol", "--from_args", os.path.join(d, "args.json")] + flag + ["--device", dev, "--out", d],
                                    cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT)
    if name.startswith(("s133_", "s134_", "s135_", "s136_", "s138_", "s143_", "s144_")) and not os.path.exists(os.path.join(d, "eval_contraction_fp32.json")):   # s136_ added 2026-10-07 (gap)
        # Theory 133-5: PASS's rho reads only the post-hoc fp32 contraction - every accepted §133 / §134 / §135 run, on the host, before
        # any keep / skip decision (kept models used to skip it)
        mod = "boxes.fsq120_posthoc" if os.path.exists(os.path.join(ROOT, "boxes", "fsq120_posthoc.py")) else "boxes.fsq120"
        with open(os.path.join(ROOT, "logs", "posthoc_contraction_%s.log" % name), "w") as lg:
            subprocess.call([sys.executable, "-u", "-m", mod, "--from_args", os.path.join(d, "args.json"), "--contraction_only", "--fp32", "--device", dev, "--out", d],
                            cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT)
    guard134 = name.startswith(("s134_copy_ff_", "s134_copy_cyc_", "s134_copy_mod_", "s134_copy_rdfa_")) or name.startswith(("s138_", "s143_"))   # §138: prefix 1/2/4/8 too   # Theory 2026-10-06: §134 guard runs, prefix always
    if name.startswith(("s130chk_", "s131x_")) or guard134:                 # Theory 57d6b89e: FAILED runs also get prefix accuracy + fp32 contraction
        try:
            la = json.load(open(ep)).get("long_acc")
        except Exception:
            la = None
        if guard134 or (la is not None and la < 0.99):
            mod = "boxes.fsq120_posthoc" if os.path.exists(os.path.join(ROOT, "boxes", "fsq120_posthoc.py")) else "boxes.fsq120"
            for flag, outf in ((["--prefix_only"], "eval_prefix.json"), (["--contraction_only", "--fp32"], "eval_contraction_fp32.json")):
                if not os.path.exists(os.path.join(d, outf)):
                    with open(os.path.join(ROOT, "logs", "posthoc_%s_%s.log" % (outf.split(".")[0], name)), "w") as lg:
                        subprocess.call([sys.executable, "-u", "-m", mod, "--from_args", os.path.join(d, "args.json")] + flag + ["--device", dev, "--out", d],
                                        cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT)
    if name.startswith("s143_") and not os.path.exists(os.path.join(d, "eval_ablate704.json")):   # note 143-2 (ef7364f7): 704-block ablation on-host
        mod = "boxes.fsq120_posthoc" if os.path.exists(os.path.join(ROOT, "boxes", "fsq120_posthoc.py")) else "boxes.fsq120"
        with open(os.path.join(ROOT, "logs", "posthoc_ablate704_%s.log" % name), "w") as lg:
            try:                                                            # own limit: the drain's ssh allows 900 s for this whole script
                subprocess.call([sys.executable, "-u", "-m", mod, "--from_args", os.path.join(d, "args.json"), "--ablate704_only", "--device", dev, "--out", d],
                                cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT, timeout=480)
            except subprocess.TimeoutExpired:
                lg.write("ABLATE704 TIMEOUT (480 s)\n")
    for pat in patterns():
        if pat in d:
            print("KEEP pattern %s" % pat); return
    if name.startswith("s128_"):
        print("KEEP s128 run"); return
    if name.startswith("s119_coconut"):
        print("KEEP s119 Coconut (SIM-CoT init)"); return
    try:
        e = json.load(open(ep))
    except Exception as ex:
        print("KEEP eval.json unreadable (%s)" % ex); return
    if name.startswith("s119_") and (e.get("intact_acc") or 0) >= 0.90:
        print("KEEP s119 competent (intact %.3f)" % e["intact_acc"]); return
    if e.get("k1_certified") is True or (e.get("k1_never_labelled_acc") or 0) >= 0.99:   # Audit: acquired self-training run (C4 evidence)
        print("KEEP acquired self-training (k1_certified %s, never-labelled %s)" % (e.get("k1_certified"), e.get("k1_never_labelled_acc"))); return
    A = accs(e)
    if not A:
        print("KEEP no accuracy field recognised"); return
    hi = {k: v for k, v in A.items() if v >= 0.99}
    if hi:
        print("KEEP exact (%s)" % ", ".join("%s %.4f" % kv for kv in hi.items())); return
    steps = []
    a = json.load(open(os.path.join(d, "args.json"))) if os.path.exists(os.path.join(d, "args.json")) else {}
    if "family" in a and "arm" in a:                                        # fsq120 run: fp32 contraction on the host first
        if not os.path.exists(os.path.join(d, "eval_contraction_fp32.json")):
            mod = "boxes.fsq120_posthoc" if os.path.exists(os.path.join(ROOT, "boxes", "fsq120_posthoc.py")) else "boxes.fsq120"   # current code, shipped by the drain
            rc = subprocess.call([sys.executable, "-u", "-m", mod, "--from_args", os.path.join(d, "args.json"), "--contraction_only", "--fp32",
                                  "--device", dev, "--out", d],                # the run's own configuration (Audit: no hand-rebuilt flags)
                                 stdout=open(os.path.join(ROOT, "logs", "gen_%s_contraction_fp32.log" % name), "w"), stderr=subprocess.STDOUT, cwd=ROOT)
            if rc != 0 or not os.path.exists(os.path.join(d, "eval_contraction_fp32.json")):
                print("KEEP fp32 contraction failed on host (rc %d)" % rc); return
            steps.append("fp32 contraction (run now)")
        else:
            steps.append("fp32 contraction (present)")
    print("SKIP not exact (%s) | posthoc: %s" % (", ".join("%s %.4f" % kv for kv in A.items()), "; ".join(steps) or "none registered"))


if __name__ == "__main__":
    main()
