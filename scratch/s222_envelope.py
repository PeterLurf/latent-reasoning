"""§222 Note 222-3 (b) (Theory / Audit 10-21): the same-state run-to-run envelope of the REGISTERED kernel (memory-efficient SDPA) on the deploy card
class, and the resume test against it. Runs scratch/s222_cert.py unchanged (subprocesses), untrained model, smoke sizes (pool 3000, held 64, per_D 3),
the t2 group (--group 3 --groups 3 = cexor_t2 / lin_t2 / noplant_t2), --steps 200 --eval_every 40 --ckpt_every 100 (--steps S scales them: ckpt S/2, kill at S/2 + 0.15 S; smokes only):
  A, B  two runs from the same state (same seeds, same card)            -> the envelope: per probe, max over steps 101-200 of |CE_A - CE_B|, and the held-out
                                                                           accuracy differences at every eval
  K, R  the same run killed after step 130 (exit 9), resumed from the step-100 checkpoint
Recorded: the envelope, R vs A and R vs B (steps 101-200 CE and the evals after step 100), B vs A. PASS (Note 222-3, Audit: vs A OR vs B): for every probe
max |CE_R - CE_A| <= max(2 x the envelope over all probes, 1e-6) and every held-out eval after the resume within 1 item of A. Writes <out>/envelope.json.
    python -B scratch/s222_envelope.py --out boxes/runs_fsq/s222_envelope_3090 --device cuda:0 [--m_rule scratch/gate_s222/m_rule.json]"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--m_rule", default="scratch/gate_s222/m_rule.json"); ap.add_argument("--steps", type=int, default=200, help="smokes only")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time(); S = a.steps; CK = S // 2; KS = CK + max(1, (3 * S) // 20)
    base = [sys.executable, "-B", "scratch/s222_cert.py", "--mode", "cert", "--model", "untrained", "--device", a.device, "--pool_n", "3000", "--hold_n", "64", "--per_D", "3",
            "--steps", str(S), "--eval_every", str(S // 5), "--group", "3", "--groups", "3", "--ckpt_every", str(CK), "--m_rule", a.m_rule]
    rc = {}
    for nm, extra in (("A", []), ("B", []), ("K", ["--stop_at", str(KS)]), ("R", ["--resume"])):
        d = os.path.join(a.out, "R" if nm in ("K", "R") else nm); os.makedirs(d, exist_ok=True)
        ll = os.path.join(d, "loss_%s.jsonl" % nm)
        with open(os.path.join(a.out, "run_%s.log" % nm), "w") as lf:
            rc[nm] = subprocess.run(base + ["--out", d, "--ckpt_dir", d, "--loss_log", ll] + extra, stdout=lf, stderr=subprocess.STDOUT).returncode
        print("ENVELOPE run", nm, "rc", rc[nm], round(time.time() - t0), "s", flush=True)
    rd = lambda p: {r["step"]: {k: float(v) for k, v in r["loss"].items()} for r in map(json.loads, open(p))}
    LA, LB, LR = rd(os.path.join(a.out, "A", "loss_A.jsonl")), rd(os.path.join(a.out, "B", "loss_B.jsonl")), rd(os.path.join(a.out, "R", "loss_R.jsonl"))
    keys = sorted(LA[1]); win = [s for s in sorted(LA) if CK < s <= S]
    env = {k: max(abs(LA[s][k] - LB[s][k]) for s in win) for k in keys}; ra = {k: max(abs(LR[s][k] - LA[s][k]) for s in win if s in LR) for k in keys}
    cv = lambda d: json.load(open(os.path.join(a.out, d, "cert_untrained.g3.json")))["arms"]
    CA, CB, CR = cv("A"), cv("B"), cv("R")
    ev = lambda X, Y: {k: [[s, round((y - x) * 64)] for (s, x), (_, y) in zip(X[k]["curve"], Y[k]["curve"]) if s > CK] for k in keys}
    E = max(env.values()); o = {"def": __doc__, "rc": rc, "probes": keys, "window_steps": [win[0], win[-1]], "kernel": "registered (memory-efficient SDPA, default)",
                                "envelope_ce_max_abs_by_probe_B_vs_A": env, "envelope_ce_max_abs": E, "resume_ce_max_abs_by_probe_R_vs_A": ra,
                                "heldout_item_diff_B_vs_A": ev(CA, CB), "heldout_item_diff_R_vs_A": ev(CA, CR), "resume_steps_R": [min(LR), max(LR)],
                                "bitwise_B_vs_A": all(LA[s] == LB[s] for s in LA), "bitwise_R_vs_A": all(LR[s] == LA[s] for s in LR)}
    rbb = {k: max(abs(LR[s][k] - LB[s][k]) for s in win if s in LR) for k in keys}; o["resume_ce_max_abs_by_probe_R_vs_B"] = rbb; o["heldout_item_diff_R_vs_B"] = ev(CB, CR)
    thr = max(2 * E, 1e-6); o["threshold_ce"] = thr
    vs = {}
    for ref, rr in (("A", ra), ("B", rbb)):                                  # Audit 10-21: A vs B is one noise sample - R passes within the envelope of EITHER
        vs[ref] = bool(all(rr[k] <= thr for k in keys) and all(abs(d_) <= 1 for k in keys for _, d_ in o["heldout_item_diff_R_vs_" + ref][k]))
    o["pass_vs"] = vs; runs_ok = all(v == 0 for n, v in rc.items() if n != "K") and rc.get("K") == 9 and min(LR) == CK + 1
    o["proposed_PASS"] = bool(runs_ok and (vs["A"] or vs["B"]))
    o["rule"] = "Note 222-3 + Audit 10-21: per probe max |CE_R - CE_X| over the window <= max(2 x the A/B envelope max, 1e-6) and every held-out eval after the resume within 1 item of X, for X = A or X = B; all runs rc 0 (K rc 9); R starts at ckpt + 1"
    try:
        import torch
        o["gpu"] = torch.cuda.get_device_name(a.device)
    except Exception:
        o["gpu"] = None
    o["sec"] = round(time.time() - t0); json.dump(o, open(os.path.join(a.out, "envelope.json"), "w", newline="\n"), indent=1)
    for d in ("A", "B", "R"):                                              # retention: the smoke checkpoints are not results (the drain fetches the dir)
        for f in ("ckpt_last.pt", "ckpt_last.pt.tmp"):
            if os.path.exists(os.path.join(a.out, d, f)):
                os.remove(os.path.join(a.out, d, f))
    print("ENVELOPE", json.dumps({k: o[k] for k in ("envelope_ce_max_abs", "resume_ce_max_abs_by_probe_R_vs_A", "bitwise_B_vs_A", "bitwise_R_vs_A", "proposed_PASS", "gpu")}), flush=True)


if __name__ == "__main__":
    main()
