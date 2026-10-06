"""fsq120 --ckpt_every / --resume parity smoke (user 2026-10-06; Audit's tolerance, fixed before reading): same card, the check runs'
flags (--probe127 --no_cert, --micro 1) on a compressed schedule (--sched_scale 0.05: p_hard ramp 100-200, L_max to 8):
  A = uninterrupted 400 steps; B = ckpt_every 200, stop_after 200; C = B's dir resumed to 400; D = tiny run with eval (ckpt deleted?).
PASS: B == A on every logged loss <= 200 (to the printed precision), C within 1e-3 of A at every logged step 200-400 (exact if the
kernels are deterministic) - revised (Audit): max|C - A| <= max(1e-3, 3 x max|A' - A|), A' = a second uninterrupted A - C's step set == A's, D has eval.json and no ckpt_last.pt. Writes <out>/eval.json.
    python -m scratch.parity_ckpt --device cuda:0 --out boxes/runs_fsq/parity_ckpt_<card>"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
F = ["--family", "s5", "--arm", "final", "--seed", "1", "--steps", "400", "--sched_scale", "0.05", "--micro", "1", "--probe127", "--no_cert",
     "--log_every", "50", "--device", a.device]


def run(sub, extra):
    d = os.path.join(a.out, sub)
    with open(os.path.join(a.out, sub + ".log"), "w") as f:
        rc = subprocess.call([sys.executable, "-u", "-m", "boxes.fsq120", *F, *extra, "--out", d], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    return rc, d


def losses(d):
    return {r["step"]: r["loss"] for r in map(json.loads, open(os.path.join(d, "log.jsonl"))) if r.get("loss") is not None}


rA, A = run("A", ["--stop_after", "400"])
rA2, A2 = run("A2", ["--stop_after", "400"])                                  # Audit: A' = A again (kernel non-determinism floor)
rB, B = run("B", ["--ckpt_every", "200", "--stop_after", "200"])
C = os.path.join(a.out, "C"); shutil.copytree(B, C)
rC, _ = run("C", ["--ckpt_every", "200", "--resume", os.path.join(C, "ckpt_last.pt"), "--stop_after", "400"])
with open(os.path.join(a.out, "D.log"), "w") as f:
    rD = subprocess.call([sys.executable, "-u", "-m", "boxes.fsq120", "--family", "s5", "--arm", "final", "--seed", "2", "--steps", "40", "--sched_scale", "0.002", "--bs", "8",
                          "--micro", "1", "--probe127", "--no_cert", "--ckpt_every", "20", "--test_n", "4", "--test_len", "8", "--ext_n", "2", "--donor_n", "4",
                          "--margin_n", "4", "--device", a.device, "--out", os.path.join(a.out, "D")], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
LA, LB, LC, LA2 = losses(A), losses(B), losses(C), losses(A2)
nd = max([abs(LA[s_] - LA2[s_]) for s_ in LA if s_ > 200 and s_ in LA2] or [float("nan")])
b_ok = sorted(LB) == [s for s in sorted(LA) if s <= 200] and all(LA[s] == LB[s] for s in LB)
c_diff = {s: abs(LA[s] - LC[s]) for s in LA if s > 200 and s in LC}
c_ok = sorted(LC) == sorted(LA) and all(LA[s] == LC[s] for s in LA if s <= 200) and bool(c_diff) and max(c_diff.values()) <= max(1e-3, 3 * nd)
d_ok = os.path.exists(os.path.join(a.out, "D", "eval.json")) and not os.path.exists(os.path.join(a.out, "D", "ckpt_last.pt"))
res = {"rc": {"A": rA, "A2": rA2, "B": rB, "C": rC, "D": rD}, "A2_max_abs_diff_gt_200": nd, "A2_identical_all": LA == LA2, "tol": max(1e-3, 3 * nd), "steps_A": sorted(LA), "B_identical_le_200": b_ok, "C_max_abs_diff_gt_200": max(c_diff.values()) if c_diff else None,
       "C_exact_gt_200": bool(c_diff) and max(c_diff.values()) == 0.0, "C_ok": c_ok, "D_ckpt_deleted_after_eval": d_ok,
       "loss_A": LA, "loss_C": LC, "PASS": bool(b_ok and c_ok and d_ok and rA == rA2 == rB == rC == rD == 0)}
for d_ in (B, C):                                                           # test checkpoints are not kept (retention)
    if os.path.exists(os.path.join(d_, "ckpt_last.pt")):
        res.setdefault("ckpt_bytes", {})[os.path.basename(d_)] = os.path.getsize(os.path.join(d_, "ckpt_last.pt")); os.remove(os.path.join(d_, "ckpt_last.pt"))
for d_ in (A, A2, B, C, os.path.join(a.out, "D")):
    if os.path.exists(os.path.join(d_, "model.pt")):
        os.remove(os.path.join(d_, "model.pt"))
json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1); print("PARITY", json.dumps({k: res[k] for k in res if not k.startswith("loss")}), flush=True)
