#!/bin/bash
# §223 build gate (PS2_v1 Addendum B item 3: off-host full-state checkpoints with a TESTED resume): boxes/codi_ps.py, the PS cell (--recode --lam 1
# --bottleneck: the D head + the M2 mask), small shape (eff_bs 8 = micro 8, 60 steps, d stages 1,2,3 with CurGate bins of 10 / cap 20 so the
# curriculum advances inside the test, ckpt every 20, log every step), under scratch/s222_det_wrap.py (math SDP + deterministic algorithms: a gate,
# not the registered kernel) so the comparison can be bitwise. REF = 60 steps uninterrupted; KILL = the same run stopped (exit 9) after step 30;
# RESUME = --resume from the step-20 checkpoint. PASS iff every per-step log record (loss, ans_ce, num_ce, recode_ce, d_max, stage, d_hist, K_mean)
# of RESUME's steps 21..60 equals REF's, and the stage advances agree.   bash scratch/s223_resume_test.sh <cuda dev>  -> logs/s223_resume.log
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
DV=${1:-cuda:2}; T=scratch/_tmp/s223_resume; rm -rf "$T"; mkdir -p "$T/ref" "$T/res"; L=logs/s223_resume.log; : > "$L"
base="python -B scratch/s222_det_wrap.py boxes/codi_ps.py --recode --lam 1 --bottleneck --device $DV --eff_bs 8 --micro 8 --steps 60 --d_stages 1,2,3 --cg_bin 10 --cg_cap 20 --ckpt_every 20 --log_every 1"
echo "$(date '+%F %T') REF" >> "$L"; $base --out $T/ref >> "$L" 2>&1; echo "ref rc $?" >> "$L"
echo "$(date '+%F %T') KILL after 30" >> "$L"; $base --out $T/res --stop_at 30 >> "$L" 2>&1; echo "kill rc $? (9 expected)" >> "$L"
echo "$(date '+%F %T') RESUME" >> "$L"; $base --out $T/res --resume >> "$L" 2>&1; echo "resume rc $?" >> "$L"
python - "$T" >> "$L" 2>&1 <<'EOF'
import json, sys
T = sys.argv[1]; keys = ("loss", "ans_ce", "num_ce", "num_acc", "recode_ce", "recode_val_ce", "d_max", "stage", "d_hist", "K_mean", "gn_preclip", "lr")
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
ref = {r["step"]: r for r in rd(T + "/ref/log.jsonl")}; res = rd(T + "/res/log.jsonl")
kill = [r for r in res if r["step"] <= 30]; rsm = [r for r in res if r["step"] > 30 or (r["step"] > 20 and r is not None)]
# the res log holds the kill run's steps 1..30 then the resumed steps 21..60 (appended): split at the second occurrence of step 21
idx = [i for i, r in enumerate(res) if r["step"] == 21]; second = res[idx[1]:] if len(idx) > 1 else []
pre = all(all(r.get(k) == ref[r["step"]].get(k) for k in keys) for r in res[:idx[1]] if len(idx) > 1) if len(idx) > 1 else False
post = bool(second) and all(all(r.get(k) == ref[r["step"]].get(k) for k in keys) for r in second)
bad = [(r["step"], k) for r in second for k in keys if r.get(k) != ref[r["step"]].get(k)][:5]
st = lambda p: [json.loads(l) for l in open(p)] if __import__("os").path.exists(p) else []
sa, sb = st(T + "/ref/stages.jsonl"), st(T + "/res/stages.jsonl")
stages_ok = [x["step"] for x in sa] == sorted({x["step"] for x in sb}) and len(sa) > 0
print("kill-run steps == ref:", pre, "| resumed steps", (second[0]["step"], second[-1]["step"]) if second else None, "== ref:", post, "first diffs", bad)
print("stage advances ref", [(x["step"], x["to"], x["reason"][:9]) for x in sa], "res steps", sorted({x["step"] for x in sb}), "agree:", stages_ok)
print("RESUME-TEST s223", "PASS" if (pre and post and stages_ok and second and second[-1]["step"] == 60) else "FAIL")
EOF
echo "$(date '+%F %T') done" >> "$L"
