#!/bin/bash
# §218 G-L10 half 2 (Theory / Audit 10-18): RESTORE-EXACT in the registered (non-deterministic) setting on the 3070 after chain4 (MSYS pid 56386):
# Bn1 = 300 steps with a ckpt at 300 (+ ckpt_fingerprint.json), Bn2 = resume -> langvb's resume_check.json (sha256 of every model / optimiser tensor,
# param groups, rng / rng_k / rng_s, torch rng, the stream's band rngs, CurGate, self counters: restored == saved). Merged into g_l10.json as half 2.
cd /d/Latent
while kill -0 56386 2>/dev/null; do sleep 60; done
G=scratch/gate_s218/g_l10_restore; mkdir -p $G
COMMON="--arch lslva --device cuda:1 --cur 4 --steps 400 --bs 8 --micro 1 --cal_n 16 --log_every 50 --per_n 2 --eval_bs 4"
python -B -m boxes.langvb $COMMON --ckpt_every 300 --stop_at 300 --out $G/Bn > $G/Bn1.log 2>&1; rc1=$?
python -B -m boxes.langvb $COMMON --ckpt_every 0 --stop_at 301 --resume $G/Bn/ckpt_last.pt --out $G/Bn > $G/Bn2.log 2>&1; rc2=$?
python - <<'PY'
import json, os
p = "scratch/gate_s218/g_l10.json"; g = json.load(open(p)) if os.path.exists(p) else {}
rc = json.load(open("scratch/gate_s218/g_l10_restore/Bn/resume_check.json"))
g["half1_deterministic_bitwise"] = {k: g.get(k) for k in ("resume_equal_301_400", "control_equal_1_400", "first_diff_AB", "n_A", "n_B", "n_C")}
g["half2_restore_exact"] = rc; g["pass"] = bool(g.get("resume_equal_301_400") and g.get("control_equal_1_400") and rc.get("equal"))
g["rule"] = "half 1: --deterministic, B 301..400 == A bitwise and C == A; half 2: registered setting, the restored state's fingerprint == the saved one"
json.dump(g, open(p, "w", newline="\n"), indent=1); print("g_l10", g["pass"], rc)
PY
find $G -name "*.pt" -delete
echo "$(date '+%F %T') l10_restore rc1 $rc1 rc2 $rc2" >> logs/s218_gates.out
