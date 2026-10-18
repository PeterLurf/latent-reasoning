#!/bin/bash
# §218 G-L10 OFF-HOST + RESUME (local half) on the 3070 (cuda:1), after scratch/s218_3070_chain.sh (MSYS pid 47893):
#   A  uninterrupted 400 steps;  B  the same args stopped at 300 (ckpt at 300) then resumed to 400;  C  a second uninterrupted run (nondeterminism control).
#   Pass: B's per-step losses 301..400 == A's, bitwise (log_every 1). D 3-4 (cur 4), bs 8, micro 1, cal_n 16.
cd /d/Latent
while kill -0 47893 2>/dev/null; do sleep 60; done
G=scratch/gate_s218/g_l10; mkdir -p $G
COMMON="--arch lslva --device cuda:1 --cur 4 --steps 400 --bs 8 --micro 1 --cal_n 16 --log_every 1 --per_n 2 --eval_bs 4"
python -B -m boxes.langvb $COMMON --ckpt_every 0 --out $G/A > $G/A.log 2>&1; rcA=$?
python -B -m boxes.langvb $COMMON --ckpt_every 300 --stop_at 300 --out $G/B > $G/B1.log 2>&1; rcB1=$?
python -B -m boxes.langvb $COMMON --ckpt_every 300 --resume $G/B/ckpt_last.pt --out $G/B > $G/B2.log 2>&1; rcB2=$?
python -B -m boxes.langvb $COMMON --ckpt_every 0 --out $G/C > $G/C.log 2>&1; rcC=$?
python - "$G" <<'PY'
import json, sys, os
G = sys.argv[1]
def losses(d):
    out = {}
    for l in open(os.path.join(G, d, "log.jsonl")):
        r = json.loads(l)
        if "loss" in r and r.get("loss") is not None:
            out[r["step"]] = (r["loss"], r["answer_ce"], r["rec_ce"])
    return out
A, B, C = losses("A"), losses("B"), losses("C")
rng = range(301, 401)
eqAB = all(A.get(s) == B.get(s) for s in rng); eqAC = all(A.get(s) == C.get(s) for s in range(1, 401))
diffAB = [s for s in rng if A.get(s) != B.get(s)][:5]
res = {"resume_equal_301_400": eqAB, "control_equal_1_400": eqAC, "first_diff_AB": diffAB, "n_A": len(A), "n_B": len(B), "n_C": len(C),
       "pass": eqAB, "note": "B = stop at 300 (ckpt_last.pt incl. model / opt / np rngs / torch rng / stream band rngs / CurGate) + resume; C = an uninterrupted re-run"}
json.dump(res, open("scratch/gate_s218/g_l10.json", "w", newline="\n"), indent=1); print("g_l10", res)
PY
echo "$(date '+%F %T') l10 rcA $rcA rcB1 $rcB1 rcB2 $rcB2 rcC $rcC" >> logs/s218_gates.out
