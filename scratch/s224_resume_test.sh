#!/bin/bash
# §224 K11 (FH_v1 §6): resume bitwise at a 200-step boundary with rng_far and the dose state restored. TREE's boxes/vbank.py (PYTHONPATH = TREE),
# the FH FT recipe on D3-T s1 (--far --fh, far_t 13,48, far_n 4, bs 16, micro 1, dose window 150,250 so the window straddles the ckpt), fp32 (Turing:
# amp off) under scratch/s222_det_wrap.py (math SDP + deterministic algorithms: a gate, not the registered kernel). REF = 300 steps uninterrupted;
# KILL = the same run stopped after 230 (ckpt at 200); RESUME = --resume KILL's ckpt_last.pt to 300 into the KILL dir (log appends). PASS iff every
# RESUME record 210..300 (all keys but sec) and the lambda-set line equal REF's, and REF's lambda was set at 250 from ratios spanning the boundary.
#   bash scratch/s224_resume_test.sh <cuda dev> [tree]   -> logs/s224_resume.log, scratch/gate_s224/k11.json
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
DV=${1:-cuda:2}; TREE=${2:-scratch/_tmp/fh224_tree}; T=scratch/_tmp/s224_resume; rm -rf "$T"; mkdir -p "$T/ref" "$T/res"; L=logs/s224_resume.log; : > "$L"
base="python -B scratch/s222_det_wrap.py $TREE/boxes/vbank.py --mode slva --init_from boxes/runs_fsq/s213_d3t_s1/model.pt --steps 12000 --lr 3e-5 --cur 8 --c_sched 0,1 --draws_log_steps 20000 --far --fh --fh_dose 150,250 --far_n 4 --bs 16 --micro 1 --ckpt_every 200 --log_every 10 --fp32 --device $DV"
echo "$(date '+%F %T') REF" >> "$L"; PYTHONPATH=$TREE $base --out $T/ref --stop_at 300 >> "$L" 2>&1; echo "ref rc $?" >> "$L"
echo "$(date '+%F %T') KILL after 230" >> "$L"; PYTHONPATH=$TREE $base --out $T/res --stop_at 230 >> "$L" 2>&1; echo "kill rc $?" >> "$L"
cp $T/res/ckpt_last.pt $T/ckpt_200.pt
echo "$(date '+%F %T') RESUME" >> "$L"; PYTHONPATH=$TREE $base --out $T/res --stop_at 300 --resume $T/ckpt_200.pt >> "$L" 2>&1; echo "resume rc $?" >> "$L"
python - "$T" >> "$L" 2>&1 <<'EOF'
import json, sys
T = sys.argv[1]; rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]  # noqa: E731
ref = rd(T + "/ref/log.jsonl"); res = rd(T + "/res/log.jsonl")
i2 = [i for i, r in enumerate(res) if r.get("step") == 210 and "loss" in r][1]                # the resumed part starts at the second step-210 record
rsm = res[i2:]; strip = lambda r: {k: v for k, v in r.items() if k != "sec"}  # noqa: E731
refm = [r for r in ref if r.get("step", 0) >= 210]
ok_rec = [strip(a) for a in refm] == [strip(b) for b in rsm]
lam_ref = [r for r in ref if "fh_lambda_set" in r]; lam_res = [r for r in rsm if "fh_lambda_set" in r]
o = {"n_ref_after": len(refm), "n_resumed": len(rsm), "records_bitwise": ok_rec, "lambda_ref": lam_ref, "lambda_resumed": lam_res,
     "first_diff": next(([strip(a), strip(b)] for a, b in zip(refm, rsm) if strip(a) != strip(b)), None)}
o["pass"] = ok_rec and len(lam_ref) == 1 and lam_ref == lam_res and lam_ref[0]["step"] == 250 and lam_ref[0]["fh_n_ratios"] == 101
json.dump(o, open("scratch/gate_s224/k11.json", "w", newline="\n"), indent=1, default=str); print("K11 pass", o["pass"])
EOF
