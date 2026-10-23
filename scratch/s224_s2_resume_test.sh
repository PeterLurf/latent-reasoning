#!/bin/bash
# §224 Stage 2 resume test (Audit 10-22: off-host full-state mirrors with a TESTED resume at the 48k recipe, across the c-anneal window): the D3-T
# recipe from scratch (s213_d3t_s1 args: slva, 48k, lr 1e-4, cur 3,5,8, c_sched 1000,8000, kmax 12, kextra 4, seed 1) + --far --fh, test-only
# --fh_dose 1050,1150 (so the dose window straddles the checkpoint), bs 16 (8 GB), under scratch/s222_det_wrap.py (math SDP + deterministic, fp32 on
# the Turing 2070; a gate setting). REF = to step 1300 uninterrupted; KILL = stopped after 1250 (ckpt every 1100 -> step 1100, inside the anneal);
# RESUME = --resume <the kill's ckpt> to 1300 in the kill dir. PASS iff every record 1150..1300 (all keys but sec, clip_frac_100 known non-state
# per Audit's K11 ruling) and the lambda-set line equal REF's.   bash scratch/s224_s2_resume_test.sh [cuda dev]  -> scratch/gate_s224/s2_resume.json
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=D:/Latent
DV=${1:-cuda:2}; T=scratch/_tmp/s224_s2res; rm -rf "$T"; mkdir -p "$T/ref" "$T/res"; L=logs/s224_s2_resume.log; : > "$L"
base="python -B scratch/s222_det_wrap.py boxes/vbank.py --mode slva --table train --seed 1 --steps 48000 --lr 1e-4 --cur 3,5,8 --c_sched 1000,8000 --kmax 12 --kextra 4 --far --fh --fh_dose 1050,1150 --bs 16 --micro 1 --ckpt_every 1100 --log_every 50 --fp32 --device $DV"
echo "$(date '+%F %T') REF" >> "$L"; $base --out $T/ref --stop_at 1300 >> "$L" 2>&1; echo "ref rc $?" >> "$L"
echo "$(date '+%F %T') KILL after 1250" >> "$L"; $base --out $T/res --stop_at 1250 >> "$L" 2>&1; echo "kill rc $?" >> "$L"
cp $T/res/ckpt_last.pt $T/ckpt_1100.pt
echo "$(date '+%F %T') RESUME" >> "$L"; $base --out $T/res --stop_at 1300 --resume $T/ckpt_1100.pt >> "$L" 2>&1; echo "resume rc $?" >> "$L"
python - "$T" >> "$L" 2>&1 <<'EOF'
import json, sys
T = sys.argv[1]; rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]  # noqa: E731
ref = rd(T + "/ref/log.jsonl"); res = rd(T + "/res/log.jsonl")
ik = [i for i, r in enumerate(res) if r.get("step") == 1250 and "loss" in r][0]   # v2: the resumed segment starts after the KILL run's last record
rsm = [r for r in res[ik + 1:] if r.get("step", 0) >= 1150]; strip = lambda r: {k: v for k, v in r.items() if k not in ("sec", "clip_frac_100")}  # noqa: E731
refm = [r for r in ref if r.get("step", 0) >= 1150]
lam_ref = [r for r in ref if "fh_lambda_set" in r]; lam_res = [r for r in rsm if "fh_lambda_set" in r]
c = [r.get("c") for r in refm if "c" in r]
o = {"n_ref_after": len(refm), "n_resumed": len(rsm), "records_bitwise": [strip(a) for a in refm] == [strip(b) for b in rsm], "lambda_ref": lam_ref,
     "lambda_resumed": lam_res, "c_range_after_resume": [min(c), max(c)] if c else None, "first_diff": next(([strip(a), strip(b)] for a, b in zip(refm, rsm) if strip(a) != strip(b)), None)}
o["pass"] = o["records_bitwise"] and len(lam_ref) == 1 and lam_ref == lam_res and lam_ref[0]["step"] == 1150 and 0 < max(c) < 20
json.dump(o, open("scratch/gate_s224/s2_resume.json", "w", newline="\n"), indent=1, default=str); print("S2-RESUME pass", o["pass"])
EOF
