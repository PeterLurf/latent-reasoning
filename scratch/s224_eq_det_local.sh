#!/bin/bash
# §224 G-FH-EQ diagnosis (b331 ALERT 08:45 10-22): eqA (--far --far_w 0) vs eqB (no far) under scratch/s222_det_wrap.py (math SDP + deterministic)
# on the 2070 (fp32, Turing), FT recipe s1 at bs 16 (8 GB), 200 steps, log every 10: bitwise expected iff the far code path is inert at weight 0.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=D:/Latent; T=scratch/_tmp/s224_eqdet; rm -rf $T; mkdir -p $T
base="python -B scratch/s222_det_wrap.py boxes/vbank.py --mode slva --table train --seed 1 --steps 12000 --lr 3e-5 --cur 8 --c_sched 0,1 --kmax 12 --kextra 4 --init_from boxes/runs_fsq/s213_d3t_s1/model.pt --draws_log_steps 20000 --bs 16 --micro 1 --stop_at 200 --log_every 10 --device cuda:2"
$base --out $T/eqB > logs/s224_eqdet_B.log 2>&1; echo "eqB rc $?"
$base --far --far_w 0 --out $T/eqA > logs/s224_eqdet_A.log 2>&1; echo "eqA rc $?"
python - "$T" <<'PY'
import json, sys
T = sys.argv[1]; rd = lambda p: [{k: v for k, v in json.loads(l).items() if k != "sec" and not k.startswith(("far_", "fh_"))} for l in open(p) if l.strip() and "loss" in l]
A, B = rd(T + "/eqA/log.jsonl"), rd(T + "/eqB/log.jsonl"); print("records", len(A), len(B), "bitwise", A == B)
d = next(((a, b) for a, b in zip(A, B) if a != b), None); print("first diff", d and {k: [d[0].get(k), d[1].get(k)] for k in d[0] if d[0].get(k) != d[1].get(k)})
PY
