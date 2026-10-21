#!/bin/bash
# Note 216-14 IDLE-HORIZON on the RTX 3070 (eval_vb's card, eval_bs 32: G-IH0 bitwise): after no python holds the 3070 (§219 T2's eval_text first),
# boxes.vbank_diag --part khorizon for the 6 models in order -> boxes/runs_fsq/<run>/eval_khorizon.json. rc is captured BEFORE any $(date).
# Audit 10-21 (G-D PASS conditions): (1) per model, eval_vb.json AND eval_hist24.json must both read card RTX 3070 / eval_bs 32 / fp32 (the run uses that
# card / bs); a mismatch skips the model and is logged (the gate could not be bitwise against both); (2) a summary line with k_max per model at the end.
# A model whose eval_khorizon.json already exists is skipped (rename it *.void_* to re-run).   nohup bash scratch/s216_khorizon_3070.sh >> logs/s216_khorizon.out 2>&1 &
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
U=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/3070/ {print $2}' | head -1); DV=cuda:1
busy() { nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader | grep "$U" | grep -qi python; }
RUNS="s213_d3t_s1 s215_ftctrl_s1 s215_ftself_s1 s216_r2_s1 s216_d3t_s2 s216_r1_s2"
echo "$(date '+%F %T') khorizon queue armed (3070 $U, $DV): waiting for the card"
while busy; do sleep 60; done; sleep 30; while busy; do sleep 60; done
for r in $RUNS; do
  [ -e boxes/runs_fsq/$r/eval_khorizon.json ] && { echo "$(date '+%F %T') $r eval_khorizon.json exists - skipped"; continue; }
  meta=$(python -c "import json, sys
r = sys.argv[1]; bad = []
for f in ('eval_vb.json', 'eval_hist24.json'):
    d = json.load(open('boxes/runs_fsq/%s/%s' % (r, f)))
    if '3070' not in str(d.get('card')) or d.get('eval_bs') != 32 or d.get('dtype', 'fp32') != 'fp32': bad.append('%s card %s bs %s dtype %s' % (f, d.get('card'), d.get('eval_bs'), d.get('dtype')))
print('OK' if not bad else 'BAD ' + '; '.join(bad))" $r)
  [ "$meta" = OK ] || { echo "$(date '+%F %T') $r SKIPPED: reference meta not 3070 / bs 32 / fp32 ($meta) - tell Theory / Audit"; continue; }
  python -B -m boxes.vbank_diag --run boxes/runs_fsq/$r --part khorizon --device $DV --eval_bs 32 > logs/s216_khorizon_$r.log 2>&1; rc=$?
  echo "$(date '+%F %T') $r khorizon rc $rc $(grep -E '^G-IH0' logs/s216_khorizon_$r.log | tail -1)"
done
python -c "import json, sys
rows = []
for r in sys.argv[1:]:
    try: d = json.load(open('boxes/runs_fsq/%s/eval_khorizon.json' % r)); rows.append('%s k_max %s G-IH0 bitwise %s min_share %s' % (r, d.get('k_max'), d['gate_IH0'].get('bitwise'), d['gate_IH0'].get('min_share_vs_hist24')))
    except Exception as e: rows.append('%s missing (%s)' % (r, e))
print('KHORIZON SUMMARY | ' + ' | '.join(rows))" $RUNS
echo "$(date '+%F %T') khorizon queue done"
