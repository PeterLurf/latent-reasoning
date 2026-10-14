#!/usr/bin/env bash
# §199 CSR seed landing (Theory / Audit 2026-10-14): when gen_b<q> has drained and <run>/eval.json exists: the path_indep probes at 4k / 8k / 12k / 16k
# (8k / 16k registered for P-A5; standard probe, loss_seen no), then note 199-A5's gradshare (scratch/s199_gradshare.py), both on <dev>; the
# ckpt_model_<step>.pt copies are deleted ONLY when all four probe JSONs and the gradshare JSON exist (retention); else they are kept and logged.
#   nohup bash scratch/s199_landing.sh <seed> <queue n> [dev cuda:1] > logs/s199_landing_s<seed>.out 2>&1 &
cd "$(dirname "$0")/.."
s=$1; q=$2; dev=${3:-cuda:1}; R=boxes/runs_fsq/s199_csr_s$s
until grep -q "gen_b$q " logs/auto_drain.log && [ -s $R/eval.json ]; do sleep 60; done
echo "$(date '+%F %T') s$s landed: probes + gradshare on $dev"
for st in 4000 8000 12000 16000; do
  [ -f $R/ckpt_model_$st.pt ] && python scratch/probe_pathindep.py $R $R/ckpt_model_$st.pt $dev --loss_seen no 2>&1 | grep -E "PATHINDEP|Error|Trace" || echo "MISSING or failed: $st"
done
python scratch/s199_gradshare.py s199_csr_s$s $dev --chunk 8 2>&1 | tail -n 6
if ls $R/pathindep_4000.json $R/pathindep_8000.json $R/pathindep_12000.json $R/pathindep_16000.json $R/eval_s199_gradshare.json > /dev/null 2>&1 && ! grep -q '"missing"' $R/eval_s199_gradshare.json; then
  rm -f $R/ckpt_model_4000.pt $R/ckpt_model_8000.pt $R/ckpt_model_12000.pt $R/ckpt_model_16000.pt; echo "ckpt_model copies deleted"
else echo "KEPT the ckpt_model copies (an output is missing)"; fi
echo "$(date '+%F %T') s$s landing done"
