#!/usr/bin/env bash
# §128 (and §127 Stage 2) post-hoc on one local card, polling as runs are fetched: for every boxes/runs_fsq/s128_* / s127_* run dir
# with eval.json + model.pt (not *.void*): fp32 --contraction_only (registered for every run with eval.json) and, for the held-out
# families cyc / mod / rdfa, --cert_only (task-language CERT + n_moore: §128's CLOSED rule). One job at a time; args from the
# run's own args.json. Stops when scratch/posthoc128.stop exists.   bash scratch/posthoc128_local.sh cuda:1
cd "$(dirname "$0")/.."
dev=${1:-cuda:1}
while [ ! -f scratch/posthoc128.stop ]; do
  did=0
  for d in boxes/runs_fsq/s128_*_s* boxes/runs_fsq/s127_*_s* boxes/runs_fsq/s130chk_*; do
    case "$d" in *void*|*slots*) continue;; esac
    [ -s "$d/eval.json" ] && [ -s "$d/model.pt" ] || continue
    A="--from_args $d/args.json"                                            # the run's own configuration (Audit 2026-10-06)
    fam=$(python -c "import json; print(json.load(open('$d/args.json'))['family'])"); n=$(basename $d)
    if [ ! -s "$d/eval_contraction_fp32.json" ]; then
      python -u -m boxes.fsq120 $A --contraction_only --fp32 --device $dev --out $d > logs/c1_${n}_contraction_fp32.log 2>&1
      echo "$(date +%T) ${n} contraction_only --fp32 rc $? ($dev)" >> logs/c1_local.log; did=1
    fi
    if [ "$fam" = cyc ] || [ "$fam" = mod ] || [ "$fam" = rdfa ]; then
      if [ ! -s "$d/eval_cert.json" ]; then
        python -u -m boxes.fsq120 $A --cert_only --device $dev --out $d > logs/c1_${n}_cert.log 2>&1
        echo "$(date +%T) ${n} cert_only rc $? ($dev)" >> logs/c1_local.log; did=1
      fi
    fi
  done
  [ $did = 0 ] && sleep 300
done
echo "$(date +%T) POSTHOC128_STOPPED $dev" >> logs/c1_local.log
