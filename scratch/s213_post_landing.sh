#!/usr/bin/env bash
# §213 post-landing (note 213-1 eval contract): local fp32, eval_bs 32, ONE local card for both cells (DEV, default cuda:1 = the 3070), card recorded.
# Per cell, in landing order, after its gen8 drain reports ALL_DRAINED and model.pt + log.jsonl are local: vbank_eval (acc, probe, vswap, stall) ->
# vbank_diag steppos -> lfk -> rs -> d3hist; then the d3hist intact gate (acc_n2 n 1..12 == eval_vb acc_n2, bitwise). Logs logs/s213_<part>_<run>.log;
# done-lines here.   nohup bash scratch/s213_post_landing.sh >> logs/s213_post_landing.out 2>&1 &
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:1}; todo="307:s213_d3t_s1 308:s213_d3f_s1"
ready() { grep -q "ALL_DRAINED" logs/auto_drain_gen_b$1.out 2>/dev/null && [ -s boxes/runs_fsq/$2/model.pt ] && [ -s boxes/runs_fsq/$2/log.jsonl ]; }
while [ -n "$todo" ]; do
  for cell in $todo; do
    n=${cell%%:*}; r=${cell#*:}; R=boxes/runs_fsq/$r
    ready "$n" "$r" || continue
    echo "$(date '+%F %T') b$n $r drained -> evals on $DEV"
    python -m boxes.vbank_eval --run $R --device $DEV --eval_bs 32 > logs/s213_eval_vb_$r.log 2>&1; echo "$(date '+%F %T') b$n $r eval_vb rc $? $(tail -n 1 logs/s213_eval_vb_$r.log | cut -c1-160)"
    for P in steppos lfk rs d3hist; do
      python -m boxes.vbank_diag --run $R --part $P --device $DEV --eval_bs 32 > logs/s213_${P}_$r.log 2>&1; echo "$(date '+%F %T') b$n $r $P rc $? $(tail -n 1 logs/s213_${P}_$r.log | cut -c1-160)"
    done
    python -c "
import json; R = '$R'; v = json.load(open(R + '/eval_vb.json'))['acc_n2']; h = json.load(open(R + '/eval_d3hist.json'))['intact']['acc_n2']
print('GATE d3hist intact == eval_vb acc_n2 (n 1..12):', all(h[k] == v[k] for k in h))" 2>&1 | sed "s/^/$(date '+%F %T') b$n $r /"
    todo=$(echo "$todo" | tr ' ' '\n' | grep -v "^$cell$" | tr '\n' ' ' | sed 's/ *$//')
  done
  [ -n "$todo" ] && sleep 300
done
echo "$(date '+%F %T') §213 post-landing evals done"
