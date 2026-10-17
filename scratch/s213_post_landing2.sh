#!/usr/bin/env bash
# §213 post-landing, continued (replaces scratch/s213_post_landing.sh after D3-T's eval_vb probe OOM at 20:54 10-16; boxes/vbank_eval.py now empties the CUDA
# cache between parts). Same contract (note 213-1): local fp32, eval_bs 32, ONE card (cuda:1 = the 3070) for both cells. $1 = the Windows pid of D3-T's
# running steppos job (left running when the old script's bash was stopped): wait for it, then D3-T lfk -> rs -> d3hist -> eval_vb --parts probe,vswap,stall
# (merged into the existing eval_vb.json, whose acc part is complete) -> d3hist gate; then D3-F (b308) after its drain: eval_vb -> steppos -> lfk -> rs ->
# d3hist -> gate. Ends with the line scratch/s213_post_halt.sh waits for.   nohup bash scratch/s213_post_landing2.sh <pid> >> logs/s213_post_landing.out 2>&1 &
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:1}
while powershell -NoProfile -Command "if (Get-Process -Id $1 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 30; done
echo "$(date '+%F %T') b307 s213_d3t_s1 steppos done (pid $1): $(tail -n 1 logs/s213_steppos_s213_d3t_s1.log | cut -c1-160)"
gate() { python -c "
import json; R = 'boxes/runs_fsq/$1'; v = json.load(open(R + '/eval_vb.json'))['acc_n2']; h = json.load(open(R + '/eval_d3hist.json'))['intact']['acc_n2']
print('GATE d3hist intact == eval_vb acc_n2 (n 1..12):', all(h[k] == v[k] for k in h))" 2>&1 | sed "s/^/$(date '+%F %T') $2 $1 /"; }
diag() { for P in "$@"; do python -m boxes.vbank_diag --run boxes/runs_fsq/$r --part $P --device $DEV --eval_bs 32 > logs/s213_${P}_$r.log 2>&1; echo "$(date '+%F %T') b$n $r $P rc $? $(tail -n 1 logs/s213_${P}_$r.log | cut -c1-160)"; done; }
n=307; r=s213_d3t_s1
diag lfk rs d3hist
python -m boxes.vbank_eval --run boxes/runs_fsq/$r --device $DEV --eval_bs 32 --parts probe,vswap,stall > logs/s213_eval_vb2_$r.log 2>&1; echo "$(date '+%F %T') b$n $r eval_vb probe,vswap,stall rc $? $(tail -n 1 logs/s213_eval_vb2_$r.log | cut -c1-160)"
gate $r b$n
n=308; r=s213_d3f_s1
until grep -q "ALL_DRAINED" logs/auto_drain_gen_b$n.out 2>/dev/null && [ -s boxes/runs_fsq/$r/model.pt ] && [ -s boxes/runs_fsq/$r/log.jsonl ]; do sleep 120; done
echo "$(date '+%F %T') b$n $r drained -> evals on $DEV"
python -m boxes.vbank_eval --run boxes/runs_fsq/$r --device $DEV --eval_bs 32 > logs/s213_eval_vb_$r.log 2>&1; echo "$(date '+%F %T') b$n $r eval_vb rc $? $(tail -n 1 logs/s213_eval_vb_$r.log | cut -c1-160)"
diag steppos lfk rs d3hist
gate $r b$n
echo "$(date '+%F %T') §213 post-landing evals done"
