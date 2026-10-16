#!/usr/bin/env bash
# §211 post-landing (registered eval_vb pass, notes 211-1 / 211-4 / 211-7 / 211-14): for each cell, after its gen8 drain reports ALL_DRAINED and model.pt +
# eval.json are local, run boxes/vbank_eval.py (fp32, eval_bs 32) on its own local card: b304 N2 -> cuda:0 (4060), b305 N1 -> cuda:2 (2070),
# b306 N3 -> cuda:1 (3070, after the §205 8k chain's CHAIN DONE). One job per card, never a co-tenant. Logs logs/s211_eval_vb_<run>.log; done-lines here.
#   nohup bash scratch/s211_post_landing.sh >> logs/s211_post_landing.out 2>&1 &
cd "$(dirname "$0")/.."
run_cell() {   # n run dev [wait_file]
  local n=$1 r=$2 d=$3 w=${4:-}
  until grep -q "ALL_DRAINED" logs/auto_drain_gen_b$n.out 2>/dev/null && [ -s boxes/runs_fsq/$r/model.pt ] && [ -s boxes/runs_fsq/$r/eval.json ]; do sleep 120; done
  if [ -n "$w" ]; then until grep -q "CHAIN DONE" "$w" 2>/dev/null; do sleep 120; done; fi
  echo "$(date '+%F %T') b$n $r drained -> eval_vb on $d"
  python -m boxes.vbank_eval --run boxes/runs_fsq/$r --device $d > logs/s211_eval_vb_$r.log 2>&1
  echo "$(date '+%F %T') b$n $r eval_vb rc $? $(tail -n 1 logs/s211_eval_vb_$r.log | cut -c1-160)"
}
run_cell 304 s211_vb_pa_s1 cuda:0 &
run_cell 305 s211_vb_carry_s1 cuda:2 &
run_cell 306 s211_vb_pamct_s1 cuda:1 logs/s205_8k_chain.out &
wait; echo "$(date '+%F %T') all three eval_vb done"
