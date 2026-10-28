#!/usr/bin/env bash
# §238 DF references ($0, local 4060 = cuda:0, fp32, eval_bs 100; DF_v1 §8 + Note 238-1): eval-only on the DF sets. Unread.
#   s228_pdir2c_s1_df (acc) + s228_pdir2c_s1_df_bs50 (acc, eval_bs 50); step 0: rcb_plaw4_s1_df / rcb_plaw4_s2_df (acc, ws, opclass, swap_prod, cue_only), rcb_pcd4_s1_df (acc).
#   nohup bash scratch/s238_refs.sh > logs/s238_refs.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
run() { echo "$(date '+%F %T') start $*"; python -B scratch/s236_reg_eval.py --law 2 "$@"; rc=$?; echo "$(date '+%F %T') rc $rc ($*)"; return $rc; }
run --run s228_pdir2c_s1 --parts acc --eval_bs 100 --device cuda:0 || exit 3
run --run s228_pdir2c_s1 --parts acc --eval_bs 50 --device cuda:0 --out boxes/runs_fsq/s228_pdir2c_s1_df_bs50 || exit 3
run --run rcb_plaw4_s1 --parts acc,ws,opclass,swap_prod,cue_only --eval_bs 100 --device cuda:0 || exit 3
run --run rcb_plaw4_s2 --parts acc,ws,opclass,swap_prod,cue_only --eval_bs 100 --device cuda:0 || exit 3
run --run rcb_pcd4_s1 --parts acc --eval_bs 100 --device cuda:0 || exit 3
for d in s228_pdir2c_s1_df s228_pdir2c_s1_df_bs50 rcb_plaw4_s1_df rcb_plaw4_s2_df rcb_pcd4_s1_df; do rm -f boxes/runs_fsq/$d/model.pt; done
echo "$(date '+%F %T') REFS DONE"
