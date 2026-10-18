#!/usr/bin/env bash
# §216 (Audit G-S13 APPROVED d8d91599): merge T1 / E1-E4 from scratch/_tmp/sd216_tree into boxes/ AFTER §214's chain ("§214 stepbyt done"; its parts
# re-import vbank_diag per process), verify the merged files reproduce scratch/gate_s216/g_s13_boxes.diff exactly, commit; then G-S7 on the final tree
# (CPU, vs the pre-merge HEAD); then, once the s1 round-2 pool exists (pool_meta.json), on the 4060: G-S0 / G-S8 at Kb 20, G-S2 (CPU), and G-S17 part 2
# (the trainer's pool load + deep-shape auto-micro on the real 444k pool: --mem_check --mem_check_self --mem_check_steps 2, mem_budget 7.0, timed).
#   nohup bash scratch/s216_merge_and_gates.sh >> logs/s216_merge.out 2>&1 &
cd "$(dirname "$0")/.."
T=scratch/_tmp/sd216_tree/boxes; say() { echo "$(date '+%F %T') $*"; }
until grep -q "§214 stepbyt done" logs/s214_post_landing.out 2>/dev/null; do sleep 120; done
git pull -q; PRE=$(git rev-parse HEAD)
for f in vbank.py vbank_diag.py; do sed 's#^ROOT = "D:/Latent"$#ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#' $T/$f > boxes/$f.new && python -m py_compile boxes/$f.new && mv boxes/$f.new boxes/$f || { say "MERGE FAILED $f"; exit 1; }; done
for f in vbank.py vbank_diag.py; do diff -u <(git show HEAD:boxes/$f) boxes/$f --label a/boxes/$f --label b/boxes/$f; done > scratch/_tmp/merge_check.diff
if cmp -s scratch/_tmp/merge_check.diff scratch/gate_s216/g_s13_boxes.diff; then say "merge == reviewed diff"; else say "MERGE DIFF MISMATCH - reverting"; git checkout -- boxes/vbank.py boxes/vbank_diag.py; exit 1; fi
git commit -q -m "§216 T1 + E1-E4 merged into boxes/ (Audit G-S13 APPROVED d8d91599; identical to scratch/gate_s216/g_s13_boxes.diff; after §214's chain)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- boxes/vbank.py boxes/vbank_diag.py && git push -q; say "merged $(git rev-parse --short HEAD)"
P=scratch/_tmp/pre_tree216; rm -rf $P; mkdir -p $P/boxes && cp boxes/*.py $P/boxes/ && rm -f $P/boxes/_*tmp*.py && git show $PRE:boxes/vbank.py > $P/boxes/vbank.py && git show $PRE:boxes/vbank_diag.py > $P/boxes/vbank_diag.py
(cd $P/boxes && grep -l "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))" *.py | xargs sed -i 's#os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#"D:/Latent"#')
OMP_NUM_THREADS=4 python scratch/gate_s215.py s7 --device cpu --tree D:/Latent --pre D:/Latent/$P > logs/s216_gs7_final.log 2>&1; rc=$?
cp scratch/gate_s215/s7.json scratch/gate_s216/s7_final.json; git checkout -- scratch/gate_s215/s7.json; say "G-S7 final rc $rc $(tail -n 1 logs/s216_gs7_final.log | cut -c1-12)"
until [ -f boxes/runs_fsq/s216_pool_s1/pool_meta.json ]; do sleep 120; done
for g in s0 s8; do python scratch/gate_s216.py $g --device cuda:0 > logs/s216_g$g.log 2>&1; say "G-${g^^} rc $? $(tail -n 1 logs/s216_g$g.log | cut -c1-12)"; done
python scratch/gate_s216.py s2 --device cpu > logs/s216_gs2.log 2>&1; say "G-S2 rc $? $(tail -n 1 logs/s216_gs2.log | cut -c1-12)"
O=D:/Latent/scratch/_tmp/s216_gs17; rm -rf $O; mkdir -p $O; t0=$(date +%s)
python -m boxes.vbank --mode slva --table train --seed 1 --steps 12000 --lr 3e-5 --cur 8 --c_sched 0,1 --kmax 12 --kextra 4 --init_from boxes/runs_fsq/s215_ftself_s1/model.pt \
  --draws_log_steps 20000 --self_pool boxes/runs_fsq/s216_pool_s1/pool_r2.jsonl --p_self 0.5 --self_kmax 20 --self_kextra 4 --mem_check --mem_check_self --mem_check_steps 2 \
  --micro 8 --self_micro 8 --mem_budget 7.0 --eval_bs 8 --device cuda:0 --out $O > $O/run.log 2>&1; rc=$?
python -c "
import json; o = {'rc': $rc, 'sec': $(( $(date +%s) - t0 )), 'log_tail': open('$O/run.log').read().splitlines()[-4:], 'note': '4060, micro 8 (8 GB card); the deep-shape selection + pool load at full 444k'}
json.dump(o, open('scratch/gate_s216/gs17_part2.json', 'w', newline='\n'), indent=1); print(o)" | sed "s/^/$(date '+%F %T') G-S17 part 2 /"
rm -f $O/*.pt
say "§216 final-tree gates done"
