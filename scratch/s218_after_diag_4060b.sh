#!/bin/bash
# 4060 (cuda:0) after the running untrained diag process (Audit + Theory 10-21; replaces s218_after_diag_4060.sh, stopped by pid while waiting):
# (1) untrained K2-anti-lin x3.0 (218-22 m-rule input) -> g_l5_known/k2_anti_x3.json;
# (2) the 218-22 branch-(ii) REFERENCE chain on the original G-L1c card (4060, bf16): s218_l1c_save.py (G-L1c re-run + save) -> disjointness record ->
#     SAME-MODEL gate vs the committed g_l1c.json (s218_l1c_samegate.py; a mismatch blocks branch (ii): the K arms still run, descriptive) ->
#     k3anti lin x10 -> k3antixor x10 -> k2anti x1.5 / 2.0 / 3.0 -> descriptive k3anti / k3antixor x1, k3sym / k3symxor x1 / x10 (tags _ref, _ref_x1);
# (3) KL-DRIFT s1 trio (216-10 / 216-12, 4060-only). One job at a time; rc captured before any $(date).
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s218_ref_chain.out; K=logs/s216_kl_base.out
until grep -qE "^WROTE k3_diag|Traceback" logs/s218_gl5_diag.log; do sleep 60; done
sleep 20
echo "$(date '+%F %T') 218-17 diag process ended (k3_diag written or a traceback; see logs/s218_gl5_diag.log)" >> $K
python -B scratch/s218_gl5_known.py --device cuda:0 --arms k2anti --k2anti_mults 3.0 --tag _x3 > logs/s218_gl5_k2anti_x3.log 2>&1; rc=$?
echo "$(date '+%F %T') 218-17 k2anti x3 (untrained, 4060) rc $rc" >> $K
python -B scratch/s218_l1c_save.py --device cuda:0 > logs/s218_l1c_save.log 2>&1; rc=$?
echo "$(date '+%F %T') l1c_save (4060) rc $rc" >> $L
if [ $rc -eq 0 ] && [ -s scratch/gate_s218/g_l1c_model/model.pt ]; then
  python -B scratch/s218_l1c_disjoint.py > logs/s218_l1c_disjoint.log 2>&1; rc=$?; echo "$(date '+%F %T') l1c disjoint rc $rc" >> $L
  python -B scratch/s218_l1c_samegate.py > logs/s218_l1c_samegate.log 2>&1; rc=$?; echo "$(date '+%F %T') l1c same-model gate rc $rc: $(tail -n 1 logs/s218_l1c_samegate.log)" >> $L
  M="--device cuda:0 --model_run scratch/gate_s218/g_l1c_model"
  run() { local tag=$1; shift; python -B scratch/s218_gl5_known.py $M "$@" > logs/s218_ref_$tag.log 2>&1; local rc=$?; echo "$(date '+%F %T') ref $tag rc $rc" >> $L; }
  run anti_x10 --arms k3anti --k3_mults 10 --tag _ref
  run antixor_x10 --arms k3antixor --k3_mults 10 --tag _ref
  run k2anti --arms k2anti --k2anti_mults 1.5,2.0,3.0 --tag _ref
  run anti_x1 --arms k3anti,k3antixor --k3_mults 1 --tag _ref_x1
  run sym --arms k3sym,k3symxor --k3_mults 1,10 --tag _ref
  echo "$(date '+%F %T') §218 reference chain (4060) done" >> $L
fi
python -B scratch/s216_kl_base.py --device cuda:0 --runs s215_ftctrl_s1,s215_ftself_s1,s216_r2_s1 > logs/s216_kl_base.log 2>&1; rc=$?
echo "$(date '+%F %T') s216 kl_base (4060, s1) rc $rc" >> $K
