#!/bin/bash
# Note 218-22 route (ii) inputs + 218-17 (a): the TRAINED-REFERENCE chain, all on the 2070 (cuda:2, Theory 10-21: one card for the whole chain, recorded),
# after §220 CODI20 frees it. (1) s218_l1c_save.py: the G-L1c OVERFIT-256 re-run (items 7973da5b, recipe unchanged) -> scratch/gate_s218/g_l1c_model/
# (model.pt + args.json) + the disjointness record (s218_l1c_disjoint.py); (2) in Theory's order, on the saved model (--model_run):
#   k3anti lin x10 -> k3antixor x10 -> k2anti at x1.5 / x2 / x3 (the m-rule) -> descriptive: k3anti / k3antixor x1, k3sym / k3symxor x1 / x10.
# JSONs under g_l5_known/ with tag _ref (and _ref_x1 for the x1 arms). rc captured before any $(date).
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s218_ref_chain.out
until grep -q "s220 codi20 (2070) rc" logs/s220_stage0.out; do sleep 60; done
python -B scratch/s218_l1c_save.py --device cuda:2 > logs/s218_l1c_save.log 2>&1; rc=$?
echo "$(date '+%F %T') l1c_save (2070) rc $rc" >> $L
[ $rc -ne 0 ] && exit 1
[ -s scratch/gate_s218/g_l1c_model/model.pt ] || { echo "$(date '+%F %T') no model.pt - stop" >> $L; exit 1; }
python -B scratch/s218_l1c_disjoint.py > logs/s218_l1c_disjoint.log 2>&1; rc=$?
echo "$(date '+%F %T') l1c disjoint rc $rc" >> $L
M="--device cuda:2 --model_run scratch/gate_s218/g_l1c_model"
run() { local tag=$1; shift; python -B scratch/s218_gl5_known.py $M "$@" > logs/s218_ref_$tag.log 2>&1; local rc=$?; echo "$(date '+%F %T') ref $tag rc $rc" >> $L; }
run anti_x10 --arms k3anti --k3_mults 10 --tag _ref
run antixor_x10 --arms k3antixor --k3_mults 10 --tag _ref
run k2anti --arms k2anti --k2anti_mults 1.5,2.0,3.0 --tag _ref
run anti_x1 --arms k3anti,k3antixor --k3_mults 1 --tag _ref_x1
run sym --arms k3sym,k3symxor --k3_mults 1,10 --tag _ref
echo "$(date '+%F %T') §218 reference chain (2070) done" >> $L
